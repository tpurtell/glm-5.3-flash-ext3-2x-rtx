#!/usr/bin/env python3
"""Benchmark B12x PCIe DCP MLA collectives against vLLM's AG/RS recipe.

Run this inside the image with both GPUs visible. The shapes are GLM-5.3
Flash's TP2/DCP2 decode shapes: thirty-two local query heads, sixty-four
global attention heads, and 512-wide query/latent outputs.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import socket
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from b12x.comm.pcie.pcie_dcp_a2a import PCIeDCPA2APool
from vllm.v1.attention.ops.common import cp_lse_ag_out_rs


TOTAL_HEADS = 64
QUERY_HEAD_DIM = 512
OUTPUT_HEAD_DIM = 512


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class _TorchGroup:
    """Small GroupCoordinator-compatible wrapper used by vLLM's combine."""

    device_group: dist.ProcessGroup
    world_size: int
    rank_in_group: int

    def all_gather(self, value: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if dim == 0:
            output = torch.empty(
                (self.world_size * value.shape[0], *value.shape[1:]),
                dtype=value.dtype,
                device=value.device,
            )
            dist.all_gather_into_tensor(output, value, group=self.device_group)
            return output
        if dim == 1 and value.ndim == 3:
            batch, local_heads, head_dim = value.shape
            rank_major = torch.empty(
                (self.world_size * batch, local_heads, head_dim),
                dtype=value.dtype,
                device=value.device,
            )
            dist.all_gather_into_tensor(rank_major, value, group=self.device_group)
            return (
                rank_major.view(self.world_size, batch, local_heads, head_dim)
                .permute(1, 0, 2, 3)
                .reshape(batch, self.world_size * local_heads, head_dim)
                .contiguous()
            )
        raise ValueError(f"unsupported all-gather dim/shape: {dim}/{value.shape}")

    def reduce_scatter(self, value: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if dim != 1 or value.ndim != 3:
            raise ValueError(f"unsupported reduce-scatter dim/shape: {dim}/{value.shape}")
        batch, heads, head_dim = value.shape
        local_heads = heads // self.world_size
        rank_major = (
            value.view(batch, self.world_size, local_heads, head_dim)
            .permute(1, 0, 2, 3)
            .contiguous()
        )
        output = torch.empty(
            (batch, local_heads, head_dim), dtype=value.dtype, device=value.device
        )
        dist.reduce_scatter_tensor(output, rank_major, group=self.device_group)
        return output


def _time_eager(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / repeats


def _capture(fn, capture_context=None) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    stream.synchronize()
    dist.barrier()
    if capture_context is None:
        with torch.cuda.graph(graph, stream=stream):
            fn()
    else:
        with capture_context(stream), torch.cuda.graph(graph, stream=stream):
            fn()
    stream.synchronize()
    dist.barrier()
    return graph


def _time_graph(graph: torch.cuda.CUDAGraph, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / repeats


def _worker(
    rank: int,
    world_size: int,
    port: int,
    batches: tuple[int, ...],
    warmup: int,
    repeats: int,
    output: str | None,
) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    group = _TorchGroup(dist.group.WORLD, world_size, rank)
    pool = PCIeDCPA2APool.from_process_group(
        process_group=dist.group.WORLD,
        device=device,
        max_batch_size=max(batches),
        total_heads=TOTAL_HEADS,
        head_dim=OUTPUT_HEAD_DIM,
        query_head_dim=QUERY_HEAD_DIM,
        single_channel=True,
    )
    pool.prepare_graph_all_gather_heads()
    pool.prepare_graph_lse_reduce_scatter(dtype=torch.bfloat16)

    if rank == 0:
        print(
            "batch,op,nccl_eager_us,b12x_eager_us,eager_speedup,"
            "nccl_graph_us,b12x_graph_us,graph_speedup",
            flush=True,
        )

    results = []
    for batch in batches:
        generator = torch.Generator(device=device).manual_seed(41000 + batch + rank)
        partial_output = torch.randn(
            batch,
            TOTAL_HEADS,
            OUTPUT_HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        partial_lse = torch.randn(
            batch,
            TOTAL_HEADS,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )

        nccl_combine = lambda: cp_lse_ag_out_rs(
            partial_output,
            partial_lse,
            cp_group=group,
            is_lse_base_on_e=True,
        )
        b12x_combine = lambda: pool.lse_reduce_scatter(
            partial_output, partial_lse, is_lse_base_on_e=True
        )

        torch.testing.assert_close(
            b12x_combine(), nccl_combine(), rtol=2e-2, atol=2e-2
        )
        torch.cuda.synchronize()

        operations = []
        for query_dtype, dtype_name in (
            (torch.bfloat16, "bf16"),
            (torch.float8_e4m3fn, "fp8"),
        ):
            local_query = torch.randn(
                batch,
                TOTAL_HEADS // world_size,
                QUERY_HEAD_DIM,
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            ).to(query_dtype)
            nccl_query = lambda local_query=local_query: group.all_gather(
                local_query, dim=1
            )
            b12x_query = lambda local_query=local_query: pool.all_gather_heads(
                local_query
            )
            b12x_result = b12x_query()
            nccl_result = nccl_query()
            if query_dtype == torch.float8_e4m3fn:
                torch.testing.assert_close(
                    b12x_result.view(torch.uint8),
                    nccl_result.view(torch.uint8),
                    rtol=0,
                    atol=0,
                )
            else:
                torch.testing.assert_close(b12x_result, nccl_result, rtol=0, atol=0)
            operations.append(
                (f"query_gather_{dtype_name}", nccl_query, b12x_query)
            )
        operations.append(("lse_combine_bf16", nccl_combine, b12x_combine))

        for name, nccl_fn, b12x_fn in operations:
            nccl_eager = _time_eager(nccl_fn, warmup, repeats)
            b12x_eager = _time_eager(b12x_fn, warmup, repeats)
            nccl_graph = _capture(nccl_fn)
            b12x_graph = _capture(b12x_fn, pool.capture)
            nccl_graph_us = _time_graph(nccl_graph, warmup, repeats)
            b12x_graph_us = _time_graph(b12x_graph, warmup, repeats)
            if rank == 0:
                print(
                    f"{batch},{name},{nccl_eager:.3f},{b12x_eager:.3f},"
                    f"{nccl_eager / b12x_eager:.3f},"
                    f"{nccl_graph_us:.3f},{b12x_graph_us:.3f},"
                    f"{nccl_graph_us / b12x_graph_us:.3f}",
                    flush=True,
                )
                results.append(
                    {
                        "batch": batch,
                        "operation": name,
                        "nccl_eager_us": nccl_eager,
                        "b12x_eager_us": b12x_eager,
                        "eager_speedup": nccl_eager / b12x_eager,
                        "nccl_graph_us": nccl_graph_us,
                        "b12x_graph_us": b12x_graph_us,
                        "graph_speedup": nccl_graph_us / b12x_graph_us,
                    }
                )
        dist.barrier()

    if rank == 0 and output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema": "glm53-b12x-dcp-a2a.v1",
                    "total_heads": TOTAL_HEADS,
                    "query_head_dim": QUERY_HEAD_DIM,
                    "output_head_dim": OUTPUT_HEAD_DIM,
                    "warmup": warmup,
                    "repeats": repeats,
                    "results": results,
                },
                indent=2,
            )
            + "\n"
        )

    del nccl_graph, b12x_graph
    gc.collect()
    torch.cuda.synchronize()
    dist.barrier()
    pool.close()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="1,2,4,8,16,24,32")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    batches = tuple(int(value) for value in args.batches.split(","))
    if not batches or min(batches) <= 0:
        raise SystemExit("--batches must contain positive integers")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    mp.spawn(
        _worker,
        args=(
            2,
            _free_port(),
            batches,
            args.warmup,
            args.repeats,
            str(args.output) if args.output is not None else None,
        ),
        nprocs=2,
        join=True,
    )


if __name__ == "__main__":
    main()
