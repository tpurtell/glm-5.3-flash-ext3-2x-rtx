#!/usr/bin/env python3
"""Measure lossless B12x one-shot against vLLM PyNCCL on local TP2."""

from __future__ import annotations

import argparse
import json
import torch
import torch.distributed as dist

from b12x.comm.pcie import OneshotAllReducePool
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator


def timed_ms(fn, iterations: int) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=6144)
    parser.add_argument("--max-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--rows", default="1,2,4,8,10,16,32,64,128,256,512")
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(__import__("os").environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    cpu_group = dist.new_group(backend="gloo")
    pynccl = PyNcclCommunicator(group=cpu_group, device=device)
    pool = OneshotAllReducePool.from_exchange_group(
        exchange_group=dist.group.WORLD,
        device=device,
        eager_buffer_bytes=args.max_bytes,
        max_size=args.max_bytes,
        single_channel=True,
        max_concurrent_channels=1,
    )
    pool.for_stream()

    results = []
    for rows in (int(item) for item in args.rows.split(",")):
        inp = torch.full(
            (rows, args.hidden_size), rank + 1, dtype=torch.bfloat16, device=device
        )
        expected = float(dist.get_world_size() * (dist.get_world_size() + 1) // 2)
        b12x_out = pool.all_reduce(inp)
        nccl_out = pynccl.all_reduce(inp)
        torch.cuda.synchronize()
        if not torch.all(b12x_out == expected) or not torch.all(nccl_out == expected):
            raise RuntimeError(f"all-reduce correctness failed for rows={rows}")

        for _ in range(5):
            pool.all_reduce(inp)
            pynccl.all_reduce(inp)
        iterations = 200 if inp.nbytes <= 1024 * 1024 else 50
        b12x_ms = timed_ms(lambda: pool.all_reduce(inp), iterations)
        nccl_ms = timed_ms(lambda: pynccl.all_reduce(inp), iterations)
        local = torch.tensor([b12x_ms, nccl_ms], dtype=torch.float64)
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local.tolist(), group=cpu_group)
        if rank == 0:
            b12x_worst = max(item[0] for item in gathered)
            nccl_worst = max(item[1] for item in gathered)
            results.append(
                {
                    "rows": rows,
                    "bytes": inp.nbytes,
                    "b12x_ms": round(b12x_worst, 6),
                    "pynccl_ms": round(nccl_worst, 6),
                    "speedup": round(nccl_worst / b12x_worst, 3),
                }
            )

    # Validate the deployment's single-channel policy across independently
    # captured graphs, and compare actual replay latency (the serving path).
    graph_results = []
    graphs = []
    capture_stream = torch.cuda.Stream(device=device)
    for rows in (1, 10, 32, 64):
        static_in = torch.full(
            (rows, args.hidden_size), rank + 1, dtype=torch.bfloat16, device=device
        )
        b12x_graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(capture_stream), pool.capture():
            b12x_graph.capture_begin()
            b12x_out = pool.all_reduce(static_in)
            b12x_graph.capture_end()

        nccl_graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(capture_stream):
            nccl_graph.capture_begin()
            nccl_out = pynccl.all_reduce(static_in)
            nccl_graph.capture_end()

        for _ in range(10):
            b12x_graph.replay()
            nccl_graph.replay()
        torch.cuda.synchronize()
        if not torch.all(b12x_out == expected) or not torch.all(nccl_out == expected):
            raise RuntimeError("CUDA graph replay correctness failed")
        iterations = 1000
        b12x_graph_ms = timed_ms(b12x_graph.replay, iterations)
        nccl_graph_ms = timed_ms(nccl_graph.replay, iterations)
        local = torch.tensor([b12x_graph_ms, nccl_graph_ms], dtype=torch.float64)
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local.tolist(), group=cpu_group)
        if rank == 0:
            b12x_worst = max(item[0] for item in gathered)
            nccl_worst = max(item[1] for item in gathered)
            graph_results.append(
                {
                    "rows": rows,
                    "bytes": static_in.nbytes,
                    "b12x_ms": round(b12x_worst, 6),
                    "pynccl_ms": round(nccl_worst, 6),
                    "speedup": round(nccl_worst / b12x_worst, 3),
                }
            )
        graphs.extend((b12x_graph, nccl_graph))

    if rank == 0:
        print(
            json.dumps(
                {
                    "eager": results,
                    "cuda_graph": graph_results,
                    "alternating_graphs": "pass",
                },
                indent=2,
            )
        )

    pool.close()
    pynccl.destroy()
    dist.destroy_process_group(cpu_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
