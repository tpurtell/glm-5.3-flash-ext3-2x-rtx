#!/usr/bin/env python3
"""Compare the deployed NCCL DCP owner shuffle with B12x PCIe staging."""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

from b12x.comm.pcie import DcpTopKOwnerExchange
from vllm.model_executor.layers.sparse_attn_indexer import (
    _unpack_b12x_dcp_gathered_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=(64, 512, 4096))
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    return parser.parse_args()


def timed_ms(fn, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    dist.barrier()
    torch.cuda.synchronize()
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    local_ms = torch.tensor(
        [start.elapsed_time(end) / iterations],
        dtype=torch.float64,
        device="cuda",
    )
    dist.all_reduce(local_ms, op=dist.ReduceOp.MAX)
    return float(local_ms.item())


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 2:
        raise RuntimeError(f"this deployed-topology benchmark requires world=2, got {world}")

    max_rows = max(args.rows)
    if any(rows <= 0 or rows % world for rows in args.rows):
        raise ValueError("every row count must be positive and divisible by world size")
    owner = DcpTopKOwnerExchange.from_process_group(
        process_group=dist.group.WORLD,
        device=torch.device("cuda", local_rank),
        max_rows=max_rows,
        topk=args.topk,
    )
    try:
        if rank == 0:
            print("rows,nccl_a2a_unpack_ms,b12x_owner_stage_ms,speedup")
        for rows in args.rows:
            owner_rows = rows // world
            width = world * args.topk
            base = torch.arange(rows * args.topk, device="cuda", dtype=torch.int32)
            local_indices = (base.reshape(rows, args.topk) + rank * 10_000_000).contiguous()
            score_bits = (
                torch.arange(rows * args.topk, device="cuda", dtype=torch.int32)
                + 0x3E800000
                + rank * 4096
            ).reshape(rows, args.topk)
            local_scores = score_bits.view(torch.float32).contiguous()

            send = torch.empty((rows, 2, args.topk), device="cuda", dtype=torch.int32)
            received = torch.empty_like(send)
            candidate_indices = torch.empty(
                (owner_rows, width), device="cuda", dtype=torch.int32
            )
            candidate_score_bits = torch.empty_like(candidate_indices)

            def nccl_shuffle() -> None:
                send[:, 0, :].copy_(local_indices)
                send[:, 1, :].copy_(local_scores.view(torch.int32))
                dist.all_to_all_single(received.view(-1), send.view(-1))
                _unpack_b12x_dcp_gathered_candidates(
                    received,
                    candidate_indices,
                    candidate_score_bits,
                    dcp_world_size=world,
                    topk_tokens=args.topk,
                )

            staged: tuple[torch.Tensor, torch.Tensor] | None = None

            def b12x_shuffle() -> None:
                nonlocal staged
                staged = owner.stage_candidates(local_indices, local_scores)

            for _ in range(args.warmup):
                nccl_shuffle()
                b12x_shuffle()
            torch.cuda.synchronize()
            assert staged is not None
            nccl_shuffle()
            b12x_shuffle()
            torch.cuda.synchronize()
            torch.testing.assert_close(staged[0], candidate_indices, rtol=0, atol=0)
            assert torch.equal(staged[1].view(torch.int32), candidate_score_bits)

            nccl_ms = timed_ms(nccl_shuffle, args.iterations)
            b12x_ms = timed_ms(b12x_shuffle, args.iterations)
            if rank == 0:
                print(f"{rows},{nccl_ms:.4f},{b12x_ms:.4f},{nccl_ms / b12x_ms:.3f}")
    finally:
        owner.close_coordinated()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
