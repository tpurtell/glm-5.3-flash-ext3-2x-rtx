#!/usr/bin/env python3
"""Exercise DFlash2's anchor/mask query contract on the GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
    _prepare_dflash_inputs_kernel,
)
from vllm.v1.worker.gpu.spec_decode.utils import get_parallel_drafting_token_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.draft_config.read_text(encoding="utf-8"))
    dflash_config = config.get("dflash_config") or {}
    mask_token_id = get_parallel_drafting_token_id(
        SimpleNamespace(dflash_config=dflash_config)
    )
    num_speculative_steps = 5
    num_query_per_req = num_speculative_steps + 1
    max_num_reqs = 4
    max_num_tokens = 32
    block_size = 128
    device = torch.device("cuda")

    def tensor(values: list[int], dtype: torch.dtype = torch.int32) -> torch.Tensor:
        return torch.tensor(values, dtype=dtype, device=device)

    # Request 0 is decode after one rejected target token. Request 1 is
    # chunked prefill, so its anchor must come from next_prefill_tokens.
    target_positions = tensor([10, 11, 12, 50, 51, 52, 53], torch.int64)
    target_query_start_loc = tensor([0, 3, 7])
    idx_mapping = tensor([2, 0])
    num_sampled = tensor([1, 0])
    num_rejected = tensor([1, 0])
    last_sampled = tensor([0, 0, 111, 0], torch.int64)
    next_prefill_tokens = tensor([222, 0, 0, 0], torch.int64)
    input_temperature = torch.tensor(
        [0.0, 0.1, 0.2, 0.3], dtype=torch.float32, device=device
    )
    input_seeds = tensor([1000, 1001, 1002, 1003], torch.int64)
    block_table = tensor([100, 101, 200, 201]).view(2, 2)

    out_input_ids = torch.full(
        (max_num_tokens,), -999, dtype=torch.int32, device=device
    )
    out_query_positions = torch.full_like(out_input_ids, -999, dtype=torch.int64)
    out_query_start_loc = torch.full(
        (max_num_reqs + 1,), -999, dtype=torch.int32, device=device
    )
    out_seq_lens = torch.full(
        (max_num_reqs,), -999, dtype=torch.int32, device=device
    )
    out_query_slot_mapping = torch.full(
        (max_num_tokens,), -999, dtype=torch.int64, device=device
    )
    out_context_positions = torch.full_like(target_positions, -999)
    out_context_slot_mapping = torch.full_like(target_positions, -999)
    out_sample_indices = torch.full(
        (max_num_reqs * num_speculative_steps,),
        -999,
        dtype=torch.int64,
        device=device,
    )
    out_sample_pos = torch.full_like(out_sample_indices, -999)
    out_sample_idx_mapping = torch.full(
        (max_num_reqs * num_speculative_steps,),
        -999,
        dtype=torch.int32,
        device=device,
    )
    out_temperature = torch.full(
        (max_num_reqs,), -999, dtype=torch.float32, device=device
    )
    out_seeds = torch.full(
        (max_num_reqs,), -999, dtype=torch.int64, device=device
    )

    _prepare_dflash_inputs_kernel[(2, 1)](
        out_input_ids,
        out_query_positions,
        out_query_start_loc,
        out_seq_lens,
        out_query_slot_mapping,
        out_context_positions,
        out_context_slot_mapping,
        out_sample_indices,
        out_sample_pos,
        out_sample_idx_mapping,
        out_temperature,
        out_seeds,
        target_positions,
        target_query_start_loc,
        idx_mapping,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        num_rejected,
        input_temperature,
        input_seeds,
        block_table,
        block_table.stride(0),
        mask_token_id,
        block_size,
        num_query_per_req,
        num_speculative_steps,
        max_num_reqs,
        max_num_tokens,
        1_048_576,
        SAMPLE_FROM_ANCHOR=False,
        PAD_SLOT_ID=PAD_SLOT_ID,
        BLOCK_SIZE=16,
        num_warps=1,
    )
    torch.cuda.synchronize()

    query_ids = out_input_ids[:12].cpu().tolist()
    query_positions = out_query_positions[:12].cpu().tolist()
    sample_indices = out_sample_indices[:10].cpu().tolist()
    sample_positions = out_sample_pos[:10].cpu().tolist()
    sample_mapping = out_sample_idx_mapping[:10].cpu().tolist()
    padded_mapping = out_sample_idx_mapping[10:].cpu().tolist()
    padded_slots = out_query_slot_mapping[12:].cpu().tolist()

    checks = {
        "config_resolves_mask_token": mask_token_id
        == int(dflash_config["mask_token_id"]),
        "decode_anchor_uses_last_sampled": query_ids[0] == 111,
        "prefill_anchor_uses_next_token": query_ids[6] == 222,
        "five_masks_follow_each_anchor": query_ids
        == [111, *([mask_token_id] * 5), 222, *([mask_token_id] * 5)],
        "rejection_rolls_back_anchor_position": query_positions[:6]
        == [12, 13, 14, 15, 16, 17],
        "prefill_positions_follow_context": query_positions[6:]
        == [54, 55, 56, 57, 58, 59],
        "only_masks_are_sampled": sample_indices
        == [1, 2, 3, 4, 5, 7, 8, 9, 10, 11],
        "mask_samples_use_own_positions": sample_positions
        == [13, 14, 15, 16, 17, 55, 56, 57, 58, 59],
        "request_state_mapping_is_preserved": sample_mapping == [2] * 5 + [0] * 5,
        "padded_samples_are_inert": padded_mapping == [-1] * 10,
        "padded_query_slots_are_inert": padded_slots == [PAD_SLOT_ID] * 20,
    }
    report = {
        "schema": "glm53-dflash2-mask-semantics.v1",
        "draft_model": str(args.draft_config.parent),
        "mask_token_id": mask_token_id,
        "num_speculative_tokens": num_speculative_steps,
        "sample_from_anchor": False,
        "query_ids": query_ids,
        "query_positions": query_positions,
        "sample_indices": sample_indices,
        "sample_positions": sample_positions,
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
