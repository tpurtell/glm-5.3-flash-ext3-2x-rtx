#!/usr/bin/env python3
"""GPU oracle for the fused k-pool score gather plus DCP id remap."""

import torch

from vllm.v1.attention.backends.mla.b12x_dcp_topk import (
    triton_gather_dcp_topk_scores_and_globalize,
)


def main() -> None:
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    rows, topk, width = 7, 512, 2048
    logits = torch.arange(rows * width, device=device, dtype=torch.float32).reshape(
        rows, width
    )
    row_starts = torch.arange(rows, device=device, dtype=torch.int32) * 3
    local = torch.arange(topk, device=device, dtype=torch.int32).expand(rows, -1).clone()
    local[2, -5:] = -1
    original = local.clone()
    scores = torch.empty((rows, topk), device=device, dtype=torch.float32)

    triton_gather_dcp_topk_scores_and_globalize(
        local,
        logits,
        scores,
        dcp_world_size=2,
        dcp_rank=1,
        cp_kv_cache_interleave_size=1,
        row_starts=row_starts,
    )
    expected_ids = torch.where(original >= 0, original * 2 + 1, -1)
    score_positions = torch.clamp(original, min=0).to(torch.int64)
    score_positions += row_starts[:, None]
    expected_scores = torch.gather(logits, 1, score_positions)
    expected_scores.masked_fill_(original < 0, -float("inf"))
    torch.testing.assert_close(local, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(scores, expected_scores, rtol=0, atol=0)
    print("B12x k-pool DCP fused score gather/global-id oracle passed")


if __name__ == "__main__":
    main()
