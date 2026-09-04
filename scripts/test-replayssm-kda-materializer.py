#!/usr/bin/env python3
"""Numerically qualify KDA ReplaySSM prefix materialization on CUDA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from vllm.third_party.flash_linear_attention.ops.kda_replayssm_spec_decode import (
    materialize_kda_replayssm_state,
)


def reference_materialize(
    checkpoint: torch.Tensor,
    d_cache: torch.Tensor,
    k_cache: torch.Tensor,
    g_cache: torch.Tensor,
    source_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    row_write_pos: torch.Tensor,
    row_cache_base: torch.Tensor,
) -> torch.Tensor:
    rows = []
    ring_len = d_cache.shape[2]
    for row in range(source_indices.shape[0]):
        source = int(source_indices[row])
        initial = bool(has_initial_state[row]) and source > 0
        if not initial:
            rows.append(torch.zeros_like(checkpoint[0], dtype=torch.float32))
            continue
        state = checkpoint[source].float()
        write_pos = int(row_write_pos[row])
        cache_base = int(row_cache_base[row])
        for offset in range(write_pos):
            physical = (cache_base + offset) & (ring_len - 1)
            decay = torch.exp(g_cache[source, :, physical, :].float())
            delta = d_cache[source, :, physical, :].float()
            key = k_cache[source, :, physical, :].float()
            state = state * decay[:, None, :] + delta[:, :, None] * key[:, None, :]
        rows.append(state)
    return torch.stack(rows).to(checkpoint.dtype)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(739_184)
    device = torch.device("cuda:0")
    slots, rows, heads, ring_len, value_dim, key_dim = 8, 5, 2, 16, 8, 16
    checkpoint = (
        torch.randn(slots, heads, value_dim, key_dim, device=device) * 0.125
    ).to(torch.bfloat16)
    d_cache = (
        torch.randn(slots, heads, ring_len, value_dim, device=device) * 0.0625
    ).to(torch.bfloat16)
    k_cache = (
        torch.randn(slots, heads, ring_len, key_dim, device=device) * 0.0625
    ).to(torch.bfloat16)
    # Negative gates keep a full-ring replay numerically well conditioned.
    g_cache = (
        -torch.rand(slots, heads, ring_len, key_dim, device=device) * 0.03125
    ).to(torch.bfloat16)

    metadata = torch.zeros(rows, 8, dtype=torch.int32, device=device)
    source_indices = metadata[:, 1]
    has_initial_state = metadata[:, 3]
    row_write_pos = metadata[:, 5]
    row_cache_base = metadata[:, 7]
    source_indices.copy_(torch.tensor([1, 3, 0, 5, 7], device=device))
    has_initial_state.copy_(torch.tensor([1, 1, 0, 1, 1], device=device))
    row_write_pos.copy_(torch.tensor([0, 4, 9, 16, 7], device=device))
    row_cache_base.copy_(torch.tensor([0, 13, 2, 5, 15], device=device))
    metadata_views = (
        source_indices,
        has_initial_state,
        row_write_pos,
        row_cache_base,
    )
    if not all(tensor.stride(0) == 8 for tensor in metadata_views):
        raise AssertionError("test metadata did not preserve a non-contiguous stride")

    expected = reference_materialize(
        checkpoint,
        d_cache,
        k_cache,
        g_cache,
        source_indices,
        has_initial_state,
        row_write_pos,
        row_cache_base,
    )
    actual = torch.empty_like(expected)
    materialize_kda_replayssm_state(
        checkpoint,
        d_cache,
        k_cache,
        g_cache,
        actual,
        source_indices,
        has_initial_state,
        row_write_pos,
        row_cache_base,
    )
    torch.cuda.synchronize(device)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)

    contiguous = torch.empty_like(expected)
    materialize_kda_replayssm_state(
        checkpoint,
        d_cache,
        k_cache,
        g_cache,
        contiguous,
        source_indices.contiguous(),
        has_initial_state.contiguous(),
        row_write_pos.contiguous(),
        row_cache_base.contiguous(),
    )
    torch.cuda.synchronize(device)
    torch.testing.assert_close(actual, contiguous, rtol=0, atol=0)
    if torch.count_nonzero(actual[2]).item() != 0:
        raise AssertionError("null/no-initial-state row was not zeroed")

    error = (actual.float() - expected.float()).abs()
    report = {
        "schema": "glm53-kda-replayssm-materializer-v1",
        "passed": True,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "shape": {
            "slots": slots,
            "rows": rows,
            "heads": heads,
            "ring_len": ring_len,
            "value_dim": value_dim,
            "key_dim": key_dim,
        },
        "metadata_stride": source_indices.stride(0),
        "maximum_absolute_error": error.max().item(),
        "mean_absolute_error": error.mean().item(),
        "contiguous_and_strided_outputs_identical": bool(
            torch.equal(actual, contiguous)
        ),
        "null_row_is_zero": torch.count_nonzero(actual[2]).item() == 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
