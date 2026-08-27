# SPDX-License-Identifier: Apache-2.0
"""Graph-safe DCP top-k remap helpers used by the B12x vLLM adapters."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _convert_dcp_local_topk_to_global_kernel(
    token_indices_ptr,
    scores_ptr,
    ti_stride0,
    ti_stride1,
    scores_stride0,
    scores_stride1,
    width: tl.constexpr,
    DCP_WORLD_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < width
    idx_ptrs = token_indices_ptr + row * ti_stride0 + offs * ti_stride1
    local_idx = tl.load(idx_ptrs, mask=mask, other=-1)
    valid = local_idx >= 0

    interleave_block = local_idx // CP_KV_CACHE_INTERLEAVE_SIZE
    interleave_offset = local_idx % CP_KV_CACHE_INTERLEAVE_SIZE
    global_idx = (
        interleave_block * DCP_WORLD_SIZE + DCP_RANK
    ) * CP_KV_CACHE_INTERLEAVE_SIZE + interleave_offset
    tl.store(idx_ptrs, tl.where(valid, global_idx, -1), mask=mask)

    score_ptrs = scores_ptr + row * scores_stride0 + offs * scores_stride1
    scores = tl.load(score_ptrs, mask=mask, other=-float("inf"))
    tl.store(score_ptrs, tl.where(valid, scores, -float("inf")), mask=mask)


def triton_convert_dcp_local_topk_to_global(
    token_indices: torch.Tensor,
    scores: torch.Tensor,
    *,
    dcp_world_size: int,
    dcp_rank: int,
    cp_kv_cache_interleave_size: int,
    BLOCK_N: int = 128,
) -> None:
    """Convert local DCP top-k ids in-place to global logical ids."""
    assert token_indices.dtype == torch.int32
    assert scores.dtype == torch.float32
    assert token_indices.shape == scores.shape
    assert token_indices.is_contiguous()
    assert scores.is_contiguous()
    width = token_indices.shape[1]
    assert width % BLOCK_N == 0, (
        f"top-k width ({width}) must be divisible by BLOCK_N ({BLOCK_N})"
    )
    grid = (token_indices.shape[0], width // BLOCK_N)
    _convert_dcp_local_topk_to_global_kernel[grid](
        token_indices,
        scores,
        token_indices.stride(0),
        token_indices.stride(1),
        scores.stride(0),
        scores.stride(1),
        width,
        DCP_WORLD_SIZE=dcp_world_size,
        DCP_RANK=dcp_rank,
        CP_KV_CACHE_INTERLEAVE_SIZE=cp_kv_cache_interleave_size,
        BLOCK_N=BLOCK_N,
    )


@triton.jit
def _gather_dcp_topk_scores_and_globalize_kernel(
    token_indices_ptr,
    logits_ptr,
    row_starts_ptr,
    scores_ptr,
    ti_stride0,
    ti_stride1,
    logits_stride0,
    logits_stride1,
    scores_stride0,
    scores_stride1,
    logits_width,
    width: tl.constexpr,
    DCP_WORLD_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
    HAS_ROW_STARTS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < width
    idx_ptrs = token_indices_ptr + row * ti_stride0 + offs * ti_stride1
    local_idx = tl.load(idx_ptrs, mask=mask, other=-1)
    valid = local_idx >= 0

    score_idx = tl.maximum(local_idx, 0)
    if HAS_ROW_STARTS:
        score_idx += tl.load(row_starts_ptr + row)
    score_valid = valid & (score_idx < logits_width)
    scores = tl.load(
        logits_ptr + row * logits_stride0 + score_idx * logits_stride1,
        mask=mask & score_valid,
        other=-float("inf"),
    ).to(tl.float32)

    interleave_block = local_idx // CP_KV_CACHE_INTERLEAVE_SIZE
    interleave_offset = local_idx % CP_KV_CACHE_INTERLEAVE_SIZE
    global_idx = (
        interleave_block * DCP_WORLD_SIZE + DCP_RANK
    ) * CP_KV_CACHE_INTERLEAVE_SIZE + interleave_offset
    tl.store(idx_ptrs, tl.where(valid, global_idx, -1), mask=mask)
    tl.store(
        scores_ptr + row * scores_stride0 + offs * scores_stride1,
        tl.where(score_valid, scores, -float("inf")),
        mask=mask,
    )


def triton_gather_dcp_topk_scores_and_globalize(
    token_indices: torch.Tensor,
    logits: torch.Tensor,
    scores: torch.Tensor,
    *,
    dcp_world_size: int,
    dcp_rank: int,
    cp_kv_cache_interleave_size: int,
    row_starts: torch.Tensor | None = None,
    BLOCK_N: int = 128,
) -> None:
    """Gather FP32 candidate scores and globalize local ids in one pass."""
    assert token_indices.dtype == torch.int32
    assert logits.dtype == torch.float32
    assert scores.dtype == torch.float32
    assert token_indices.shape == scores.shape
    assert logits.ndim == 2 and logits.shape[0] == token_indices.shape[0]
    assert token_indices.is_contiguous() and scores.is_contiguous()
    width = token_indices.shape[1]
    assert width % BLOCK_N == 0, (
        f"top-k width ({width}) must be divisible by BLOCK_N ({BLOCK_N})"
    )
    if row_starts is not None:
        assert row_starts.shape[0] == token_indices.shape[0]
    row_starts_arg = row_starts if row_starts is not None else token_indices
    grid = (token_indices.shape[0], width // BLOCK_N)
    _gather_dcp_topk_scores_and_globalize_kernel[grid](
        token_indices,
        logits,
        row_starts_arg,
        scores,
        token_indices.stride(0),
        token_indices.stride(1),
        logits.stride(0),
        logits.stride(1),
        scores.stride(0),
        scores.stride(1),
        logits.shape[1],
        width,
        DCP_WORLD_SIZE=dcp_world_size,
        DCP_RANK=dcp_rank,
        CP_KV_CACHE_INTERLEAVE_SIZE=cp_kv_cache_interleave_size,
        HAS_ROW_STARTS=row_starts is not None,
        BLOCK_N=BLOCK_N,
    )


@triton.jit
def _gather_topk_ids_by_position_kernel(
    candidate_ids_ptr,
    positions_ptr,
    out_ptr,
    cand_stride0,
    cand_stride1,
    pos_stride0,
    pos_stride1,
    out_stride0,
    out_stride1,
    topk: tl.constexpr,
    candidate_width: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < topk
    pos = tl.load(
        positions_ptr + row * pos_stride0 + offs * pos_stride1,
        mask=mask,
        other=-1,
    )
    valid = (pos >= 0) & (pos < candidate_width)
    gathered = tl.load(
        candidate_ids_ptr + row * cand_stride0 + pos * cand_stride1,
        mask=mask & valid,
        other=-1,
    )
    tl.store(
        out_ptr + row * out_stride0 + offs * out_stride1,
        tl.where(valid, gathered, -1),
        mask=mask,
    )


def triton_gather_topk_ids_by_position(
    candidate_ids: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    *,
    BLOCK_N: int = 128,
) -> None:
    """Gather final ids from flattened candidate ids by int32 positions."""
    assert candidate_ids.dtype == torch.int32
    assert positions.dtype == torch.int32
    assert out.dtype == torch.int32
    assert candidate_ids.ndim == 2
    assert positions.ndim == 2
    assert out.shape == positions.shape
    assert candidate_ids.shape[0] == positions.shape[0]
    assert positions.shape[1] % BLOCK_N == 0, (
        f"top-k width ({positions.shape[1]}) must be divisible by BLOCK_N ({BLOCK_N})"
    )
    grid = (positions.shape[0], positions.shape[1] // BLOCK_N)
    _gather_topk_ids_by_position_kernel[grid](
        candidate_ids,
        positions,
        out,
        candidate_ids.stride(0),
        candidate_ids.stride(1),
        positions.stride(0),
        positions.stride(1),
        out.stride(0),
        out.stride(1),
        positions.shape[1],
        candidate_ids.shape[1],
        BLOCK_N=BLOCK_N,
    )
