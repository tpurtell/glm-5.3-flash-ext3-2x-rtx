#!/usr/bin/env python3
"""Use B12x PCIe staging for profitable GLM DCP owner top-k exchanges."""

from __future__ import annotations

import sys
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, found {count}: {old[:160]!r}")
    path.write_text(text.replace(old, new, 1))


def port(root: Path) -> None:
    envs = root / "envs.py"
    replace_once(
        envs,
        "    VLLM_DCP_TOPK_OWNER_MERGE: bool = False\n",
        "    VLLM_DCP_TOPK_OWNER_MERGE: bool = False\n"
        "    VLLM_B12X_DCP_TOPK_OWNER_EXCHANGE: bool = True\n"
        "    VLLM_B12X_DCP_TOPK_MIN_ROWS: int = 128\n"
        "    VLLM_B12X_DCP_TOPK_MAX_ROWS: int = 8192\n",
    )
    replace_once(
        envs,
        '    "VLLM_DCP_TOPK_OWNER_MERGE": lambda: bool(\n'
        '        int(os.getenv("VLLM_DCP_TOPK_OWNER_MERGE", "0"))\n'
        "    ),\n",
        '    "VLLM_DCP_TOPK_OWNER_MERGE": lambda: bool(\n'
        '        int(os.getenv("VLLM_DCP_TOPK_OWNER_MERGE", "0"))\n'
        "    ),\n"
        '    "VLLM_B12X_DCP_TOPK_OWNER_EXCHANGE": lambda: bool(\n'
        '        int(os.getenv("VLLM_B12X_DCP_TOPK_OWNER_EXCHANGE", "1"))\n'
        "    ),\n"
        '    "VLLM_B12X_DCP_TOPK_MIN_ROWS": lambda: int(\n'
        '        os.getenv("VLLM_B12X_DCP_TOPK_MIN_ROWS", "128")\n'
        "    ),\n"
        '    "VLLM_B12X_DCP_TOPK_MAX_ROWS": lambda: int(\n'
        '        os.getenv("VLLM_B12X_DCP_TOPK_MAX_ROWS", "8192")\n'
        "    ),\n",
    )

    indexer = root / "model_executor/layers/sparse_attn_indexer.py"
    replace_once(
        indexer,
        "from vllm.forward_context import get_forward_context\n",
        "from vllm.forward_context import get_forward_context\n"
        "from vllm.logger import init_logger\n",
    )
    replace_once(
        indexer,
        "RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024\n",
        "RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024\n"
        "logger = init_logger(__name__)\n",
    )
    replace_once(
        indexer,
        '_B12X_PREFILL_PAGED_ROUTE = "packed_contiguous"\n',
        '_B12X_PREFILL_PAGED_ROUTE = "packed_contiguous"\n'
        "_B12X_DCP_TOPK_OWNER_EXCHANGES: dict[tuple[int, int, int], object] = {}\n",
    )

    group_helper = '''def _get_owner_merge_dcp_group(expected_world_size: int):
    """Return the collective group that owns the indexer's KV shards.
'''
    if indexer.read_text().count(group_helper) != 1:
        raise RuntimeError("could not locate DCP owner group helper")

    owner_anchor = '''    return group


def _merge_b12x_dcp_topk_by_owner(
'''
    owner_helpers = '''    return group


def _get_b12x_dcp_topk_owner_exchange(
    *,
    device: torch.device,
    dcp_world_size: int,
    topk_tokens: int,
    required_rows: int,
):
    """Lazily allocate the persistent B12x IPC channel outside KV profiling."""
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    key = (int(device_index), int(dcp_world_size), int(topk_tokens))
    existing = _B12X_DCP_TOPK_OWNER_EXCHANGES.get(key)
    if existing is not None:
        if int(existing.max_rows) < int(required_rows):
            raise RuntimeError(
                "B12x DCP owner channel is smaller than the active prefill: "
                f"channel={existing.max_rows}, rows={required_rows}"
            )
        return existing

    group = _get_owner_merge_dcp_group(dcp_world_size)
    process_group = getattr(group, "device_group", None)
    if process_group is None:
        communicator = getattr(group, "device_communicator", None)
        process_group = getattr(communicator, "device_group", None)
    if process_group is None:
        raise RuntimeError("DCP group does not expose its CUDA process group")

    max_rows = max(int(required_rows), int(envs.VLLM_B12X_DCP_TOPK_MAX_ROWS))
    if max_rows <= 0:
        raise ValueError("VLLM_B12X_DCP_TOPK_MAX_ROWS must be positive")
    max_rows -= max_rows % int(dcp_world_size)
    if max_rows < int(required_rows):
        max_rows += int(dcp_world_size)

    from b12x.comm.pcie import DcpTopKOwnerExchange

    exchange = DcpTopKOwnerExchange.from_exchange_group(
        exchange_group=process_group,
        device=device,
        max_rows=max_rows,
        topk=int(topk_tokens),
    )
    _B12X_DCP_TOPK_OWNER_EXCHANGES[key] = exchange
    logger.info_once(
        "Using B12x PCIe DCP owner top-k exchange (world=%d, max_rows=%d, "
        "topk=%d, min_rows=%d).",
        int(dcp_world_size),
        max_rows,
        int(topk_tokens),
        _b12x_dcp_topk_owner_min_rows(int(topk_tokens)),
    )
    return exchange


def _b12x_dcp_topk_owner_min_rows(topk_tokens: int) -> int:
    min_rows = int(envs.VLLM_B12X_DCP_TOPK_MIN_ROWS)
    if min_rows <= 0:
        raise ValueError("VLLM_B12X_DCP_TOPK_MIN_ROWS must be positive")
    # Qualified on the public 2x SM120 PCIe topology. Smaller candidate sets
    # need more rows to amortize the direct IPC launch.
    qualified = {512: 1024, 1024: 512, 2048: 128}
    return max(min_rows, qualified.get(int(topk_tokens), min_rows))


def _use_b12x_dcp_topk_owner_exchange(
    *, rows: int, dcp_world_size: int, topk_tokens: int
) -> bool:
    # The released recipe qualifies the direct PCIe path on one DCP2 group.
    return bool(
        envs.VLLM_B12X_DCP_TOPK_OWNER_EXCHANGE
        and int(dcp_world_size) == 2
        and int(rows) >= _b12x_dcp_topk_owner_min_rows(topk_tokens)
    )


def _merge_b12x_kpool_dcp_topk_by_owner(
    *,
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
    dcp_world_size: int,
    dcp_rank: int,
    cp_kv_cache_interleave_size: int,
    row_starts: torch.Tensor | None,
) -> bool:
    """Owner-shard the exact GLM k-pool merge with direct B12x PCIe staging."""
    if dcp_world_size <= 1 or topk_indices.numel() == 0:
        return False
    rows = int(topk_indices.shape[0])
    if rows % int(dcp_world_size) != 0:
        return False
    if not _use_b12x_dcp_topk_owner_exchange(
        rows=rows,
        dcp_world_size=dcp_world_size,
        topk_tokens=topk_tokens,
    ):
        return False

    from b12x.attention.nsa_indexer.tiled_topk import run_row_topk

    from vllm.v1.attention.backends.mla.b12x_dcp_topk import (
        triton_gather_dcp_topk_scores_and_globalize,
        triton_gather_topk_ids_by_position,
    )

    owner_rows = rows // int(dcp_world_size)
    candidate_width = int(dcp_world_size * topk_tokens)
    local_scores, candidate_lengths, owner_values, owner_positions, owner_indices = (
        current_workspace_manager().get_simultaneous(
            ((rows, int(topk_tokens)), torch.float32),
            ((owner_rows,), torch.int32),
            ((owner_rows, int(topk_tokens)), torch.float32),
            ((owner_rows, int(topk_tokens)), torch.int32),
            ((owner_rows, int(topk_tokens)), torch.int32),
        )
    )
    triton_gather_dcp_topk_scores_and_globalize(
        topk_indices,
        logits,
        local_scores,
        dcp_world_size=dcp_world_size,
        dcp_rank=dcp_rank,
        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        row_starts=row_starts,
    )
    owner_exchange = _get_b12x_dcp_topk_owner_exchange(
        device=topk_indices.device,
        dcp_world_size=dcp_world_size,
        topk_tokens=topk_tokens,
        required_rows=rows,
    )
    candidate_indices, candidate_scores = owner_exchange.stage_candidates(
        topk_indices,
        local_scores,
    )
    candidate_lengths.fill_(candidate_width)
    run_row_topk(
        row_logits=candidate_scores,
        lengths=candidate_lengths,
        topk=int(topk_tokens),
        output_values=owner_values,
        output_indices=owner_positions,
    )
    triton_gather_topk_ids_by_position(
        candidate_indices,
        owner_positions,
        owner_indices,
    )
    _dcp_all_gather_first_dim_into(
        _get_owner_merge_dcp_group(dcp_world_size),
        owner_indices,
        topk_indices,
    )
    return True


def _merge_b12x_dcp_topk_by_owner(
'''
    replace_once(indexer, owner_anchor, owner_helpers)

    old_workspace = '''    owner_rows = rows // int(dcp_world_size)
    candidate_width = int(dcp_world_size * topk_tokens)
    (
        candidates,
        received_candidates,
        candidate_indices,
        candidate_score_bits,
        candidate_lengths,
        owner_values,
        owner_positions,
        owner_indices,
    ) = current_workspace_manager().get_simultaneous(
        ((rows, 2, int(topk_tokens)), torch.int32),
        ((rows, 2, int(topk_tokens)), torch.int32),
        ((owner_rows, candidate_width), torch.int32),
        ((owner_rows, candidate_width), torch.int32),
        ((owner_rows,), torch.int32),
        ((owner_rows, int(topk_tokens)), torch.float32),
        ((owner_rows, int(topk_tokens)), torch.int32),
        ((owner_rows, int(topk_tokens)), torch.int32),
    )
    candidates[:, 0, :].copy_(topk_indices)
    candidates[:, 1, :].copy_(topk_scores.view(torch.int32))

    _dcp_all_to_all_first_dim_into(
        _get_owner_merge_dcp_group(dcp_world_size),
        candidates,
        received_candidates,
    )
    _unpack_b12x_dcp_gathered_candidates(
        received_candidates,
        candidate_indices,
        candidate_score_bits,
        dcp_world_size=dcp_world_size,
        topk_tokens=int(topk_tokens),
    )
    candidate_lengths.fill_(candidate_width)
    run_row_topk(
        row_logits=candidate_score_bits.view(torch.float32),
'''
    new_workspace = '''    owner_rows = rows // int(dcp_world_size)
    candidate_width = int(dcp_world_size * topk_tokens)
    use_b12x_pcie_owner = _use_b12x_dcp_topk_owner_exchange(
        rows=rows,
        dcp_world_size=dcp_world_size,
        topk_tokens=topk_tokens,
    )
    if use_b12x_pcie_owner:
        owner_exchange = _get_b12x_dcp_topk_owner_exchange(
            device=topk_indices.device,
            dcp_world_size=dcp_world_size,
            topk_tokens=topk_tokens,
            required_rows=rows,
        )
        candidate_indices, candidate_scores = owner_exchange.stage_candidates(
            topk_indices,
            topk_scores,
        )
        candidate_score_bits = candidate_scores.view(torch.int32)
    else:
        (
            candidates,
            received_candidates,
            candidate_indices,
            candidate_score_bits,
        ) = current_workspace_manager().get_simultaneous(
            ((rows, 2, int(topk_tokens)), torch.int32),
            ((rows, 2, int(topk_tokens)), torch.int32),
            ((owner_rows, candidate_width), torch.int32),
            ((owner_rows, candidate_width), torch.int32),
        )
        candidates[:, 0, :].copy_(topk_indices)
        candidates[:, 1, :].copy_(topk_scores.view(torch.int32))
        _dcp_all_to_all_first_dim_into(
            _get_owner_merge_dcp_group(dcp_world_size),
            candidates,
            received_candidates,
        )
        _unpack_b12x_dcp_gathered_candidates(
            received_candidates,
            candidate_indices,
            candidate_score_bits,
            dcp_world_size=dcp_world_size,
            topk_tokens=int(topk_tokens),
        )

    candidate_lengths, owner_values, owner_positions, owner_indices = (
        current_workspace_manager().get_simultaneous(
            ((owner_rows,), torch.int32),
            ((owner_rows, int(topk_tokens)), torch.float32),
            ((owner_rows, int(topk_tokens)), torch.int32),
            ((owner_rows, int(topk_tokens)), torch.int32),
        )
    )
    candidate_lengths.fill_(candidate_width)
    run_row_topk(
        row_logits=candidate_score_bits.view(torch.float32),
'''
    replace_once(indexer, old_workspace, new_workspace)

    kpool = root / "model_executor/layers/sparse_attn_indexer_kpool.py"
    replace_once(
        kpool,
        "    _merge_b12x_dcp_topk,\n"
        "    _merge_dcp_topk_global,\n",
        "    _merge_b12x_dcp_topk,\n"
        "    _merge_b12x_kpool_dcp_topk_by_owner,\n"
        "    _merge_dcp_topk_global,\n",
    )
    replace_once(
        kpool,
        "            _merge_dcp_topk_global(\n"
        "                logits,\n"
        "                topk_dst,\n"
        "                select_k,\n"
        "                dcp_rank,\n"
        "                dcp_world_size,\n"
        "                cp_kv_cache_interleave_size,\n"
        "                row_starts=chunk.cu_seqlen_ks,\n"
        "            )\n",
        "            used_b12x_owner = _merge_b12x_kpool_dcp_topk_by_owner(\n"
        "                logits=logits,\n"
        "                topk_indices=topk_dst,\n"
        "                topk_tokens=select_k,\n"
        "                dcp_rank=dcp_rank,\n"
        "                dcp_world_size=dcp_world_size,\n"
        "                cp_kv_cache_interleave_size=(\n"
        "                    cp_kv_cache_interleave_size\n"
        "                ),\n"
        "                row_starts=chunk.cu_seqlen_ks,\n"
        "            )\n"
        "            if not used_b12x_owner:\n"
        "                _merge_dcp_topk_global(\n"
        "                    logits,\n"
        "                    topk_dst,\n"
        "                    select_k,\n"
        "                    dcp_rank,\n"
        "                    dcp_world_size,\n"
        "                    cp_kv_cache_interleave_size,\n"
        "                    row_starts=chunk.cu_seqlen_ks,\n"
        "                )\n",
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} VLLM_PACKAGE_ROOT")
    port(Path(sys.argv[1]))
