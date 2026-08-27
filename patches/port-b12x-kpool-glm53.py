#!/usr/bin/env python3
"""Route GLM-5.3's pool-compressed sparse indexer through B12x on SM120.

The day-zero GLM image has a newer kpool-aware indexer than the qualified
B12x/vLLM source image.  Preserve all of that implementation's cache writes,
tail handling, pool expansion, and fallback paths; replace only the paged
decode score+top-k pair with B12x's fused exact selector.  The B12x source is
still the pinned user fork installed at /opt/b12x.
"""

from __future__ import annotations

import sys
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, found {count}: {old[:160]!r}")
    path.write_text(text.replace(old, new, 1))


def insert_after_once(path: Path, anchor: str, addition: str) -> None:
    replace_once(path, anchor, anchor + addition)


def port(root: Path) -> None:
    kpool = root / "model_executor/layers/sparse_attn_indexer_kpool.py"

    insert_after_once(
        kpool,
        "from vllm.v1.worker.workspace import current_workspace_manager\n",
        "from vllm.model_executor.layers.sparse_attn_indexer import (\n"
        "    _b12x_sparse_indexer_requested,\n"
        "    _ensure_b12x_sparse_indexer_supported,\n"
        "    _merge_b12x_dcp_topk,\n"
        "    _reserve_b12x_paged_indexer_scratch,\n"
        "    _run_b12x_paged_topk,\n"
        ")\n",
    )

    insert_after_once(
        kpool,
        "MXFP4_BLOCK_SIZE = 32\n",
        "\n"
        "\n"
        "def _use_b12x_kpool_indexer() -> bool:\n"
        "    raw = os.environ.get(\"VLLM_USE_B12X_KPOOL_INDEXER\")\n"
        "    if raw is not None:\n"
        "        return bool(int(raw))\n"
        "    return _b12x_sparse_indexer_requested()\n",
    )

    # The memory profiler must see the B12x caller-owned scratch even though
    # the fake/dummy forward has no real attention metadata.  Context lengths
    # here are pool rows, not model-token rows.
    insert_after_once(
        kpool,
        "        worst_decode_tokens = 0\n"
        "        if cfg is not None:\n"
        "            sched = cfg.scheduler_config\n"
        "            num_spec = (\n"
        "                cfg.speculative_config.num_speculative_tokens\n"
        "                if cfg.speculative_config is not None\n"
        "                else 0\n"
        "            )\n"
        "            worst_decode_tokens = min(\n"
        "                sched.max_num_seqs * (num_spec + 1),\n"
        "                sched.max_num_batched_tokens,\n"
        "            )\n",
        "        if _use_b12x_kpool_indexer() and index_kpool > 1:\n"
        "            _ensure_b12x_sparse_indexer_supported()\n"
        "            _reserve_b12x_paged_indexer_scratch(\n"
        "                q_rows=max(1, worst_decode_tokens),\n"
        "                num_q_heads=int(q_quant.shape[1]),\n"
        "                topk_tokens=max(1, topk_tokens // index_kpool),\n"
        "                total_k_rows=max(1, max_model_len // index_kpool),\n"
        "                device=q_quant.device,\n"
        "                shared_page_table=False,\n"
        "            )\n",
    )

    # Keep the raw rank-3 cache live for B12x.  DeepGEMM still receives its
    # rank-4 quant view on fallback (prefill, FP4, padded/spec edge cases).
    replace_once(
        kpool,
        "        kv_cache_raw = kv_cache  # raw [num_blocks, block_size, head_dim+4] for writes\n"
        "        kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache)\n",
        "        kv_cache_raw = kv_cache  # raw [num_blocks, block_size, head_dim+4] for writes\n"
        "        kv_cache_quant_view = kv_cache_as_quant_view(\n"
        "            kv_cache, head_dim, use_fp4_cache\n"
        "        )\n",
    )

    old = """        logits = fp8_fp4_paged_mqa_logits(
            (padded_q_quant_cast, padded_q_scale),
            kv_cache,
            padded_weights[:num_padded_tokens],
            seq_lens,
            decode_metadata.block_table,
            decode_metadata.schedule_metadata,
            max_model_len=max_model_len,
            clean_logits=False,
        )
        num_rows = logits.shape[0]
        # kpool: logits are pool-granular -> select topk_tokens//kpool pools,
        # then expand each pool back to its kpool tokens.
        select_k = topk_tokens // index_kpool if index_kpool > 1 else topk_tokens
        if index_kpool > 1:
            pool_topk = torch.empty(
                (num_rows, select_k), dtype=torch.int32, device=logits.device
            )
            topk_dst = pool_topk
        else:
            topk_dst = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        if current_platform.is_cuda() and select_k in (512, 1024, 2048):
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.persistent_topk(
                logits,
                seq_lens,
                topk_dst,
                topk_workspace,
                select_k,
                attn_metadata_narrowed.max_seq_len,
            )
        else:
            if current_platform.is_xpu():
                xpu_ops.top_k_per_row_decode(  # type: ignore[attr-defined]
                    logits,
                    next_n,
                    seq_lens,
                    topk_dst,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    select_k,
                )
            else:
                torch.ops._C.top_k_per_row_decode(
                    logits,
                    next_n,
                    seq_lens,
                    topk_dst,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    select_k,
                )
"""
    new = """        # kpool: the cache and seq_lens are pool-granular, so select
        # topk_tokens//kpool pools and expand them back to model-token ids below.
        select_k = topk_tokens // index_kpool if index_kpool > 1 else topk_tokens
        num_rows = num_padded_tokens
        if index_kpool > 1:
            pool_topk = torch.empty(
                (num_rows, select_k),
                dtype=torch.int32,
                device=padded_q_quant_decode_tokens.device,
            )
            topk_dst = pool_topk
        else:
            topk_dst = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        # B12x consumes one rank-1 sequence length and one page-table row per
        # query. Native speculative decode stores seq_lens as [B, next_n], so
        # normalize that graph-stably. Retain the official DeepGEMM path for
        # padded/non-FP8 edge cases until they have their own exact B12x route.
        b12x_seq_lens = seq_lens
        b12x_block_table = decode_metadata.block_table
        if b12x_seq_lens.dim() == 2:
            b12x_batch_size, b12x_next_n = b12x_seq_lens.shape
            if num_padded_tokens == b12x_batch_size * b12x_next_n:
                b12x_seq_lens = b12x_seq_lens.reshape(-1).contiguous()
                b12x_block_table = b12x_block_table.repeat_interleave(
                    b12x_next_n, dim=0
                ).contiguous()
        use_b12x_kpool = (
            _use_b12x_kpool_indexer()
            and index_kpool > 1
            and not use_fp4_cache
            and not decode_metadata.requires_padding
            and b12x_seq_lens.dim() == 1
        )
        if use_b12x_kpool:
            _ensure_b12x_sparse_indexer_supported()
            cfg = get_current_vllm_config_or_none()
            from vllm.distributed import get_dcp_group

            dcp_group = get_dcp_group()
            dcp_world_size = int(dcp_group.world_size)
            dcp_rank = int(dcp_group.rank_in_group)
            cp_kv_cache_interleave_size = 1
            if cfg is not None:
                cp_kv_cache_interleave_size = int(
                    cfg.parallel_config.cp_kv_cache_interleave_size
                )
            topk_scores = None
            if dcp_world_size > 1:
                topk_scores = torch.empty(
                    (num_rows, select_k),
                    dtype=torch.float32,
                    device=pool_topk.device,
                )
            _run_b12x_paged_topk(
                q_fp8=padded_q_quant_decode_tokens.reshape(
                    num_rows, *padded_q_quant_decode_tokens.shape[2:]
                ).contiguous(),
                weights=padded_weights[:num_rows].reshape(
                    num_rows, -1
                ).contiguous(),
                kv_cache=kv_cache_raw,
                seq_lens=b12x_seq_lens[:num_rows],
                block_table=b12x_block_table[:num_rows],
                schedule_metadata=decode_metadata.schedule_metadata,
                active_width=getattr(decode_metadata, "active_width", None),
                topk_indices=topk_dst,
                topk_tokens=select_k,
                topk_scores=topk_scores,
            )
            _merge_b12x_dcp_topk(
                topk_indices=topk_dst,
                topk_scores=topk_scores,
                topk_tokens=select_k,
                dcp_world_size=dcp_world_size,
                dcp_rank=dcp_rank,
                cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            )
        else:
            logits = fp8_fp4_paged_mqa_logits(
                (padded_q_quant_cast, padded_q_scale),
                kv_cache_quant_view,
                padded_weights[:num_padded_tokens],
                seq_lens,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len=max_model_len,
                clean_logits=False,
            )
            num_rows = logits.shape[0]
            if current_platform.is_cuda() and select_k in (512, 1024, 2048):
                workspace_manager = current_workspace_manager()
                (topk_workspace,) = workspace_manager.get_simultaneous(
                    ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
                )
                torch.ops._C.persistent_topk(
                    logits,
                    seq_lens,
                    topk_dst,
                    topk_workspace,
                    select_k,
                    attn_metadata_narrowed.max_seq_len,
                )
            else:
                if current_platform.is_xpu():
                    xpu_ops.top_k_per_row_decode(  # type: ignore[attr-defined]
                        logits,
                        next_n,
                        seq_lens,
                        topk_dst,
                        num_rows,
                        logits.stride(0),
                        logits.stride(1),
                        select_k,
                    )
                else:
                    torch.ops._C.top_k_per_row_decode(
                        logits,
                        next_n,
                        seq_lens,
                        topk_dst,
                        num_rows,
                        logits.stride(0),
                        logits.stride(1),
                        select_k,
                    )
"""
    replace_once(kpool, old, new)

    # A single startup line is enough to prove the kpool implementation—not
    # merely the sparse MLA consumer—has selected B12x.
    insert_after_once(
        kpool,
        "        self.use_fp4_cache = use_fp4_cache\n",
        "        self.use_b12x_kpool_indexer = (\n"
        "            _use_b12x_kpool_indexer()\n"
        "            and not use_fp4_cache\n"
        "        )\n"
        "        if self.use_b12x_kpool_indexer:\n"
        "            _ensure_b12x_sparse_indexer_supported()\n"
        "            logger.info_once(\n"
        "                \"Using B12x fused paged score+top-k for GLM kpool decode.\"\n"
        "            )\n",
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} VLLM_PACKAGE_ROOT")
    port(Path(sys.argv[1]))
    print("B12x GLM-5.3 kpool indexer port applied")
