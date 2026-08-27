#!/usr/bin/env python3
"""Adapt the qualified B12x DCP integration to the newer GLM vLLM tree."""

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
    indexer = root / "model_executor/layers/sparse_attn_indexer.py"
    text = indexer.read_text()
    text = text.replace(
        "from vllm.v1.attention.backends.mla.sparse_utils import (\n"
        "        triton_convert_dcp_local_topk_to_global,\n"
        "        triton_gather_topk_ids_by_position,\n"
        "    )",
        "from vllm.v1.attention.backends.mla.b12x_dcp_topk import (\n"
        "        triton_convert_dcp_local_topk_to_global,\n"
        "        triton_gather_topk_ids_by_position,\n"
        "    )",
    )
    text = text.replace(
        "from vllm.v1.attention.backends.mla.sparse_utils import (\n"
        "        triton_convert_dcp_local_topk_to_global,\n"
        "    )",
        "from vllm.v1.attention.backends.mla.b12x_dcp_topk import (\n"
        "        triton_convert_dcp_local_topk_to_global,\n"
        "    )",
    )
    if "from vllm.v1.attention.backends.mla.sparse_utils import (\n        triton_convert_dcp_local_topk_to_global" in text:
        raise RuntimeError("not all B12x DCP helper imports were redirected")
    text = text.replace(
        "from vllm.distributed import get_dcp_group as get_indexer_dcp_group\n",
        "from vllm.distributed import get_dcp_group\n",
    )
    # The qualified adapter imported the old group accessor inside several
    # functions.  The current tree no longer exports it, and some call sites
    # do not leave a blank line after the import, so remove it line-wise.
    text = "".join(
        line
        for line in text.splitlines(keepends=True)
        if "from vllm.distributed.parallel_state import get_indexer_dcp_group"
        not in line
    )
    text = text.replace(
        "get_indexer_dcp_group(", "_b12x_compat_dcp_group("
    )
    logger_anchor = "RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024\n"
    if text.count(logger_anchor) != 1:
        raise RuntimeError("could not locate sparse-indexer constants anchor")
    text = text.replace(
        logger_anchor,
        logger_anchor
        + "\n"
        + "\n"
        + "def _b12x_compat_dcp_group(expected_world_size: int | None = None):\n"
        + "    group = get_dcp_group()\n"
        + "    if (\n"
        + "        expected_world_size is not None\n"
        + "        and int(group.world_size) != int(expected_world_size)\n"
        + "    ):\n"
        + "        raise RuntimeError(\n"
        + "            f\"DCP group has {group.world_size} ranks, \"\n"
        + "            f\"expected {expected_world_size}.\"\n"
        + "        )\n"
        + "    return group\n",
        1,
    )
    if "get_indexer_dcp_group(" in text:
        raise RuntimeError("not all legacy DCP group calls were adapted")
    indexer.write_text(text)

    # Preserve the newer single-tile compaction optimization while restoring
    # the graph-stable preallocated output API expected by the B12x backend.
    sparse_utils = root / "v1/attention/backends/mla/sparse_utils.py"
    replace_once(
        sparse_utils,
        "    return_valid_counts: bool = False,\n"
        "    compact_valid_to_front: bool = True,\n"
        ") -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:\n"
        "    \"\"\"Filter global per-request indices to this DCP rank's local slots.\n",
        "    return_valid_counts: bool = False,\n"
        "    compact_valid_to_front: bool = True,\n"
        "    out: torch.Tensor | None = None,\n"
        "    valid_counts: torch.Tensor | None = None,\n"
        ") -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:\n"
        "    \"\"\"Filter global per-request indices to this DCP rank's local slots.\n",
    )
    replace_once(
        sparse_utils,
        "    if dcp_size == 1:\n"
        "        return triton_convert_req_index_to_global_index(\n",
        "    if dcp_size == 1:\n"
        "        assert out is None and valid_counts is None, (\n"
        "            \"preallocated out/valid_counts are only supported on the DCP path\"\n"
        "        )\n"
        "        return triton_convert_req_index_to_global_index(\n",
    )
    replace_once(
        sparse_utils,
        "    if compact_valid_to_front:\n"
        "        out = torch.full_like(token_indices_c, -1)\n"
        "    else:\n"
        "        out = torch.empty_like(token_indices_c)\n"
        "\n"
        "    valid_counts: torch.Tensor | None = None\n"
        "    if count_valid:\n"
        "        # Zero-init only matters for the atomic accumulation path.\n"
        "        alloc = torch.empty if single_tile else torch.zeros\n"
        "        valid_counts = alloc(num_tokens, dtype=torch.int32, device=token_indices.device)\n",
        "    if out is None:\n"
        "        out = (\n"
        "            torch.full_like(token_indices_c, -1)\n"
        "            if compact_valid_to_front\n"
        "            else torch.empty_like(token_indices_c)\n"
        "        )\n"
        "    else:\n"
        "        assert out.dtype == torch.int32 and out.device == token_indices.device\n"
        "        assert out.shape == token_indices_c.shape\n"
        "        if compact_valid_to_front:\n"
        "            out.fill_(-1)\n"
        "\n"
        "    if count_valid:\n"
        "        if valid_counts is None:\n"
        "            # Zero-init only matters for the atomic accumulation path.\n"
        "            alloc = torch.empty if single_tile else torch.zeros\n"
        "            valid_counts = alloc(\n"
        "                num_tokens, dtype=torch.int32, device=token_indices.device\n"
        "            )\n"
        "        else:\n"
        "            assert valid_counts.dtype == torch.int32\n"
        "            assert valid_counts.shape[0] == num_tokens\n"
        "            if not single_tile:\n"
        "                valid_counts.zero_()\n"
        "    else:\n"
        "        valid_counts = None\n",
    )

    # Pool-compressed caches need to shard *pool rows*, not model-token rows.
    # A completed global pool p belongs to rank (p // interleave) % world and
    # maps to that rank's dense local-pool index.  The original compressor
    # kernel indexed the rank-local block table with global pool ids, which is
    # correct only for DCP=1 and can silently write/read the wrong physical row.
    compressor_utils = root / "v1/attention/backends/mla/compressor_utils.py"
    replace_once(
        compressor_utils,
        "    COMPRESS_RATIO: tl.constexpr,\n"
        "    PAD_ID: tl.constexpr,\n",
        "    COMPRESS_RATIO: tl.constexpr,\n"
        "    DCP_WORLD_SIZE: tl.constexpr,\n"
        "    DCP_RANK: tl.constexpr,\n"
        "    DCP_INTERLEAVE: tl.constexpr,\n"
        "    PAD_ID: tl.constexpr,\n",
    )
    replace_once(
        compressor_utils,
        "        is_valid = (pos + 1) % COMPRESS_RATIO == 0\n"
        "        pos_after_compress = pos // COMPRESS_RATIO\n\n"
        "        block_ids = pos_after_compress // block_size\n",
        "        completes_pool = (pos + 1) % COMPRESS_RATIO == 0\n"
        "        global_pool = pos // COMPRESS_RATIO\n"
        "        pool_group = global_pool // DCP_INTERLEAVE\n"
        "        pool_offset = global_pool % DCP_INTERLEAVE\n"
        "        is_local = pool_group % DCP_WORLD_SIZE == DCP_RANK\n"
        "        is_valid = completes_pool & is_local\n"
        "        local_pool = (\n"
        "            pool_group // DCP_WORLD_SIZE * DCP_INTERLEAVE + pool_offset\n"
        "        )\n\n"
        "        block_ids = local_pool // block_size\n",
    )
    replace_once(
        compressor_utils,
        "        slot_ids = block_numbers * block_size + pos_after_compress % block_size\n",
        "        slot_ids = block_numbers * block_size + local_pool % block_size\n",
    )
    replace_once(
        compressor_utils,
        "    compress_ratio: int,\n"
        "    out: torch.Tensor | None = None,\n",
        "    compress_ratio: int,\n"
        "    dcp_world_size: int = 1,\n"
        "    dcp_rank: int = 0,\n"
        "    cp_kv_cache_interleave_size: int = 1,\n"
        "    out: torch.Tensor | None = None,\n",
    )
    replace_once(
        compressor_utils,
        "    if out is not None:\n",
        "    assert dcp_world_size >= 1\n"
        "    assert 0 <= dcp_rank < dcp_world_size\n"
        "    assert cp_kv_cache_interleave_size >= 1\n"
        "    assert block_size % cp_kv_cache_interleave_size == 0, (\n"
        "        f\"compressed cache block_size ({block_size}) must be divisible by \"\n"
        "        f\"DCP interleave ({cp_kv_cache_interleave_size})\"\n"
        "    )\n"
        "    if out is not None:\n",
    )
    replace_once(
        compressor_utils,
        "        compress_ratio,\n"
        "        PAD_ID=-1,\n",
        "        compress_ratio,\n"
        "        DCP_WORLD_SIZE=dcp_world_size,\n"
        "        DCP_RANK=dcp_rank,\n"
        "        DCP_INTERLEAVE=cp_kv_cache_interleave_size,\n"
        "        PAD_ID=-1,\n",
    )

    metadata = root / "v1/attention/backends/mla/indexer.py"
    replace_once(
        metadata,
        "        if self.dcp_world_size > 1 and self.compress_ratio > 1:\n"
        "            raise NotImplementedError(\n"
        "                \"DCP is not supported with sparse indexer KV compression \"\n"
        "                f\"(compress_ratio={self.compress_ratio}).\"\n"
        "            )\n\n",
        "        if self.dcp_world_size > 1 and self.compress_ratio > 1:\n"
        "            logger.info_once(\n"
        "                \"Using DCP-sharded sparse indexer KV compression \"\n"
        "                \"(compress_ratio=%d, world_size=%d, interleave=%d).\",\n"
        "                self.compress_ratio,\n"
        "                self.dcp_world_size,\n"
        "                self.cp_kv_cache_interleave_size,\n"
        "            )\n\n",
    )
    replace_once(
        metadata,
        "                self.compress_ratio,\n"
        "                out=self.compressed_slot_mapping_buffer,\n",
        "                self.compress_ratio,\n"
        "                dcp_world_size=self.dcp_world_size,\n"
        "                dcp_rank=self.dcp_rank,\n"
        "                cp_kv_cache_interleave_size=(\n"
        "                    self.cp_kv_cache_interleave_size\n"
        "                ),\n"
        "                out=self.compressed_slot_mapping_buffer,\n",
    )

    # Decode bounds must follow the same ownership unit as the compressed
    # cache.  Compress the global per-token bound first, then localize pool
    # rows.  localize(tokens)//ratio is wrong at pool boundaries (for example
    # six tokens -> one pool, but 3//4 on both DCP2 ranks -> zero pools).
    old_decode_order = (
        "            # DCP: localize the now-expanded per-token global bounds to this\n"
        "            # rank's owned KV. Done here (after expansion) so each token's global\n"
        "            # causal length is localized individually; see the comment above.\n"
        "            if dcp_local_seq_lens is not None:\n"
        "                seq_lens = self._dcp_localize_decode_seq_lens(\n"
        "                    seq_lens, num_decodes, seq_lens_is_buffer_view\n"
        "                )\n\n"
        "            # For DeepseekV4 (compress_ratio > 1), the indexer KV cache stores\n"
        "            # compressed tokens. Convert uncompressed seq_lens to compressed.\n"
        "            if self.compress_ratio > 1:\n"
        "                if seq_lens_is_buffer_view:\n"
        "                    seq_lens //= self.compress_ratio\n"
        "                else:\n"
        "                    # Copy to avoid mutating shared state; keeps CG address stable.\n"
        "                    self.expanded_seq_lens_buffer[:num_decodes] = (\n"
        "                        seq_lens // self.compress_ratio\n"
        "                    )\n"
        "                    self.expanded_seq_lens_buffer[num_decodes:num_decode_tokens] = 0\n"
        "                    seq_lens = self.expanded_seq_lens_buffer[:num_decode_tokens]\n"
    )
    new_decode_order = (
        "            # A compressed DCP cache is sharded in compressed-row units.\n"
        "            # First convert every expanded global token bound to pool rows,\n"
        "            # then localize those rows to this rank.\n"
        "            if self.compress_ratio > 1:\n"
        "                if seq_lens_is_buffer_view:\n"
        "                    seq_lens //= self.compress_ratio\n"
        "                else:\n"
        "                    self.expanded_seq_lens_buffer[:num_decode_tokens] = (\n"
        "                        seq_lens // self.compress_ratio\n"
        "                    )\n"
        "                    seq_lens = self.expanded_seq_lens_buffer[:num_decode_tokens]\n\n"
        "            if dcp_local_seq_lens is not None:\n"
        "                if self.compress_ratio > 1:\n"
        "                    seq_lens = self._dcp_localize_decode_seq_lens(\n"
        "                        seq_lens, num_decode_tokens, True\n"
        "                    )\n"
        "                else:\n"
        "                    seq_lens = self._dcp_localize_decode_seq_lens(\n"
        "                        seq_lens, num_decodes, seq_lens_is_buffer_view\n"
        "                    )\n"
    )
    replace_once(metadata, old_decode_order, new_decode_order)

    # The newer unified MLA DCP wrapper probes the standard FlashMLA metadata
    # fields even for sparse backends.  B12x owns its unified decode/prefill
    # metadata directly, so both standard sub-metadata objects are absent; an
    # explicit None property lets the wrapper take its documented sparse
    # fallback (`seq_lens`) without coupling B12x to the FlashMLA dataclasses.
    b12x_mla = root / "v1/attention/backends/mla/b12x_mla_sparse.py"
    replace_once(
        b12x_mla,
        "    block_size: int = 64\n"
        "    topk_tokens: int = 2048\n\n\n"
        "class B12xMLASparseMetadataBuilder",
        "    block_size: int = 64\n"
        "    topk_tokens: int = 2048\n\n"
        "    @property\n"
        "    def decode(self) -> None:\n"
        "        return None\n\n"
        "    @property\n"
        "    def prefill(self) -> None:\n"
        "        return None\n\n\n"
        "class B12xMLASparseMetadataBuilder",
    )

    mla_attention = root / "model_executor/layers/attention/mla_attention.py"
    replace_once(
        mla_attention,
        "                seq_lens = (\n"
        "                    attn_metadata.decode.seq_lens\n"
        "                    if attn_metadata.decode is not None\n"
        "                    else cast(torch.Tensor, attn_metadata.seq_lens)[  # type: ignore[attr-defined]\n"
        "                        : attn_metadata.num_decodes\n"
        "                    ]\n"
        "                )\n"
        "                query_start_loc = attn_metadata.query_start_loc[\n"
        "                    : attn_metadata.num_decodes + 1\n"
        "                ]\n",
        "                if self.impl.is_sparse and attn_metadata.decode is None:\n"
        "                    # Sparse B12x routes decode and prefill tokens through\n"
        "                    # one MQA call, so combine every request.  The standard\n"
        "                    # wrapper's num_decodes slice is empty on pure prefill.\n"
        "                    seq_lens = cast(torch.Tensor, attn_metadata.seq_lens)[\n"
        "                        : attn_metadata.num_reqs\n"
        "                    ]\n"
        "                    query_start_loc = attn_metadata.query_start_loc[\n"
        "                        : attn_metadata.num_reqs + 1\n"
        "                    ]\n"
        "                else:\n"
        "                    seq_lens = (\n"
        "                        attn_metadata.decode.seq_lens\n"
        "                        if attn_metadata.decode is not None\n"
        "                        else cast(torch.Tensor, attn_metadata.seq_lens)[  # type: ignore[attr-defined]\n"
        "                            : attn_metadata.num_decodes\n"
        "                        ]\n"
        "                    )\n"
        "                    query_start_loc = attn_metadata.query_start_loc[\n"
        "                        : attn_metadata.num_decodes + 1\n"
        "                    ]\n",
    )

    kv_interface = root / "v1/kv_cache_interface.py"
    replace_once(
        kv_interface,
        "    def max_num_blocks_per_req(self, vllm_config: VllmConfig, max_len: int) -> int:\n"
        "        # One block per request for the request's whole lifetime; caps the\n",
        "    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:\n"
        "        # The tail is replicated one-page scratch on every DCP rank, not\n"
        "        # a sharded sliding window.  Avoid the base SW DCP assertion.\n"
        "        return self.page_size_bytes\n\n"
        "    def max_num_blocks_per_req(self, vllm_config: VllmConfig, max_len: int) -> int:\n"
        "        # One block per request for the request's whole lifetime; caps the\n",
    )

    # Mamba recurrent state is already explicitly replicated across DCP ranks:
    # MambaManager.__init__ undoes the base manager's DCP block-size scaling.
    # Prefix lookup is therefore rank-identical and does not need the old
    # blanket DCP prohibition (PCP remains guarded).
    cache_manager = root / "v1/core/single_type_kv_cache_manager.py"
    replace_once(
        cache_manager,
        "        assert dcp_world_size == 1, \"DCP not support mamba now.\"\n"
        "        assert pcp_world_size == 1, \"PCP not support mamba now.\"\n",
        "        assert pcp_world_size == 1, \"PCP not support mamba now.\"\n",
    )

    # Hybrid DCP only needs to reason about cache groups that participate in
    # prefix caching.  KpoolTailSpec is replicated one-page request scratch and
    # explicitly opts out, so skip it before enforcing the Full/Mamba whitelist.
    coordinator = root / "v1/core/kv_cache_coordinator.py"
    replace_once(
        coordinator,
        "            for g in kv_cache_config.kv_cache_groups:\n"
        "                assert isinstance(g.kv_cache_spec, (FullAttentionSpec, MambaSpec)), (\n"
        "                    \"DCP with hybrid KV cache layouts only supports \"\n"
        "                    \"full-attention and Mamba groups, got: \"\n"
        "                    f\"{type(g.kv_cache_spec).__name__}.\"\n"
        "                )\n",
        "            for g in kv_cache_config.kv_cache_groups:\n"
        "                if not g.kv_cache_spec.participates_in_prefix_caching:\n"
        "                    continue\n"
        "                assert isinstance(g.kv_cache_spec, (FullAttentionSpec, MambaSpec)), (\n"
        "                    \"DCP with hybrid KV cache layouts only supports \"\n"
        "                    \"full-attention and Mamba groups, got: \"\n"
        "                    f\"{type(g.kv_cache_spec).__name__}.\"\n"
        "                )\n",
    )

    # The GLM k-pool prefill path predates DCP.  Gather only this rank's pool
    # rows, compute its local exact top-k, then merge scored candidates into
    # the global pool top-k before expanding pools back to model-token ids.
    kpool = root / "model_executor/layers/sparse_attn_indexer_kpool.py"
    replace_once(
        kpool,
        "    _merge_b12x_dcp_topk,\n",
        "    _merge_b12x_dcp_topk,\n"
        "    _merge_dcp_topk_global,\n",
    )
    replace_once(
        kpool,
        "        k_quant_full, k_scale_full = workspace_manager.get_simultaneous(\n"
        "            values_spec,\n"
        "            scales_spec,\n"
        "        )\n"
        "        for chunk in prefill_metadata.chunks if not short_prefill else ():\n"
        "            k_quant = k_quant_full[: chunk.total_seq_lens]\n"
        "            k_scale = k_scale_full[: chunk.total_seq_lens]\n\n"
        "            if not chunk.skip_kv_gather:\n"
        "                ops.cp_gather_indexer_k_quant_cache(\n"
        "                    kv_cache,\n"
        "                    k_quant,\n"
        "                    k_scale,\n"
        "                    chunk.block_table,\n"
        "                    chunk.cu_seq_lens,\n"
        "                )\n",
        "        k_quant_full, k_scale_full = workspace_manager.get_simultaneous(\n"
        "            values_spec,\n"
        "            scales_spec,\n"
        "        )\n"
        "        cfg = get_current_vllm_config_or_none()\n"
        "        from vllm.distributed import get_dcp_group\n\n"
        "        dcp_group = get_dcp_group()\n"
        "        dcp_world_size = int(dcp_group.world_size)\n"
        "        dcp_rank = int(dcp_group.rank_in_group)\n"
        "        cp_kv_cache_interleave_size = 1\n"
        "        if cfg is not None:\n"
        "            cp_kv_cache_interleave_size = int(\n"
        "                cfg.parallel_config.cp_kv_cache_interleave_size\n"
        "            )\n"
        "        for chunk in prefill_metadata.chunks if not short_prefill else ():\n"
        "            assert chunk.local_cu_seq_lens is not None\n"
        "            k_quant = k_quant_full[: chunk.max_local_total_seq_lens]\n"
        "            k_scale = k_scale_full[: chunk.max_local_total_seq_lens]\n\n"
        "            if not chunk.skip_kv_gather and chunk.local_total_seq_lens > 0:\n"
        "                ops.cp_gather_indexer_k_quant_cache(\n"
        "                    kv_cache,\n"
        "                    k_quant,\n"
        "                    k_scale,\n"
        "                    chunk.block_table,\n"
        "                    chunk.local_cu_seq_lens,\n"
        "                )\n",
    )
    # Insert after either CUDA/XPU top-k implementation has filled topk_dst.
    replace_once(
        kpool,
        "                    select_k,\n"
        "                )\n\n"
        "            if index_kpool > 1:\n"
        "                pool_ids = pool_topk.to(torch.int64)\n",
        "                    select_k,\n"
        "                )\n\n"
        "            _merge_dcp_topk_global(\n"
        "                logits,\n"
        "                topk_dst,\n"
        "                select_k,\n"
        "                dcp_rank,\n"
        "                dcp_world_size,\n"
        "                cp_kv_cache_interleave_size,\n"
        "                row_starts=chunk.cu_seqlen_ks,\n"
        "            )\n\n"
        "            if index_kpool > 1:\n"
        "                pool_ids = pool_topk.to(torch.int64)\n",
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} VLLM_PACKAGE_ROOT")
    port(Path(sys.argv[1]))
    print("B12x DCP compatibility port for GLM-5.3 applied")
