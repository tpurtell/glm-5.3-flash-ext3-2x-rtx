#!/usr/bin/env python3
"""Port the qualified B12x vLLM adapter onto the GLM-5.3 day-zero tree.

The copied adapter is deliberately kept as provenance rather than rewritten
here.  This script changes only the newer vLLM registration surface, DCP group
aliases, environment declarations, and GLM-5.3's NoPE-to-GLM_NSA padding.
"""

from __future__ import annotations

import sys
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, found {count}: {old[:120]!r}")
    path.write_text(text.replace(old, new, 1))


def insert_after_once(path: Path, anchor: str, addition: str) -> None:
    replace_once(path, anchor, anchor + addition)


def port(root: Path) -> None:
    registry = root / "v1/attention/backends/registry.py"
    insert_after_once(
        registry,
        "    FLASHINFER_MLA_SPARSE = (\n"
        "        \"vllm.v1.attention.backends.mla.flashinfer_mla_sparse.\"\n"
        "        \"FlashInferMLASparseTRTLLMBackend\"\n"
        "    )\n",
        "    B12X_MLA_SPARSE = (\n"
        "        \"vllm.v1.attention.backends.mla.b12x_mla_sparse.\"\n"
        "        \"B12xMLASparseBackend\"\n"
        "    )\n",
    )

    cuda = root / "platforms/cuda.py"
    replace_once(
        cuda,
        "        elif device_capability.major == 12:\n"
        "            return [\n"
        "                AttentionBackendEnum.TRITON_MLA,\n"
        "                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,\n"
        "            ]\n",
        "        elif device_capability.major == 12:\n"
        "            return [\n"
        "                AttentionBackendEnum.TRITON_MLA,\n"
        "                AttentionBackendEnum.B12X_MLA_SPARSE,\n"
        "                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,\n"
        "            ]\n",
    )

    mla_attention = root / "model_executor/layers/attention/mla_attention.py"
    replace_once(
        mla_attention,
        '    if backend_name == "FLASHINFER_MLA_SPARSE_SM120" and kv_cache_dtype in (\n',
        '    if backend_name in ("FLASHINFER_MLA_SPARSE_SM120", "B12X_MLA_SPARSE") and kv_cache_dtype in (\n',
    )

    envs = root / "envs.py"
    insert_after_once(
        envs,
        "    VLLM_SPARSE_INDEXER_MAX_LOGITS_MB: int = 512\n",
        "    VLLM_USE_B12X_SPARSE_INDEXER: bool = False\n"
        "    VLLM_DCP_GLOBAL_TOPK: bool = False\n"
        "    VLLM_DCP_QUERY_SPLIT: bool = False\n"
        "    VLLM_DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS: int = 8192\n"
        "    VLLM_DCP_TOPK_OWNER_MERGE: bool = False\n"
        "    VLLM_B12X_MLA_CKV_GATHER: bool = False\n"
        "    VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS: int = 131072\n"
        "    VLLM_B12X_MLA_CKV_GATHER_MIN_TOKENS: int = 8192\n"
        "    VLLM_B12X_MLA_CKV_PREFETCH_DEPTH: int = 0\n"
        "    VLLM_B12X_MLA_CKV_PREFETCH_WORKSPACE_MIB: int = 0\n",
    )
    insert_after_once(
        envs,
        '    "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB": lambda: int(\n'
        '        os.getenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "512")\n'
        '    ),\n',
        '    "VLLM_USE_B12X_SPARSE_INDEXER": lambda: bool(\n'
        '        int(os.getenv("VLLM_USE_B12X_SPARSE_INDEXER", "0"))\n'
        '    ),\n'
        '    "VLLM_DCP_GLOBAL_TOPK": lambda: bool(\n'
        '        int(os.getenv("VLLM_DCP_GLOBAL_TOPK", "0"))\n'
        '    ),\n'
        '    "VLLM_DCP_QUERY_SPLIT": lambda: bool(\n'
        '        int(os.getenv("VLLM_DCP_QUERY_SPLIT", "0"))\n'
        '    ),\n'
        '    "VLLM_DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS": lambda: int(\n'
        '        os.getenv("VLLM_DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS", "8192")\n'
        '    ),\n'
        '    "VLLM_DCP_TOPK_OWNER_MERGE": lambda: bool(\n'
        '        int(os.getenv("VLLM_DCP_TOPK_OWNER_MERGE", "0"))\n'
        '    ),\n'
        '    "VLLM_B12X_MLA_CKV_GATHER": lambda: bool(\n'
        '        int(os.getenv("VLLM_B12X_MLA_CKV_GATHER", "0"))\n'
        '    ),\n'
        '    "VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS": lambda: int(\n'
        '        os.getenv("VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS", "131072")\n'
        '    ),\n'
        '    "VLLM_B12X_MLA_CKV_GATHER_MIN_TOKENS": lambda: int(\n'
        '        os.getenv("VLLM_B12X_MLA_CKV_GATHER_MIN_TOKENS", "8192")\n'
        '    ),\n'
        '    "VLLM_B12X_MLA_CKV_PREFETCH_DEPTH": lambda: int(\n'
        '        os.getenv("VLLM_B12X_MLA_CKV_PREFETCH_DEPTH", "0")\n'
        '    ),\n'
        '    "VLLM_B12X_MLA_CKV_PREFETCH_WORKSPACE_MIB": lambda: int(\n'
        '        os.getenv("VLLM_B12X_MLA_CKV_PREFETCH_WORKSPACE_MIB", "0")\n'
        '    ),\n',
    )

    indexer = root / "model_executor/layers/sparse_attn_indexer.py"
    replace_once(
        indexer,
        "from vllm.distributed import (\n"
        "    get_indexer_dcp_group,\n"
        "    get_query_split_group,\n"
        ")\n",
        "from vllm.distributed import get_dcp_group as get_indexer_dcp_group\n"
        "from vllm.distributed import get_dcp_group as get_query_split_group\n",
    )

    deepseek_v2 = root / "model_executor/models/deepseek_v2.py"
    insert_after_once(
        deepseek_v2,
        "        self.topk_indices_buffer = topk_indices_buffer\n",
        "        attention_backend = vllm_config.attention_config.backend\n"
        "        attention_backend_name = (\n"
        "            attention_backend\n"
        "            if isinstance(attention_backend, str)\n"
        "            else getattr(attention_backend, \"name\", None)\n"
        "        )\n"
        "        self.output_physical_slots = (\n"
        "            attention_backend_name == \"B12X_MLA_SPARSE\"\n"
        "            and vllm_config.parallel_config.decode_context_parallel_size == 1\n"
        "        )\n",
    )
    replace_once(
        deepseek_v2,
        "            self.topk_indices_buffer,\n"
        "        )\n\n"
        "        self.is_inplace_rope = is_inplace_rope\n",
        "            self.topk_indices_buffer,\n"
        "            output_physical_slots=self.output_physical_slots,\n"
        "            num_q_heads=self.n_head,\n"
        "        )\n\n"
        "        self.is_inplace_rope = is_inplace_rope\n",
    )

    # GLM-5.3 has its own kpool-aware indexer wrapper rather than the generic
    # DeepSeek Indexer above.  Its current implementation returns logical
    # token positions; the B12x adapter converts those to the MLA cache's
    # physical slots using its own graph-stable metadata buffers.
    glm_attention = root / "models/glm5next/nvidia/attention.py"
    insert_after_once(
        glm_attention,
        "        self.topk_indices_buffer = topk_indices_buffer\n",
        "        self.output_physical_slots = False\n",
    )

    backend = root / "v1/attention/backends/mla/b12x_mla_sparse.py"

    # The original adapter allocated request-id and converted-index buffers
    # only for DCP.  GLM's kpool indexer also needs them at DCP=1 because it
    # produces logical token positions rather than physical MLA cache slots.
    replace_once(
        backend,
        "        else:\n"
        "            self.req_id_per_token_buffer = None\n"
        "            self.page_table_1_buffer = None\n"
        "            self.nsa_cache_seqlens_buffer = None\n"
        "            self.req_ids_arange = None\n"
        "            self.ckv_page_table_1_buffer = None\n",
        "        else:\n"
        "            self.req_id_per_token_buffer = torch.empty(\n"
        "                (max_tokens,), dtype=torch.int32, device=device\n"
        "            )\n"
        "            self.page_table_1_buffer = torch.empty(\n"
        "                (max_tokens, self.topk_tokens),\n"
        "                dtype=torch.int32,\n"
        "                device=device,\n"
        "            )\n"
        "            self.nsa_cache_seqlens_buffer = torch.empty(\n"
        "                (max_tokens,), dtype=torch.int32, device=device\n"
        "            )\n"
        "            self.req_ids_arange = torch.arange(\n"
        "                max_tokens, dtype=torch.int32, device=device\n"
        "            )\n"
        "            self.ckv_page_table_1_buffer = None\n",
    )
    replace_once(
        backend,
        "        if cm.max_query_len <= 1 and num_tokens == cm.num_reqs:\n"
        "            if use_dcp:\n"
        "                assert self.req_ids_arange is not None\n"
        "                req_id_per_token_tensor = self.req_ids_arange[:num_tokens]\n",
        "        if cm.max_query_len <= 1 and num_tokens == cm.num_reqs:\n"
        "            assert self.req_ids_arange is not None\n"
        "            req_id_per_token_tensor = self.req_ids_arange[:num_tokens]\n"
        "            if use_dcp:\n",
    )
    replace_once(
        backend,
        "        cm = common_attn_metadata\n"
        "        num_tokens = cm.num_actual_tokens\n"
        "        if cm.max_query_len <= 1 and num_tokens == cm.num_reqs:\n",
        "        cm = common_attn_metadata\n"
        "        num_tokens = cm.num_actual_tokens\n"
        "        batch_topology = getattr(cm, \"batch_topology\", None)\n"
        "        if cm.max_query_len <= 1 and num_tokens == cm.num_reqs:\n",
    )
    replace_once(
        backend,
        "        elif cm.batch_topology is not None:\n"
        "            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (\n"
        "                cm.batch_topology.split_decodes_and_prefills(\n",
        "        elif batch_topology is not None:\n"
        "            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (\n"
        "                batch_topology.split_decodes_and_prefills(\n",
    )
    replace_once(
        backend,
        "            if cm.batch_topology is not None:\n"
        "                starts = cm.batch_topology.query_start_loc_np[: cm.num_reqs + 1]\n"
        "                query_lens = cm.batch_topology.query_lens_np\n"
        "                req_id_per_token_np = cm.batch_topology.req_id_per_token_np\n",
        "            if batch_topology is not None:\n"
        "                starts = batch_topology.query_start_loc_np[: cm.num_reqs + 1]\n"
        "                query_lens = batch_topology.query_lens_np\n"
        "                req_id_per_token_np = batch_topology.req_id_per_token_np\n",
    )
    replace_once(
        backend,
        "            req_ids = None\n"
        "            if use_dcp:\n"
        "                req_ids = np.zeros((num_tokens,), dtype=np.int32)\n"
        "                if num_query_tokens:\n"
        "                    req_ids[:num_query_tokens] = req_id_per_token_np\n",
        "            req_ids = np.zeros((num_tokens,), dtype=np.int32)\n"
        "            if num_query_tokens:\n"
        "                req_ids[:num_query_tokens] = req_id_per_token_np\n",
    )

    # Convert each logical per-request top-k token position into a flat
    # physical MLA-cache slot.  The output tensor is builder-owned, so this is
    # allocation-free and CUDA-graph safe.
    insert_after_once(
        backend,
        "def _mask_page_table_after_nsa_len(\n"
        "    page_table: torch.Tensor,\n"
        "    nsa_cache_seqlens: torch.Tensor,\n"
        ") -> None:\n"
        "    width = page_table.shape[1]\n"
        "    if width == 0 or page_table.shape[0] == 0:\n"
        "        return\n"
        "    block_n = 128\n"
        "    _mask_page_table_after_nsa_len_kernel[\n"
        "        (page_table.shape[0], triton.cdiv(width, block_n))\n"
        "    ](\n"
        "        page_table,\n"
        "        nsa_cache_seqlens,\n"
        "        page_table.stride(0),\n"
        "        page_table.stride(1),\n"
        "        width,\n"
        "        BLOCK_N=block_n,\n"
        "    )\n"
        "\n",
        "\n"
        "@triton.jit\n"
        "def _logical_topk_to_physical_slots_kernel(\n"
        "    req_id_ptr,\n"
        "    block_table_ptr,\n"
        "    logical_ptr,\n"
        "    physical_ptr,\n"
        "    bt_stride0,\n"
        "    bt_stride1,\n"
        "    logical_stride0,\n"
        "    logical_stride1,\n"
        "    physical_stride0,\n"
        "    physical_stride1,\n"
        "    max_blocks,\n"
        "    BLOCK_SIZE: tl.constexpr,\n"
        "    TOPK: tl.constexpr,\n"
        "    BLOCK_N: tl.constexpr,\n"
        "):\n"
        "    row = tl.program_id(0)\n"
        "    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)\n"
        "    col_mask = cols < TOPK\n"
        "    logical = tl.load(\n"
        "        logical_ptr + row * logical_stride0 + cols * logical_stride1,\n"
        "        mask=col_mask,\n"
        "        other=-1,\n"
        "    )\n"
        "    block_offset = logical // BLOCK_SIZE\n"
        "    valid = col_mask & (logical >= 0) & (block_offset < max_blocks)\n"
        "    req_id = tl.load(req_id_ptr + row)\n"
        "    physical_block = tl.load(\n"
        "        block_table_ptr\n"
        "        + req_id * bt_stride0\n"
        "        + block_offset * bt_stride1,\n"
        "        mask=valid,\n"
        "        other=-1,\n"
        "    )\n"
        "    physical = physical_block * BLOCK_SIZE + logical % BLOCK_SIZE\n"
        "    physical = tl.where(valid & (physical_block >= 0), physical, -1)\n"
        "    tl.store(\n"
        "        physical_ptr + row * physical_stride0 + cols * physical_stride1,\n"
        "        physical,\n"
        "        mask=col_mask,\n"
        "    )\n"
        "\n"
        "\n"
        "def _logical_topk_to_physical_slots(\n"
        "    req_ids: torch.Tensor,\n"
        "    block_table: torch.Tensor,\n"
        "    logical: torch.Tensor,\n"
        "    physical: torch.Tensor,\n"
        "    block_size: int,\n"
        ") -> None:\n"
        "    if logical.shape != physical.shape:\n"
        "        raise ValueError(\"logical/physical top-k shapes must match\")\n"
        "    if logical.shape[0] != req_ids.shape[0]:\n"
        "        raise ValueError(\"request-id rows must match top-k rows\")\n"
        "    block_n = 128\n"
        "    _logical_topk_to_physical_slots_kernel[\n"
        "        (logical.shape[0], triton.cdiv(logical.shape[1], block_n))\n"
        "    ](\n"
        "        req_ids,\n"
        "        block_table,\n"
        "        logical,\n"
        "        physical,\n"
        "        block_table.stride(0),\n"
        "        block_table.stride(1),\n"
        "        logical.stride(0),\n"
        "        logical.stride(1),\n"
        "        physical.stride(0),\n"
        "        physical.stride(1),\n"
        "        block_table.shape[1],\n"
        "        BLOCK_SIZE=block_size,\n"
        "        TOPK=logical.shape[1],\n"
        "        BLOCK_N=block_n,\n"
        "    )\n"
        "\n",
    )
    replace_once(backend, "        return [576]\n", "        return [512, 576]\n")
    insert_after_once(
        backend,
        "    supports_mha_prefill: bool = False\n",
        "    supports_dense_mha_prefill: bool = False\n",
    )
    replace_once(
        backend,
        '        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]\n'
        '        self.v_head_dim: int = mla_args.get("v_head_dim", 512)\n'
        "        # GLM_NSA contract: q_head_dim = kv_lora_rank (512) + qk_rope (64) = 576.\n"
        "        self.q_head_dim = self.kv_lora_rank + self.qk_rope_head_dim\n",
        '        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]\n'
        '        self.v_head_dim: int = mla_args.get("v_head_dim", 512)\n'
        "        # GLM-5.3 is NoPE (rope dim zero), while the current B12x GLM_NSA\n"
        "        # kernel intentionally shares the 512+64 packed-cache ABI. Add an\n"
        "        # exact-zero RoPE lane, matching the official SM120 backend.\n"
        "        self.rope_pad = 0\n"
        "        if self.qk_rope_head_dim == 0:\n"
        "            if self.kv_lora_rank != 512:\n"
        "                raise NotImplementedError(\n"
        "                    \"B12X_MLA_SPARSE NoPE padding requires kv_lora_rank=512\"\n"
        "                )\n"
        "            self.rope_pad = 64\n"
        "        self.q_head_dim = (\n"
        "            self.kv_lora_rank + self.qk_rope_head_dim + self.rope_pad\n"
        "        )\n",
    )
    replace_once(
        backend,
        "        expects_physical_slots = self.dcp_world_size == 1\n"
        "        if (\n"
        "            indexer is not None\n"
        "            and bool(indexer.output_physical_slots) != expects_physical_slots\n"
        "        ):\n"
        "            expected = \"physical\" if expects_physical_slots else \"logical\"\n"
        "            raise RuntimeError(\n"
        "                f\"B12X_MLA_SPARSE requires {expected} sparse-indexer output \"\n"
        "                f\"when dcp_world_size={self.dcp_world_size}\"\n"
        "            )\n",
        "        self._indexer_outputs_physical_slots = bool(\n"
        "            getattr(indexer, \"output_physical_slots\", False)\n"
        "        )\n"
        "        if self.dcp_world_size > 1 and self._indexer_outputs_physical_slots:\n"
        "            raise RuntimeError(\n"
        "                \"DCP B12X_MLA_SPARSE requires logical sparse-indexer output\"\n"
        "            )\n",
    )
    replace_once(
        backend,
        "        if not self._kv_fp8_rope:\n",
        "        # The packed GLM_NSA record reserves a BF16 RoPE tail even\n"
        "        # for GLM-5.3 NoPE. Keep it exactly zero.\n"
        "        if self.rope_pad:\n"
        "            k_pe = k_pe.new_zeros((k_pe.shape[0], 1, self.rope_pad))\n"
        "        if not self._kv_fp8_rope:\n",
    )
    replace_once(
        backend,
        "            q_all = q_buffer[:, :num_input_heads]\n"
        "            ops.concat_mla_q(ql_nope, q_pe, q_all)\n",
        "            q_all = q_buffer[:, :num_input_heads]\n"
        "            if self.rope_pad:\n"
        "                q_all.zero_()\n"
        "                q_all[..., : ql_nope.shape[-1]].copy_(ql_nope)\n"
        "            else:\n"
        "                ops.concat_mla_q(ql_nope, q_pe, q_all)\n",
    )
    replace_once(
        backend,
        "            if not exact_workspace_alias:\n"
        "                q_all.copy_(q.contiguous())\n",
        "            if not exact_workspace_alias:\n"
        "                if self.rope_pad and q.shape[-1] + self.rope_pad == self.q_head_dim:\n"
        "                    q_all.zero_()\n"
        "                    q_all[..., : q.shape[-1]].copy_(q.contiguous())\n"
        "                else:\n"
        "                    q_all.copy_(q.contiguous())\n",
    )
    replace_once(
        backend,
        "        else:\n"
        "            # Without DCP, the b12x indexer writes flat physical cache slots\n"
        "            # directly into the shared top-k buffer.\n"
        "            selected_indices = topk_indices\n"
        "            nsa_cache_seqlens = per_token_cache\n",
        "        else:\n"
        "            if self._indexer_outputs_physical_slots:\n"
        "                selected_indices = topk_indices\n"
        "                nsa_cache_seqlens = per_token_cache\n"
        "            else:\n"
        "                # GLM-5.3's kpool indexer returns logical token positions.\n"
        "                # Resolve them through this layer's MLA block table into\n"
        "                # the physical slots consumed by the B12x unified kernel.\n"
        "                assert attn_metadata.req_id_per_token is not None\n"
        "                assert attn_metadata.page_table_1 is not None\n"
        "                assert attn_metadata.nsa_cache_seqlens is not None\n"
        "                selected_indices = attn_metadata.page_table_1[\n"
        "                    :num_actual_toks, : topk_indices.shape[1]\n"
        "                ]\n"
        "                _logical_topk_to_physical_slots(\n"
        "                    attn_metadata.req_id_per_token[:num_actual_toks],\n"
        "                    attn_metadata.block_table,\n"
        "                    topk_indices,\n"
        "                    selected_indices,\n"
        "                    self.block_size,\n"
        "                )\n"
        "                nsa_cache_seqlens = attn_metadata.nsa_cache_seqlens[\n"
        "                    :num_actual_toks\n"
        "                ]\n"
        "                nsa_cache_seqlens.copy_(per_token_cache)\n"
        "                nsa_cache_seqlens.clamp_max_(topk_indices.shape[1])\n"
        "                _mask_page_table_after_nsa_len(\n"
        "                    selected_indices, nsa_cache_seqlens\n"
        "                )\n",
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} VLLM_PACKAGE_ROOT")
    port(Path(sys.argv[1]))
    print("B12x sparse MLA/indexer port for GLM-5.3 applied")
