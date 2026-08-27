#!/usr/bin/env python3
"""Harden the day-zero GLM runtime for dynamic long-prefill on SM12x."""

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
    # The day-zero gate enables Programmatic Dependent Launch on every
    # capability >= 9.  The closely related dual-Spark qualification found
    # KDA recurrent-state races on SM12x, where PDL has not been validated.
    cuda_platform = root / "platforms/cuda.py"
    replace_once(
        cuda_platform,
        "        return major >= 9\n",
        "        # PDL state-kernel ordering is not qualified on SM12x.\n"
        "        return major in (9, 10)\n",
    )

    # The top-k kernels only promise min(k, valid) outputs.  Initializing the
    # remainder makes short/local DCP rows deterministically invalid instead
    # of allowing uninitialized pool ids to address unwritten cache rows.
    kpool = root / "model_executor/layers/sparse_attn_indexer_kpool.py"
    replace_once(
        kpool,
        "                pool_topk = torch.empty(\n"
        "                    (num_rows, select_k), dtype=torch.int32, device=logits.device\n"
        "                )\n",
        "                pool_topk = torch.full(\n"
        "                    (num_rows, select_k),\n"
        "                    -1,\n"
        "                    dtype=torch.int32,\n"
        "                    device=logits.device,\n"
        "                )\n",
    )
    replace_once(
        kpool,
        "            pool_topk = torch.empty(\n"
        "                (num_rows, select_k),\n"
        "                dtype=torch.int32,\n"
        "                device=padded_q_quant_decode_tokens.device,\n"
        "            )\n",
        "            pool_topk = torch.full(\n"
        "                (num_rows, select_k),\n"
        "                -1,\n"
        "                dtype=torch.int32,\n"
        "                device=padded_q_quant_decode_tokens.device,\n"
        "            )\n",
    )

    kpool_ops = root / "models/glm5next/nvidia/ops/kpool_compress.py"
    replace_once(
        kpool_ops,
        "    hist_out = tl.where(pid >= 0, hist_val, -1)\n",
        "    hist_out = tl.where((pid >= 0) & (pid < pool_len), hist_val, -1)\n",
    )

    # Hybrid DCP packs attention pages across ranks, but Mamba recurrent state
    # remains replicated and its metadata still addresses every local-kernel
    # boundary.  The day-zero worker let the packed group spec shrink Mamba's
    # input block table to cdiv(max_len, block_size * DCP), yielding only 46
    # columns at 300K.  The GDN builder correctly needed 91 columns at its 3328
    # token kernel granularity and crashed at column 47.  Capacity must cover
    # the kernel-granular Mamba address space; this is only metadata memory.
    gpu_runner = root / "v1/worker/gpu/model_runner.py"
    replace_once(
        gpu_runner,
        "            max_num_blocks = cdiv(\n"
        "                block_table_max_model_len, spec.block_size * self.dcp_size\n"
        "            )\n"
        "            # For Mamba/Hybrid Model, KVCaches need extra blocks for speculative tokens\n"
        "            if isinstance(spec, MambaSpec):\n",
        "            max_num_blocks = cdiv(\n"
        "                block_table_max_model_len, spec.block_size * self.dcp_size\n"
        "            )\n"
        "            # Mamba state is replicated under DCP; its table is not sharded.\n"
        "            if isinstance(spec, MambaSpec):\n"
        "                max_num_blocks = cdiv(block_table_max_model_len, spec.block_size)\n"
        "            # For Mamba/Hybrid Model, KVCaches need extra blocks for speculative tokens\n"
        "            if isinstance(spec, MambaSpec):\n",
    )

    # Port the metadata plumbing pattern from vLLM PR #51540 to GLM's KDA
    # wrapper.  vLLM already derives chunk segmentation on the host, but GLM
    # discarded it and asked FLA to derive it again from a reused device
    # cu_seqlens buffer.  FLA's identity cache then returned the previous
    # 8192-token segmentation for a pressure-limited 6656-token chunk.  Pass a
    # distinct host-derived non-spec segmentation all the way into KDA.
    gdn = root / "v1/attention/backends/gdn_attn.py"
    replace_once(
        gdn,
        "    chunk_indices: torch.Tensor | None = None\n"
        "    chunk_offsets: torch.Tensor | None = None\n",
        "    chunk_indices: torch.Tensor | None = None\n"
        "    chunk_offsets: torch.Tensor | None = None\n"
        "    non_spec_chunk_indices: torch.Tensor | None = None\n"
        "    non_spec_chunk_offsets: torch.Tensor | None = None\n",
    )
    replace_once(
        gdn,
        "        chunk_indices: torch.Tensor | None = None\n"
        "        chunk_offsets: torch.Tensor | None = None\n"
        "        prefill_query_start_loc: torch.Tensor | None = None\n",
        "        chunk_indices: torch.Tensor | None = None\n"
        "        chunk_offsets: torch.Tensor | None = None\n"
        "        non_spec_chunk_indices: torch.Tensor | None = None\n"
        "        non_spec_chunk_offsets: torch.Tensor | None = None\n"
        "        prefill_query_start_loc: torch.Tensor | None = None\n",
    )
    replace_once(
        gdn,
        "                chunk_offsets = async_tensor_h2d(\n"
        "                    prepare_chunk_offsets(prefill_query_start_loc_cpu, FLA_CHUNK_SIZE),\n"
        "                    device=gpu_device,\n"
        "                )\n\n"
        "        if num_prefills > 0:\n",
        "                chunk_offsets = async_tensor_h2d(\n"
        "                    prepare_chunk_offsets(prefill_query_start_loc_cpu, FLA_CHUNK_SIZE),\n"
        "                    device=gpu_device,\n"
        "                )\n\n"
        "            # GLM KDA runs over the whole non-spec batch, including\n"
        "            # plain decodes in a mixed step, so it needs segmentation\n"
        "            # distinct from the decode-peeled prefill metadata above.\n"
        "            model_type = getattr(\n"
        "                self.vllm_config.model_config.hf_config, \"model_type\", None\n"
        "            )\n"
        "            if model_type == \"glm5_next\":\n"
        "                from vllm.third_party.flash_linear_attention.ops.index import (\n"
        "                    prepare_chunk_indices as prepare_non_spec_chunk_indices,\n"
        "                    prepare_chunk_offsets as prepare_non_spec_chunk_offsets,\n"
        "                )\n\n"
        "                assert non_spec_query_start_loc_cpu is not None\n"
        "                non_spec_chunk_indices = async_tensor_h2d(\n"
        "                    prepare_non_spec_chunk_indices(\n"
        "                        non_spec_query_start_loc_cpu, FLA_CHUNK_SIZE\n"
        "                    ),\n"
        "                    device=query_start_loc.device,\n"
        "                )\n"
        "                non_spec_chunk_offsets = async_tensor_h2d(\n"
        "                    prepare_non_spec_chunk_offsets(\n"
        "                        non_spec_query_start_loc_cpu, FLA_CHUNK_SIZE\n"
        "                    ),\n"
        "                    device=query_start_loc.device,\n"
        "                )\n\n"
        "        if num_prefills > 0:\n",
    )
    replace_once(
        gdn,
        "            has_initial_state=has_initial_state,\n"
        "            chunk_indices=chunk_indices,\n",
        "            has_initial_state=has_initial_state,\n"
        "            chunk_indices=chunk_indices,\n"
        "            non_spec_chunk_indices=non_spec_chunk_indices,\n"
        "            non_spec_chunk_offsets=non_spec_chunk_offsets,\n",
    )

    fla_kda = root / "third_party/flash_linear_attention/ops/kda.py"
    replace_once(
        fla_kda,
        "    cu_seqlens: torch.Tensor | None = None,\n"
        "    safe_gate: bool = False,\n"
        "    lower_bound: float = -5.0,\n"
        "):\n"
        "    chunk_size = FLA_CHUNK_SIZE\n"
        "    chunk_indices = (\n"
        "        prepare_chunk_indices(cu_seqlens, chunk_size)\n"
        "        if cu_seqlens is not None\n"
        "        else None\n"
        "    )\n"
        "    g = fused_kda_gate_chunk_cumsum(\n",
        "    cu_seqlens: torch.Tensor | None = None,\n"
        "    chunk_indices: torch.Tensor | None = None,\n"
        "    chunk_offsets: torch.Tensor | None = None,\n"
        "    safe_gate: bool = False,\n"
        "    lower_bound: float = -5.0,\n"
        "):\n"
        "    chunk_size = FLA_CHUNK_SIZE\n"
        "    if chunk_indices is None and cu_seqlens is not None:\n"
        "        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)\n"
        "    g = fused_kda_gate_chunk_cumsum(\n",
    )
    replace_once(
        fla_kda,
        "    use_qk_l2norm_in_kernel: bool = False,\n"
        "    cu_seqlens: torch.Tensor | None = None,\n"
        "    safe_gate: bool = False,\n"
        "    lower_bound: float = -5.0,\n"
        "    **kwargs,\n"
        "):\n"
        "    \"\"\"Run chunk KDA from raw gate projection using fused gate+cumsum.\"\"\"\n",
        "    use_qk_l2norm_in_kernel: bool = False,\n"
        "    cu_seqlens: torch.Tensor | None = None,\n"
        "    chunk_indices: torch.Tensor | None = None,\n"
        "    chunk_offsets: torch.Tensor | None = None,\n"
        "    safe_gate: bool = False,\n"
        "    lower_bound: float = -5.0,\n"
        "    **kwargs,\n"
        "):\n"
        "    \"\"\"Run chunk KDA from raw gate projection using fused gate+cumsum.\"\"\"\n",
    )
    replace_once(
        fla_kda,
        "        output_final_state=output_final_state,\n"
        "        cu_seqlens=cu_seqlens,\n"
        "        safe_gate=safe_gate,\n"
        "        lower_bound=lower_bound,\n"
        "    )\n"
        "    return o, final_state\n",
        "        output_final_state=output_final_state,\n"
        "        cu_seqlens=cu_seqlens,\n"
        "        chunk_indices=chunk_indices,\n"
        "        chunk_offsets=chunk_offsets,\n"
        "        safe_gate=safe_gate,\n"
        "        lower_bound=lower_bound,\n"
        "    )\n"
        "    return o, final_state\n",
    )

    # The day-zero FLA solve_tril API already accepts precomputed chunk
    # indices, but KDA fails to forward them.  This was the final stale-cache
    # path reached when the scheduler changed a long-prefill chunk from 8192
    # to 6656 tokens under KV pressure.
    replace_once(
        fla_kda,
        "    A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)\n",
        "    A = solve_tril(\n"
        "        A=A,\n"
        "        cu_seqlens=cu_seqlens,\n"
        "        chunk_indices=chunk_indices,\n"
        "        output_dtype=k.dtype,\n"
        "    )\n",
    )

    # The recurrent state-update kernel has the same issue for chunk offsets.
    # Its API already supports a caller-supplied map, so carry the host-derived
    # offsets beside the indices instead of rebuilding them on the GPU.
    replace_once(
        fla_kda,
        "    chunk_indices: torch.Tensor | None = None,\n"
        "    chunk_size: int = FLA_CHUNK_SIZE,\n"
        "):\n"
        "    # `g` must already be chunk-local cumulatively-summed",
        "    chunk_indices: torch.Tensor | None = None,\n"
        "    chunk_offsets: torch.Tensor | None = None,\n"
        "    chunk_size: int = FLA_CHUNK_SIZE,\n"
        "):\n"
        "    # `g` must already be chunk-local cumulatively-summed",
    )
    replace_once(
        fla_kda,
        "        cu_seqlens=cu_seqlens,\n"
        "        chunk_indices=chunk_indices,\n"
        "        use_exp2=True,\n",
        "        cu_seqlens=cu_seqlens,\n"
        "        chunk_indices=chunk_indices,\n"
        "        chunk_offsets=chunk_offsets,\n"
        "        use_exp2=True,\n",
    )
    replace_once(
        fla_kda,
        "        cu_seqlens=cu_seqlens,\n"
        "        chunk_indices=chunk_indices,\n"
        "        chunk_size=chunk_size,\n"
        "    )\n\n\n"
        "def chunk_kda(\n",
        "        cu_seqlens=cu_seqlens,\n"
        "        chunk_indices=chunk_indices,\n"
        "        chunk_offsets=chunk_offsets,\n"
        "        chunk_size=chunk_size,\n"
        "    )\n\n\n"
        "def chunk_kda(\n",
    )

    glm_kda = root / "models/glm5next/nvidia/kda.py"
    replace_once(
        glm_kda,
        "                use_qk_l2norm_in_kernel=True,\n"
        "                cu_seqlens=non_spec_query_start_loc,\n"
        "                safe_gate=safe_gate,\n",
        "                use_qk_l2norm_in_kernel=True,\n"
        "                cu_seqlens=non_spec_query_start_loc,\n"
        "                chunk_indices=attn_metadata_narrowed.non_spec_chunk_indices,\n"
        "                chunk_offsets=attn_metadata_narrowed.non_spec_chunk_offsets,\n"
        "                safe_gate=safe_gate,\n",
    )

    # Narrow auxiliary cache groups (GLM's one-block K-pool tail is one)
    # deliberately have block-table rows that do not cover raw token positions.
    # Both runner generations used the raw logical block index without checking
    # it against the row stride, causing an out-of-bounds device read.  CUDA can
    # hide this by returning data from another mapped allocation, so make every
    # out-of-range lane an explicit PAD slot in both kernels.
    block_table_v1 = root / "v1/worker/block_table.py"
    replace_once(
        block_table_v1,
        "        block_numbers = tl.load(\n"
        "            block_table_ptr + row_offset + block_indices,\n"
        "            mask=mask & is_local,\n"
        "            other=0,\n"
        "        ).to(tl.int64)\n"
        "        slot_offsets = local_block_offsets % block_size\n"
        "        slot_ids = block_numbers * block_size + slot_offsets\n"
        "        slot_ids = tl.where(is_local, slot_ids, PAD_ID)\n",
        "        in_range = block_indices < block_table_stride\n"
        "        block_numbers = tl.load(\n"
        "            block_table_ptr + row_offset + block_indices,\n"
        "            mask=mask & is_local & in_range,\n"
        "            other=0,\n"
        "        ).to(tl.int64)\n"
        "        slot_offsets = local_block_offsets % block_size\n"
        "        slot_ids = block_numbers * block_size + slot_offsets\n"
        "        slot_ids = tl.where(is_local & in_range, slot_ids, PAD_ID)\n",
    )

    block_table_v2 = root / "v1/worker/gpu/block_table.py"
    replace_once(
        block_table_v2,
        "        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)\n"
        "        positions = tl.load(pos + offset, mask=offset < end_idx, other=0)\n\n"
        "        block_indices = positions // (block_size * CP_SIZE)\n"
        "        block_offsets = positions % (block_size * CP_SIZE)\n"
        "        block_numbers = tl.load(\n"
        "            block_table_ptr + req_state_idx * block_table_stride + block_indices\n"
        "        )\n\n"
        "        if CP_SIZE == 1:\n"
        "            # Common case: Context parallelism is not used.\n"
        "            slot_ids = block_numbers * block_size + block_offsets\n"
        "        else:\n"
        "            # Context parallelism is used.\n"
        "            is_local = block_offsets // CP_INTERLEAVE % CP_SIZE == cp_rank\n"
        "            rounds = block_offsets // (CP_INTERLEAVE * CP_SIZE)\n"
        "            remainder = block_offsets % CP_INTERLEAVE\n"
        "            local_offsets = rounds * CP_INTERLEAVE + remainder\n"
        "            slot_ids = block_numbers * block_size + local_offsets\n"
        "            slot_ids = tl.where(is_local, slot_ids, PAD_ID)\n\n"
        "        tl.store(slot_mapping_ptr + offset, slot_ids, mask=offset < end_idx)\n",
        "        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)\n"
        "        token_mask = offset < end_idx\n"
        "        positions = tl.load(pos + offset, mask=token_mask, other=0)\n\n"
        "        block_indices = positions // (block_size * CP_SIZE)\n"
        "        block_offsets = positions % (block_size * CP_SIZE)\n"
        "        in_range = block_indices < block_table_stride\n"
        "        block_numbers = tl.load(\n"
        "            block_table_ptr + req_state_idx * block_table_stride + block_indices,\n"
        "            mask=token_mask & in_range,\n"
        "            other=0,\n"
        "        )\n\n"
        "        if CP_SIZE == 1:\n"
        "            # Common case: Context parallelism is not used.\n"
        "            slot_ids = block_numbers * block_size + block_offsets\n"
        "            slot_ids = tl.where(in_range, slot_ids, PAD_ID)\n"
        "        else:\n"
        "            # Context parallelism is used.\n"
        "            is_local = block_offsets // CP_INTERLEAVE % CP_SIZE == cp_rank\n"
        "            rounds = block_offsets // (CP_INTERLEAVE * CP_SIZE)\n"
        "            remainder = block_offsets % CP_INTERLEAVE\n"
        "            local_offsets = rounds * CP_INTERLEAVE + remainder\n"
        "            slot_ids = block_numbers * block_size + local_offsets\n"
        "            slot_ids = tl.where(is_local & in_range, slot_ids, PAD_ID)\n\n"
        "        tl.store(slot_mapping_ptr + offset, slot_ids, mask=token_mask)\n",
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} VLLM_PACKAGE_ROOT")
    port(Path(sys.argv[1]))
    print("GLM-5.3 SM12x long-prefill stability port applied")
