#!/usr/bin/env python3
"""Replicate DFlash sliding KV while the GLM target uses DCP sharding."""

from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(path: Path, old: str, new: str, description: str) -> None:
    source = path.read_text(encoding="utf-8")
    if old not in source:
        if new in source:
            print(f"[skip] {description}")
            return
        raise RuntimeError(f"{description}: insertion point not found in {path}")
    if source.count(old) != 1:
        raise RuntimeError(f"{description}: insertion point is ambiguous in {path}")
    path.write_text(source.replace(old, new), encoding="utf-8")
    print(f"[ok]   {description}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("vllm_root", type=Path)
    args = parser.parse_args()

    interface = args.vllm_root / "v1/kv_cache_interface.py"
    replace_once(
        interface,
        "    indexes_kv_by_block_stride: bool = False\n\n"
        "    @property\n"
        "    def unpadded_page_size_bytes",
        "    indexes_kv_by_block_stride: bool = False\n"
        "    # A small speculative drafter can replicate its cache even when the\n"
        "    # target model shards long-context KV with DCP.\n"
        "    dcp_shard_count_override: int | None = None\n\n"
        "    @property\n"
        "    def unpadded_page_size_bytes",
        "add a per-attention DCP shard-count override",
    )
    replace_once(
        interface,
        "        parallel_config = vllm_config.parallel_config\n"
        "        kv_shard_count = parallel_config.decode_context_parallel_size\n"
        "        return cdiv(max_len, self.block_size * kv_shard_count)\n",
        "        parallel_config = vllm_config.parallel_config\n"
        "        kv_shard_count = (\n"
        "            self.dcp_shard_count_override\n"
        "            or parallel_config.decode_context_parallel_size\n"
        "        )\n"
        "        return cdiv(max_len, self.block_size * kv_shard_count)\n",
        "size draft block tables as replicated rather than DCP-sharded",
    )
    replace_once(
        interface,
        "        assert vllm_config.parallel_config.decode_context_parallel_size == 1, (\n"
        "            \"DCP not support sliding window.\"\n"
        "        )\n",
        "        dcp_shard_count = (\n"
        "            self.dcp_shard_count_override\n"
        "            or vllm_config.parallel_config.decode_context_parallel_size\n"
        "        )\n"
        "        assert dcp_shard_count == 1, \"DCP not support sliding window.\"\n",
        "admit an explicitly replicated draft sliding cache under target DCP",
    )
    replace_once(
        interface,
        "    def is_uniform_with_collection(\n"
        "        self, kv_cache_specs: dict[str, KVCacheSpec]\n"
        "    ) -> bool:\n"
        "        return all(\n"
        "            isinstance(spec, SlidingWindowSpec)\n"
        "            and spec.sliding_window == self.sliding_window\n"
        "            for spec in kv_cache_specs.values()\n"
        "        )\n\n\n"
        "@dataclass(frozen=True, kw_only=True)\n"
        "class SlidingWindowMLASpec(SlidingWindowSpec):",
        "    def is_uniform_with_collection(\n"
        "        self, kv_cache_specs: dict[str, KVCacheSpec]\n"
        "    ) -> bool:\n"
        "        return all(\n"
        "            isinstance(spec, SlidingWindowSpec)\n"
        "            and spec.sliding_window == self.sliding_window\n"
        "            for spec in kv_cache_specs.values()\n"
        "        )\n\n"
        "    @property\n"
        "    def participates_in_prefix_caching(self) -> bool:\n"
        "        # The replicated DFlash window is request-local scratch. Its\n"
        "        # manager blocks cannot share the target DCP2 hash granularity,\n"
        "        # and reusing a non-causal draft window would be incorrect.\n"
        "        # Normal sliding-window models remain cacheable.\n"
        "        return self.dcp_shard_count_override is None\n\n\n"
        "@dataclass(frozen=True, kw_only=True)\n"
        "class SlidingWindowMLASpec(SlidingWindowSpec):",
        "exclude replicated DFlash scratch from target prefix caching",
    )

    attention = (
        args.vllm_root / "model_executor/layers/attention/attention.py"
    )
    replace_once(
        attention,
        "                kv_quant_mode=quant_mode,\n"
        "                sliding_window=self.sliding_window,\n"
        "            ).real_page_size_bytes\n",
        "                kv_quant_mode=quant_mode,\n"
        "                sliding_window=self.sliding_window,\n"
        "                dcp_shard_count_override=getattr(\n"
        "                    self, \"dcp_shard_count_override\", None\n"
        "                ),\n"
        "            ).real_page_size_bytes\n",
        "carry the DFlash replication marker into SW page sizing",
    )
    replace_once(
        attention,
        "                sliding_window=self.sliding_window,\n"
        "                page_size_padded=shared_page,\n"
        "            )\n"
        "        elif self.kv_cache_dtype.startswith(\"turboquant_\"):",
        "                sliding_window=self.sliding_window,\n"
        "                page_size_padded=shared_page,\n"
        "                dcp_shard_count_override=getattr(\n"
        "                    self, \"dcp_shard_count_override\", None\n"
        "                ),\n"
        "            )\n"
        "        elif self.kv_cache_dtype.startswith(\"turboquant_\"):",
        "carry the DFlash replication marker into its final KV spec",
    )
    replace_once(
        attention,
        "            sw_block_size = _largest_kernel_block_within(\n"
        "                self.attn_backend, sw_per_token, shared_page, block_size\n"
        "            )\n"
        "            return SlidingWindowSpec(\n",
        "            sw_block_size = _largest_kernel_block_within(\n"
        "                self.attn_backend, sw_per_token, shared_page, block_size\n"
        "            )\n"
        "            if getattr(self, \"dcp_shard_count_override\", None) == 1:\n"
        "                # All cache groups draw IDs from one vLLM BlockPool.\n"
        "                # FlashAttention's generic 16-token minimum makes a\n"
        "                # 2K DFlash window consume 257 IDs per long request,\n"
        "                # and every ID also reserves GLM target tensors. A\n"
        "                # 128-token page minimizes that shared-pool tax while\n"
        "                # remaining a native FA multiple and keeping padding 0.\n"
        "                sw_block_size = 128\n"
        "            return SlidingWindowSpec(\n",
        "coarsen replicated DFlash KV pages for the shared global block pool",
    )

    model = args.vllm_root / "model_executor/models/qwen3_dflash.py"
    replace_once(
        model,
        "            sinks=self.attention_sink_bias,\n"
        "        )\n"
        "        self.causal = causal\n",
        "            sinks=self.attention_sink_bias,\n"
        "        )\n"
        "        # Target DCP shards GLM's long MLA history; this fixed 2K draft\n"
        "        # window is intentionally replicated on each tensor-parallel rank.\n"
        "        self.attn.dcp_shard_count_override = 1\n"
        "        # AttentionImplBase.__new__ currently reads the process-global\n"
        "        # DCP group rather than this draft model's copied parallel config.\n"
        "        # Override the instance so dense FlashAttention does not split a\n"
        "        # replicated DFlash cache or run a cross-rank LSE reduction.\n"
        "        self.attn.impl.dcp_world_size = 1\n"
        "        self.attn.impl.dcp_rank = 0\n"
        "        self.attn.impl.total_cp_world_size = 1\n"
        "        self.attn.impl.total_cp_rank = 0\n"
        "        self.attn.impl.need_to_return_lse_for_decode = False\n"
        "        self.causal = causal\n",
        "mark DFlash attention implementation and KV as replicated",
    )

    flash_attn = args.vllm_root / "v1/attention/backends/flash_attn.py"
    replace_once(
        flash_attn,
        "        except AssertionError:\n"
        "            # DCP might not be initialized in testing\n"
        "            self.dcp_world_size = 1\n"
        "            self.dcp_rank = 0\n\n"
        "        # Fused draft decode reuses the captured metadata object across draft\n",
        "        except AssertionError:\n"
        "            # DCP might not be initialized in testing\n"
        "            self.dcp_world_size = 1\n"
        "            self.dcp_rank = 0\n"
        "        if kv_cache_spec.dcp_shard_count_override == 1:\n"
        "            # DFlash's small dense window is fully replicated even when\n"
        "            # the target's MLA history is DCP-sharded. The builder must\n"
        "            # follow the cache spec, not the process-global DCP group.\n"
        "            self.dcp_world_size = 1\n"
        "            self.dcp_rank = 0\n\n"
        "        # Fused draft decode reuses the captured metadata object across draft\n",
        "build FlashAttention metadata as DCP1 for replicated draft cache",
    )

    cp_utils = args.vllm_root / "v1/worker/cp_utils.py"
    replace_once(
        cp_utils,
        "        for layer in layers.values():\n"
        "            get_attn_backend = getattr(layer, \"get_attn_backend\", None)\n",
        "        for layer in layers.values():\n"
        "            if getattr(layer, \"dcp_shard_count_override\", None) == 1:\n"
        "                # The DFlash2 draft window is deliberately replicated.\n"
        "                # Do not require target-DCP LSE support from its dense\n"
        "                # FlashAttention implementation.\n"
        "                continue\n"
        "            get_attn_backend = getattr(layer, \"get_attn_backend\", None)\n",
        "exclude replicated DFlash attention from target CP compatibility checks",
    )

    speculator = (
        args.vllm_root
        / "v1/worker/gpu/spec_decode/dflash/speculator.py"
    )
    replace_once(
        speculator,
        "        return replace(\n"
        "            self.vllm_config,\n"
        "            attention_config=replace(\n",
        "        return replace(\n"
        "            self.vllm_config,\n"
        "            parallel_config=replace(\n"
        "                self.vllm_config.parallel_config,\n"
        "                decode_context_parallel_size=1,\n"
        "            ),\n"
        "            attention_config=replace(\n",
        "build DFlash metadata as replicated DCP1 attention",
    )
    replace_once(
        speculator,
        "        self.query_cudagraph_manager = DFlashCudaGraphManager(\n"
        "            self.vllm_config,\n",
        "        self.query_cudagraph_manager = DFlashCudaGraphManager(\n"
        "            self.attn_vllm_config,\n",
        "capture DFlash graphs with its replicated DCP1 config",
    )
    replace_once(
        speculator,
        "        with set_forward_context(\n"
        "            attn_metadata,\n"
        "            self.vllm_config,\n",
        "        with set_forward_context(\n"
        "            attn_metadata,\n"
        "            self.attn_vllm_config,\n",
        "run DFlash attention under its replicated DCP1 forward context",
    )
    dflash_utils = (
        args.vllm_root
        / "v1/worker/gpu/spec_decode/dflash/utils.py"
    )
    replace_once(
        dflash_utils,
        "    draft_vllm_config = replace(\n"
        "        vllm_config,\n"
        "        attention_config=replace(\n",
        "    draft_vllm_config = replace(\n"
        "        vllm_config,\n"
        "        # The target shards MLA history with DCP, but DFlash2 is a\n"
        "        # small GQA drafter whose 2K window is replicated. Building\n"
        "        # its Attention objects with the target DCP topology makes the\n"
        "        # dense backend consume rank-local lengths and corrupts every\n"
        "        # proposal even when its KV allocation is replicated.\n"
        "        parallel_config=replace(\n"
        "            vllm_config.parallel_config,\n"
        "            decode_context_parallel_size=1,\n"
        "        ),\n"
        "        attention_config=replace(\n",
        "construct the DFlash model with replicated DCP1 attention",
    )

    coordinator = args.vllm_root / "v1/core/kv_cache_coordinator.py"
    replace_once(
        coordinator,
        "                kv_cache_group_id=i,\n"
        "                dcp_world_size=dcp_world_size,\n"
        "                pcp_world_size=pcp_world_size,\n",
        "                kv_cache_group_id=i,\n"
        "                dcp_world_size=(\n"
        "                    getattr(\n"
        "                        kv_cache_group.kv_cache_spec,\n"
        "                        \"dcp_shard_count_override\",\n"
        "                        None,\n"
        "                    )\n"
        "                    or dcp_world_size\n"
        "                ),\n"
        "                pcp_world_size=pcp_world_size,\n",
        "construct the DFlash sliding-window manager with DCP1",
    )
    replace_once(
        coordinator,
        "                assert isinstance(g.kv_cache_spec, (FullAttentionSpec, MambaSpec)), (\n"
        "                    \"DCP with hybrid KV cache layouts only supports \"\n"
        "                    \"full-attention and Mamba groups, got: \"\n"
        "                    f\"{type(g.kv_cache_spec).__name__}.\"\n"
        "                )\n",
        "                is_replicated_sliding_window = (\n"
        "                    isinstance(g.kv_cache_spec, SlidingWindowSpec)\n"
        "                    and g.kv_cache_spec.dcp_shard_count_override == 1\n"
        "                )\n"
        "                assert isinstance(\n"
        "                    g.kv_cache_spec, (FullAttentionSpec, MambaSpec)\n"
        "                ) or is_replicated_sliding_window, (\n"
        "                    \"DCP with hybrid KV cache layouts only supports \"\n"
        "                    \"full-attention, Mamba, and explicitly replicated \"\n"
        "                    \"sliding-window groups, got: \"\n"
        "                    f\"{type(g.kv_cache_spec).__name__}.\"\n"
        "                )\n",
        "admit the replicated DFlash cache in the hybrid DCP coordinator",
    )
    replace_once(
        coordinator,
        "        for manager in self.single_type_managers:\n"
        "            num_tokens_to_cache = aligned_num_computed_tokens\n"
        "            # EAGLE groups match one block past each aligned boundary and drop\n",
        "        for manager, group in zip(\n"
        "            self.single_type_managers, self.kv_cache_config.kv_cache_groups\n"
        "        ):\n"
        "            if not group.kv_cache_spec.participates_in_prefix_caching:\n"
        "                continue\n"
        "            num_tokens_to_cache = aligned_num_computed_tokens\n"
        "            # EAGLE groups match one block past each aligned boundary and drop\n",
        "skip draft scratch when materializing prefix-cache hashes",
    )


if __name__ == "__main__":
    main()
