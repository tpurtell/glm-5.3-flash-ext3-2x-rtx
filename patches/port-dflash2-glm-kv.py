#!/usr/bin/env python3
"""Keep GLM-5.3's shared hybrid cache layout when DFlash adds draft KV."""

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


def replace_exact(
    path: Path, old: str, new: str, count: int, description: str
) -> None:
    source = path.read_text(encoding="utf-8")
    matches = source.count(old)
    if matches == 0 and source.count(new) == count:
        print(f"[skip] {description}")
        return
    if matches != count:
        raise RuntimeError(
            f"{description}: expected {count} insertion points in {path}, got {matches}"
        )
    path.write_text(source.replace(old, new), encoding="utf-8")
    print(f"[ok]   {description}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("vllm_root", type=Path)
    args = parser.parse_args()
    path = args.vllm_root / "v1/core/kv_cache_utils.py"

    replace_once(
        path,
        "    attn_specs = {\n"
        "        k: v\n"
        "        for k, v in kv_cache_spec.items()\n"
        "        if not isinstance(v, (MambaSpec, KpoolTailSpec))\n"
        "    }\n"
        "    if not mamba_specs or not all(\n"
        "        type(s) is MLAAttentionSpec for s in attn_specs.values()\n"
        "    ):\n"
        "        return None\n"
        "    mla_specs = cast(dict[str, MLAAttentionSpec], attn_specs)\n",
        "    # A DFlash drafter registers its own ordinary sliding-attention KV\n"
        "    # layers in the same global spec. Keep those in independent groups\n"
        "    # instead of letting them disable GLM's MLA/mamba slot-sharing path.\n"
        "    mla_specs = {\n"
        "        k: v for k, v in kv_cache_spec.items() if type(v) is MLAAttentionSpec\n"
        "    }\n"
        "    draft_specs = {\n"
        "        k: v\n"
        "        for k, v in kv_cache_spec.items()\n"
        "        if not isinstance(v, (MambaSpec, KpoolTailSpec))\n"
        "        and type(v) is not MLAAttentionSpec\n"
        "    }\n"
        "    if not mamba_specs or not mla_specs:\n"
        "        return None\n"
        "    mla_specs = cast(dict[str, MLAAttentionSpec], mla_specs)\n",
        "separate DFlash cache layers from GLM target cache layers",
    )
    replace_once(
        path,
        "    uniform_spec = UniformTypeKVCacheSpecs.from_specs(attn_specs)\n"
        "    assert uniform_spec is not None\n\n"
        "    # Tail cache:",
        "    uniform_spec = UniformTypeKVCacheSpecs.from_specs(mla_specs)\n"
        "    assert uniform_spec is not None\n"
        "    draft_group = None\n"
        "    if draft_specs:\n"
        "        draft_uniform = UniformTypeKVCacheSpecs.from_specs(draft_specs)\n"
        "        if draft_uniform is None:\n"
        "            return None\n"
        "        draft_group = KVCacheGroupSpec(list(draft_specs), draft_uniform)\n\n"
        "    # Tail cache:",
        "construct an independent uniform DFlash cache group",
    )
    replace_once(
        path,
        "        [KVCacheGroupSpec(list(attn_specs), uniform_spec)]\n"
        "        + ([tail_group] if tail_group is not None else [])\n"
        "        + create_kv_cache_group_specs(padded_specs, mamba_grouped_names)\n",
        "        [KVCacheGroupSpec(list(mla_specs), uniform_spec)]\n"
        "        + ([tail_group] if tail_group is not None else [])\n"
        "        + create_kv_cache_group_specs(padded_specs, mamba_grouped_names)\n"
        "        + ([draft_group] if draft_group is not None else [])\n",
        "append DFlash after GLM target cache groups",
    )
    replace_once(
        path,
        "        list[str],\n"
        "        int,\n"
        "    ]\n"
        "    | None\n"
        "):\n"
        "    \"\"\"Detect the `_get_kv_cache_groups_glm5_next` layout from the\n",
        "        list[str],\n"
        "        int,\n"
        "        list[KVCacheGroupSpec],\n"
        "    ]\n"
        "    | None\n"
        "):\n"
        "    \"\"\"Detect the `_get_kv_cache_groups_glm5_next` layout from the\n",
        "extend the GLM layout descriptor with draft groups",
    )
    replace_once(
        path,
        "      - (attn_group, mamba_groups, mla_names, idx_names, mla_page, idx_page,\n"
        "         tail_names, tail_page)\n",
        "      - (attn_group, mamba_groups, mla_names, idx_names, mla_page, idx_page,\n"
        "         tail_names, tail_page, draft_groups)\n",
        "document DFlash groups in the GLM layout descriptor",
    )
    replace_once(
        path,
        "    attn_group: KVCacheGroupSpec | None = None\n"
        "    tail_group: KVCacheGroupSpec | None = None\n"
        "    for g in uniform_groups:\n"
        "        group_inner = cast(UniformTypeKVCacheSpecs, g.kv_cache_spec).kv_cache_specs\n"
        "        if all(type(s) is MLAAttentionSpec for s in group_inner.values()):\n"
        "            attn_group = g\n"
        "        elif all(isinstance(s, KpoolTailSpec) for s in group_inner.values()):\n"
        "            tail_group = g\n",
        "    attn_group: KVCacheGroupSpec | None = None\n"
        "    tail_group: KVCacheGroupSpec | None = None\n"
        "    draft_groups: list[KVCacheGroupSpec] = []\n"
        "    for g in uniform_groups:\n"
        "        group_inner = cast(UniformTypeKVCacheSpecs, g.kv_cache_spec).kv_cache_specs\n"
        "        if all(type(s) is MLAAttentionSpec for s in group_inner.values()):\n"
        "            attn_group = g\n"
        "        elif all(isinstance(s, KpoolTailSpec) for s in group_inner.values()):\n"
        "            tail_group = g\n"
        "        elif all(\n"
        "            isinstance(s, AttentionSpec) and type(s) is not MLAAttentionSpec\n"
        "            for s in group_inner.values()\n"
        "        ):\n"
        "            draft_groups.append(g)\n"
        "        else:\n"
        "            return None\n",
        "recognize independent DFlash groups in GLM layout detection",
    )
    replace_once(
        path,
        "        tail_names,\n"
        "        tail_page,\n"
        "    )\n\n\n"
        "def get_kv_cache_config_from_groups",
        "        tail_names,\n"
        "        tail_page,\n"
        "        draft_groups,\n"
        "    )\n\n\n"
        "def get_kv_cache_config_from_groups",
        "return draft groups from GLM layout detection",
    )
    replace_once(
        path,
        "        _, _, mla_names, idx_names, mla_page, idx_page, _, _ = glm5\n"
        "        return len(mla_names) * mla_page + len(idx_names) * idx_page\n",
        "        _, _, mla_names, idx_names, mla_page, idx_page, _, _, draft_groups = glm5\n"
        "        return (\n"
        "            len(mla_names) * mla_page\n"
        "            + len(idx_names) * idx_page\n"
        "            + sum(g.kv_cache_spec.page_size_bytes for g in draft_groups)\n"
        "        )\n",
        "include DFlash tensors in GLM per-block accounting",
    )
    replace_exact(
        path,
        "            tail_names,\n"
        "            _tail_page,\n"
        "        ) = glm5n\n",
        "            tail_names,\n"
        "            _tail_page,\n"
        "            draft_groups,\n"
        "        ) = glm5n\n",
        2,
        "unpack DFlash groups in GLM allocation and max-memory accounting",
    )
    replace_once(
        path,
        "        per_block = len(mla_names) * mla_page + len(idx_names) * idx_page\n",
        "        per_block = (\n"
        "            len(mla_names) * mla_page\n"
        "            + len(idx_names) * idx_page\n"
        "            + sum(g.kv_cache_spec.page_size_bytes for g in draft_groups)\n"
        "        )\n",
        "budget DFlash tensors in GLM cache allocation",
    )
    replace_once(
        path,
        "            for i in range(len(idx_names))\n"
        "        ]\n"
        "    elif _use_packed_kv_cache_config",
        "            for i in range(len(idx_names))\n"
        "        ] + [\n"
        "            KVCacheTensor(\n"
        "                size=group_spec.kv_cache_specs[layer_name].page_size_bytes\n"
        "                * num_blocks,\n"
        "                shared_by=[layer_name],\n"
        "            )\n"
        "            for group in draft_groups\n"
        "            for group_spec in [cast(UniformTypeKVCacheSpecs, group.kv_cache_spec)]\n"
        "            for layer_name in group.layer_names\n"
        "        ]\n"
        "    elif _use_packed_kv_cache_config",
        "allocate independent DFlash tensors beside shared GLM tensors",
    )
    replace_once(
        path,
        "        if tail_names:\n"
        "            # Tail: 1 block/req (KpoolTailSpec.max_admission_blocks_per_request\n"
        "            # == 1), drawn from the shared pool.\n"
        "            blocks_needed += 1\n"
        "        return blocks_needed * (len(mla_names) * mla_page + len(idx_names) * idx_page)\n",
        "        if tail_names:\n"
        "            # Tail: 1 block/req (KpoolTailSpec.max_admission_blocks_per_request\n"
        "            # == 1), drawn from the shared pool.\n"
        "            blocks_needed += 1\n"
        "        blocks_needed += sum(\n"
        "            cast(UniformTypeKVCacheSpecs, group.kv_cache_spec).max_memory_usage_pages(\n"
        "                vllm_config\n"
        "            )\n"
        "            for group in draft_groups\n"
        "        )\n"
        "        per_block = (\n"
        "            len(mla_names) * mla_page\n"
        "            + len(idx_names) * idx_page\n"
        "            + sum(g.kv_cache_spec.page_size_bytes for g in draft_groups)\n"
        "        )\n"
        "        return blocks_needed * per_block\n",
        "include DFlash demand in GLM max-memory accounting",
    )


if __name__ == "__main__":
    main()
