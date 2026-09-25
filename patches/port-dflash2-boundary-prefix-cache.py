#!/usr/bin/env python3
"""Let DFlash2 prefix-cache hits land on the aligned boundary.

With DFlash2, no KV group is annotated as EAGLE, and the GLM MTP scoping in
port-glm53-mtp-prefix-cache.py only recognizes unwrapped ``model.layers.*``
draft layers. The drafter's replicated sliding-window group does not match, so
vLLM's all-group fallback marks every group as EAGLE, including the three
target-only KDA groups. Every hit then drops one block, and the scheduler
backs its Mamba checkpoint off by one more block: each request recomputes
roughly two blocks of an already-cached prefix.

DFlash2 context KV is a per-position projection of the target hidden state, so
the drafter needs no lookahead block. When GLM53_DFLASH_BOUNDARY_LOOKUP=1,
clear the EAGLE groups and keep the scheduler back-off only while some group
still drops its last matching block. Unset or 0 keeps the current behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path


def replace_once(path: Path, old: str, new: str, description: str) -> None:
    text = path.read_text()
    if new in text:
        print(f"[skip] {description}")
        return
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{description}: expected one anchor, found {count}")
    path.write_text(text.replace(old, new, 1))
    print(f"[ok]   {description}")


def main(root: Path) -> None:
    coordinator = root / "v1/core/kv_cache_coordinator.py"
    scheduler = root / "v1/core/sched/scheduler.py"

    replace_once(
        coordinator,
        "            if not self.eagle_group_ids:\n"
        "                self.eagle_group_ids = set(\n"
        "                    range(len(kv_cache_config.kv_cache_groups))\n"
        "                )\n"
        "\n"
        "        self.single_type_managers = tuple(\n",
        "            if not self.eagle_group_ids:\n"
        "                self.eagle_group_ids = set(\n"
        "                    range(len(kv_cache_config.kv_cache_groups))\n"
        "                )\n"
        "        # DFlash2 context KV is a per-position projection of the target\n"
        "        # hidden state: no group needs the EAGLE lookahead block.\n"
        "        import os\n"
        "\n"
        "        if use_eagle and os.environ.get(\"GLM53_DFLASH_BOUNDARY_LOOKUP\") == \"1\":\n"
        "            logger.info(\n"
        "                \"DFlash2 boundary prefix lookup: EAGLE block drop disabled \"\n"
        "                \"(was groups %s)\", sorted(self.eagle_group_ids)\n"
        "            )\n"
        "            self.eagle_group_ids = set()\n"
        "\n"
        "        self.single_type_managers = tuple(\n",
        "scope DFlash2 prefix hits to the aligned boundary",
    )
    replace_once(
        scheduler,
        "        last_cache_position = request.num_tokens - request.num_tokens % block_size\n"
        "        if self.use_eagle:\n"
        "            last_cache_position = max(last_cache_position - block_size, 0)\n",
        "        last_cache_position = request.num_tokens - request.num_tokens % block_size\n"
        "        # Back off only while some group still drops its last matching block.\n"
        "        if self.use_eagle and self.kv_cache_manager.coordinator.eagle_group_ids:\n"
        "            last_cache_position = max(last_cache_position - block_size, 0)\n",
        "keep the Mamba checkpoint back-off only for EAGLE groups",
    )
    for path in (coordinator, scheduler):
        compile(path.read_text(), str(path), "exec")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} VLLM_ROOT")
    main(Path(sys.argv[1]))
