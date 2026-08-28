#!/usr/bin/env python3
"""Keep vLLM's EAGLE/MTP cache drop scoped to GLM's draft MLA group.

GLM-5.3's target layers are wrapped as ``language_model.model.layers.*`` while
the MTP layer is registered as ``model.layers.*``. vLLM's generic fallback
marks every hybrid group as EAGLE when no group was annotated, including the
three target-only KDA groups. Detect the GLM wrapper and mark only reusable
groups containing an unwrapped draft layer; retain the conservative fallback
for all other model layouts.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(root: Path) -> None:
    path = root / "v1/core/kv_cache_coordinator.py"
    text = path.read_text()
    old = (
        "        # Conservatively fall back to flag all groups when no group is flagged.\n"
        "        if use_eagle and not self.eagle_group_ids:\n"
        "            self.eagle_group_ids = set(range(len(kv_cache_config.kv_cache_groups)))\n"
    )
    new = (
        "        # GLM's wrapped target layers and unwrapped MTP layer share the MLA\n"
        "        # group. Mark only that group as EAGLE; target-only KDA groups must\n"
        "        # retain their exact aligned prefix checkpoints. Other model layouts\n"
        "        # keep vLLM's conservative all-group fallback.\n"
        "        if use_eagle and not self.eagle_group_ids:\n"
        "            is_wrapped_glm = any(\n"
        "                name.startswith(\"language_model.model.layers.\")\n"
        "                for group in kv_cache_config.kv_cache_groups\n"
        "                for name in group.layer_names\n"
        "            )\n"
        "            if is_wrapped_glm:\n"
        "                self.eagle_group_ids = {\n"
        "                    i\n"
        "                    for i, group in enumerate(kv_cache_config.kv_cache_groups)\n"
        "                    if group.kv_cache_spec.participates_in_prefix_caching\n"
        "                    and any(\n"
        "                        name.startswith(\"model.layers.\")\n"
        "                        for name in group.layer_names\n"
        "                    )\n"
        "                }\n"
        "            if not self.eagle_group_ids:\n"
        "                self.eagle_group_ids = set(\n"
        "                    range(len(kv_cache_config.kv_cache_groups))\n"
        "                )\n"
    )
    if new in text:
        print("[skip] scope GLM MTP prefix-cache drop")
    else:
        count = text.count(old)
        if count != 1:
            raise RuntimeError(
                "scope GLM MTP prefix-cache drop: "
                f"expected one anchor, found {count}"
            )
        path.write_text(text.replace(old, new, 1))
        print("[ok]   scope GLM MTP prefix-cache drop")
    compile(path.read_text(), str(path), "exec")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} VLLM_ROOT")
    main(Path(sys.argv[1]))
