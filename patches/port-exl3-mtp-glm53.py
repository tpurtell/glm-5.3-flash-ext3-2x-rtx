#!/usr/bin/env python3
"""Improve GLM-5.3 MTP/EXL3 loader diagnostics while porting draft weights."""

from __future__ import annotations

import sys
from pathlib import Path


def main(root: Path) -> None:
    path = root / "models/glm5next/nvidia/mtp.py"
    text = path.read_text()
    old = "                    param = params_dict[name_mapped]\n"
    new = (
        "                    if name_mapped not in params_dict:\n"
        "                        draft_prefix = (\n"
        "                            f\"model.layers.{spec_layer}.mtp_block.mlp.experts\"\n"
        "                        )\n"
        "                        draft_params = sorted(\n"
        "                            key for key in params_dict if draft_prefix in key\n"
        "                        )\n"
        "                        quant_debug = {\n"
        '                            "type": type(self.quant_config).__name__,\n'
        '                            "standard_fused_moe": getattr(\n'
        '                                self.quant_config, "standard_fused_moe", None\n'
        "                            ),\n"
        '                            "draft_prefix_matches": getattr(\n'
        '                                self.quant_config, "_moe_prefix_is_exl3",\n'
        "                                lambda *_: None,\n"
        "                            )(draft_prefix + \".routed_experts\"),\n"
        '                            "layer_45_storage": sorted(\n'
        "                                key\n"
        '                                for key in getattr(\n'
        '                                    self.quant_config, "tensor_storage", {}\n'
        "                                )\n"
        '                                if "layers.45" in key\n'
        "                            )[:12],\n"
        "                        }\n"
        "                        raise KeyError(\n"
        "                            f\"{name_mapped}; registered draft expert params: \"\n"
        "                            f\"{draft_params[:32]}; quantization: {quant_debug}\"\n"
        "                        )\n"
        "                    param = params_dict[name_mapped]\n"
    )
    if new in text:
        print("[skip] MTP EXL3 parameter diagnostics")
        return
    if text.count(old) != 1:
        raise RuntimeError(f"MTP expert parameter anchor count: {text.count(old)}")
    path.write_text(text.replace(old, new, 1))
    compile(path.read_text(), str(path), "exec")
    print("[ok]   MTP EXL3 parameter diagnostics")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} VLLM_ROOT")
    main(Path(sys.argv[1]))
