#!/usr/bin/env python3
"""Port the proven vLLM EXL3/B12x MoE method to GLM-5.3.

The source quantization method comes from tpurtell's working DeepSeek-V4
image.  Its fast standard-checkpoint path is already model-independent after
configuration: this patch registers the method in the newer GLM vLLM tree,
admits GLM's multimodal checkpoint prefix, and leaves non-expert GLM tensors
unquantized so they retain the checkpoint's native BF16/FP32 representation.
"""

from __future__ import annotations

import sys
from pathlib import Path


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text()
    if new in text:
        print(f"[skip] {label}")
        return
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    path.write_text(text.replace(old, new, 1))
    print(f"[ok]   {label}")


def main(root: Path) -> None:
    registry = root / "model_executor/layers/quantization/__init__.py"
    replace_once(
        registry,
        '    "deepseek_v4_fp8",\n    "online",',
        '    "deepseek_v4_fp8",\n    "exl3",\n    "online",',
        "register EXL3 quantization name",
    )
    replace_once(
        registry,
        "    from .experts_int8 import ExpertsInt8Config\n",
        "    from .experts_int8 import ExpertsInt8Config\n"
        "    from .exl3 import Exl3Config\n",
        "import EXL3 config",
    )
    replace_once(
        registry,
        '        "deepseek_v4_fp8": DeepseekV4FP8Config,\n'
        '        "humming": HummingConfig,',
        '        "deepseek_v4_fp8": DeepseekV4FP8Config,\n'
        '        "exl3": Exl3Config,\n'
        '        "humming": HummingConfig,',
        "map EXL3 config",
    )

    exl3 = root / "model_executor/layers/quantization/exl3.py"
    replace_once(
        exl3,
        "        self._configure_standard_deepseek(hf_config)\n",
        "        self._configure_standard_fused_moe(hf_config)\n",
        "call generalized standard MoE detector",
    )
    replace_once(
        exl3,
        "    def _configure_standard_deepseek(\n"
        "        self, hf_config: PretrainedConfig | None\n"
        "    ) -> None:\n"
        '        """Use b12x for a standard, unsliced, uniform DeepSeek EXL3 MoE."""\n\n'
        '        if hf_config is None or getattr(hf_config, "model_type", None) != "deepseek_v4":\n'
        "            return\n",
        "    def _configure_standard_fused_moe(\n"
        "        self, hf_config: PretrainedConfig | None\n"
        "    ) -> None:\n"
        '        """Use b12x for supported standard, unsliced, uniform EXL3 MoEs."""\n\n'
        '        model_type = getattr(hf_config, "model_type", None)\n'
        "        if model_type not in {\n"
        '            "deepseek_v4",\n'
        '            "glm5_next",\n'
        '            "glm5_next_text",\n'
        "        }:\n"
        "            return\n",
        "admit GLM-5.3 standard expert checkpoints",
    )
    replace_once(
        exl3,
        "        standard_expert = re.compile(\n"
        '            r"^(?:model\\.layers\\.\\d+|mtp\\.\\d+)\\.mlp\\.experts\\.\\d+\\."\n'
        '            r"(?:gate_proj|up_proj|down_proj)$"\n'
        "        )\n",
        "        standard_expert = re.compile(\n"
        '            r"^(?:model(?:\\.language_model)?\\.layers\\.\\d+|mtp\\.\\d+)"\n'
        '            r"\\.mlp\\.experts\\.\\d+\\."\n'
        '            r"(?:gate_proj|up_proj|down_proj)$"\n'
        "        )\n",
        "accept GLM multimodal checkpoint prefixes",
    )
    replace_once(
        exl3,
        "        self.standard_fused_moe = True\n"
        "        # GPTQModel emits standard Transformers expert names while NVIDIA's\n",
        "        self.standard_fused_moe = True\n"
        "        logger.info_once(\n"
        '            "EXL3 standard fused MoE enabled for %s: K%s %s via b12x",\n'
        "            model_type,\n"
        "            int(bits),\n"
        "            self.codebook,\n"
        "        )\n"
        '        if model_type != "deepseek_v4":\n'
        "            # GLM's vLLM implementation and checkpoint both use the\n"
        "            # standard gate_proj/down_proj/up_proj spelling.\n"
        "            return\n"
        "        # GPTQModel emits standard Transformers expert names while NVIDIA's\n",
        "avoid DeepSeek-only name aliases for GLM",
    )
    replace_once(
        exl3,
        "        if not base and self.standard_fused_moe:\n",
        "        if (\n"
        "            not base\n"
        "            and self.standard_fused_moe\n"
        '            and getattr(hf_config, "model_type", None) == "deepseek_v4"\n'
        "        ):\n",
        "retain native GLM non-expert tensors",
    )
    replace_once(
        exl3,
        "\n\nclass Exl3Config(QuantizationConfig):\n",
        "\n\nclass _B12xSmallNBF16Method(UnquantizedLinearMethod):\n"
        '    """Route narrow native-BF16 GLM linears through B12x at decode."""\n\n'
        "    def process_weights_after_loading(self, layer: Any) -> None:\n"
        "        super().process_weights_after_loading(layer)\n"
        "        from b12x.gemm import bf16_gemv\n\n"
        "        weight = layer.weight\n"
        "        if weight.ndim == 2 and weight.is_cuda and weight.dtype == torch.bfloat16:\n"
        "            layer.b12x_gemv_weight = weight.data.detach().clone().contiguous()\n"
        "            bf16_gemv.precompile(layer.b12x_gemv_weight)\n\n"
        "    def apply(\n"
        "        self, layer: Any, x: torch.Tensor, bias: torch.Tensor | None = None\n"
        "    ) -> torch.Tensor:\n"
        "        weight = getattr(layer, \"b12x_gemv_weight\", None)\n"
        "        if (\n"
        "            weight is not None\n"
        "            and bias is None\n"
        "            and x.dtype == torch.bfloat16\n"
        "            and weight.dtype == torch.bfloat16\n"
        "        ):\n"
        "            from b12x.gemm import bf16_gemv\n\n"
        "            x_2d = x.reshape(-1, x.shape[-1])\n"
        "            output = bf16_gemv.mm(x_2d, weight)\n"
        "            return output.reshape(*x.shape[:-1], weight.shape[0])\n"
        "        return super().apply(layer, x, bias)\n"
        "\n\nclass Exl3Config(QuantizationConfig):\n",
        "add B12x native-BF16 small-N linear method",
    )
    replace_once(
        exl3,
        "                if self._base_quant_config is not None:\n"
        "                    return self._base_quant_config.get_quant_method(layer, prefix)\n"
        "                return UnquantizedLinearMethod()\n"
        "            return Exl3LinearMethod(self)\n",
        "                if self._base_quant_config is not None:\n"
        "                    return self._base_quant_config.get_quant_method(layer, prefix)\n"
        "                from b12x.gemm.bf16_gemv import (\n"
        "                    SMALL_N_GEMV_MAX_OUT,\n"
        "                    SMALL_N_GEMV_MIN_IN,\n"
        "                    is_disabled as b12x_bf16_gemv_disabled,\n"
        "                )\n\n"
        "                out_size = int(getattr(layer, \"output_size\", 0) or 0)\n"
        "                in_size = int(getattr(layer, \"input_size\", 0) or 0)\n"
        "                enabled = os.getenv(\n"
        "                    \"B12X_EXL3_BF16_GEMV\", \"1\"\n"
        "                ) not in (\"\", \"0\", \"false\", \"False\")\n"
        "                if (\n"
        "                    enabled\n"
        "                    and not b12x_bf16_gemv_disabled()\n"
        "                    and 0 < out_size <= SMALL_N_GEMV_MAX_OUT\n"
        "                    and in_size >= SMALL_N_GEMV_MIN_IN\n"
        "                ):\n"
        "                    logger.info_once(\n"
        "                        \"B12x BF16 small-N GEMV enabled for native GLM linears.\"\n"
        "                    )\n"
        "                    return _B12xSmallNBF16Method()\n"
        "                return UnquantizedLinearMethod()\n"
        "            return Exl3LinearMethod(self)\n",
        "route narrow native-BF16 GLM linears through B12x",
    )
    replace_once(
        exl3,
        "        topk = int(topk_ids.shape[1])\n"
        "        device_index = x.device.index\n"
        "        key = (\n",
        "        topk = int(topk_ids.shape[1])\n"
        "        bf16_epilogue = os.getenv(\n"
        '            "B12X_EXL3_BF16_EPILOGUE", "1"\n'
        '        ) not in ("", "0", "false", "False")\n'
        "        device_index = x.device.index\n"
        "        key = (\n",
        "select B12x BF16 EXL3 epilogue policy",
    )
    replace_once(
        exl3,
        "            prefill_block_m,\n"
        "            layer.exl3_trellis_tile_config,\n"
        "        )\n"
        "        runtime = _RANK_SLICED_RUNTIMES.get(key)\n",
        "            prefill_block_m,\n"
        "            layer.exl3_trellis_tile_config,\n"
        "            bf16_epilogue,\n"
        "        )\n"
        "        runtime = _RANK_SLICED_RUNTIMES.get(key)\n",
        "fingerprint B12x BF16 EXL3 epilogue policy",
    )
    replace_once(
        exl3,
        '                quant_mode="w4a16",\n'
        "                w4a16_block_size_m=plan_block_m,\n"
        "            )\n",
        '                quant_mode="w4a16",\n'
        "                w4a16_block_size_m=plan_block_m,\n"
        "                full_rotation_output_dtype=(\n"
        "                    torch.bfloat16 if bf16_epilogue else torch.float32\n"
        "                ),\n"
        "            )\n",
        "request B12x BF16 EXL3 epilogue output",
    )

    compile(registry.read_text(), str(registry), "exec")
    compile(exl3.read_text(), str(exl3), "exec")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} VLLM_ROOT")
    main(Path(sys.argv[1]))
