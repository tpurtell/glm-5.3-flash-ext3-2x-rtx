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
        "        candidates = [prefix]\n",
        "        candidates = [prefix]\n"
        '        mtp_checkpoint_prefix = prefix.replace(".mtp_block.", ".", 1)\n'
        '        mtp_checkpoint_prefix = mtp_checkpoint_prefix.replace(\n'
        '            ".routed_experts.", ".", 1\n'
        "        )\n"
        "        if mtp_checkpoint_prefix != prefix:\n"
        "            candidates.append(mtp_checkpoint_prefix)\n"
        '            if mtp_checkpoint_prefix.startswith("model."):\n'
        "                candidates.append(\n"
        '                    "model.language_model."\n'
        '                    + mtp_checkpoint_prefix.removeprefix("model.")\n'
        "                )\n",
        "resolve GLM draft-layer prefixes to checkpoint EXL3 metadata",
    )
    replace_once(
        exl3,
        '        if prefix.startswith("model."):\n'
        '            candidates.append(prefix.removeprefix("model."))\n',
        '        if prefix.startswith("model."):\n'
        '            text_prefix = prefix.removeprefix("model.")\n'
        "            candidates.extend(\n"
        "                (text_prefix, f\"model.language_model.{text_prefix}\")\n"
        "            )\n",
        "resolve pre-wrapper GLM text prefixes during draft construction",
    )
    replace_once(
        exl3,
        "        if isinstance(layer, RoutedExperts):\n"
        "            if not self._moe_prefix_is_exl3(prefix, layer):\n",
        "        if isinstance(layer, RoutedExperts):\n"
        "            is_standard_mtp = (\n"
        "                self.standard_fused_moe and \".mtp_block.\" in prefix\n"
        "            )\n"
        "            if (\n"
        "                not is_standard_mtp\n"
        "                and not self._moe_prefix_is_exl3(prefix, layer)\n"
        "            ):\n",
        "instantiate standard GLM MTP experts as EXL3 before weight mapping",
    )
    replace_once(
        exl3,
        "            layer.exl3_max_num_batched_tokens = int(\n"
        "                scheduler_config.max_num_batched_tokens\n"
        "            )\n"
        "            # Stamp the layer role while the model-construction config context\n",
        "            is_draft = (\n"
        '                getattr(vllm_config.model_config, "runner_type", None) == "draft"\n'
        "            )\n"
        "            # A standard MTP forward has one row per live request, not one\n"
        "            # row per target-prefill token. Planning its EXL3 prefill arena\n"
        "            # for max_num_batched_tokens duplicated the target's ~GiB-scale\n"
        "            # arena even though that capacity was unreachable. Keep its\n"
        "            # independently captured runtime, but size it to concurrency.\n"
        "            layer.exl3_max_num_batched_tokens = int(\n"
        "                scheduler_config.max_num_seqs\n"
        "                if is_draft\n"
        "                else scheduler_config.max_num_batched_tokens\n"
        "            )\n"
        "            # Stamp the layer role while the model-construction config context\n",
        "size GLM MTP EXL3 arena to request concurrency",
    )
    replace_once(
        exl3,
        "        max_batched_tokens = int(layer.exl3_max_num_batched_tokens)\n"
        "        prefill_plan_enabled = prefill_trellis and max_batched_tokens > max_trellis_m\n",
        "        scheduler_max_batched_tokens = int(layer.exl3_max_num_batched_tokens)\n"
        "        # B12x prefill scratch grows with the planned token capacity. A\n"
        "        # single scheduler batch does not need one monolithic MoE launch:\n"
        "        # expert routing is token-independent, so larger batches can be\n"
        "        # tiled through a much smaller persistent arena without changing\n"
        "        # results. This is especially valuable on large EXL3 models, where\n"
        "        # an 8192-row plan otherwise reserves roughly 1.8 GiB per GPU.\n"
        "        prefill_capacity = _positive_env_int(\n"
        "            \"VLLM_EXL3_PREFILL_CAPACITY\", 1024\n"
        "        )\n"
        "        max_batched_tokens = min(\n"
        "            scheduler_max_batched_tokens, prefill_capacity\n"
        "        )\n"
        "        prefill_plan_enabled = prefill_trellis and max_batched_tokens > max_trellis_m\n",
        "bound persistent B12x prefill arena capacity",
    )
    replace_once(
        exl3,
        "        if runtime[\"prefill_plan\"] is not None and m > runtime[\"max_trellis_m\"]:\n"
        "            if m > runtime[\"max_batched_tokens\"]:\n"
        "                raise ValueError(\n"
        "                    \"EXL3 batch exceeds its planned capacity: \"\n"
        "                    f\"m={m}, capacity={runtime['max_batched_tokens']}\"\n"
        "                )\n"
        "            binding = runtime[\"api\"].bind(\n"
        "                runtime[\"prefill_plan\"],\n"
        "                scratch=runtime[\"prefill_scratch\"],\n"
        "                a=x,\n"
        "                experts=layer.exl3_trellis_weights,\n"
        "                topk_weights=topk_weights,\n"
        "                topk_ids=topk_ids,\n"
        "            )\n"
        "            output = runtime[\"api\"].run(binding=binding)\n"
        "            return output.to(x.dtype)\n",
        "        if runtime[\"prefill_plan\"] is not None and m > runtime[\"max_trellis_m\"]:\n"
        "            capacity = int(runtime[\"max_batched_tokens\"])\n"
        "            if m <= capacity:\n"
        "                binding = runtime[\"api\"].bind(\n"
        "                    runtime[\"prefill_plan\"],\n"
        "                    scratch=runtime[\"prefill_scratch\"],\n"
        "                    a=x,\n"
        "                    experts=layer.exl3_trellis_weights,\n"
        "                    topk_weights=topk_weights,\n"
        "                    topk_ids=topk_ids,\n"
        "                )\n"
        "                output = runtime[\"api\"].run(binding=binding)\n"
        "                return output.to(x.dtype)\n"
        "\n"
        "            # Tile a large scheduler prefill through the bounded arena.\n"
        "            # Each route is independent, so concatenating the per-tile\n"
        "            # MoE outputs is exactly the monolithic operation.\n"
        "            output = torch.empty(\n"
        "                (m, x.shape[1]), dtype=x.dtype, device=x.device\n"
        "            )\n"
        "            for start in range(0, m, capacity):\n"
        "                end = min(start + capacity, m)\n"
        "                current_m = end - start\n"
        "                if current_m <= runtime[\"max_trellis_m\"]:\n"
        "                    plan = runtime[\"trellis_plan\"]\n"
        "                    scratch = runtime[\"trellis_scratch\"]\n"
        "                else:\n"
        "                    plan = runtime[\"prefill_plan\"]\n"
        "                    scratch = runtime[\"prefill_scratch\"]\n"
        "                binding = runtime[\"api\"].bind(\n"
        "                    plan,\n"
        "                    scratch=scratch,\n"
        "                    a=x[start:end],\n"
        "                    experts=layer.exl3_trellis_weights,\n"
        "                    topk_weights=topk_weights[start:end],\n"
        "                    topk_ids=topk_ids[start:end],\n"
        "                )\n"
        "                tile_output = runtime[\"api\"].run(binding=binding)\n"
        "                output[start:end].copy_(tile_output)\n"
        "            return output\n",
        "tile large EXL3 prefills through bounded B12x arena",
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
