#!/usr/bin/env python3
"""Warm every GLM5 mHC TileLang split-K variant before serving."""

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
    warmup = root / "model_executor/warmup/deepseek_v4_mhc_warmup.py"

    replace_once(
        warmup,
        '"""Warm up DeepSeek V4 mHC TileLang kernels before serving requests.\n',
        '"""Warm up DeepSeek V4 and GLM5 mHC TileLang kernels before serving.\n',
        "describe GLM5 coverage",
    )
    replace_once(
        warmup,
        '        if module.__class__.__name__ != "DeepseekV4DecoderLayer":\n'
        "            continue\n",
        '        if module.__class__.__name__ not in {\n'
        '            "DeepseekV4DecoderLayer",\n'
        '            "Glm5NextDecoderLayer",\n'
        "        }:\n"
        "            continue\n",
        "discover GLM5 mHC layers",
    )

    old_layer_warmup = '''def _warmup_layer_mhc(
    layer: torch.nn.Module,
    token_sizes: list[int],
) -> None:
    max_tokens = max(token_sizes)
    hidden_size = int(layer.hidden_size)
    hc_mult = int(layer.hc_mult)
    device = layer.hc_attn_fn.device
    residual = torch.zeros(
        max_tokens,
        hc_mult,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )

    for size in token_sizes:
        residual_slice = residual[:size]
        for fn, scale, base in (
            (layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base),
            (layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base),
        ):
            layer_input, post_mix, comb_mix = layer.hc_pre(
                residual_slice,
                fn,
                scale,
                base,
            )
            layer.hc_post(layer_input, residual_slice, post_mix, comb_mix)
'''
    new_layer_warmup = '''def _warmup_layer_mhc(
    layer: torch.nn.Module,
    token_sizes: list[int],
) -> None:
    max_tokens = max(token_sizes)
    hidden_size = int(layer.hidden_size)
    hc_mult = int(getattr(layer, "hc_mult", getattr(layer, "n", 0)))
    if hc_mult <= 0:
        raise RuntimeError("mHC warmup layer has no residual-stream count")
    device = layer.hc_attn_fn.device
    residual = torch.zeros(
        max_tokens,
        hc_mult,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )

    if layer.__class__.__name__ == "Glm5NextDecoderLayer":
        # GLM returns (post, comb, input), unlike the older DeepSeek wrapper,
        # and production fuses every inter-layer/post-attention hc_post with
        # the following hc_pre + RMSNorm. Warm both entry points with the real
        # scalar contract so TileLang cannot discover them during a request.
        input_norm = layer.input_layernorm
        post_norm = layer.post_attention_layernorm
        for size in token_sizes:
            residual_slice = residual[:size]
            post_mix, comb_mix, layer_input = layer.hc_pre(
                residual_slice,
                layer.hc_attn_fn,
                layer.hc_attn_scale,
                layer.hc_attn_base,
                norm_weight=input_norm.weight.data,
                norm_eps=input_norm.variance_epsilon,
            )
            layer.hc_post(layer_input, residual_slice, post_mix, comb_mix)
            layer.hc_fused_post_pre(
                layer_input,
                residual_slice,
                post_mix,
                comb_mix,
                layer.hc_ffn_fn,
                layer.hc_ffn_scale,
                layer.hc_ffn_base,
                norm_weight=post_norm.weight.data,
                norm_eps=post_norm.variance_epsilon,
            )
        return

    for size in token_sizes:
        residual_slice = residual[:size]
        for fn, scale, base in (
            (layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base),
            (layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base),
        ):
            layer_input, post_mix, comb_mix = layer.hc_pre(
                residual_slice,
                fn,
                scale,
                base,
            )
            layer.hc_post(layer_input, residual_slice, post_mix, comb_mix)
'''
    replace_once(
        warmup,
        old_layer_warmup,
        new_layer_warmup,
        "warm GLM5 standalone and fused mHC paths",
    )

    replace_once(
        warmup,
        '    if model_type is not None and model_type != "deepseek_v4":\n'
        "        return\n",
        "    if model_type is not None and model_type not in {\n"
        '        "deepseek_v4",\n'
        '        "glm5_next",\n'
        '        "glm5_next_text",\n'
        "    }:\n"
        "        return\n",
        "admit GLM5 model types",
    )
    replace_once(
        warmup,
        "    if not token_sizes:\n"
        "        return\n\n"
        "    started = time.perf_counter()\n",
        "    if layer.__class__.__name__ == \"Glm5NextDecoderLayer\":\n"
        "        # DeepGEMM chooses a static split-K from ceil(tokens / 64).\n"
        "        # A prefill tail can occupy any bucket, so powers of two and\n"
        "        # CUDA-graph sizes alone are insufficient. One representative\n"
        "        # per bucket covers every tail through max_num_batched_tokens.\n"
        "        token_sizes = _normalize_token_sizes(\n"
        "            [*token_sizes, *range(64, max_tokens + 1, 64)],\n"
        "            max_tokens=max_tokens,\n"
        "        )\n"
        "    if not token_sizes:\n"
        "        return\n\n"
        "    started = time.perf_counter()\n",
        "cover every GLM5 split-K bucket",
    )
    replace_once(
        warmup,
        '        "Warming up DeepSeek V4 mHC TileLang kernels for token sizes: %s",\n',
        '        "Warming up mHC TileLang kernels for %s token sizes: %s",\n'
        "        layer.__class__.__name__,\n",
        "identify warmed model family",
    )
    replace_once(
        warmup,
        '        "DeepSeek V4 mHC TileLang warmup finished in %.2f seconds.",\n',
        '        "mHC TileLang warmup finished in %.2f seconds.",\n',
        "use a model-neutral completion message",
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} VLLM_PACKAGE_ROOT")
    main(Path(sys.argv[1]).resolve())
