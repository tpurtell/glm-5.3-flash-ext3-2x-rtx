#!/usr/bin/env python3
"""Expose GLM-5.3 mHC layer taps to DFlash2's EAGLE3-style interface."""

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
    path = args.vllm_root / "models/glm5next/nvidia/model.py"

    replace_once(
        path,
        "from vllm.model_executor.models.interfaces import (\n"
        "    HasInnerState,\n",
        "from vllm.model_executor.models.interfaces import (\n"
        "    EagleModelMixin,\n"
        "    HasInnerState,\n",
        "import EAGLE3 layer-tap mixin",
    )
    replace_once(
        path,
        "    SupportsPP,\n"
        "    SupportsReplaySSM,\n",
        "    SupportsEagle3,\n"
        "    SupportsPP,\n"
        "    SupportsReplaySSM,\n",
        "import EAGLE3 target interface",
    )
    replace_once(
        path,
        "class Glm5NextModel(nn.Module):\n",
        "class Glm5NextModel(nn.Module, EagleModelMixin):\n",
        "mark GLM text body as auxiliary-hidden-state capable",
    )
    replace_once(
        path,
        "        for layer in self._active_layers:\n"
        "            hidden_states, residual, post, comb = layer(\n"
        "                positions, hidden_states, residual, post, comb\n"
        "            )\n\n"
        "        if not get_pp_group().is_last_rank:\n",
        "        aux_hidden_states: list[torch.Tensor] = []\n"
        "        for layer in self._active_layers:\n"
        "            hidden_states, residual, post, comb = layer(\n"
        "                positions, hidden_states, residual, post, comb\n"
        "            )\n"
        "            if layer.layer_idx + 1 in self.aux_hidden_state_layers:\n"
        "                # DFlash taps the post-layer target representation. GLM's\n"
        "                # mHC path defers hc_post to the next layer, so materialize\n"
        "                # a read-only view here and contract its residual streams.\n"
        "                if layer.mhc and post is not None:\n"
        "                    aux_hidden_state = hc_contract(\n"
        "                        layer.hc_post(\n"
        "                            hidden_states, residual, post, comb\n"
        "                        ),\n"
        "                        layer.n,\n"
        "                    )\n"
        "                else:\n"
        "                    aux_hidden_state = hidden_states\n"
        "                if self.is_sequence_parallel:\n"
        "                    aux_hidden_state = sp_all_gather(aux_hidden_state)[\n"
        "                        :full_num_tokens\n"
        "                    ]\n"
        "                aux_hidden_states.append(aux_hidden_state)\n\n"
        "        if not get_pp_group().is_last_rank:\n",
        "capture DFlash2 hidden-state taps after GLM mHC layers",
    )
    replace_once(
        path,
        "        hidden_states = self.norm(hidden_states)\n"
        "        return hidden_states\n\n"
        "    def load_weights",
        "        hidden_states = self.norm(hidden_states)\n"
        "        if aux_hidden_states:\n"
        "            return hidden_states, aux_hidden_states\n"
        "        return hidden_states\n\n"
        "    def load_weights",
        "return GLM target taps to the DFlash2 runner",
    )
    replace_once(
        path,
        "    SupportsPP,\n"
        "    MixtureOfExperts,\n"
        "    IsHybrid,\n"
        "    SupportsReplaySSM,\n"
        "):\n"
        "    def __init__(self, *, vllm_config: VllmConfig, prefix: str = \"\"):\n",
        "    SupportsPP,\n"
        "    MixtureOfExperts,\n"
        "    IsHybrid,\n"
        "    SupportsReplaySSM,\n"
        "    SupportsEagle3,\n"
        "):\n"
        "    def __init__(self, *, vllm_config: VllmConfig, prefix: str = \"\"):\n",
        "mark GLM causal target as EAGLE3-compatible",
    )
    replace_once(
        path,
        "class Glm5NextForConditionalGeneration(\n"
        "    Glm4vForConditionalGeneration,\n"
        "    HasInnerState,\n"
        "    IsHybrid,\n"
        "    SupportsReplaySSM,\n"
        "):\n",
        "class Glm5NextForConditionalGeneration(\n"
        "    Glm4vForConditionalGeneration,\n"
        "    HasInnerState,\n"
        "    IsHybrid,\n"
        "    SupportsReplaySSM,\n"
        "    SupportsEagle3,\n"
        "):\n",
        "mark GLM multimodal target as EAGLE3-compatible",
    )


if __name__ == "__main__":
    main()
