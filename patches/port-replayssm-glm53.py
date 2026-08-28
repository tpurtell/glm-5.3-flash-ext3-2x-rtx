#!/usr/bin/env python3
"""Adapt upstream ReplaySSM speculative rollback to GLM-5.3's KDA layer."""

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
        raise RuntimeError(f"{path}: {label}: expected one anchor, found {count}")
    path.write_text(text.replace(old, new, 1))
    print(f"[ok]   {label}")


def port_model(root: Path) -> None:
    model = root / "models/glm5next/nvidia/model.py"
    replace_once(
        model,
        "    SupportsPP,\n)",
        "    SupportsPP,\n    SupportsReplaySSM,\n)",
        "import ReplaySSM model marker",
    )
    replace_once(
        model,
        "    nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid\n)",
        "    nn.Module,\n"
        "    HasInnerState,\n"
        "    SupportsPP,\n"
        "    MixtureOfExperts,\n"
        "    IsHybrid,\n"
        "    SupportsReplaySSM,\n"
        ")",
        "mark GLM text model ReplaySSM-capable",
    )
    replace_once(
        model,
        "    Glm4vForConditionalGeneration, HasInnerState, IsHybrid\n)",
        "    Glm4vForConditionalGeneration,\n"
        "    HasInnerState,\n"
        "    IsHybrid,\n"
        "    SupportsReplaySSM,\n"
        ")",
        "mark GLM wrapper ReplaySSM-capable",
    )
    replace_once(
        model,
        "    ) -> tuple[torch.dtype, torch.dtype]:\n"
        "        return MambaStateDtypeCalculator.kda_state_dtype(\n"
        "            vllm_config.model_config.dtype, vllm_config.cache_config.mamba_cache_dtype\n"
        "        )\n",
        "    ) -> tuple[torch.dtype, ...]:\n"
        "        base = MambaStateDtypeCalculator.kda_state_dtype(\n"
        "            vllm_config.model_config.dtype,\n"
        "            vllm_config.cache_config.mamba_cache_dtype,\n"
        "        )\n"
        "        if vllm_config.cache_config.use_replayssm_spec:\n"
        "            return MambaStateDtypeCalculator.append_kda_replayssm_spec_ring(\n"
        "                base, vllm_config.model_config.dtype\n"
        "            )\n"
        "        return base\n",
        "size GLM KDA ReplaySSM ring dtypes",
    )
    old_shape = (
        "        return MambaStateShapeCalculator.kda_state_shape(\n"
        "            tp_size,\n"
        "            hf_config.linear_num_heads,\n"
        "            hf_config.linear_head_dim,\n"
        "            conv_kernel_size=hf_config.linear_conv_kernel_dim,\n"
        "            num_spec=num_spec,\n"
        "        )\n"
    )
    new_shape = (
        "        base = MambaStateShapeCalculator.kda_state_shape(\n"
        "            tp_size,\n"
        "            hf_config.linear_num_heads,\n"
        "            hf_config.linear_head_dim,\n"
        "            conv_kernel_size=hf_config.linear_conv_kernel_dim,\n"
        "            num_spec=num_spec,\n"
        "        )\n"
        "        if vllm_config.cache_config.use_replayssm_spec:\n"
        "            return MambaStateShapeCalculator.append_kda_replayssm_spec_ring(\n"
        "                base,\n"
        "                hf_config.linear_num_heads,\n"
        "                hf_config.linear_head_dim,\n"
        "                tp_size,\n"
        "                vllm_config.cache_config.replayssm_buffer_len,\n"
        "                num_spec,\n"
        "            )\n"
        "        return base\n"
    )
    replace_once(model, old_shape, new_shape, "size GLM KDA ReplaySSM ring shapes")


def port_kda(root: Path) -> None:
    kda = root / "models/glm5next/nvidia/kda.py"
    replace_once(
        kda,
        "        return MambaStateDtypeCalculator.kda_state_dtype(\n"
        "            self.model_config.dtype, self.cache_config.mamba_cache_dtype\n"
        "        )\n",
        "        base = MambaStateDtypeCalculator.kda_state_dtype(\n"
        "            self.model_config.dtype, self.cache_config.mamba_cache_dtype\n"
        "        )\n"
        "        if self.cache_config.use_replayssm_spec:\n"
        "            return MambaStateDtypeCalculator.append_kda_replayssm_spec_ring(\n"
        "                base, self.model_config.dtype\n"
        "            )\n"
        "        return base\n",
        "bind ReplaySSM dtypes in GLM KDA layer",
    )
    old_layer_shape = (
        "        return MambaStateShapeCalculator.kda_state_shape(\n"
        "            self.tp_size,\n"
        "            self.num_heads,\n"
        "            self.head_dim,\n"
        "            conv_kernel_size=self.conv_size,\n"
        "            num_spec=self.num_spec,\n"
        "        )\n"
    )
    new_layer_shape = (
        "        base = MambaStateShapeCalculator.kda_state_shape(\n"
        "            self.tp_size,\n"
        "            self.num_heads,\n"
        "            self.head_dim,\n"
        "            conv_kernel_size=self.conv_size,\n"
        "            num_spec=self.num_spec,\n"
        "        )\n"
        "        if self.cache_config is not None and self.cache_config.use_replayssm_spec:\n"
        "            return MambaStateShapeCalculator.append_kda_replayssm_spec_ring(\n"
        "                base,\n"
        "                self.num_heads,\n"
        "                self.head_dim,\n"
        "                self.tp_size,\n"
        "                self.cache_config.replayssm_buffer_len,\n"
        "                self.num_spec,\n"
        "            )\n"
        "        return base\n"
    )
    replace_once(kda, old_layer_shape, new_layer_shape, "bind ReplaySSM shapes in KDA")
    replace_once(
        kda,
        "        self._conv_state_dim_first = is_conv_state_dim_first()\n",
        "        self._conv_state_dim_first = is_conv_state_dim_first()\n"
        "        self.use_replayssm_spec = vllm_config.cache_config.use_replayssm_spec\n",
        "cache KDA ReplaySSM dispatch flag",
    )
    replace_once(
        kda,
        "        (conv_state, recurrent_state) = constant_caches\n",
        "        if self.use_replayssm_spec:\n"
        "            (\n"
        "                conv_state,\n"
        "                recurrent_state,\n"
        "                replay_d_cache,\n"
        "                replay_k_cache,\n"
        "                replay_g_cache,\n"
        "            ) = constant_caches\n"
        "        else:\n"
        "            conv_state, recurrent_state = constant_caches\n"
        "            replay_d_cache = replay_k_cache = replay_g_cache = None\n",
        "unpack compact KDA cache tensors",
    )

    old_spec = (
        "            core_attn_out_spec, _ = fused_recurrent_kda(\n"
        "                q=_rearr(q_spec),\n"
        "                k=_rearr(k_spec),\n"
        "                v=_rearr(v_spec),\n"
        "                g=g1_spec,\n"
        "                beta=beta_spec,\n"
        "                initial_state=recurrent_state,\n"
        "                use_qk_l2norm_in_kernel=True,\n"
        "                cu_seqlens=spec_query_start_loc[: num_spec_decodes + 1],\n"
        "                ssm_state_indices=spec_state_indices_tensor,\n"
        "                num_accepted_tokens=num_accepted_tokens,\n"
        "                out=spec_out,\n"
        "                sigmoid_beta=True,\n"
        "                a_log=self.A_log,\n"
        "                g_bias=self.dt_bias,\n"
        "                compute_gate=True,\n"
        "                lower_bound=lower_bound,\n"
        "            )\n"
    )
    new_spec = (
        "            if self.use_replayssm_spec:\n"
        "                from vllm.third_party.flash_linear_attention.ops.kda_replayssm_spec_decode import (\n"
        "                    kda_replayssm_spec_decode,\n"
        "                )\n"
        "\n"
        "                assert replay_d_cache is not None\n"
        "                assert replay_k_cache is not None\n"
        "                assert replay_g_cache is not None\n"
        "                assert attn_metadata_narrowed.spec_write_pos_d is not None\n"
        "                assert attn_metadata_narrowed.spec_cache_base_d is not None\n"
        "                assert attn_metadata_narrowed.spec_is_flush_d is not None\n"
        "                if spec_out is None:\n"
        "                    spec_out = torch.empty_like(_rearr(v_spec))\n"
        "                core_attn_out_spec = kda_replayssm_spec_decode(\n"
        "                    q=_rearr(q_spec),\n"
        "                    k=_rearr(k_spec),\n"
        "                    v=_rearr(v_spec),\n"
        "                    raw_g=g1_spec,\n"
        "                    raw_beta=beta_spec,\n"
        "                    a_log=self.A_log,\n"
        "                    g_bias=self.dt_bias,\n"
        "                    checkpoint=recurrent_state,\n"
        "                    d_cache=replay_d_cache,\n"
        "                    k_cache=replay_k_cache,\n"
        "                    g_cache=replay_g_cache,\n"
        "                    out=spec_out,\n"
        "                    query_start_loc=spec_query_start_loc[: num_spec_decodes + 1],\n"
        "                    state_indices=spec_state_indices_tensor[:num_spec_decodes, 0],\n"
        "                    write_pos=attn_metadata_narrowed.spec_write_pos_d,\n"
        "                    cache_base=attn_metadata_narrowed.spec_cache_base_d,\n"
        "                    is_flush=attn_metadata_narrowed.spec_is_flush_d,\n"
        "                    max_spec_len=self.num_spec + 1,\n"
        "                    lower_bound=lower_bound,\n"
        "                )\n"
        "            else:\n"
        "                core_attn_out_spec, _ = fused_recurrent_kda(\n"
        "                    q=_rearr(q_spec),\n"
        "                    k=_rearr(k_spec),\n"
        "                    v=_rearr(v_spec),\n"
        "                    g=g1_spec,\n"
        "                    beta=beta_spec,\n"
        "                    initial_state=recurrent_state,\n"
        "                    use_qk_l2norm_in_kernel=True,\n"
        "                    cu_seqlens=spec_query_start_loc[: num_spec_decodes + 1],\n"
        "                    ssm_state_indices=spec_state_indices_tensor,\n"
        "                    num_accepted_tokens=num_accepted_tokens,\n"
        "                    out=spec_out,\n"
        "                    sigmoid_beta=True,\n"
        "                    a_log=self.A_log,\n"
        "                    g_bias=self.dt_bias,\n"
        "                    compute_gate=True,\n"
        "                    lower_bound=lower_bound,\n"
        "                )\n"
    )
    replace_once(kda, old_spec, new_spec, "route GLM KDA verify through ReplaySSM")

    old_gather = (
        "            initial_state = gather_initial_states(\n"
        "                recurrent_state, non_spec_state_indices_tensor, has_initial_state\n"
        "            )\n"
    )
    new_gather = old_gather + (
        "            if self.use_replayssm_spec:\n"
        "                from vllm.third_party.flash_linear_attention.ops.kda_replayssm_spec_decode import (\n"
        "                    materialize_kda_replayssm_state,\n"
        "                )\n"
        "\n"
        "                source = attn_metadata_narrowed.replayssm_prefill_source_state_indices\n"
        "                row_wp = attn_metadata_narrowed.replayssm_prefill_write_pos\n"
        "                row_base = attn_metadata_narrowed.replayssm_prefill_cache_base\n"
        "                assert source is not None and row_wp is not None and row_base is not None\n"
        "                assert replay_d_cache is not None\n"
        "                assert replay_k_cache is not None\n"
        "                assert replay_g_cache is not None\n"
        "                materialize_kda_replayssm_state(\n"
        "                    recurrent_state,\n"
        "                    replay_d_cache,\n"
        "                    replay_k_cache,\n"
        "                    replay_g_cache,\n"
        "                    initial_state,\n"
        "                    source,\n"
        "                    has_initial_state,\n"
        "                    row_wp,\n"
        "                    row_base,\n"
        "                )\n"
    )
    replace_once(kda, old_gather, new_gather, "reconstruct ReplaySSM prefix state")


def port_validation(root: Path) -> None:
    config = root / "config/vllm.py"
    replace_once(
        config,
        "            if self.cache_config.mamba_cache_mode != \"none\":\n"
        "                raise ValueError(\n"
        "                    \"ReplaySSM speculative decoding does not support prefix \"\n"
        "                    \"caching; pass --mamba-cache-mode none\"\n"
        "                )\n",
        "            if self.cache_config.mamba_cache_mode not in (\"none\", \"align\"):\n"
        "                raise ValueError(\n"
        "                    \"ReplaySSM speculative decoding supports only mamba cache \"\n"
        "                    \"modes 'none' and 'align'\"\n"
        "                )\n"
        "            if (\n"
        "                self.cache_config.mamba_cache_mode == \"align\"\n"
        "                and self.model_config is not None\n"
        "                and self.model_config.architecture\n"
        "                not in {\"Glm5NextForConditionalGeneration\", \"Glm5NextForCausalLM\"}\n"
        "            ):\n"
        "                raise ValueError(\n"
        "                    \"ReplaySSM speculative align mode is currently qualified \"\n"
        "                    \"only for GLM-5.3 KDA\"\n"
        "                )\n",
        "qualify compact KDA rollback with align prefix cache",
    )


def main(root: Path) -> None:
    port_model(root)
    port_kda(root)
    port_validation(root)
    for path in (
        root / "models/glm5next/nvidia/model.py",
        root / "models/glm5next/nvidia/kda.py",
        root / "config/vllm.py",
    ):
        compile(path.read_text(), str(path), "exec")
    print("GLM-5.3 compact KDA ReplaySSM port applied")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} VLLM_ROOT")
    main(Path(sys.argv[1]))
