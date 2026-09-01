#!/usr/bin/env python3
"""Add real replicated-input B12x EP execution to the EXL3 vLLM adapter.

vLLM owns global/local expert placement and the final TP-group all-reduce.
B12x owns route filtering, execution planning, and the rank-local partial.  In
particular, this port does not remap top-k routes in Python and does not fall
back to a generic MoE implementation.
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
    exl3 = root / "model_executor/layers/quantization/exl3.py"

    replace_once(
        exl3,
        "_SPARKINFER_FUSED_MOE_API: Any | None = None\n"
        "_SPARKINFER_MIXED_TRELLIS_API: Any | None = None\n",
        "_SPARKINFER_FUSED_MOE_API: Any | None = None\n"
        "_SPARKINFER_EP_MOE_API: Any | None = None\n"
        "_SPARKINFER_MIXED_TRELLIS_API: Any | None = None\n",
        "reserve the B12x EP API handle",
    )
    replace_once(
        exl3,
        "    _SPARKINFER_FUSED_MOE_API = fused_moe\n"
        "    return fused_moe\n\n\n"
        "def _load_b12x_mixed_trellis() -> Any:\n",
        "    _SPARKINFER_FUSED_MOE_API = fused_moe\n"
        "    return fused_moe\n\n\n"
        "def _load_b12x_ep_moe() -> Any:\n"
        "    \"\"\"Resolve B12x's replicated-input EP API lazily.\"\"\"\n\n"
        "    global _SPARKINFER_EP_MOE_API\n"
        "    if _SPARKINFER_EP_MOE_API is not None:\n"
        "        return _SPARKINFER_EP_MOE_API\n"
        "    try:\n"
        "        from b12x.moe import ep_moe\n"
        "    except Exception as exc:\n"
        "        raise RuntimeError(\n"
        "            \"EXL3 expert parallelism requires b12x.moe.ep_moe. \"\n"
        "            \"Install the recipe's pinned B12x build.\"\n"
        "        ) from exc\n"
        "    _SPARKINFER_EP_MOE_API = ep_moe\n"
        "    return ep_moe\n\n\n"
        "def _load_b12x_mixed_trellis() -> Any:\n",
        "load the public B12x EP API",
    )

    old_parameter_header = '''class Exl3MoEParameter(BasevLLMParameter):
    """EXL3 tensors keyed by expert/projection, optionally in one GPU slab."""

    def __new__(
        cls,
        *,
        weight_loader,
        num_experts: int = 0,
        shard_ids: tuple[str, ...] = (),
        preallocate: bool = False,
        tp_rank: int = 0,
        tp_size: int = 1,
        tp_slice_dim: int | None = None,
    ):
        del num_experts, shard_ids, preallocate, tp_rank, tp_size, tp_slice_dim
        data = torch.empty(0, dtype=torch.uint8)
        return super().__new__(cls, data=data, weight_loader=weight_loader)

    def __init__(
        self,
        *,
        weight_loader,
        num_experts: int = 0,
        shard_ids: tuple[str, ...] = (),
        preallocate: bool = False,
        tp_rank: int = 0,
        tp_size: int = 1,
        tp_slice_dim: int | None = None,
    ):
        self.exl3_tensors: dict[tuple[int, str], torch.Tensor] = {}
        self.exl3_backing: torch.Tensor | None = None
        self.exl3_num_experts = int(num_experts)
        self.exl3_shard_ids = tuple(shard_ids)
        self.exl3_preallocate = bool(preallocate)
        self.exl3_tp_rank = int(tp_rank)
        self.exl3_tp_size = int(tp_size)
        self.exl3_tp_slice_dim = tp_slice_dim
        super().__init__(data=self.data, weight_loader=weight_loader)

    def load_exl3_weight(
'''
    new_parameter_header = '''class Exl3MoEParameter(BasevLLMParameter):
    """EXL3 tensors keyed by rank-local expert/projection."""

    def __new__(
        cls,
        *,
        weight_loader,
        num_experts: int = 0,
        shard_ids: tuple[str, ...] = (),
        preallocate: bool = False,
        tp_rank: int = 0,
        tp_size: int = 1,
        tp_slice_dim: int | None = None,
        global_to_local: tuple[int, ...] | None = None,
    ):
        del (
            num_experts,
            shard_ids,
            preallocate,
            tp_rank,
            tp_size,
            tp_slice_dim,
            global_to_local,
        )
        data = torch.empty(0, dtype=torch.uint8)
        return super().__new__(cls, data=data, weight_loader=weight_loader)

    def __init__(
        self,
        *,
        weight_loader,
        num_experts: int = 0,
        shard_ids: tuple[str, ...] = (),
        preallocate: bool = False,
        tp_rank: int = 0,
        tp_size: int = 1,
        tp_slice_dim: int | None = None,
        global_to_local: tuple[int, ...] | None = None,
    ):
        self.exl3_tensors: dict[tuple[int, str], torch.Tensor] = {}
        self.exl3_backing: torch.Tensor | None = None
        self.exl3_num_experts = int(num_experts)
        self.exl3_shard_ids = tuple(shard_ids)
        self.exl3_preallocate = bool(preallocate)
        self.exl3_tp_rank = int(tp_rank)
        self.exl3_tp_size = int(tp_size)
        self.exl3_tp_slice_dim = tp_slice_dim
        self.exl3_global_to_local = global_to_local
        super().__init__(data=self.data, weight_loader=weight_loader)

    def map_global_expert_id(self, expert_id: int) -> int:
        expert_id = int(expert_id)
        if self.exl3_global_to_local is None:
            return expert_id
        if not 0 <= expert_id < len(self.exl3_global_to_local):
            raise ValueError(
                f"EXL3 global expert {expert_id} is outside "
                f"[0, {len(self.exl3_global_to_local)})"
            )
        return int(self.exl3_global_to_local[expert_id])

    def load_exl3_weight(
'''
    replace_once(
        exl3,
        old_parameter_header,
        new_parameter_header,
        "teach EXL3 parameters the canonical global-to-local map",
    )
    replace_once(
        exl3,
        "    del weight_name\n"
        "    param.load_exl3_weight(\n"
        "        loaded_weight,\n"
        "        expert_id=expert_id,\n"
        "        shard_id=shard_id,\n"
        "    )\n"
        "    return True if return_success else None\n",
        "    del weight_name\n"
        "    local_expert_id = param.map_global_expert_id(expert_id)\n"
        "    if local_expert_id < 0:\n"
        "        return False if return_success else None\n"
        "    param.load_exl3_weight(\n"
        "        loaded_weight,\n"
        "        expert_id=local_expert_id,\n"
        "        shard_id=shard_id,\n"
        "    )\n"
        "    return True if return_success else None\n",
        "map checkpoint expert IDs before custom EXL3 loading",
    )

    replace_once(
        exl3,
        "        del extra_weight_attrs\n"
        "        if params_dtype not in (torch.bfloat16, torch.float16):\n"
        "            raise ValueError(\n"
        "                f\"EXL3 MoE requires BF16 or FP16 activations, got {params_dtype}\"\n"
        "            )\n"
        "        if self.moe.moe_parallel_config.use_ep:\n"
        "            raise NotImplementedError(\n"
        "                \"EXL3 correctness MoE currently supports TP but not expert parallelism\"\n"
        "            )\n",
        "        global_num_experts = int(\n"
        "            extra_weight_attrs.pop(\"global_num_experts\", num_experts)\n"
        "        )\n"
        "        if params_dtype not in (torch.bfloat16, torch.float16):\n"
        "            raise ValueError(\n"
        "                f\"EXL3 MoE requires BF16 or FP16 activations, got {params_dtype}\"\n"
        "            )\n"
        "        use_ep = bool(self.moe.moe_parallel_config.use_ep)\n"
        "        if use_ep and params_dtype != torch.bfloat16:\n"
        "            raise TypeError(\n"
        "                \"B12x replicated-input EXL3 EP requires BF16 activations\"\n"
        "            )\n"
        "        if use_ep and self.moe.moe_parallel_config.enable_eplb:\n"
        "            raise NotImplementedError(\n"
        "                \"EXL3 EP supports static vLLM expert placement, not EPLB\"\n"
        "            )\n"
        "        canonical_map = layer.expert_map if use_ep else None\n"
        "        if use_ep:\n"
        "            if canonical_map is None:\n"
        "                raise RuntimeError(\n"
        "                    \"vLLM enabled EP without a canonical expert map\"\n"
        "                )\n"
        "            global_to_local = tuple(\n"
        "                int(value) for value in canonical_map.detach().cpu().tolist()\n"
        "            )\n"
        "            if len(global_to_local) != global_num_experts:\n"
        "                raise ValueError(\n"
        "                    \"EXL3 EP expert-map width does not match global experts: \"\n"
        "                    f\"map={len(global_to_local)}, global={global_num_experts}\"\n"
        "                )\n"
        "            mapped = sorted(value for value in global_to_local if value >= 0)\n"
        "            if mapped != list(range(num_experts)):\n"
        "                raise ValueError(\n"
        "                    \"EXL3 EP expert map must cover every local slot exactly once\"\n"
        "                )\n"
        "            local_to_global_list = [-1] * num_experts\n"
        "            for global_id, local_id in enumerate(global_to_local):\n"
        "                if local_id >= 0:\n"
        "                    local_to_global_list[local_id] = global_id\n"
        "            local_to_global = tuple(local_to_global_list)\n"
        "            layer.register_buffer(\n"
        "                \"exl3_ep_expert_map_tensor\",\n"
        "                canonical_map.detach().clone().to(torch.int32).contiguous(),\n"
        "                persistent=False,\n"
        "            )\n"
        "        else:\n"
        "            global_to_local = None\n"
        "            local_to_global = tuple(range(num_experts))\n"
        "        layer.exl3_global_num_experts = global_num_experts\n"
        "        layer.exl3_local_to_global = local_to_global\n",
        "validate and retain vLLM's static EP placement",
    )
    replace_once(
        exl3,
        "        rank_sliced = self.quant_config.rank_sliced_metadata is not None\n"
        "        fused_trellis = rank_sliced or self.quant_config.standard_fused_moe\n",
        "        rank_sliced = self.quant_config.rank_sliced_metadata is not None\n"
        "        if use_ep and rank_sliced:\n"
        "            raise NotImplementedError(\n"
        "                \"EXL3 EP requires a standard checkpoint, not pre-sliced metadata\"\n"
        "            )\n"
        "        fused_trellis = rank_sliced or self.quant_config.standard_fused_moe\n",
        "reject incompatible pre-sliced EP checkpoints",
    )
    replace_once(
        exl3,
        "            if expected_experts != num_experts:\n"
        "                raise ValueError(\n"
        "                    \"rank-sliced EXL3 expert count does not match the model: \"\n"
        "                    f\"checkpoint={expected_experts}, model={num_experts}\"\n"
        "                )\n",
        "            if expected_experts != global_num_experts:\n"
        "                raise ValueError(\n"
        "                    \"rank-sliced EXL3 expert count does not match the model: \"\n"
        "                    f\"checkpoint={expected_experts}, model={global_num_experts}\"\n"
        "                )\n",
        "compare pre-sliced metadata with global expert count",
    )
    replace_once(
        exl3,
        "                        tp_slice_dim=tp_slice_dim,\n"
        "                    ),\n",
        "                        tp_slice_dim=tp_slice_dim,\n"
        "                        global_to_local=global_to_local,\n"
        "                    ),\n",
        "attach the expert map to every EXL3 parameter",
    )
    replace_once(
        exl3,
        "        for expert_id in range(layer.local_num_experts):\n"
        "            for shard_id, projection in projections.items():\n"
        "                prefix = f\"{layer.layer_name}.{expert_id}.{projection}\"\n",
        "        for expert_id in range(layer.local_num_experts):\n"
        "            global_expert_id = layer.exl3_local_to_global[expert_id]\n"
        "            for shard_id, projection in projections.items():\n"
        "                prefix = (\n"
        "                    f\"{layer.layer_name}.{global_expert_id}.{projection}\"\n"
        "                )\n",
        "validate local payloads against their global metadata names",
    )

    replace_once(
        exl3,
        "    def _prepare_rank_sliced_weights(self, layer: RoutedExperts) -> None:\n"
        "        if getattr(layer, \"exl3_mixed_bitrate\", False):\n"
        "            self._prepare_mixed_rank_sliced_weights(layer)\n"
        "            return\n",
        "    def _prepare_rank_sliced_weights(self, layer: RoutedExperts) -> None:\n"
        "        if getattr(layer, \"exl3_mixed_bitrate\", False):\n"
        "            if layer.use_ep:\n"
        "                raise NotImplementedError(\n"
        "                    \"mixed-bitrate EXL3 EP needs a unified prepared-expert \"\n"
        "                    \"payload; this release's uniform K3 model is supported\"\n"
        "                )\n"
        "            self._prepare_mixed_rank_sliced_weights(layer)\n"
        "            return\n",
        "refuse the unrelated mixed-bitrate EP path explicitly",
    )
    replace_once(
        exl3,
        "        layer.exl3_trellis_tile_config = tile_config\n\n"
        "    def get_fused_moe_quant_config(\n",
        "        layer.exl3_trellis_tile_config = tile_config\n"
        "        if layer.use_ep:\n"
        "            ep_api = _load_b12x_ep_moe()\n"
        "            layer.exl3_prepared_ep_map = ep_api.prepare_expert_map(\n"
        "                layer.exl3_ep_expert_map_tensor,\n"
        "                local_num_experts=num_experts,\n"
        "                global_num_experts=int(layer.exl3_global_num_experts),\n"
        "                device=w13.device,\n"
        "            )\n"
        "            logger.info_once(\n"
        "                \"EXL3 B12x EP active: %d global / %d local experts; \"\n"
        "                \"rank-local partials use vLLM's final all-reduce\",\n"
        "                int(layer.exl3_global_num_experts),\n"
        "                num_experts,\n"
        "            )\n\n"
        "    def get_fused_moe_quant_config(\n",
        "prepare the immutable B12x EP map after device placement",
    )

    replace_once(
        exl3,
        "        topk = int(topk_ids.shape[1])\n"
        "        bf16_epilogue = os.getenv(\n",
        "        topk = int(topk_ids.shape[1])\n"
        "        use_ep = bool(layer.use_ep)\n"
        "        if use_ep and min_trellis_m != 1:\n"
        "            raise ValueError(\n"
        "                \"EXL3 EP requires VLLM_EXL3_TRELLIS_MIN_M=1 so no \"\n"
        "                \"non-EP parity path can be reached\"\n"
        "            )\n"
        "        bf16_epilogue = os.getenv(\n",
        "keep every EP shape on the B12x path",
    )
    replace_once(
        exl3,
        "            int(layer.local_num_experts),\n"
        "            topk,\n",
        "            int(layer.local_num_experts),\n"
        "            int(layer.exl3_global_num_experts),\n"
        "            use_ep,\n"
        "            topk,\n",
        "fingerprint EP topology in the runtime cache",
    )
    replace_once(
        exl3,
        "        api = _load_b12x_fused_moe()\n\n"
        "        def _plan_with_scratch(plan_max_tokens: int, plan_block_m: int):\n"
        "            caps = api.Caps(\n"
        "                max_tokens=plan_max_tokens,\n"
        "                num_topk=topk,\n"
        "                # vLLM supplies final top-k IDs/weights to bind(); the fused-MoE\n"
        "                # router workspace is unused. A zero route-workspace request\n"
        "                # still lets the W4A16 core derive route_E from weight_E.\n"
        "                route_num_experts=0,\n"
        "                device=x.device,\n"
        "                weight_plan=layer.exl3_trellis_weights.plan,\n"
        "                quant_mode=\"w4a16\",\n"
        "                w4a16_block_size_m=plan_block_m,\n"
        "                full_rotation_output_dtype=(\n"
        "                    torch.bfloat16 if bf16_epilogue else torch.float32\n"
        "                ),\n"
        "            )\n"
        "            plan = api.plan(caps)\n"
        "            scratch_spec = plan.scratch_specs()[0]\n"
        "            scratch = torch.empty(\n"
        "                scratch_spec.shape,\n"
        "                dtype=scratch_spec.dtype,\n"
        "                device=scratch_spec.device,\n"
        "            )\n"
        "            return plan, scratch\n\n"
        "        trellis_plan, trellis_scratch = _plan_with_scratch(max_trellis_m, block_m)\n",
        "        api = _load_b12x_ep_moe() if use_ep else _load_b12x_fused_moe()\n\n"
        "        def _plan_with_scratch(plan_max_tokens: int, plan_block_m: int):\n"
        "            if use_ep:\n"
        "                caps = api.Caps(\n"
        "                    max_tokens=plan_max_tokens,\n"
        "                    num_topk=topk,\n"
        "                    global_num_experts=int(layer.exl3_global_num_experts),\n"
        "                    device=x.device,\n"
        "                    weight_plan=layer.exl3_trellis_weights.plan,\n"
        "                )\n"
        "            else:\n"
        "                caps = api.Caps(\n"
        "                    max_tokens=plan_max_tokens,\n"
        "                    num_topk=topk,\n"
        "                    # vLLM supplies final top-k IDs/weights to bind(); the\n"
        "                    # fused-MoE router workspace is unused.\n"
        "                    route_num_experts=0,\n"
        "                    device=x.device,\n"
        "                    weight_plan=layer.exl3_trellis_weights.plan,\n"
        "                    quant_mode=\"w4a16\",\n"
        "                    w4a16_block_size_m=plan_block_m,\n"
        "                    full_rotation_output_dtype=(\n"
        "                        torch.bfloat16 if bf16_epilogue else torch.float32\n"
        "                    ),\n"
        "                )\n"
        "            plan = api.plan(caps)\n"
        "            scratch_spec = plan.scratch_specs()[0]\n"
        "            scratch = torch.empty(\n"
        "                scratch_spec.shape,\n"
        "                dtype=scratch_spec.dtype,\n"
        "                device=scratch_spec.device,\n"
        "            )\n"
        "            output = (\n"
        "                torch.empty(\n"
        "                    (plan_max_tokens, int(layer.exl3_hidden_size)),\n"
        "                    dtype=torch.bfloat16,\n"
        "                    device=x.device,\n"
        "                )\n"
        "                if use_ep\n"
        "                else None\n"
        "            )\n"
        "            return plan, scratch, output\n\n"
        "        trellis_plan, trellis_scratch, trellis_output = _plan_with_scratch(\n"
        "            max_trellis_m, block_m\n"
        "        )\n",
        "plan B12x EP scratch and stable partial-output storage",
    )
    replace_once(
        exl3,
        "        prefill_plan = None\n"
        "        prefill_scratch = None\n"
        "        if prefill_plan_enabled:\n"
        "            prefill_plan, prefill_scratch = _plan_with_scratch(\n"
        "                max_batched_tokens, prefill_block_m\n"
        "            )\n",
        "        prefill_plan = None\n"
        "        prefill_scratch = None\n"
        "        prefill_output = None\n"
        "        if prefill_plan_enabled:\n"
        "            prefill_plan, prefill_scratch, prefill_output = (\n"
        "                _plan_with_scratch(max_batched_tokens, prefill_block_m)\n"
        "            )\n",
        "retain stable EP prefill output storage",
    )
    replace_once(
        exl3,
        "            \"api\": api,\n"
        "            \"trellis_plan\": trellis_plan,\n"
        "            \"trellis_scratch\": trellis_scratch,\n"
        "            \"prefill_plan\": prefill_plan,\n"
        "            \"prefill_scratch\": prefill_scratch,\n",
        "            \"api\": api,\n"
        "            \"use_ep\": use_ep,\n"
        "            \"trellis_plan\": trellis_plan,\n"
        "            \"trellis_scratch\": trellis_scratch,\n"
        "            \"trellis_output\": trellis_output,\n"
        "            \"prefill_plan\": prefill_plan,\n"
        "            \"prefill_scratch\": prefill_scratch,\n"
        "            \"prefill_output\": prefill_output,\n",
        "publish EP topology and output buffers in the runtime",
    )

    apply_anchor = '''    def _apply_rank_sliced(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
'''
    apply_with_helper = '''    @staticmethod
    def _bind_rank_sliced(
        runtime: dict[str, Any],
        layer: RoutedExperts,
        *,
        plan: Any,
        scratch: torch.Tensor,
        output: torch.Tensor | None,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "scratch": scratch,
            "a": x,
            "experts": layer.exl3_trellis_weights,
            "topk_weights": topk_weights,
            "topk_ids": topk_ids,
        }
        if runtime["use_ep"]:
            if output is None:
                raise RuntimeError("EXL3 EP runtime has no stable output buffer")
            kwargs.update(
                expert_map=layer.exl3_prepared_ep_map,
                output=output[: x.shape[0]],
            )
        return runtime["api"].bind(plan, **kwargs)

    def _apply_rank_sliced(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
'''
    replace_once(
        exl3,
        apply_anchor,
        apply_with_helper,
        "centralize allocation-free TP/EP binding",
    )
    replace_once(
        exl3,
        "        runtime = self._rank_sliced_runtime(layer, x, topk_ids)\n"
        "        m = int(x.shape[0])\n",
        "        runtime = self._rank_sliced_runtime(layer, x, topk_ids)\n"
        "        if runtime[\"use_ep\"]:\n"
        "            logger.info_once(\n"
        "                \"Dispatching EXL3 routes through B12x replicated-input EP; \"\n"
        "                \"vLLM will all-reduce the rank-local partial.\"\n"
        "            )\n"
        "        m = int(x.shape[0])\n",
        "log actual B12x EP dispatch",
    )
    replace_once(
        exl3,
        "            binding = runtime[\"api\"].bind(\n"
        "                runtime[\"trellis_plan\"],\n"
        "                scratch=runtime[\"trellis_scratch\"],\n"
        "                a=x,\n"
        "                experts=layer.exl3_trellis_weights,\n"
        "                topk_weights=topk_weights,\n"
        "                topk_ids=topk_ids,\n"
        "            )\n",
        "            binding = self._bind_rank_sliced(\n"
        "                runtime,\n"
        "                layer,\n"
        "                plan=runtime[\"trellis_plan\"],\n"
        "                scratch=runtime[\"trellis_scratch\"],\n"
        "                output=runtime[\"trellis_output\"],\n"
        "                x=x,\n"
        "                topk_weights=topk_weights,\n"
        "                topk_ids=topk_ids,\n"
        "            )\n",
        "bind the decode EP partial",
    )
    replace_once(
        exl3,
        "                binding = runtime[\"api\"].bind(\n"
        "                    runtime[\"prefill_plan\"],\n"
        "                    scratch=runtime[\"prefill_scratch\"],\n"
        "                    a=x,\n"
        "                    experts=layer.exl3_trellis_weights,\n"
        "                    topk_weights=topk_weights,\n"
        "                    topk_ids=topk_ids,\n"
        "                )\n",
        "                binding = self._bind_rank_sliced(\n"
        "                    runtime,\n"
        "                    layer,\n"
        "                    plan=runtime[\"prefill_plan\"],\n"
        "                    scratch=runtime[\"prefill_scratch\"],\n"
        "                    output=runtime[\"prefill_output\"],\n"
        "                    x=x,\n"
        "                    topk_weights=topk_weights,\n"
        "                    topk_ids=topk_ids,\n"
        "                )\n",
        "bind the monolithic prefill EP partial",
    )
    replace_once(
        exl3,
        "                binding = runtime[\"api\"].bind(\n"
        "                    plan,\n"
        "                    scratch=scratch,\n"
        "                    a=x[start:end],\n"
        "                    experts=layer.exl3_trellis_weights,\n"
        "                    topk_weights=topk_weights[start:end],\n"
        "                    topk_ids=topk_ids[start:end],\n"
        "                )\n",
        "                plan_output = (\n"
        "                    runtime[\"trellis_output\"]\n"
        "                    if plan is runtime[\"trellis_plan\"]\n"
        "                    else runtime[\"prefill_output\"]\n"
        "                )\n"
        "                binding = self._bind_rank_sliced(\n"
        "                    runtime,\n"
        "                    layer,\n"
        "                    plan=plan,\n"
        "                    scratch=scratch,\n"
        "                    output=plan_output,\n"
        "                    x=x[start:end],\n"
        "                    topk_weights=topk_weights[start:end],\n"
        "                    topk_ids=topk_ids[start:end],\n"
        "                )\n",
        "bind each tiled prefill EP partial",
    )
    replace_once(
        exl3,
        "        if layer.expert_map is not None:\n"
        "            raise NotImplementedError(\"EXL3 MoE expert maps/EPLB are not supported\")\n",
        "        if layer.expert_map is not None and not layer.use_ep:\n"
        "            raise NotImplementedError(\n"
        "                \"EXL3 expert maps are supported only for static B12x EP\"\n"
        "            )\n",
        "admit only the implemented static EP expert map",
    )

    compile(exl3.read_text(), str(exl3), "exec")
    print("EXL3 B12x EP port applied")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} VLLM_ROOT")
    main(Path(sys.argv[1]))
