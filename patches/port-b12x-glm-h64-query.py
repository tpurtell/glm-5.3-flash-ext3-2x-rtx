#!/usr/bin/env python3
"""Route qualified GLM H64 absorbed-query projection through B12x.

The B12x kernel fuses the BF16 per-head BMM with RoPE-query concatenation for
the exact GLM geometry [64, M, 192] x [64, 192, 512] -> [M, 64, 576].  Its
planner deliberately enables automatic serving only for packed decode M=2..16;
all other shapes retain vLLM's native BMM path.
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
    mla = root / "model_executor/layers/attention/mla_attention.py"
    replace_once(
        mla,
        "import math\nfrom abc import abstractmethod\n",
        "import math\nimport os\nfrom abc import abstractmethod\n",
        "import environment policy support",
    )
    replace_once(
        mla,
        "_FP8_DTYPE = current_platform.fp8_dtype()\n",
        "_FP8_DTYPE = current_platform.fp8_dtype()\n"
        "_B12X_GLM_H64_PREWARMED: set[tuple[object, ...]] = set()\n",
        "track GLM H64 prewarm state",
    )
    replace_once(
        mla,
        "            if self.dcp_q_replicate:\n"
        "                self.W_UK_T_dcp_qrep = get_dcp_group().all_gather(\n"
        "                    self.W_UK_T.contiguous(), dim=0\n"
        "                )\n\n"
        "        # If we should not load quant weights, we initialize the scales to 1.0\n",
        "            if self.dcp_q_replicate:\n"
        "                self.W_UK_T_dcp_qrep = get_dcp_group().all_gather(\n"
        "                    self.W_UK_T.contiguous(), dim=0\n"
        "                )\n"
        "            self._prewarm_b12x_glm_h64_query_projection()\n\n"
        "        # If we should not load quant weights, we initialize the scales to 1.0\n",
        "prewarm B12x GLM H64 query kernels after weight loading",
    )
    replace_once(
        mla,
        "    def process_weights_after_loading(self, act_dtype: torch.dtype):\n",
        "    def _prewarm_b12x_glm_h64_query_projection(self) -> None:\n"
        "        if self.attn_backend.get_name() != \"B12X_MLA_SPARSE\":\n"
        "            return\n"
        "        policy = os.getenv(\n"
        "            \"VLLM_B12X_GLM_H64_QUERY_PROJ\", \"auto\"\n"
        "        ).strip().lower()\n"
        "        if policy == \"disable\":\n"
        "            return\n"
        "        from b12x.gemm import mla_query_projection\n\n"
        "        weight = (\n"
        "            self.W_UK_T_dcp_qrep\n"
        "            if self.W_UK_T_dcp_qrep is not None\n"
        "            else self.W_UK_T\n"
        "        )\n"
        "        if weight is None:\n"
        "            return\n"
        "        probe = mla_query_projection.plan_glm_h64_bf16(\n"
        "            workload=\"packed_decode\",\n"
        "            policy=policy,\n"
        "            query_rows=2,\n"
        "            num_heads=int(weight.shape[0]),\n"
        "            nope_dim=int(weight.shape[1]),\n"
        "            latent_dim=int(weight.shape[2]),\n"
        "            output_dtype=torch.bfloat16,\n"
        "            device=weight.device,\n"
        "        )\n"
        "        if not probe.h64_supported:\n"
        "            return\n"
        "        key = (weight.device, weight.dtype, tuple(weight.shape))\n"
        "        if key in _B12X_GLM_H64_PREWARMED:\n"
        "            return\n"
        "        mla_query_projection.prewarm_glm_h64_bf16(\n"
        "            weight, range(2, 17), synchronize=True, nope=True\n"
        "        )\n"
        "        _B12X_GLM_H64_PREWARMED.add(key)\n"
        "        logger.info_once(\n"
        "            \"Prewarmed B12x GLM H64 BF16 query projection for decode M=2..16.\"\n"
        "        )\n\n"
        "    def process_weights_after_loading(self, act_dtype: torch.dtype):\n",
        "add B12x GLM H64 prewarm helper",
    )
    replace_once(
        mla,
        "            if self.is_aiter_triton_fp4_bmm_enabled:\n",
        "            b12x_fused_mqa_q: torch.Tensor | None = None\n"
        "            if self.is_aiter_triton_fp4_bmm_enabled:\n",
        "initialize fused GLM H64 query result",
    )
    replace_once(
        mla,
        "                if self.q_pad_num_heads is not None:\n"
        "                    mqa_ql_nope = mqa_q_nope.new_empty((self.q_pad_num_heads, B, L))\n"
        "                    mqa_ql_nope.resize_((N, B, L))\n"
        "                else:\n"
        "                    mqa_ql_nope = mqa_q_nope.new_empty((N, B, L))\n\n"
        "                # Multiply (N, B, P) x (N, P, L) -> (N, B, L)\n"
        "                torch.bmm(mqa_q_nope, W_UK_T, out=mqa_ql_nope)\n\n"
        "                # Convert from (N, B, L) to (B, N, L)\n"
        "                mqa_ql_nope = mqa_ql_nope.transpose(0, 1)\n\n"
        "            if fp8_attention and self.impl.supports_quant_query_input:\n",
        "                b12x_plan = None\n"
        "                if self.attn_backend.get_name() == \"B12X_MLA_SPARSE\":\n"
        "                    from b12x.gemm import mla_query_projection\n\n"
        "                    policy = os.getenv(\n"
        "                        \"VLLM_B12X_GLM_H64_QUERY_PROJ\", \"auto\"\n"
        "                    ).strip().lower()\n"
        "                    b12x_plan = mla_query_projection.plan_glm_h64_bf16(\n"
        "                        workload=\"packed_decode\",\n"
        "                        policy=policy,\n"
        "                        query_rows=B,\n"
        "                        num_heads=N,\n"
        "                        nope_dim=P,\n"
        "                        latent_dim=L,\n"
        "                        output_dtype=torch.bfloat16,\n"
        "                        device=mqa_q_nope.device,\n"
        "                    )\n"
        "                if b12x_plan is not None and b12x_plan.use_sparkinfer:\n"
        "                    b12x_fused_mqa_q = self.impl.get_fused_mla_query_output(\n"
        "                        B,\n"
        "                        N,\n"
        "                        torch.bfloat16,\n"
        "                        dcp_query_replicated=qrep_decode,\n"
        "                    )\n"
        "                    if b12x_fused_mqa_q is None:\n"
        "                        b12x_fused_mqa_q = mqa_q_nope.new_empty((B, N, L + 64))\n"
        "                    mla_query_projection.run_glm_h64_bf16(\n"
        "                        mqa_q_nope, W_UK_T, mqa_q_pe, b12x_fused_mqa_q\n"
        "                    )\n"
        "                else:\n"
        "                    if self.q_pad_num_heads is not None:\n"
        "                        mqa_ql_nope = mqa_q_nope.new_empty(\n"
        "                            (self.q_pad_num_heads, B, L)\n"
        "                        )\n"
        "                        mqa_ql_nope.resize_((N, B, L))\n"
        "                    else:\n"
        "                        mqa_ql_nope = mqa_q_nope.new_empty((N, B, L))\n\n"
        "                    # Multiply (N, B, P) x (N, P, L) -> (N, B, L)\n"
        "                    torch.bmm(mqa_q_nope, W_UK_T, out=mqa_ql_nope)\n"
        "                    mqa_ql_nope = mqa_ql_nope.transpose(0, 1)\n\n"
        "            if b12x_fused_mqa_q is not None:\n"
        "                mqa_q = b12x_fused_mqa_q\n"
        "            elif fp8_attention and self.impl.supports_quant_query_input:\n",
        "run qualified B12x GLM H64 query projection",
    )

    backend = root / "v1/attention/backends/mla/b12x_mla_sparse.py"
    replace_once(
        backend,
        "    def supports_fused_mla_query_output(\n"
        "        self,\n"
        "        num_heads: int,\n"
        "        output_dtype: torch.dtype,\n"
        "    ) -> bool:\n",
        "    def supports_fused_mla_query_output(\n"
        "        self,\n"
        "        num_heads: int,\n"
        "        output_dtype: torch.dtype,\n"
        "        *,\n"
        "        dcp_query_replicated: bool = False,\n"
        "    ) -> bool:\n",
        "allow explicit replicated-query fused-output checks",
    )
    replace_once(
        backend,
        "            self.dcp_world_size == 1\n"
        "            and output_dtype == torch.bfloat16\n",
        "            (self.dcp_world_size == 1 or dcp_query_replicated)\n"
        "            and output_dtype == torch.bfloat16\n",
        "admit DCP replicated GLM H64 query output",
    )
    replace_once(
        backend,
        "    def get_fused_mla_query_output(\n"
        "        self,\n"
        "        num_tokens: int,\n"
        "        num_heads: int,\n"
        "        output_dtype: torch.dtype,\n"
        "    ) -> torch.Tensor | None:\n",
        "    def get_fused_mla_query_output(\n"
        "        self,\n"
        "        num_tokens: int,\n"
        "        num_heads: int,\n"
        "        output_dtype: torch.dtype,\n"
        "        *,\n"
        "        dcp_query_replicated: bool = False,\n"
        "    ) -> torch.Tensor | None:\n",
        "thread DCP replicated query state into workspace selection",
    )
    replace_once(
        backend,
        "            not self.supports_fused_mla_query_output(num_heads, output_dtype)\n",
        "            not self.supports_fused_mla_query_output(\n"
        "                num_heads,\n"
        "                output_dtype,\n"
        "                dcp_query_replicated=dcp_query_replicated,\n"
        "            )\n",
        "validate DCP replicated query workspace selection",
    )

    compile(mla.read_text(), str(mla), "exec")
    compile(backend.read_text(), str(backend), "exec")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} VLLM_ROOT")
    main(Path(sys.argv[1]))
