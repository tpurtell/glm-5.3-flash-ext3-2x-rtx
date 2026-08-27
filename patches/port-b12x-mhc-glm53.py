#!/usr/bin/env python3
"""Use B12x for the one GLM mHC shape where it is a measured winner."""

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
    mhc = root / "model_executor/layers/mhc.py"
    replace_once(
        mhc,
        "import torch\n",
        "import os\n\nimport torch\n",
        "import B12x mHC policy support",
    )
    replace_once(
        mhc,
        "from vllm.model_executor.custom_op import CustomOp\n",
        "from vllm.logger import init_logger\n"
        "from vllm.model_executor.custom_op import CustomOp\n",
        "import mHC logger",
    )
    replace_once(
        mhc,
        "HAS_AITER_MHC = is_aiter_found_and_supported()\n",
        "HAS_AITER_MHC = is_aiter_found_and_supported()\n"
        "logger = init_logger(__name__)\n",
        "initialize mHC logger",
    )

    old = '''    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        tile_n: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.ops.vllm.mhc_fused_post_pre_tilelang(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            tile_n,
            norm_weight,
            norm_eps,
        )
'''
    new = '''    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        tile_n: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Qualified on dual SM120 PCIe: the exact GLM H4096 decode-M=1
        # specialization is 1.23x faster than TileLang.  TileLang remains the
        # measured winner for M>=16 and serves every non-GLM/non-exact shape.
        b12x_policy = os.getenv("VLLM_USE_B12X_MHC", "auto").strip().lower()
        use_b12x = (
            b12x_policy not in {"0", "disable", "off"}
            and int(x.shape[0]) == 1
            and x.dtype == torch.bfloat16
            and residual.dtype == torch.bfloat16
            and tuple(residual.shape[1:]) == (4, 4096)
            and fn.dtype == torch.float32
            and tuple(fn.shape) == (24, 4 * 4096)
            and abs(float(rms_eps) - 1.0e-5) <= 1.0e-12
            and abs(float(hc_pre_eps) - 1.0e-6) <= 1.0e-12
            and abs(float(hc_sinkhorn_eps) - 1.0e-6) <= 1.0e-12
            and float(hc_post_mult_value) == 2.0
            and int(sinkhorn_repeat) == 20
            and int(n_splits) == 1
            and int(tile_n) == 1
            and norm_weight is not None
            and abs(float(norm_eps) - 1.0e-5) <= 1.0e-12
        )
        if use_b12x:
            from b12x.norm import mhc as b12x_mhc

            logger.info_once(
                "Using B12x fused GLM H4096 mHC post+pre for decode M=1."
            )
            residual_cur, post, comb, y = b12x_mhc.run_post_pre(
                x,
                residual,
                post_layer_mix,
                comb_res_mix,
                fn,
                hc_scale,
                hc_base,
                rms_eps=rms_eps,
                hc_eps=hc_pre_eps,
                sinkhorn_iters=sinkhorn_repeat,
                norm_weight=norm_weight,
                norm_eps=norm_eps,
            )
            # Preserve vLLM's public post-mix ABI [tokens, 4, 1]; B12x uses
            # the equivalent compact [tokens, 4] representation internally.
            return residual_cur, post.unsqueeze(-1), comb, y
        return torch.ops.vllm.mhc_fused_post_pre_tilelang(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            tile_n,
            norm_weight,
            norm_eps,
        )
'''
    replace_once(mhc, old, new, "route qualified GLM decode mHC through B12x")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} VLLM_PACKAGE_ROOT")
    main(Path(sys.argv[1]).resolve())
