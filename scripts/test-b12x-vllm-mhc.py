#!/usr/bin/env python3
"""Targeted oracle and dispatch test for the GLM B12x mHC adapter."""

from __future__ import annotations

import os

import torch

from b12x.norm import mhc as b12x_mhc
from vllm.model_executor.layers.mhc import MHCFusedPostPreOp


def make_inputs(tokens: int, seed: int):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    device = torch.device("cuda:0")
    x = torch.randn((tokens, 4096), generator=generator).to(
        device=device, dtype=torch.bfloat16
    )
    residual = torch.randn((tokens, 4, 4096), generator=generator).to(
        device=device, dtype=torch.bfloat16
    )
    post = torch.randn((tokens, 4), generator=generator).to(
        device=device, dtype=torch.float32
    )
    comb = torch.softmax(
        torch.randn((tokens, 4, 4), generator=generator).to(device), dim=-1
    )
    fn = (
        torch.randn((24, 4 * 4096), generator=generator).to(device) / 64
    ).contiguous()
    scale = torch.randn((3,), generator=generator).to(device).contiguous()
    bias = torch.randn((24,), generator=generator).to(device).contiguous()
    norm_weight = torch.ones((4096,), dtype=torch.bfloat16, device=device)
    return x, residual, post, comb, fn, scale, bias, norm_weight


def run(op: MHCFusedPostPreOp, inputs):
    x, residual, post, comb, fn, scale, bias, norm_weight = inputs
    return op.forward_cuda(
        x,
        residual,
        post,
        comb,
        fn,
        scale,
        bias,
        1.0e-5,
        1.0e-6,
        1.0e-6,
        2.0,
        20,
        1,
        1,
        norm_weight,
        1.0e-5,
    )


def main() -> None:
    assert torch.cuda.get_device_capability() in {(12, 0), (12, 1)}
    # The method is self-free; bypass CustomOp.__init__, which appropriately
    # requires a full VllmConfig context in production.
    op = object.__new__(MHCFusedPostPreOp)
    original = b12x_mhc.run_post_pre
    calls: list[int] = []

    def tracked(*args, **kwargs):
        calls.append(int(args[0].shape[0]))
        return original(*args, **kwargs)

    b12x_mhc.run_post_pre = tracked
    os.environ["VLLM_USE_B12X_MHC"] = "auto"
    inputs_m1 = make_inputs(1, 202_608_27)
    b12x_outputs = run(op, inputs_m1)
    torch.cuda.synchronize()
    assert calls == [1], calls

    os.environ["VLLM_USE_B12X_MHC"] = "disable"
    tilelang_outputs = run(op, inputs_m1)
    torch.cuda.synchronize()
    assert calls == [1], calls
    tolerances = (2.0e-2, 2.0e-3, 2.0e-3, 1.6e-2)
    for actual, expected, atol in zip(
        b12x_outputs, tilelang_outputs, tolerances, strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=atol)

    os.environ["VLLM_USE_B12X_MHC"] = "auto"
    run(op, make_inputs(2, 202_608_28))
    torch.cuda.synchronize()
    assert calls == [1], f"M=2 must stay on TileLang, got calls={calls}"
    print("B12x GLM mHC M=1 oracle and fail-closed dispatch passed")


if __name__ == "__main__":
    main()
