#!/usr/bin/env python3
"""Measure C1 DFlash decode on the code-agent task at existing KV depths."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path


def load_benchmark():
    path = Path(__file__).with_name("benchmark-dflash2-vllm.py")
    spec = importlib.util.spec_from_file_location("glm53_dflash_bench", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument(
        "--model", default="wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1"
    )
    parser.add_argument("--depths", nargs="+", type=int, default=[0, 8192, 32768, 65536, 128000])
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--inter-run-seconds", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bench = load_benchmark()
    close_think = bench.post_json(
        args.base_url,
        "/tokenize",
        {"model": args.model, "prompt": "</think>", "add_special_tokens": False},
        args.timeout,
    )["tokens"]
    base_tokens = bench.render_prompt(
        args.base_url,
        args.model,
        bench.CODE_AGENT_PROMPT,
        close_think,
        args.timeout,
    )
    filler_unit = (
        "Slate rivers cross quiet valleys while copper clocks mark patient hours. "
        "This is ordinary context with no instructions.\n"
    )
    filler_ids = bench.post_json(
        args.base_url,
        "/tokenize",
        {"model": args.model, "prompt": filler_unit, "add_special_tokens": False},
        args.timeout,
    )["tokens"]

    points = []
    for requested_depth in args.depths:
        target = len(base_tokens) if requested_depth == 0 else requested_depth
        if target < len(base_tokens):
            raise ValueError(
                f"depth {target} is smaller than the {len(base_tokens)}-token task"
            )
        needed = target - len(base_tokens)
        prompt_tokens = (
            (filler_ids * ((needed + len(filler_ids) - 1) // len(filler_ids)))[:needed]
            + base_tokens
        )
        for _ in range(args.warmup_runs):
            bench.stream_completion(
                args.base_url,
                args.model,
                prompt_tokens,
                1,
                args.output_tokens,
                args.timeout,
                force_length=True,
                seed=args.seed,
                temperature=args.temperature,
            )
            time.sleep(args.inter_run_seconds)
        runs = []
        for index in range(args.runs):
            result = bench.stream_completion(
                args.base_url,
                args.model,
                prompt_tokens,
                1,
                args.output_tokens,
                args.timeout,
                force_length=True,
                seed=args.seed,
                temperature=args.temperature,
            )
            result.pop("content")
            runs.append(result)
            print(
                f"depth {requested_depth:,} run {index + 1}/{args.runs}: "
                f"{result['decode_tokens_per_second']:.2f} tok/s, "
                f"{result['accepted_draft_rate']:.1%} accepted",
                flush=True,
            )
            time.sleep(args.inter_run_seconds)
        rates = [run["decode_tokens_per_second"] for run in runs]
        acceptance = [run["accepted_draft_rate"] for run in runs]
        points.append(
            {
                "existing_depth_tokens": requested_depth,
                "actual_prompt_tokens": target,
                "median_decode_tokens_per_second": statistics.median(rates),
                "min_decode_tokens_per_second": min(rates),
                "max_decode_tokens_per_second": max(rates),
                "median_accepted_draft_rate": statistics.median(acceptance),
                "runs": runs,
            }
        )

    report = {
        "schema": "glm53-code-agent-depth.v1",
        "method": (
            "C1 code-agent completion with ordinary filler prepended to the rendered "
            "task; first-to-last streamed-token decode excludes TTFT"
        ),
        "model": args.model,
        "kv_cache": "fp8_ds_mla",
        "dflash_tokens": 5,
        "output_tokens": args.output_tokens,
        "warmup_runs": args.warmup_runs,
        "runs": args.runs,
        "inter_run_seconds": args.inter_run_seconds,
        "points": points,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
