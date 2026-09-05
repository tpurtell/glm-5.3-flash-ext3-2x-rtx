#!/usr/bin/env python3
"""Repeat one long cached prompt to exercise prefix reuse and recycled page IDs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path


def load_long_context():
    path = Path(__file__).with_name("test-long-context.py")
    spec = importlib.util.spec_from_file_location("glm53_long_context", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prefix_hits(base_url: str) -> float:
    with urllib.request.urlopen(base_url.rstrip("/") + "/metrics", timeout=300) as response:
        body = response.read().decode("utf-8", errors="replace")
    total = 0.0
    for line in body.splitlines():
        if line.startswith("vllm:prefix_cache_hits_total{"):
            total += float(line.rsplit(" ", 1)[-1])
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument(
        "--model", default="wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1"
    )
    parser.add_argument("--tokens", type=int, default=128_000)
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    long_context = load_long_context()
    nonce = f"prefix-replay-{uuid.uuid4().hex}"
    cache_salt = uuid.uuid4().hex
    prompt = long_context.exact_prompt(
        args.base_url, args.model, args.tokens, nonce
    )
    before_hits = prefix_hits(args.base_url)
    passes = []
    for pass_index in range(2):
        started = time.perf_counter()
        result = long_context.post(
            args.base_url,
            "/v1/completions",
            {
                "model": args.model,
                "prompt": prompt,
                "max_tokens": 64,
                "temperature": 0,
                "seed": 20260901,
                "cache_salt": cache_salt,
            },
            args.timeout,
        )
        elapsed = time.perf_counter() - started
        output = result["choices"][0].get("text") or ""
        passes.append(
            {
                "pass": pass_index + 1,
                "passed": long_context.NEEDLE in output,
                "request_seconds": elapsed,
                "usage": result.get("usage"),
                "output": output,
            }
        )
    after_hits = prefix_hits(args.base_url)
    report = {
        "schema": "glm53-prefix-replay.v1",
        "passed": all(item["passed"] for item in passes) and after_hits > before_hits,
        "target_prompt_tokens": args.tokens,
        "cache_salt": cache_salt,
        "prefix_cache_hits_before": before_hits,
        "prefix_cache_hits_after": after_hits,
        "prefix_cache_hit_delta": after_hits - before_hits,
        "passes": passes,
        "high_recycled_page_rationale": (
            "The identical second pass exercises the production allocator/cache replay "
            "path that previously exposed 32-bit pool-offset overflow; focused B12x "
            "tests separately force page IDs past the exact 2^31/stride boundary."
        ),
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(rendered, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
