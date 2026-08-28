#!/usr/bin/env python3
"""Measure aggregate pure-decode throughput at selected concurrencies."""

from __future__ import annotations

import argparse
import http.client
import json
import statistics
import time
from pathlib import Path
from urllib.parse import urlparse


def request_once(
    base_url: str, model: str, concurrency: int, output_tokens: int, seed: int
) -> dict:
    parsed = urlparse(base_url)
    connection_type = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    connection = connection_type(parsed.hostname, parsed.port, timeout=3600)
    path = parsed.path.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a long, coherent continuation about systems engineering, "
                    "without headings or a conclusion."
                ),
            }
        ],
        "n": concurrency,
        "max_tokens": output_tokens,
        "min_tokens": output_tokens,
        "ignore_eos": True,
        "temperature": 0.7,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
        "cache_prompt": False,
    }
    started = time.perf_counter()
    connection.request(
        "POST",
        path,
        body=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    if response.status != 200:
        error = response.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {response.status}: {error}")

    token_times: list[list[float]] = [[] for _ in range(concurrency)]
    completion_tokens = None
    while True:
        raw_line = response.readline()
        if not raw_line:
            break
        observed = time.perf_counter()
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        event = json.loads(data)
        if isinstance(event.get("usage"), dict):
            completion_tokens = event["usage"].get("completion_tokens")
        for choice in event.get("choices", []):
            index = choice.get("index")
            if not isinstance(index, int) or not 0 <= index < concurrency:
                continue
            token_ids = choice.get("token_ids")
            if isinstance(token_ids, list):
                token_times[index].extend([observed] * len(token_ids))
    connection.close()

    expected = concurrency * output_tokens
    observed_tokens = sum(len(times) for times in token_times)
    if observed_tokens != expected:
        raise RuntimeError(
            f"stream exposed {observed_tokens} token IDs, expected {expected}; "
            f"usage reported {completion_tokens}"
        )
    first_token = min(times[0] for times in token_times if times)
    last_token = max(times[-1] for times in token_times if times)
    duration = last_token - first_token
    decode_tokens = sum(len(times) - 1 for times in token_times)
    return {
        "concurrency": concurrency,
        "output_tokens_per_sequence": output_tokens,
        "completion_tokens": completion_tokens,
        "decode_tokens": decode_tokens,
        "decode_seconds": duration,
        "decode_tokens_per_second": decode_tokens / duration,
        "ttft_ms": (first_token - started) * 1000,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--profile", required=True, choices=["nvfp4", "fp8"])
    parser.add_argument("--mtp-tokens", type=int, required=True)
    parser.add_argument(
        "--mtp-policy",
        choices=["off", "static", "adaptive"],
        default="static",
    )
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 16])
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument(
        "--warmup-tokens",
        type=int,
        default=0,
        help="warmup length; 0 uses --output-tokens so the full decode path is warm",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=2,
        help="full warmup requests per concurrency (two covers lazy JIT shapes)",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    points = []
    warmup_tokens = args.warmup_tokens or args.output_tokens
    for concurrency in args.concurrency:
        for _ in range(args.warmup_runs):
            request_once(
                args.base_url, args.model, concurrency, warmup_tokens, args.seed
            )
        runs = []
        for run in range(args.runs):
            result = request_once(
                args.base_url,
                args.model,
                concurrency,
                args.output_tokens,
                args.seed,
            )
            runs.append(result)
            print(
                f"{args.profile} {args.mtp_policy} MTP{args.mtp_tokens} "
                f"C{concurrency} "
                f"run {run + 1}/{args.runs}: "
                f"{result['decode_tokens_per_second']:.2f} tok/s",
                flush=True,
            )
        rates = [run["decode_tokens_per_second"] for run in runs]
        points.append(
            {
                "concurrency": concurrency,
                "aggregate_decode_tokens_per_second": {
                    "median": statistics.median(rates),
                    "min": min(rates),
                    "max": max(rates),
                },
                "runs": runs,
            }
        )

    report = {
        "schema": "glm53-decode-concurrency.v2",
        "method": (
            "one depth-0 prompt with n parallel continuations; aggregate decode "
            "timing spans first to last streamed token and excludes TTFT; the "
            "same sampling seed is repeated so MTP acceptance is comparable"
        ),
        "model": args.model,
        "kv_cache_profile": args.profile,
        "mtp_tokens": args.mtp_tokens,
        "mtp_policy": args.mtp_policy,
        "output_tokens_per_sequence": args.output_tokens,
        "warmup_runs_per_point": args.warmup_runs,
        "runs_per_point": args.runs,
        "points": points,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
