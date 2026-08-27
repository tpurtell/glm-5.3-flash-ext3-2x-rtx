#!/usr/bin/env python3
"""Run glmrt's seven semantic content contracts against a stock vLLM API."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="brandonmusic/GLM-5.3-Flash-EXL3-4bpw")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="Optional override for each contract's visible completion budget.",
    )
    args = parser.parse_args()

    glmrt_tools = Path(__file__).resolve().parents[2] / "glmrt/python/tools"
    sys.path.insert(0, str(glmrt_tools))
    from bench_real_full_mtp_acceptance import (  # type: ignore[import-not-found]
        CASES,
        WEIGHTED_CASE_IDS,
        validate_case_content,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    def post(path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            args.base_url.rstrip("/") + path,
            data=json.dumps(payload, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            return json.load(response)

    close_think = post(
        "/tokenize",
        {"model": args.model, "prompt": "</think>", "add_special_tokens": False},
    )["tokens"]
    results = []
    for case_id in WEIGHTED_CASE_IDS:
        case = CASES[case_id]
        render = post(
            "/v1/chat/completions/render",
            {
                "model": args.model,
                "messages": [{"role": "user", "content": case.prompt}],
                "reasoning_effort": "low",
            },
        )
        payload = {
            "model": args.model,
            # The published template opens <think> unconditionally. Closing it
            # here makes this a visible-answer quality test, not a reasoning-
            # token budget test.
            "prompt": render["token_ids"] + close_think,
            "temperature": 0,
            "max_tokens": args.max_tokens or case.max_tokens,
            "add_special_tokens": False,
        }
        started = time.perf_counter()
        body = post("/v1/completions", payload)
        seconds = time.perf_counter() - started
        choice = body["choices"][0]
        content = choice.get("text") or ""
        validation = validate_case_content(case_id, content)
        result = {
            "case": case_id,
            "category": case.category,
            "passed": validation["quality_contract_passed"],
            "issues": validation["quality_contract_issues"],
            "seconds": round(seconds, 3),
            "prompt_tokens": body["usage"]["prompt_tokens"],
            "completion_tokens": body["usage"]["completion_tokens"],
            "finish_reason": choice["finish_reason"],
            "preview": content[:240].replace("\n", "\\n"),
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    summary = {
        "summary": True,
        "passed": sum(item["passed"] for item in results),
        "total": len(results),
        "request_seconds": round(sum(item["seconds"] for item in results), 3),
        "completion_tokens": sum(item["completion_tokens"] for item in results),
    }
    print(json.dumps(summary), flush=True)
    with args.output.open("w", encoding="utf-8") as output:
        for result in results:
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
        output.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
