#!/usr/bin/env python3
"""Cold multi-needle retrieval qualification at an exact prompt-token count."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import urllib.request
import uuid
from pathlib import Path


DEFAULT_MODEL = "wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1"
FACTS = (
    (0.05, "CINDER-05", "azurite-4831"),
    (0.25, "JUNIPER-25", "topaz-7614"),
    (0.50, "LANTERN-50", "cobalt-2097"),
    (0.75, "MARBLE-75", "saffron-6382"),
    (0.95, "ORBIT-95", "willow-1459"),
    (0.99, "QUARTZ-99", "indigo-8726"),
)


def post(base_url: str, path: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def stream_completion(
    base_url: str, payload: dict, timeout: float
) -> tuple[str, dict | None, float, float]:
    """Return output, usage, TTFT, and total request time from an SSE stream."""
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_token_at: float | None = None
    output_parts: list[str] = []
    usage: dict | None = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("usage") is not None:
                usage = event["usage"]
            for choice in event.get("choices") or ():
                text = choice.get("text") or ""
                if text:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    output_parts.append(text)
    finished = time.perf_counter()
    if first_token_at is None:
        raise RuntimeError("stream completed without an output token")
    return (
        "".join(output_parts),
        usage,
        first_token_at - started,
        finished - started,
    )


def tokenize(base_url: str, model: str, prompt: str) -> list[int]:
    return post(
        base_url, "/tokenize", {"model": model, "prompt": prompt}, 600
    )["tokens"]


def detokenize(base_url: str, model: str, tokens: list[int]) -> str:
    return post(
        base_url, "/detokenize", {"model": model, "tokens": tokens}, 600
    )["prompt"]


def build_exact_prompt(
    base_url: str, model: str, target_tokens: int, nonce: str
) -> tuple[str, list[dict]]:
    prefix = (
        f"Cold long-context retrieval qualification {nonce}.\n"
        "The document contains six audit records. Remember every KEY=VALUE pair. "
        "All other prose is distractor material.\n"
    )
    suffix = (
        "\nEND OF DOCUMENT. Return all six audit records in document order, exactly "
        "as KEY=VALUE, one per line. Return nothing else.\nANSWER:\n"
    )
    filler = (
        "Ordinary archive prose describes quiet rivers, copper clocks, patient "
        "engineers, and slate valleys; it contains no audit key or value.\n"
    )
    prefix_ids = tokenize(base_url, model, prefix)
    suffix_ids = tokenize(base_url, model, suffix)
    filler_ids = tokenize(base_url, model, filler)
    fact_ids = [
        tokenize(
            base_url,
            model,
            f"\nAUDIT RECORD: {key}={value}\n",
        )
        for _, key, value in FACTS
    ]
    fixed = len(prefix_ids) + len(suffix_ids) + sum(map(len, fact_ids))
    if target_tokens <= fixed + len(FACTS):
        raise ValueError("target token count is too small for the qualification")

    # Place each record near its requested absolute depth, filling every gap
    # with a deterministic canonical token stream.
    result = list(prefix_ids)
    placements: list[dict] = []
    for (fraction, key, value), encoded in zip(FACTS, fact_ids):
        desired = round(target_tokens * fraction)
        gap = max(1, desired - len(result))
        repeats = math.ceil(gap / len(filler_ids))
        result.extend((filler_ids * repeats)[:gap])
        placements.append(
            {
                "fraction": fraction,
                "key": key,
                "value": value,
                "token_offset_before_record": len(result),
            }
        )
        result.extend(encoded)

    tail = target_tokens - len(result) - len(suffix_ids)
    if tail < 1:
        raise RuntimeError("record placement left no room for the question")
    result.extend((filler_ids * math.ceil(tail / len(filler_ids)))[:tail])
    result.extend(suffix_ids)

    # Decode/encode can canonicalize the two splice boundaries. Adjust only
    # the final distractor span until the HTTP-visible prompt is exact.
    for _ in range(8):
        prompt = detokenize(base_url, model, result)
        roundtrip = tokenize(base_url, model, prompt)
        delta = target_tokens - len(roundtrip)
        if delta == 0:
            break
        suffix_len = len(suffix_ids)
        tail_start = len(result) - suffix_len - tail
        tail += delta
        if tail < 1:
            raise RuntimeError("canonicalization exhausted the final filler span")
        replacement = (filler_ids * math.ceil(tail / len(filler_ids)))[:tail]
        result = result[:tail_start] + replacement + result[-suffix_len:]
    else:
        raise RuntimeError("could not canonicalize the exact prompt-token count")

    if len(roundtrip) != target_tokens:
        raise RuntimeError(
            f"prompt roundtrip has {len(roundtrip)} tokens, expected {target_tokens}"
        )
    for _, key, value in FACTS:
        record = f"{key}={value}"
        if prompt.count(record) != 1:
            raise RuntimeError(f"prompt does not contain exactly one {record!r}")
    return prompt, placements


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokens", type=int, default=1_000_000)
    parser.add_argument("--nonce", default="dflash2-fp8-release")
    parser.add_argument("--cache-salt")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=14_400)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cache_salt = args.cache_salt or uuid.uuid4().hex
    started = time.perf_counter()
    prompt, placements = build_exact_prompt(
        args.base_url, args.model, args.tokens, args.nonce
    )
    built = time.perf_counter()
    output, usage, ttft_seconds, request_seconds = stream_completion(
        args.base_url,
        {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "cache_salt": cache_salt,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        args.timeout,
    )
    finished = time.perf_counter()
    checks = [
        {
            "key": key,
            "expected": value,
            "passed": f"{key}={value}" in output,
        }
        for _, key, value in FACTS
    ]
    report = {
        "schema": "glm53-cold-multi-needle.v1",
        "passed": all(check["passed"] for check in checks),
        "target_prompt_tokens": args.tokens,
        "nonce": args.nonce,
        "cache_salt": cache_salt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "placements": placements,
        "checks": checks,
        "usage": usage,
        "build_seconds": round(built - started, 3),
        "ttft_seconds": round(ttft_seconds, 3),
        "stream_after_first_token_seconds": round(
            request_seconds - ttft_seconds, 3
        ),
        "request_seconds": round(request_seconds, 3),
        "wall_seconds_after_build": round(finished - built, 3),
        "output": output,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
