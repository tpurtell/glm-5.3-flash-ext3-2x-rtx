#!/usr/bin/env python3
"""Run an exact-token-count long-context needle test through the OpenAI API."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request


MODEL = "brandonmusic/GLM-5.3-Flash-EXL3-4bpw"
NEEDLE = "SPARK-B12X-5.3-128K-7429"


def post(base_url: str, path: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def tokenize(base_url: str, prompt: str) -> list[int]:
    result = post(base_url, "/tokenize", {"model": MODEL, "prompt": prompt}, 300)
    return result["tokens"]


def detokenize(base_url: str, tokens: list[int]) -> str:
    result = post(base_url, "/detokenize", {"model": MODEL, "tokens": tokens}, 300)
    return result["prompt"]


def exact_prompt(base_url: str, target_tokens: int) -> str:
    prefix = (
        "Long-context memory test. Store this exact secret code and do not alter it:\n"
        f"{NEEDLE}\n"
        "The following material is distractor text.\n"
    )
    suffix = (
        "\nEnd of distractor text. What was the exact secret code near the beginning? "
        "Respond with only that code and nothing else.\nAnswer:"
    )
    prefix_ids = tokenize(base_url, prefix)
    suffix_ids = tokenize(base_url, suffix)
    required_filler = target_tokens - len(prefix_ids) - len(suffix_ids)
    if required_filler <= 0:
        raise ValueError("target is too small for the test framing")

    unit = (
        "Slate rivers cross quiet valleys while copper clocks mark patient hours. "
        "This sentence contains no secret code and is only ordinary filler.\n"
    )
    # Produce surplus canonical tokenizer ids, then splice the exact count.
    unit_ids = tokenize(base_url, unit)
    repeats = required_filler // max(1, len(unit_ids)) + 2
    filler_ids = tokenize(base_url, unit * repeats)
    while len(filler_ids) < required_filler:
        repeats *= 2
        filler_ids = tokenize(base_url, unit * repeats)

    ids = prefix_ids + filler_ids[:required_filler] + suffix_ids
    prompt = detokenize(base_url, ids)
    roundtrip = tokenize(base_url, prompt)
    if len(roundtrip) != target_tokens:
        # Token boundaries can merge at the two splice points. Adjust only the
        # filler slice until decode->encode reaches the requested exact count.
        delta = target_tokens - len(roundtrip)
        required_filler += delta
        if required_filler <= 0 or required_filler > len(filler_ids):
            raise RuntimeError("could not adjust filler to the exact token count")
        ids = prefix_ids + filler_ids[:required_filler] + suffix_ids
        prompt = detokenize(base_url, ids)
        roundtrip = tokenize(base_url, prompt)
    if len(roundtrip) != target_tokens:
        raise RuntimeError(
            f"prompt roundtrip has {len(roundtrip)} tokens, expected {target_tokens}"
        )
    return prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--tokens", type=int, default=128_000)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=7200)
    args = parser.parse_args()

    started = time.perf_counter()
    prompt = exact_prompt(args.base_url, args.tokens)
    built = time.perf_counter()
    result = post(
        args.base_url,
        "/v1/completions",
        {
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": args.max_tokens,
            "temperature": 0,
        },
        args.timeout,
    )
    finished = time.perf_counter()
    text = result["choices"][0]["text"]
    passed = NEEDLE in text
    report = {
        "passed": passed,
        "target_prompt_tokens": args.tokens,
        "usage": result.get("usage"),
        "build_seconds": round(built - started, 3),
        "request_seconds": round(finished - built, 3),
        "needle": NEEDLE,
        "output": text,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
