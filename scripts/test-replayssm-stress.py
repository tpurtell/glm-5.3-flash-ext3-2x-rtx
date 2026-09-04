#!/usr/bin/env python3
"""Matched rolling-batch regression for GLM KDA ReplaySSM corruption.

The workload shape is based on Samuel Cardillo's public 32K/C4 reproducer,
with exact-token fixtures, raw SSE receipts, and a third unique-prefix phase.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import statistics
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUBWORD_LOOP = re.compile(r"([A-Za-z]{2,32})(?:\1){7,}")
CHAR_LOOP = re.compile(r"([^\s])\1{31,}", re.DOTALL)


@dataclass(frozen=True)
class Phase:
    name: str
    chat_template_kwargs: dict[str, object]
    unique_prefix: bool
    require_marker: bool


PHASES = (
    Phase(
        "thinking-off-shared-prefix",
        {"enable_thinking": False},
        unique_prefix=False,
        require_marker=True,
    ),
    Phase(
        "thinking-max-shared-prefix",
        {"enable_thinking": True, "reasoning_effort": "max"},
        unique_prefix=False,
        require_marker=False,
    ),
    Phase(
        "thinking-max-unique-prefix",
        {"enable_thinking": True, "reasoning_effort": "max"},
        unique_prefix=True,
        require_marker=False,
    ),
)


def request_json(
    base_url: str,
    path: str,
    payload: dict[str, Any] | None,
    timeout: int,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} did not return a JSON object")
    return value


def health(base_url: str, timeout: int) -> dict[str, Any]:
    started = time.perf_counter()
    request = urllib.request.Request(base_url.rstrip("/") + "/health")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            response.read()
    except Exception as error:
        return {
            "ok": False,
            "error": repr(error),
            "seconds": round(time.perf_counter() - started, 3),
        }
    return {
        "ok": status == 200,
        "status": status,
        "seconds": round(time.perf_counter() - started, 3),
    }


def token_count(
    base_url: str,
    model: str,
    content: str,
    chat_template_kwargs: dict[str, object],
    timeout: int,
) -> int:
    result = request_json(
        base_url,
        "/tokenize",
        {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "add_generation_prompt": True,
            "chat_template_kwargs": chat_template_kwargs,
        },
        timeout,
    )
    return int(result["count"])


def fixture_text(prefix: str, repeats: int, filler: int, marker: str) -> str:
    record = (
        "Archive replay-ssm-32768: ordinary numbered engineering records; "
        "this is inert reference text. Each record describes a completed "
        "checksum audit. "
    )
    suffix = (
        f"\nThe unique retrieval marker is {marker}. State exactly this marker "
        "and then say that archive replay-ssm-32768 was audited."
    )
    return prefix + record * repeats + " x" * filler + suffix


def build_exact_fixture(
    base_url: str,
    model: str,
    target_tokens: int,
    chat_template_kwargs: dict[str, object],
    timeout: int,
    *,
    nonce: str | None,
) -> tuple[str, str, int]:
    marker = "NEEDLE-ReplaySSM-32768-739184"
    prefix = "" if nonce is None else f"Unique request prefix: {nonce}.\n"
    low, high = 0, target_tokens
    while low + 1 < high:
        middle = (low + high) // 2
        candidate = fixture_text(prefix, middle, 0, marker)
        if (
            token_count(
                base_url,
                model,
                candidate,
                chat_template_kwargs,
                timeout,
            )
            < target_tokens
        ):
            low = middle
        else:
            high = middle
    repeats = low
    filler = 0
    for _ in range(12):
        candidate = fixture_text(prefix, repeats, filler, marker)
        count = token_count(
            base_url,
            model,
            candidate,
            chat_template_kwargs,
            timeout,
        )
        delta = target_tokens - count
        if delta == 0:
            return candidate, marker, count
        filler += delta
        if filler < 0:
            repeats -= 1
            filler = 0
    raise RuntimeError(
        f"could not construct {target_tokens} tokens for nonce={nonce!r}"
    )


def dominant_repeated_ngram(text: str) -> tuple[str | None, int, float]:
    words = re.findall(r"[A-Za-z0-9_-]+", text.lower())
    best: tuple[str | None, int, float] = (None, 0, 0.0)
    if not words:
        return best
    for size in range(2, 9):
        positions: dict[tuple[str, ...], list[int]] = {}
        for index in range(len(words) - size + 1):
            positions.setdefault(tuple(words[index : index + size]), []).append(index)
        for ngram, starts in positions.items():
            count = 0
            next_start = 0
            for start in starts:
                if start >= next_start:
                    count += 1
                    next_start = start + size
            dominance = count * size / len(words)
            if count >= 12 and dominance >= 0.60 and dominance > best[2]:
                best = (" ".join(ngram), count, dominance)
    return best


def detect_loop(text: str) -> dict[str, Any]:
    ngram, count, dominance = dominant_repeated_ngram(text)
    subword = SUBWORD_LOOP.search(text)
    character = CHAR_LOOP.search(text)
    return {
        "detected": bool(subword or character or ngram),
        "subword": subword.group(1) if subword else None,
        "character": character.group(1) if character else None,
        "ngram": ngram,
        "ngram_count": count,
        "ngram_dominance": round(dominance, 6),
    }


def stream_one(
    base_url: str,
    payload: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
    )
    raw_sse: list[str] = []
    content: list[str] = []
    reasoning: list[str] = []
    finish_reason = None
    usage = None
    saw_done = False
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line.startswith("data:"):
                continue
            raw_sse.append(line)
            body = line[5:].strip()
            if body == "[DONE]":
                saw_done = True
                break
            event = json.loads(body)
            if not isinstance(event, dict):
                raise RuntimeError("SSE event is not an object")
            if event.get("error") is not None:
                raise RuntimeError(f"server SSE error: {event['error']}")
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                if isinstance(delta.get("content"), str):
                    content.append(delta["content"])
                for key in ("reasoning", "reasoning_content"):
                    if isinstance(delta.get(key), str):
                        reasoning.append(delta[key])
                finish_reason = choice.get("finish_reason") or finish_reason
    if not saw_done:
        raise RuntimeError("SSE stream ended without [DONE]")
    if finish_reason is None:
        raise RuntimeError("SSE stream ended without finish_reason")
    if usage is None:
        raise RuntimeError("SSE stream ended without usage")
    answer = "".join(content)
    thought = "".join(reasoning)
    combined = thought + "\n" + answer
    if not combined.strip():
        raise RuntimeError("SSE stream contained empty generated output")
    return {
        "seconds": round(time.perf_counter() - started, 3),
        "finish_reason": finish_reason,
        "usage": usage,
        "content": answer,
        "reasoning": thought,
        "loop": detect_loop(combined),
        "raw_sse": raw_sse,
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests-per-phase", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--target-tokens", type=int, default=32_768)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--seed", type=int, default=739_184)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.requests_per_phase < 1 or args.concurrency < 1:
        raise SystemExit("request counts must be positive")
    args.output.mkdir(parents=True, exist_ok=False)

    models = request_json(args.base_url, "/v1/models", None, args.timeout)
    before_health = health(args.base_url, args.timeout)
    if not before_health["ok"]:
        raise RuntimeError(f"server unhealthy before stress: {before_health}")

    all_results: list[dict[str, Any]] = []
    phase_summaries: list[dict[str, Any]] = []
    for phase in PHASES:
        fixtures: list[tuple[str, str, int]] = []
        if phase.unique_prefix:
            for index in range(args.requests_per_phase):
                fixtures.append(
                    build_exact_fixture(
                        args.base_url,
                        args.model,
                        args.target_tokens,
                        phase.chat_template_kwargs,
                        args.timeout,
                        nonce=f"request-{index:08d}",
                    )
                )
        else:
            shared = build_exact_fixture(
                args.base_url,
                args.model,
                args.target_tokens,
                phase.chat_template_kwargs,
                args.timeout,
                nonce=None,
            )
            fixtures = [shared] * args.requests_per_phase

        def run(index: int) -> dict[str, Any]:
            content, marker, count = fixtures[index]
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0,
                "top_p": 1,
                "seed": args.seed,
                "max_tokens": args.max_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
                "chat_template_kwargs": phase.chat_template_kwargs,
            }
            result: dict[str, Any] = {
                "phase": phase.name,
                "index": index,
                "fixture_tokens": count,
                "fixture_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "marker": marker,
                "error": None,
            }
            try:
                result.update(stream_one(args.base_url, payload, args.timeout))
                generated = result["reasoning"] + "\n" + result["content"]
                result["marker_found"] = marker in generated
                result["passed"] = (
                    not result["loop"]["detected"]
                    and (result["marker_found"] or not phase.require_marker)
                )
            except Exception as error:
                result.update(
                    {
                        "error": repr(error),
                        "loop": {"detected": False},
                        "marker_found": False,
                        "passed": False,
                    }
                )
            write_json(args.output / f"{phase.name}-{index:04d}.json", result)
            print(
                json.dumps(
                    {
                        "phase": phase.name,
                        "index": index,
                        "passed": result["passed"],
                        "error": result["error"],
                        "loop": result["loop"]["detected"],
                    }
                ),
                flush=True,
            )
            return result

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as pool:
            phase_results = list(pool.map(run, range(args.requests_per_phase)))
        all_results.extend(phase_results)
        phase_health = health(args.base_url, args.timeout)
        phase_summary = {
            "phase": phase.name,
            "requests": len(phase_results),
            "passes": sum(bool(item["passed"]) for item in phase_results),
            "errors": sum(item["error"] is not None for item in phase_results),
            "loops": sum(bool(item["loop"]["detected"]) for item in phase_results),
            "marker_misses": sum(not item["marker_found"] for item in phase_results),
            "median_seconds": round(
                statistics.median(
                    item["seconds"]
                    for item in phase_results
                    if item.get("seconds") is not None
                ),
                3,
            )
            if any(item.get("seconds") is not None for item in phase_results)
            else None,
            "health_after": phase_health,
        }
        phase_summaries.append(phase_summary)
        write_json(args.output / f"{phase.name}-summary.json", phase_summary)

    passed = all(item["passed"] for item in all_results) and all(
        phase["health_after"]["ok"] for phase in phase_summaries
    )
    summary = {
        "schema": "glm53-kda-replayssm-rolling-stress-v1",
        "passed": passed,
        "source_workload": (
            "https://github.com/samuelcardillo/"
            "glm-5.3-flash-2x-rtx-pro-6000-blackwell/commit/"
            "1755c0f0c01b98463a7b87ab613a6c894b569298"
        ),
        "model": args.model,
        "target_tokens": args.target_tokens,
        "requests_per_phase": args.requests_per_phase,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "models_response": models,
        "health_before": before_health,
        "phases": phase_summaries,
        "total_requests": len(all_results),
        "total_passes": sum(bool(item["passed"]) for item in all_results),
        "total_errors": sum(item["error"] is not None for item in all_results),
        "total_loops": sum(bool(item["loop"]["detected"]) for item in all_results),
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
