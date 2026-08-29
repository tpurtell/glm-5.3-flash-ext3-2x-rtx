#!/usr/bin/env python3
"""Benchmark DFlash2 on a code-agent task or glmrt's seven-case blend."""

from __future__ import annotations

import argparse
import http.client
import importlib.util
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path
from types import ModuleType
from urllib.parse import urlparse


CODE_AGENT_PROMPT = """You are editing an async Python task runner. Fix the cancellation and
exception-handling bugs in this implementation, preserve result ordering, and add precise type
hints. Return only the complete replacement Python module.

```python
import asyncio

async def run_all(factories, limit=8):
    sem = asyncio.Semaphore(limit)
    results = []
    async def one(factory):
        async with sem:
            results.append(await factory())
    tasks = [asyncio.create_task(one(factory)) for factory in factories]
    try:
        await asyncio.gather(*tasks)
    except Exception:
        for task in tasks:
            task.cancel()
    return results
```
"""


def load_contracts() -> ModuleType:
    path = Path(__file__).with_name("test-content-vllm.py")
    spec = importlib.util.spec_from_file_location("glmrt_vllm_contracts", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load content contracts from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def post_json(base_url: str, path: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def metrics(base_url: str, timeout: float) -> dict[str, float]:
    with urllib.request.urlopen(
        base_url.rstrip("/") + "/metrics", timeout=timeout
    ) as response:
        text = response.read().decode("utf-8", errors="replace")
    wanted = {
        "vllm:spec_decode_num_draft_tokens_total": "draft_tokens",
        "vllm:spec_decode_num_accepted_tokens_total": "accepted_tokens",
    }
    values = {name: 0.0 for name in wanted.values()}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        metric_name = line.split("{", 1)[0].split(" ", 1)[0]
        destination = wanted.get(metric_name)
        if destination is not None:
            values[destination] += float(line.rsplit(" ", 1)[-1])
    return values


def render_prompt(
    base_url: str, model: str, prompt: str, close_think: list[int], timeout: float
) -> list[int]:
    rendered = post_json(
        base_url,
        "/v1/chat/completions/render",
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "reasoning_effort": "low",
        },
        timeout,
    )
    return rendered["token_ids"] + close_think


def stream_completion(
    base_url: str,
    model: str,
    prompt_tokens: list[int],
    concurrency: int,
    output_tokens: int,
    timeout: float,
    *,
    force_length: bool,
    seed: int,
    temperature: float,
) -> dict:
    parsed = urlparse(base_url)
    connection_type = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    connection = connection_type(parsed.hostname, parsed.port, timeout=timeout)
    payload = {
        "model": model,
        "prompt": prompt_tokens,
        "add_special_tokens": False,
        "n": concurrency,
        "max_tokens": output_tokens,
        "temperature": temperature,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
        "cache_prompt": False,
    }
    if force_length:
        payload.update({"min_tokens": output_tokens, "ignore_eos": True})

    before = metrics(base_url, timeout)
    started = time.perf_counter()
    connection.request(
        "POST",
        parsed.path.rstrip("/") + "/v1/completions",
        body=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    if response.status != 200:
        error = response.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {response.status}: {error}")

    token_times: list[list[float]] = [[] for _ in range(concurrency)]
    content = ["" for _ in range(concurrency)]
    finish_reasons: list[str | None] = [None for _ in range(concurrency)]
    usage = None
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
            usage = event["usage"]
        for choice in event.get("choices", []):
            index = choice.get("index")
            if not isinstance(index, int) or not 0 <= index < concurrency:
                continue
            content[index] += choice.get("text") or ""
            if choice.get("finish_reason") is not None:
                finish_reasons[index] = choice["finish_reason"]
            token_ids = choice.get("token_ids")
            if isinstance(token_ids, list):
                token_times[index].extend([observed] * len(token_ids))
    connection.close()
    ended = time.perf_counter()
    after = metrics(base_url, timeout)

    nonempty = [times for times in token_times if times]
    if len(nonempty) != concurrency:
        raise RuntimeError(
            f"only {len(nonempty)} of {concurrency} streams exposed token IDs"
        )
    first_token = min(times[0] for times in nonempty)
    last_token = max(times[-1] for times in nonempty)
    decode_tokens = sum(max(0, len(times) - 1) for times in token_times)
    decode_seconds = last_token - first_token
    drafted = after["draft_tokens"] - before["draft_tokens"]
    accepted = after["accepted_tokens"] - before["accepted_tokens"]
    return {
        "concurrency": concurrency,
        "prompt_tokens": len(prompt_tokens),
        "completion_tokens": sum(len(times) for times in token_times),
        "completion_tokens_by_sequence": [len(times) for times in token_times],
        "decode_tokens": decode_tokens,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": decode_tokens / decode_seconds,
        "ttft_ms": (first_token - started) * 1000,
        "request_seconds": ended - started,
        "draft_tokens": int(drafted),
        "accepted_draft_tokens": int(accepted),
        "accepted_draft_rate": accepted / drafted if drafted else 0.0,
        "finish_reasons": finish_reasons,
        "content": content,
        "usage": usage,
    }


def summarize_runs(runs: list[dict]) -> dict:
    rates = [run["decode_tokens_per_second"] for run in runs]
    acceptances = [run["accepted_draft_rate"] for run in runs]
    return {
        "median_decode_tokens_per_second": statistics.median(rates),
        "min_decode_tokens_per_second": min(rates),
        "max_decode_tokens_per_second": max(rates),
        "median_accepted_draft_rate": statistics.median(acceptances),
        "draft_tokens": sum(run["draft_tokens"] for run in runs),
        "accepted_draft_tokens": sum(run["accepted_draft_tokens"] for run in runs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument(
        "--model", default="wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3-v1"
    )
    parser.add_argument("--suite", choices=("code-agent", "blend"), required=True)
    parser.add_argument("--dflash-tokens", type=int, required=True)
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 16])
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    close_think = post_json(
        args.base_url,
        "/tokenize",
        {"model": args.model, "prompt": "</think>", "add_special_tokens": False},
        args.timeout,
    )["tokens"]
    report: dict = {
        "schema": "glm53-dflash2-vllm.v1",
        "model": args.model,
        "kv_cache": "fp8_ds_mla",
        "dflash_tokens": args.dflash_tokens,
        "suite": args.suite,
        "method": (
            "decode timing spans the first through last streamed token and excludes "
            "TTFT; DFlash acceptance is the matching Prometheus counter delta"
        ),
    }

    if args.suite == "code-agent":
        prompt_tokens = render_prompt(
            args.base_url, args.model, CODE_AGENT_PROMPT, close_think, args.timeout
        )
        points = []
        for concurrency in args.concurrency:
            for _ in range(args.warmup_runs):
                stream_completion(
                    args.base_url,
                    args.model,
                    prompt_tokens,
                    concurrency,
                    args.output_tokens,
                    args.timeout,
                    force_length=True,
                    seed=args.seed,
                    temperature=args.temperature,
                )
            runs = []
            for run_index in range(args.runs):
                result = stream_completion(
                    args.base_url,
                    args.model,
                    prompt_tokens,
                    concurrency,
                    args.output_tokens,
                    args.timeout,
                    force_length=True,
                    seed=args.seed,
                    temperature=args.temperature,
                )
                result.pop("content")
                runs.append(result)
                print(
                    f"K{args.dflash_tokens} C{concurrency} run "
                    f"{run_index + 1}/{args.runs}: "
                    f"{result['decode_tokens_per_second']:.2f} tok/s, "
                    f"{result['accepted_draft_rate']:.1%} accepted",
                    flush=True,
                )
            points.append(
                {
                    "concurrency": concurrency,
                    **summarize_runs(runs),
                    "runs": runs,
                }
            )
        report.update(
            {
                "workload": CODE_AGENT_PROMPT,
                "output_tokens_per_sequence": args.output_tokens,
                "warmup_runs_per_point": args.warmup_runs,
                "runs_per_point": args.runs,
                "temperature": args.temperature,
                "points": points,
            }
        )
    else:
        contracts = load_contracts()
        cases = []
        all_runs = []
        for case_id, case in contracts.CASES.items():
            prompt_tokens = render_prompt(
                args.base_url, args.model, case.prompt, close_think, args.timeout
            )
            runs = []
            for repeat in range(args.runs):
                result = stream_completion(
                    args.base_url,
                    args.model,
                    prompt_tokens,
                    1,
                    case.max_tokens,
                    args.timeout,
                    force_length=False,
                    seed=args.seed,
                    temperature=0.0,
                )
                content = result.pop("content")[0]
                result.update(contracts.validate_case_content(case_id, content))
                result["content_preview"] = content[:240].replace("\n", "\\n")
                result["repeat"] = repeat + 1
                runs.append(result)
                all_runs.append(result)
                print(
                    f"K{args.dflash_tokens} {case_id} repeat {repeat + 1}/{args.runs}: "
                    f"{result['decode_tokens_per_second']:.2f} tok/s, "
                    f"{result['accepted_draft_rate']:.1%} accepted, "
                    f"quality={result['quality_contract_passed']}",
                    flush=True,
                )
            cases.append(
                {
                    "case": case_id,
                    "category": case.category,
                    **summarize_runs(runs),
                    "quality_passed": all(
                        run["quality_contract_passed"] for run in runs
                    ),
                    "runs": runs,
                }
            )
        total_decode_tokens = sum(run["decode_tokens"] for run in all_runs)
        total_decode_seconds = sum(run["decode_seconds"] for run in all_runs)
        total_drafts = sum(run["draft_tokens"] for run in all_runs)
        total_accepted = sum(run["accepted_draft_tokens"] for run in all_runs)
        report.update(
            {
                "glmrt_standard_seven_case_blend": True,
                "repeats_per_case": args.runs,
                "cases": cases,
                "aggregate": {
                    "decode_tokens_per_second": total_decode_tokens
                    / total_decode_seconds,
                    "draft_tokens": total_drafts,
                    "accepted_draft_tokens": total_accepted,
                    "accepted_draft_rate": total_accepted / total_drafts
                    if total_drafts
                    else 0.0,
                    "quality_passed": all(
                        run["quality_contract_passed"] for run in all_runs
                    ),
                },
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
