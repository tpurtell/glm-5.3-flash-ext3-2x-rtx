#!/usr/bin/env python3
"""Exercise real GLM+DFlash serving paths before Docker reports healthy."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
import urllib.error
import urllib.request


PROMPT = """You are editing an async Python worker pool. Return only a complete
replacement Python module that preserves input ordering, cancels sibling tasks
after any exception, awaits every cancellation, and includes precise type hints.
"""

# One token per repetition with the pinned GLM tokenizer.  Sixteen thousand
# tokens crosses the production sparse-indexer/owner-merge thresholds and
# resolves their large-prefill kernels before readiness.  It remains cheap
# enough for an ordinary container start and is not a benchmark fixture.
LONG_PREFILL_PROMPT = " warm" * 16384


def request_json(
    base_url: str,
    path: str,
    payload: dict | None,
    timeout: float,
) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def request_ok(base_url: str, path: str, timeout: float) -> None:
    request = urllib.request.Request(base_url.rstrip("/") + path)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response.read()


def server_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-pid", type=int, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument(
        "--ready-timeout",
        type=float,
        default=float(os.getenv("VLLM_ENGINE_READY_TIMEOUT_S", "3600")),
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=float(os.getenv("GLM53_STARTUP_WARMUP_TIMEOUT_S", "1800")),
    )
    parser.add_argument("--passes", type=int, default=4)
    args = parser.parse_args()

    deadline = time.monotonic() + args.ready_timeout
    while True:
        if not server_alive(args.server_pid):
            raise SystemExit("vLLM exited before startup warmup")
        try:
            request_ok(args.base_url, "/health", 3)
            break
        except (OSError, urllib.error.URLError):
            if time.monotonic() >= deadline:
                raise SystemExit("timed out waiting for the internal vLLM health endpoint")
            time.sleep(2)

    models = request_json(args.base_url, "/v1/models", None, 30).get("data") or []
    if not models or not isinstance(models[0].get("id"), str):
        raise SystemExit("vLLM returned no served model for startup warmup")
    model = models[0]["id"]

    started = time.perf_counter()
    greedy = request_json(
        args.base_url,
        "/v1/completions",
        {
            "model": model,
            "prompt": PROMPT,
            "n": 1,
            "max_tokens": 32,
            "min_tokens": 32,
            "ignore_eos": True,
            "temperature": 0,
            "seed": 20260901,
            "cache_prompt": False,
            "cache_salt": "glm53-release-warmup-greedy-c1",
        },
        args.request_timeout,
    )
    if len(greedy.get("choices") or []) != 1:
        raise SystemExit("startup greedy C1 warmup did not return one choice")
    print(
        "GLM release startup greedy C1 warmup completed in "
        f"{time.perf_counter() - started:.2f}s",
        flush=True,
    )

    # Exercise the normal rendered-chat token-id path as well.  Its greedy
    # DFlash verification specialization differs from the fixed-length raw
    # completion above and otherwise JITs on the first chat/tool request.
    close_think = request_json(
        args.base_url,
        "/tokenize",
        {"model": model, "prompt": "</think>", "add_special_tokens": False},
        30,
    ).get("tokens")
    rendered = request_json(
        args.base_url,
        "/v1/chat/completions/render",
        {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "reasoning_effort": "low",
        },
        30,
    ).get("token_ids")
    if not isinstance(close_think, list) or not isinstance(rendered, list):
        raise SystemExit("startup rendered-chat warmup did not return token ids")
    started = time.perf_counter()
    rendered_greedy = request_json(
        args.base_url,
        "/v1/completions",
        {
            "model": model,
            "prompt": rendered + close_think,
            "n": 1,
            "max_tokens": 64,
            "temperature": 0,
            "seed": 20260901,
            "add_special_tokens": False,
        },
        args.request_timeout,
    )
    if len(rendered_greedy.get("choices") or []) != 1:
        raise SystemExit("startup rendered-chat warmup did not return one choice")
    print(
        "GLM release startup rendered-chat warmup completed in "
        f"{time.perf_counter() - started:.2f}s",
        flush=True,
    )

    # GLM changes its sparse K-pool tier at ordinary code-context depth.  The
    # 8K tier uses a 512-wide top-k and the PCIe DCP owner exchange, which is a
    # distinct specialization from both short decode and the 16K bulk-prefill
    # path below.  Match benchmark-code-agent-depth.py's exact token-id shape.
    depth_filler = request_json(
        args.base_url,
        "/tokenize",
        {
            "model": model,
            "prompt": (
                "Slate rivers cross quiet valleys while copper clocks mark "
                "patient hours. This is ordinary context with no instructions.\n"
            ),
            "add_special_tokens": False,
        },
        30,
    ).get("tokens")
    depth_task = rendered + close_think
    if not isinstance(depth_filler, list) or not depth_filler:
        raise SystemExit("startup 8K depth warmup could not tokenize filler")
    depth_needed = 8192 - len(depth_task)
    if depth_needed <= 0:
        raise SystemExit("startup rendered task unexpectedly exceeds 8K tokens")
    depth_prompt = (
        depth_filler
        * ((depth_needed + len(depth_filler) - 1) // len(depth_filler))
    )[:depth_needed] + depth_task
    started = time.perf_counter()
    depth_result = request_json(
        args.base_url,
        "/v1/completions",
        {
            "model": model,
            "prompt": depth_prompt,
            "n": 1,
            "max_tokens": 1,
            "min_tokens": 1,
            "ignore_eos": True,
            "temperature": 0,
            "seed": 20260901,
            "add_special_tokens": False,
            "cache_prompt": False,
            "cache_salt": "glm53-release-warmup-code-depth-8k",
        },
        args.request_timeout,
    )
    if len(depth_result.get("choices") or []) != 1:
        raise SystemExit("startup 8K code-depth warmup did not return one choice")
    print(
        "GLM release startup 8K code-depth warmup completed in "
        f"{time.perf_counter() - started:.2f}s",
        flush=True,
    )

    started = time.perf_counter()
    long_prefill = request_json(
        args.base_url,
        "/v1/completions",
        {
            "model": model,
            "prompt": LONG_PREFILL_PROMPT,
            "n": 1,
            "max_tokens": 1,
            "min_tokens": 1,
            "ignore_eos": True,
            "temperature": 0,
            "seed": 20260901,
            "cache_prompt": False,
            "cache_salt": "glm53-release-warmup-long-prefill",
        },
        args.request_timeout,
    )
    if len(long_prefill.get("choices") or []) != 1:
        raise SystemExit("startup long-prefill warmup did not return one choice")
    print(
        "GLM release startup long-prefill warmup completed in "
        f"{time.perf_counter() - started:.2f}s",
        flush=True,
    )

    # Compact ReplaySSM's aligned-prefix materializer is not used by the
    # production DFlash2 profile.  When an MTP+ReplaySSM profile is selected,
    # replay the same cached prefix before readiness so its cursor reset and
    # state reconstruction kernels cannot first-JIT on a user's request.
    if os.getenv("GLM53_REPLAYSSM_ACTIVE", "0") == "1":
        started = time.perf_counter()
        replay = request_json(
            args.base_url,
            "/v1/completions",
            {
                "model": model,
                "prompt": LONG_PREFILL_PROMPT,
                "n": 1,
                "max_tokens": 1,
                "min_tokens": 1,
                "ignore_eos": True,
                "temperature": 0,
                "seed": 20260901,
                "cache_salt": "glm53-release-warmup-replayssm-prefix",
            },
            args.request_timeout,
        )
        replay_again = request_json(
            args.base_url,
            "/v1/completions",
            {
                "model": model,
                "prompt": LONG_PREFILL_PROMPT,
                "n": 1,
                "max_tokens": 1,
                "min_tokens": 1,
                "ignore_eos": True,
                "temperature": 0,
                "seed": 20260901,
                "cache_salt": "glm53-release-warmup-replayssm-prefix",
            },
            args.request_timeout,
        )
        if len(replay.get("choices") or []) != 1 or len(
            replay_again.get("choices") or []
        ) != 1:
            raise SystemExit("startup ReplaySSM prefix warmup returned bad choices")
        print(
            "GLM release startup ReplaySSM prefix warmup completed in "
            f"{time.perf_counter() - started:.2f}s",
            flush=True,
        )

        # Stagger prompt lengths so a short request is decoding while later
        # requests are still prefilling. ReplaySSM deliberately keeps this
        # mixed lifecycle eager; exercising it here both checks the dispatch
        # and resolves its real-shape kernels before the ready marker.
        def mixed_request(item: tuple[int, int]) -> dict:
            request_index, repetitions = item
            time.sleep(request_index * 0.15)
            return request_json(
                args.base_url,
                "/v1/completions",
                {
                    "model": model,
                    "prompt": " mixed" * repetitions + PROMPT,
                    "n": 1,
                    "max_tokens": 128,
                    "min_tokens": 128,
                    "ignore_eos": True,
                    "temperature": 0,
                    "seed": 20260901,
                    "cache_salt": (
                        f"glm53-release-warmup-replayssm-mixed-{request_index}"
                    ),
                },
                args.request_timeout,
            )

        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            mixed_results = list(
                pool.map(mixed_request, enumerate((64, 1024, 2048, 4096)))
            )
        if any(len(result.get("choices") or []) != 1 for result in mixed_results):
            raise SystemExit("startup ReplaySSM mixed warmup returned bad choices")
        print(
            "GLM release startup ReplaySSM mixed prefill/decode warmup completed in "
            f"{time.perf_counter() - started:.2f}s",
            flush=True,
        )

    for pass_index in range(args.passes):
        started = time.perf_counter()
        result = request_json(
            args.base_url,
            "/v1/completions",
            {
                "model": model,
                "prompt": PROMPT,
                "n": 16,
                "max_tokens": 256,
                "min_tokens": 256,
                "ignore_eos": True,
                "temperature": 0.2,
                "seed": 20260901,
                "cache_prompt": False,
                "cache_salt": f"glm53-release-warmup-{pass_index}",
            },
            args.request_timeout,
        )
        choices = result.get("choices") or []
        if len(choices) != 16:
            raise SystemExit(
                f"startup warmup pass {pass_index + 1} returned "
                f"{len(choices)} choices, expected 16"
            )
        print(
            f"GLM release startup warmup {pass_index + 1}/{args.passes} "
            f"completed in {time.perf_counter() - started:.2f}s",
            flush=True,
        )
        time.sleep(1)


if __name__ == "__main__":
    main()
