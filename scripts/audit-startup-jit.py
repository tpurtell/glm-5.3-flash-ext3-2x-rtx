#!/usr/bin/env python3
"""Record whether a container compiled kernels after its release-ready marker."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


READY_TEXT = "GLM release startup warmup complete; container is ready."
JIT_TEXT = "JIT compilation during inference:"
KERNEL_PATTERN = re.compile(r"JIT compilation during inference: ([^.]+)")


def command(*args: str) -> str:
    return subprocess.run(
        args,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    inspect = json.loads(command("docker", "inspect", args.container))[0]
    lines = command("docker", "logs", "--timestamps", args.container).splitlines()
    ready_indices = [index for index, line in enumerate(lines) if READY_TEXT in line]
    if len(ready_indices) != 1:
        raise SystemExit(
            f"expected exactly one ready marker, observed {len(ready_indices)}"
        )
    ready_index = ready_indices[0]
    before = lines[:ready_index]
    after = lines[ready_index + 1 :]

    def jit_kernels(group: list[str]) -> list[str]:
        kernels = []
        for line in group:
            match = KERNEL_PATTERN.search(line)
            if match:
                kernels.append(match.group(1))
        return kernels

    post_ready_jit = [line for line in after if JIT_TEXT in line]
    report = {
        "schema": "glm53-release-startup-jit-audit.v1",
        "container": args.container,
        "image": inspect["Config"]["Image"],
        "image_id": inspect["Image"],
        "container_state": inspect["State"]["Status"],
        "health_status_at_audit": (
            (inspect["State"].get("Health") or {}).get("Status")
            if inspect["State"]["Status"] == "running"
            else None
        ),
        "container_stopped_after_test": inspect["State"]["Status"] == "exited",
        "ready_marker": lines[ready_index],
        "pre_ready_jit_count": sum(JIT_TEXT in line for line in before),
        "pre_ready_jit_kernels": sorted(set(jit_kernels(before))),
        "post_ready_request_count": sum('\"POST ' in line for line in after),
        "post_ready_health_200_count": sum(
            '"GET /health HTTP/1.1" 200 OK' in line for line in after
        ),
        "post_ready_http_404_count": sum(" 404 Not Found" in line for line in after),
        "post_ready_jit_count": len(post_ready_jit),
        "post_ready_jit_kernels": sorted(set(jit_kernels(after))),
        "passed": not post_ready_jit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
