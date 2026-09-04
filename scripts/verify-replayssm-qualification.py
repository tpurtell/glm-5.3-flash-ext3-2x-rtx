#!/usr/bin/env python3
"""Fail closed on the matched ReplaySSM/full-state rolling-stress receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PHASES = {
    "thinking-off-shared-prefix",
    "thinking-max-shared-prefix",
    "thinking-max-unique-prefix",
}
FATAL_LOG_MARKERS = (
    "ReplaySSM prefill row count mismatch",
    "ReplaySSM prefill source/state row count mismatch",
    "EngineCore died",
)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def argument(command: list[str], flag: str) -> str:
    positions = [index for index, item in enumerate(command) if item == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ValueError(f"command does not contain exactly one {flag}")
    return command[positions[0] + 1]


def validate_environment(
    report: dict[str, Any],
    *,
    model: str,
    power_limit: float,
    use_replayssm: bool,
) -> dict[str, Any]:
    if report.get("model") != model or report.get("api_health_status") != 200:
        raise ValueError("environment has the wrong model or unhealthy API")
    gpus = report.get("gpus")
    if not isinstance(gpus, list) or len(gpus) != 2:
        raise ValueError("environment does not describe exactly two GPUs")
    if any(abs(float(gpu["power.limit"]) - power_limit) > 0.05 for gpu in gpus):
        raise ValueError("environment was not measured at the requested power cap")
    container = report.get("container")
    if not isinstance(container, dict):
        raise ValueError("environment has no container identity")
    labels = container.get("image_labels") or {}
    if labels.get("io.tpurtell.replayssm.mixed-graph-fix") != (
        "c51c3856f7f8ba50af3b3a60ff48e7d6a1fa303c"
    ):
        raise ValueError("image lacks the qualified ReplaySSM mixed-graph fix label")
    command = container.get("command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise ValueError("container command is invalid")
    required_flags = {
        "--enable-prefix-caching",
        "--enable-expert-parallel",
        "--no-enable-flashinfer-autotune",
    }
    if not required_flags.issubset(command):
        raise ValueError("container command omits a required release flag")
    expected_values = {
        "--served-model-name": model,
        "--tensor-parallel-size": "2",
        "--decode-context-parallel-size": "2",
        "--dcp-comm-backend": "ag_rs",
        "--attention-backend": "B12X_MLA_SPARSE",
        # Full-state native MTP carries one recurrent state per verify token;
        # use a matched 262K diagnostic profile so both variants can allocate.
        "--max-model-len": "262144",
        "--max-num-batched-tokens": "2048",
        "--max-num-seqs": "16",
        "--max-cudagraph-capture-size": "64",
        "--gpu-memory-utilization": "0.950",
        "--kv-cache-dtype": "fp8_ds_mla",
        "--mamba-cache-mode": "align",
    }
    for flag, expected in expected_values.items():
        actual = argument(command, flag)
        if actual != expected:
            raise ValueError(f"{flag} differs: {actual!r} != {expected!r}")
    speculative = json.loads(argument(command, "--speculative-config"))
    if speculative.get("method") != "mtp" or speculative.get(
        "num_speculative_tokens"
    ) != 5:
        raise ValueError("qualification did not use native MTP K5")
    schedule = speculative.get("num_speculative_tokens_per_batch_size")
    if not isinstance(schedule, list) or not schedule:
        raise ValueError("qualification did not use the adaptive-MTP schedule")
    if ("--use-replayssm" in command) != use_replayssm:
        raise ValueError("ReplaySSM command state differs from receipt variant")
    if use_replayssm and argument(command, "--replayssm-buffer-len") != "10":
        raise ValueError("ReplaySSM qualification did not use buffer length 10")
    environment = container.get("environment") or {}
    expected_environment = {
        "VLLM_ADAPTIVE_MTP": "1",
        "GLM53_REPLAYSSM_ACTIVE": "1" if use_replayssm else "0",
        "VLLM_B12X_DCP_A2A": "1",
        "VLLM_USE_B12X_SPARSE_INDEXER": "1",
    }
    for key, expected in expected_environment.items():
        if environment.get(key) != expected:
            raise ValueError(f"environment {key} differs")
    return {
        "image_id": container.get("image_id"),
        "model_artifact": report.get("model_artifact"),
        "command": command,
    }


def validate_variant(
    root: Path,
    *,
    model: str,
    power_limit: float,
    use_replayssm: bool,
) -> dict[str, Any]:
    environment = validate_environment(
        load(root / "environment.json"),
        model=model,
        power_limit=power_limit,
        use_replayssm=use_replayssm,
    )
    stress = load(root / "stress" / "summary.json")
    expected = {
        "schema": "glm53-kda-replayssm-rolling-stress-v1",
        "passed": True,
        "model": model,
        "target_tokens": 32768,
        "requests_per_phase": 40,
        "concurrency": 4,
        "max_tokens": 512,
        "total_requests": 120,
        "total_passes": 120,
        "total_errors": 0,
        "total_loops": 0,
    }
    for key, value in expected.items():
        if stress.get(key) != value:
            raise ValueError(f"{root.name} stress {key} differs")
    phases = stress.get("phases")
    if not isinstance(phases, list) or {item.get("phase") for item in phases} != PHASES:
        raise ValueError(f"{root.name} has an incomplete phase set")
    for phase in phases:
        if (
            phase.get("requests") != 40
            or phase.get("passes") != 40
            or phase.get("errors") != 0
            or phase.get("loops") != 0
            or phase.get("health_after", {}).get("ok") is not True
        ):
            raise ValueError(f"{root.name} phase failed: {phase!r}")
    jit = load(root / "startup-jit-audit.json")
    if jit.get("passed") is not True or jit.get("post_ready_jit_count") != 0:
        raise ValueError(f"{root.name} compiled a kernel after readiness")
    log = (root / "server.log").read_text(encoding="utf-8", errors="replace")
    found = [marker for marker in FATAL_LOG_MARKERS if marker in log]
    if found:
        raise ValueError(f"{root.name} server log contains fatal markers: {found}")
    return {
        "environment": environment,
        "stress": stress,
        "jit": jit,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--power-limit", type=float, default=400)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    materializer = load(root / "kda-materializer.json")
    expected_materializer = {
        "schema": "glm53-kda-replayssm-materializer-v1",
        "passed": True,
        "metadata_stride": 8,
        "contiguous_and_strided_outputs_identical": True,
        "null_row_is_zero": True,
    }
    for key, value in expected_materializer.items():
        if materializer.get(key) != value:
            raise ValueError(f"KDA materializer {key} differs")
    if float(materializer.get("maximum_absolute_error", 1.0)) > 0.02:
        raise ValueError("KDA materializer numerical error exceeds tolerance")
    replay = validate_variant(
        root / "replayssm",
        model=args.model,
        power_limit=args.power_limit,
        use_replayssm=True,
    )
    control = validate_variant(
        root / "full-state-control",
        model=args.model,
        power_limit=args.power_limit,
        use_replayssm=False,
    )
    if replay["environment"]["image_id"] != control["environment"]["image_id"]:
        raise ValueError("ReplaySSM and full-state control used different images")
    candidate_image_id = (root / "candidate-image-id.txt").read_text(
        encoding="utf-8"
    ).strip()
    if candidate_image_id != replay["environment"]["image_id"]:
        raise ValueError("materializer and server tests used different images")
    if replay["environment"]["model_artifact"] != control["environment"][
        "model_artifact"
    ]:
        raise ValueError("ReplaySSM and full-state control used different model artifacts")
    receipt_paths = (
        Path("candidate-image-id.txt"),
        Path("kda-materializer.json"),
        Path("replayssm/environment.json"),
        Path("replayssm/stress/summary.json"),
        Path("replayssm/startup-jit-audit.json"),
        Path("replayssm/server.log"),
        Path("full-state-control/environment.json"),
        Path("full-state-control/stress/summary.json"),
        Path("full-state-control/startup-jit-audit.json"),
        Path("full-state-control/server.log"),
    )
    report = {
        "schema": "glm53-kda-replayssm-qualification-v1",
        "passed": True,
        "model": args.model,
        "image_id": replay["environment"]["image_id"],
        "model_artifact": replay["environment"]["model_artifact"],
        "power_limit_watts_per_gpu": args.power_limit,
        "kda_materializer": materializer,
        "replayssm_requests": replay["stress"]["total_requests"],
        "full_state_control_requests": control["stress"]["total_requests"],
        "receipts": {
            str(path): {
                "bytes": (root / path).stat().st_size,
                "sha256": sha256_file(root / path),
            }
            for path in receipt_paths
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
