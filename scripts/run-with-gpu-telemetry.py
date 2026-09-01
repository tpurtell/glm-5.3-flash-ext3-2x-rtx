#!/usr/bin/env python3
"""Run a benchmark while retaining GPU, power, throttle, and PCIe telemetry."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import shlex
import subprocess
import time
from pathlib import Path


QUERY_FIELDS = [
    "index",
    "uuid",
    "pci.bus_id",
    "pstate",
    "utilization.gpu",
    "utilization.memory",
    "power.draw",
    "power.draw.instant",
    "power.limit",
    "clocks.current.sm",
    "clocks.current.memory",
    "temperature.gpu",
    "clocks_event_reasons.active",
    "clocks_event_reasons.sw_power_cap",
    "clocks_event_reasons.hw_slowdown",
    "clocks_event_reasons.hw_thermal_slowdown",
    "clocks_event_reasons.hw_power_brake_slowdown",
    "clocks_event_reasons.sw_thermal_slowdown",
]


def snapshot(started: float) -> dict:
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    rows = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(QUERY_FIELDS):
            raise RuntimeError(
                f"nvidia-smi returned {len(values)} columns, expected "
                f"{len(QUERY_FIELDS)}: {line!r}"
            )
        rows.append(dict(zip(QUERY_FIELDS, values, strict=True)))
    return {"elapsed_seconds": time.monotonic() - started, "gpus": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-seconds", type=float, default=1.0)
    parser.add_argument(
        "--continuous-dmon",
        action="store_true",
        help="Opt into intrusive continuous NVML/dmon sampling; do not use its TPS as a headline result.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after --")
    if args.sample_seconds <= 0:
        parser.error("--sample-seconds must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dmon_path = args.output.with_suffix(".dmon.log")
    started_wall = dt.datetime.now(dt.timezone.utc)
    started = time.monotonic()
    # NVML polling can hold this driver's global path long enough to perturb
    # short multi-GPU decode samples. Take detailed snapshots only at the two
    # boundaries. Continuous dmon is an explicit characterization mode whose
    # timing result must not be used as a performance headline.
    samples = [snapshot(started)]
    dmon = None
    if args.continuous_dmon:
        dmon = subprocess.Popen(
            [
                "nvidia-smi",
                "dmon",
                "--select",
                "pucvmet",
                "--delay",
                str(max(1, round(args.sample_seconds))),
                "--options",
                "DT",
                "--filename",
                str(dmon_path),
            ]
        )
    child = subprocess.Popen(command)
    try:
        returncode = child.wait()
    except BaseException:
        child.terminate()
        child.wait(timeout=30)
        raise
    finally:
        if dmon is not None:
            dmon.terminate()
            try:
                dmon.wait(timeout=10)
            except subprocess.TimeoutExpired:
                dmon.kill()
                dmon.wait(timeout=10)
    samples.append(snapshot(started))

    ended_wall = dt.datetime.now(dt.timezone.utc)
    report = {
        "schema": "glm53-gpu-telemetry.v2",
        "command": command,
        "command_shell": shlex.join(command),
        "returncode": returncode,
        "started_at": started_wall.isoformat(),
        "ended_at": ended_wall.isoformat(),
        "duration_seconds": time.monotonic() - started,
        "detailed_snapshots": "before and after only",
        "continuous_dmon": args.continuous_dmon,
        "dmon_sample_interval_seconds": (
            args.sample_seconds if args.continuous_dmon else None
        ),
        "hostname": platform.node(),
        "kernel": platform.release(),
        "query_fields": QUERY_FIELDS,
        "samples": samples,
        "dmon_log": dmon_path.name if args.continuous_dmon else None,
        "dmon_metrics": (
            "power, temperature, utilization, clocks, power/thermal violations, "
            "memory, ECC/PCIe replay errors, and PCIe RX/TX MB/s"
            if args.continuous_dmon
            else None
        ),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
