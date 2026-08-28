#!/usr/bin/env python3
"""CPU-only guards for the adaptive MTP controller copied into the image."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


path = Path(__file__).parents[1] / "patches" / "adaptive_mtp.py"
spec = importlib.util.spec_from_file_location("adaptive_mtp", path)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
Controller = module.AdaptiveMTPController


def controller() -> object:
    return Controller(
        max_depth=5,
        min_depth=0,
        probe_interval=4,
        probe_interval_max=16,
    )


def main() -> None:
    # The production default never drops below K1; K0 remains an explicit
    # opt-in for hardware where a resident MTP layer makes scalar decode win.
    default_state = Controller(max_depth=5)
    assert default_state.select(["floor"], 8, 1) == 1
    for _ in range(8):
        default_state.observe("floor", 8, 1, 0)
        default_state.select(["floor"], 8, 1)
    assert default_state.select(["floor"], 8, 1) == 1

    # C1 free-form acceptance remains profitable at K5.
    state = controller()
    assert state.select(["a"], 1, 5) == 5
    for _ in range(8):
        state.observe("a", 1, 5, 2)
        state.select(["a"], 1, 5)
    assert state.select(["a"], 1, 5) == 5

    # A full evidence window contracts K within the same request.
    state = controller()
    assert state.select(["a"], 8, 5) == 5
    for _ in range(8):
        state.observe("a", 8, 5, 1)
        state.select(["a"], 8, 5)
    assert state.select(["a"], 8, 5) == 4

    # Sustained full windows grow the request estimate.
    state = controller()
    assert state.select(["a"], 8, 1) == 1
    for _ in range(8):
        state.observe("a", 8, 1, 1)
        state.select(["a"], 8, 1)
    assert state.select(["a"], 8, 1) == 2

    # K0 waits, then runs a complete K1 probe window. A failed probe backs
    # off instead of bouncing to K1 after one lucky token.
    state = controller()
    assert state.select(["a"], 8, 1) == 1
    for _ in range(8):
        state.observe("a", 8, 1, 0)
        state.select(["a"], 8, 1)
    assert [state.select(["a"], 8, 1) for _ in range(4)] == [0, 0, 1, 1]
    for _ in range(8):
        state.observe("a", 8, 1, 0)
        state.select(["a"], 8, 1)
    assert state.select(["a"], 8, 1) == 0
    assert state.snapshot()["a"]["probe_interval"] == 8

    # Request histories are isolated.
    state = controller()
    assert state.select(["a"], 1, 5) == 5
    assert state.select(["b"], 8, 1) == 1
    for _ in range(8):
        state.observe("b", 8, 1, 0)
        state.select(["b"], 8, 1)
    assert state.select(["a"], 1, 5) == 5

    state = controller()
    assert state.select(["a"], 1, 5) == 5
    for _ in range(32):
        state.observe("a", 1, 4, 0)
    assert state.snapshot()["a"]["history"] == 0

    # One shared execution K is the rounded arithmetic mean of request-local
    # predictions: (5 + 3 + 2) / 3 -> K3.
    state = controller()
    assert state.select(["a"], 1, 5) == 5
    assert state.select(["b"], 1, 3) == 3
    assert state.select(["c"], 1, 2) == 2
    assert state.select(["a", "b", "c"], 3, 5) == 3
    state.finish({"b"})
    assert "b" not in state.snapshot()

    # Once feedback contracts K5 to K4, already-queued K5 results belong to
    # the old epoch and cannot cascade K4 further.
    state = controller()
    for _ in range(12):
        assert state.select(["a"], 1, 5) == 5
    for _ in range(8):
        state.observe("a", 8, 5, 1)
    assert state.snapshot()["a"]["current"] == 4
    for _ in range(4):
        state.observe("a", 8, 5, 0)
    assert state.snapshot()["a"]["current"] == 4

    print("adaptive MTP policy guards passed")


if __name__ == "__main__":
    main()
