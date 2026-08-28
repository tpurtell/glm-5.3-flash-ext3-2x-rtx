"""Feedback-adaptive, request-local MTP depth estimator.

Each request keeps its own acceptance history and can change its preferred K
throughout a decode. vLLM executes one K for a whole scheduler step, so the
controller combines the live requests' estimates with an arithmetic mean.
At C1 the request estimate is therefore the executed K exactly.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def adaptive_mtp_enabled() -> bool:
    raw = os.getenv("VLLM_ADAPTIVE_MTP", "0").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"VLLM_ADAPTIVE_MTP must be 0 or 1; got {raw!r}")


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    value = default if raw is None else int(raw)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {value}")
    return value


def _env_thresholds() -> tuple[float, ...]:
    raw = os.getenv("VLLM_ADAPTIVE_MTP_LOAD_THRESHOLDS")
    if raw is None:
        # Required mean accepted/proposed ratio by runtime batch bucket
        # 1, 2, 4, 8, 16+.  Draft and verify work becomes progressively less
        # attractive as the target batch saturates the two GPUs.
        # K1 is the measured winner for the recurrent-cache-limited C8
        # execution bucket on 2x RTX PRO 6000, so do not require near-perfect
        # acceptance there. Expansion uses a separate, higher threshold.
        return (0.28, 0.45, 0.55, 0.75, 0.90)
    values = tuple(float(item.strip()) for item in raw.split(","))
    if len(values) != 5 or any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError(
            "VLLM_ADAPTIVE_MTP_LOAD_THRESHOLDS must contain five comma-"
            "separated ratios in [0,1] for batch buckets 1,2,4,8,16+"
        )
    return values


@dataclass(frozen=True)
class AdaptiveMTPObservation:
    depth: int
    proposed: int
    accepted: int

    @property
    def acceptance_ratio(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def full_window(self) -> bool:
        return self.proposed > 0 and self.accepted == self.proposed


@dataclass
class _RequestState:
    current: int | None = None
    history: deque[AdaptiveMTPObservation] = field(default_factory=deque)
    pending_depths: deque[tuple[int, bool, int]] = field(default_factory=deque)
    scalar_decisions: int = 0
    probe_interval: int = 0
    probe_active: bool = False
    epoch: int = 0


class AdaptiveMTPController:
    """Estimate K per request, then select one shared K for the batch."""

    def __init__(
        self,
        max_depth: int,
        *,
        min_depth: int = 1,
        history_limit: int = 16,
        decision_window: int = 8,
        probe_interval: int = 32,
        probe_interval_max: int = 256,
        load_thresholds: tuple[float, ...] | None = None,
    ) -> None:
        if max_depth < 1:
            raise ValueError("adaptive MTP requires max_depth >= 1")
        if not 0 <= min_depth <= max_depth:
            raise ValueError("min_depth must be between 0 and max_depth")
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.decision_window = min(history_limit, decision_window)
        self.history_limit = max(history_limit, self.decision_window)
        self.probe_interval = probe_interval
        self.probe_interval_max = max(probe_interval, probe_interval_max)
        self.load_thresholds = load_thresholds or _env_thresholds()
        self._states: dict[str, _RequestState] = {}

    @classmethod
    def from_env(cls, max_depth: int) -> AdaptiveMTPController:
        return cls(
            max_depth=max_depth,
            min_depth=_env_int("VLLM_ADAPTIVE_MTP_MIN_DEPTH", 1, minimum=0),
            history_limit=_env_int("VLLM_ADAPTIVE_MTP_HISTORY", 16),
            decision_window=_env_int("VLLM_ADAPTIVE_MTP_DECISION_WINDOW", 8),
            probe_interval=_env_int("VLLM_ADAPTIVE_MTP_PROBE_INTERVAL", 32),
            probe_interval_max=_env_int(
                "VLLM_ADAPTIVE_MTP_PROBE_INTERVAL_MAX", 256
            ),
        )

    @staticmethod
    def batch_bucket(batch_size: int) -> int:
        if batch_size <= 1:
            return 1
        if batch_size <= 2:
            return 2
        if batch_size <= 4:
            return 4
        if batch_size <= 8:
            return 8
        return 16

    def _state(self, request_id: str) -> _RequestState:
        state = self._states.setdefault(request_id, _RequestState())
        if state.probe_interval == 0:
            state.probe_interval = self.probe_interval
        return state

    def _estimate(self, request_id: str, prior_depth: int) -> int:
        """Return one request's preferred K for its next draft."""
        state = self._state(request_id)
        if state.current is None:
            state.current = max(
                self.min_depth, min(self.max_depth, prior_depth)
            )

        estimate = state.current
        if estimate == 0 and not state.probe_active:
            state.scalar_decisions += 1
            if state.scalar_decisions >= state.probe_interval:
                state.probe_active = True
        if estimate == 0 and state.probe_active:
            # Keep voting for K1 until a complete evidence window has been
            # measured. With independently aged requests, the arithmetic
            # mean can initially remain K0; persistent votes eventually form
            # a real fused-batch probe without overriding the mean policy.
            estimate = 1
        return estimate

    def select(
        self,
        request_ids: list[str],
        batch_size: int,
        prior_depth: int,
    ) -> int:
        """Return the half-up rounded mean of live request estimates."""
        if not request_ids:
            return 0
        estimates = [self._estimate(req_id, prior_depth) for req_id in request_ids]
        selected = min(
            self.max_depth,
            (sum(estimates) + len(estimates) // 2) // len(estimates),
        )
        for request_id, estimate in zip(request_ids, estimates):
            state = self._state(request_id)
            is_probe = state.current == 0 and selected > 0
            state.pending_depths.append((selected, is_probe, state.epoch))
            if is_probe:
                state.scalar_decisions = 0
        return selected

    def _transition(
        self,
        request_id: str,
        state: _RequestState,
        bucket: int,
        new_depth: int,
        observation: AdaptiveMTPObservation,
    ) -> None:
        old_depth = state.current if state.current is not None else observation.depth
        if new_depth == old_depth:
            return
        logger.info(
            "Adaptive MTP request=%s batch_bucket=%d K%d->K%d "
            "accepted=%d/%d evidence=%d",
            request_id,
            bucket,
            old_depth,
            new_depth,
            observation.accepted,
            observation.proposed,
            len(state.history),
        )
        state.current = new_depth
        state.epoch += 1
        state.history.clear()
        state.scalar_decisions = 0
        state.probe_active = False

    def observe(
        self,
        request_id: str,
        batch_size: int,
        proposed: int,
        accepted: int,
    ) -> None:
        """Observe one request's completed target-verification step."""
        bucket = self.batch_bucket(batch_size)
        state = self._state(request_id)
        # Proposal happens after target verification in a scheduler step, so
        # feedback for it arrives one step later. K0 produces no observation;
        # discard those queued scalar decisions until the executed K matches.
        while state.pending_depths and state.pending_depths[0][0] == 0:
            state.pending_depths.popleft()
        if not state.pending_depths:
            return
        depth, was_probe, epoch = state.pending_depths.popleft()
        if depth == 0 or proposed != depth:
            return
        # Scheduler/model execution is pipelined. Evidence selected before a
        # K transition belongs to the old policy epoch and must not trigger a
        # second transition after the new K is already live.
        if epoch != state.epoch:
            return
        accepted = min(proposed, accepted)
        observation = AdaptiveMTPObservation(
            depth=depth,
            proposed=proposed,
            accepted=accepted,
        )
        state.history.append(observation)
        while len(state.history) > self.history_limit:
            state.history.popleft()

        old_depth = state.current if state.current is not None else depth
        threshold = self.load_thresholds[(1, 2, 4, 8, 16).index(bucket)]
        if was_probe:
            recent = list(state.history)[-self.decision_window :]
            if len(recent) < self.decision_window:
                return
            ratio = sum(item.accepted for item in recent) / sum(
                item.proposed for item in recent
            )
            if ratio >= threshold:
                # Promote only after a complete probe window. This avoids the
                # K0<->K1 sawtooth caused by treating one lucky token as a
                # durable workload phase change.
                mean_accepted = (
                    sum(item.accepted for item in recent) + len(recent) // 2
                ) // len(recent)
                new_depth = max(1, min(self.max_depth, mean_accepted))
                state.probe_interval = self.probe_interval
            else:
                new_depth = 0
                state.probe_interval = min(
                    self.probe_interval_max, state.probe_interval * 2
                )
                state.epoch += 1
                state.history.clear()
                state.scalar_decisions = 0
                state.probe_active = False
                return
        else:
            recent = list(state.history)[-self.decision_window :]
            if len(recent) < self.decision_window:
                return
            ratio = sum(item.accepted for item in recent) / sum(
                item.proposed for item in recent
            )
            if ratio < threshold:
                # If the fused batch tested below this request's estimate, a
                # miss there disproves every deeper K too. Otherwise move one
                # rung at a time and gather a fresh evidence epoch.
                new_depth = min(old_depth - 1, depth - 1)
                new_depth = max(self.min_depth, new_depth)
            elif (
                ratio >= 0.95
                and depth >= old_depth
                and old_depth < self.max_depth
            ):
                new_depth = min(self.max_depth, max(old_depth + 1, depth))
            else:
                return

        self._transition(request_id, state, bucket, new_depth, observation)

    def finish(self, request_ids: set[str]) -> None:
        for request_id in request_ids:
            self._states.pop(request_id, None)

    def snapshot(self) -> dict[str, dict[str, int]]:
        return {
            request_id: {
                "current": -1 if state.current is None else state.current,
                "history": len(state.history),
                "probe_interval": state.probe_interval,
                "probe_active": int(state.probe_active),
                "epoch": state.epoch,
            }
            for request_id, state in sorted(self._states.items())
        }
