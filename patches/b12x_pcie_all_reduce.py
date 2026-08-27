# SPDX-License-Identifier: Apache-2.0
"""vLLM communicator adapter for B12x PCIe one-shot all-reduce.

This adapter deliberately stays small.  B12x owns IPC allocation, CUDA graph
registration, and the SM120 transport kernels; vLLM owns backend dispatch and
falls through to PyNCCL for tensors above the configured one-shot cutoff.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from b12x.comm.pcie import OneshotAllReducePool
from vllm.logger import init_logger


logger = init_logger(__name__)


def _parse_byte_size(value: str) -> int:
    normalized = value.strip().upper()
    suffixes = {
        "KIB": 1024,
        "KB": 1024,
        "K": 1024,
        "MIB": 1024 * 1024,
        "MB": 1024 * 1024,
        "M": 1024 * 1024,
    }
    for suffix in sorted(suffixes, key=len, reverse=True):
        if normalized.endswith(suffix):
            return int(normalized[: -len(suffix)]) * suffixes[suffix]
    return int(normalized)


class B12xPcieAllReduce:
    """The subset of ``CustomAllreduce`` consumed by ``CudaCommunicator``."""

    def __init__(
        self,
        group: ProcessGroup,
        device_group: ProcessGroup,
        device: int | str | torch.device,
    ) -> None:
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("B12x PCIe all-reduce requires a CUDA device")

        self.group = group
        self.device_group = device_group
        self.device = device
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.disabled = True
        self._capturing = False
        self._closed = False
        self.max_size = _parse_byte_size(
            os.getenv("VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE", "384KB")
        )
        self.enable_eager = os.getenv("VLLM_B12X_PCIE_EAGER", "0") not in (
            "",
            "0",
            "false",
            "False",
        )
        if self.max_size < 16:
            raise ValueError(
                "VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE must be at least 16 bytes"
            )

        # A single channel is sufficient for vLLM's serial TP graph replay and
        # avoids inventing process-local stream IDs as distributed identities.
        # The B12x channel itself is explicitly non-stream-affine in this mode.
        self.runtime = OneshotAllReducePool.from_exchange_group(
            exchange_group=device_group,
            device=device,
            eager_buffer_bytes=self.max_size,
            max_size=self.max_size,
            single_channel=True,
            max_concurrent_channels=1,
        )
        self.runtime.for_stream()
        self.disabled = False
        logger.info_once(
            "Enabled B12x PCIe one-shot all-reduce "
            "(world_size=%d, max_size=%d bytes, graph_channel=single, eager=%s).",
            self.world_size,
            self.max_size,
            self.enable_eager,
            scope="global",
        )

    def backend_name(self) -> str:
        return "B12X_PCIE_ONESHOT"

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        if self.disabled or self._closed:
            return False
        # On this TP2 PCIe host the Python/eager B12x path is launch-bound,
        # while captured replay beats PyNCCL.  Preserve NCCL for uncaptured
        # calls unless explicitly requested for diagnostics.
        if not (
            self.enable_eager
            or self._capturing
            or torch.cuda.is_current_stream_capturing()
        ):
            return False
        return self.runtime.for_stream().should_allreduce(inp)

    def custom_all_reduce(self, inp: torch.Tensor) -> torch.Tensor | None:
        if not self.should_custom_ar(inp):
            return None
        return self.runtime.all_reduce(inp)

    @contextmanager
    def capture(self, stream: torch.cuda.Stream | None = None):
        if self.disabled or self._closed:
            yield
            return
        old_capturing = self._capturing
        self._capturing = True
        try:
            with self.runtime.capture(stream=stream):
                yield
        finally:
            self._capturing = old_capturing

    def custom_all_gather(self, inp: torch.Tensor) -> None:
        del inp
        return None

    def custom_reduce_scatter(self, inp: torch.Tensor) -> None:
        del inp
        return None

    def close(self) -> None:
        if self._closed:
            return
        self.runtime.close()
        self._closed = True
        self.disabled = True

    def __del__(self) -> None:
        # B12x intentionally quarantines live IPC runtimes at GC.  Explicit,
        # coordinated close happens while vLLM's CPU process group still lives.
        return
