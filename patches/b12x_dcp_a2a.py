# SPDX-License-Identifier: Apache-2.0
"""Captured B12x PCIe collectives for vLLM's MLA DCP path.

The eager B12x launch is intentionally not selected: on TP2 PCIe, NCCL wins
there.  During vLLM CUDA-graph warmup/capture, this module switches the paired
MLA query-gather and LSE reduce-scatter to the graph-specialized B12x channel.
Long-prefill rows and uncaptured calls retain vLLM's normal AG/RS path.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import torch

from b12x.comm.pcie.pcie_dcp_a2a import PCIeDCPA2APool
from vllm.logger import init_logger


logger = init_logger(__name__)

_runtime: "B12xMLADCPA2A | None" = None
_capture_depth = 0


def _enabled() -> bool:
    return os.getenv("VLLM_B12X_DCP_A2A", "1") not in (
        "",
        "0",
        "false",
        "False",
    )


class B12xMLADCPA2A:
    def __init__(
        self,
        *,
        process_group,
        device: torch.device,
        max_batch_size: int,
        total_heads: int,
        output_head_dim: int,
        query_head_dim: int,
    ) -> None:
        self.device = torch.device(device)
        self.max_batch_size = int(max_batch_size)
        self.total_heads = int(total_heads)
        self.output_head_dim = int(output_head_dim)
        self.query_head_dim = int(query_head_dim)
        self.pool = PCIeDCPA2APool.from_process_group(
            process_group=process_group,
            device=self.device,
            max_batch_size=self.max_batch_size,
            total_heads=self.total_heads,
            head_dim=self.output_head_dim,
            query_head_dim=self.query_head_dim,
            single_channel=True,
            max_concurrent_channels=1,
        )
        # CuTe compilation is not legal once torch's graph capture has begun.
        self.pool.prepare_graph_all_gather_heads()
        self.pool.prepare_graph_lse_reduce_scatter(dtype=torch.bfloat16)

    def _captured_shape(self, value: torch.Tensor) -> bool:
        return (
            0 < int(value.shape[0]) <= self.max_batch_size
            and (_capture_depth > 0 or torch.cuda.is_current_stream_capturing())
        )

    def can_gather(self, query: torch.Tensor) -> bool:
        return (
            self._captured_shape(query)
            and query.ndim == 3
            and int(query.shape[1]) * self.pool.world_size == self.total_heads
            and int(query.shape[2]) == self.query_head_dim
            and query.dtype
            in (torch.bfloat16, torch.float16, torch.float8_e4m3fn)
            and query.is_contiguous()
        )

    def can_combine(
        self, partial_output: torch.Tensor, partial_lse: torch.Tensor
    ) -> bool:
        return (
            self._captured_shape(partial_output)
            and tuple(partial_output.shape[1:])
            == (self.total_heads, self.output_head_dim)
            and partial_output.dtype in (torch.bfloat16, torch.float16)
            and tuple(partial_lse.shape)
            == (int(partial_output.shape[0]), self.total_heads)
            and partial_lse.dtype == torch.float32
        )

    def gather(self, query: torch.Tensor) -> torch.Tensor:
        return self.pool.all_gather_heads(query)

    def combine(
        self,
        partial_output: torch.Tensor,
        partial_lse: torch.Tensor,
        *,
        is_lse_base_on_e: bool,
    ) -> torch.Tensor:
        return self.pool.lse_reduce_scatter(
            partial_output,
            partial_lse,
            is_lse_base_on_e=is_lse_base_on_e,
        )

    def close(self) -> None:
        self.pool.close()


def get_b12x_mla_dcp_a2a(
    *,
    process_group,
    device: torch.device,
    max_batch_size: int,
    total_heads: int,
    output_head_dim: int,
    query_head_dim: int,
) -> B12xMLADCPA2A | None:
    global _runtime
    if not _enabled():
        return None
    geometry = (
        int(max_batch_size),
        int(total_heads),
        int(output_head_dim),
        int(query_head_dim),
    )
    # This is the measured GLM-5.3 Flash MLA geometry. Fail closed for other
    # MLA models instead of allocating an unqualified process-global channel.
    if geometry[1:] != (64, 512, 512):
        return None
    if _runtime is None:
        _runtime = B12xMLADCPA2A(
            process_group=process_group,
            device=device,
            max_batch_size=geometry[0],
            total_heads=geometry[1],
            output_head_dim=geometry[2],
            query_head_dim=geometry[3],
        )
        logger.info_once(
            "Enabled captured B12x PCIe MLA DCP A2A "
            "(batch<=%d, heads=%d, query=%d, output=%d; eager fallback=vLLM).",
            geometry[0],
            geometry[1],
            geometry[3],
            geometry[2],
            scope="global",
        )
    configured = (
        _runtime.max_batch_size,
        _runtime.total_heads,
        _runtime.output_head_dim,
        _runtime.query_head_dim,
    )
    if configured != geometry or _runtime.device != torch.device(device):
        raise RuntimeError(
            "B12x MLA DCP A2A was initialized with incompatible geometry: "
            f"existing={configured}/{_runtime.device}, requested={geometry}/{device}"
        )
    return _runtime


@contextmanager
def capture_b12x_mla_dcp_a2a(
    stream: torch.cuda.Stream | None = None,
) -> Iterator[None]:
    global _capture_depth
    runtime = _runtime
    if runtime is None:
        yield
        return
    _capture_depth += 1
    try:
        with runtime.pool.capture(stream=stream):
            yield
    finally:
        _capture_depth -= 1


def close_b12x_mla_dcp_a2a() -> None:
    global _runtime
    runtime = _runtime
    if runtime is None:
        return
    runtime.close()
    _runtime = None
