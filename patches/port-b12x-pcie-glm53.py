#!/usr/bin/env python3
"""Attach the B12x PCIe communicator without replacing current vLLM code."""

from __future__ import annotations

import sys
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, found {count}: {old[:120]!r}")
    path.write_text(text.replace(old, new, 1))


def port(root: Path) -> None:
    communicator = root / "distributed/device_communicators/cuda_communicator.py"
    replace_once(
        communicator,
        "        if use_custom_allreduce and self.aiter_ar_comm is None and self.world_size > 1:\n"
        "            # Initialize a custom fast all-reduce implementation.\n"
        "            self.ca_comm = CustomAllreduce(\n"
        "                group=self.cpu_group,\n"
        "                device=self.device,\n"
        "                symm_mem_enabled=(\n"
        "                    self.symm_mem_comm is not None and not self.symm_mem_comm.disabled\n"
        "                ),\n"
        "            )\n",
        "        if use_custom_allreduce and self.aiter_ar_comm is None and self.world_size > 1:\n"
        "            # Keep B12x behind the existing custom-all-reduce slot so\n"
        "            # unsupported shapes naturally fall through to PyNCCL.\n"
        "            import os\n"
        "\n"
        "            use_b12x_pcie = (\n"
        "                os.getenv(\"VLLM_ENABLE_PCIE_ALLREDUCE\", \"0\")\n"
        "                not in (\"\", \"0\", \"false\", \"False\")\n"
        "                and os.getenv(\"VLLM_PCIE_ALLREDUCE_BACKEND\", \"b12x\").lower()\n"
        "                == \"b12x\"\n"
        "            )\n"
        "            if use_b12x_pcie:\n"
        "                from vllm.distributed.device_communicators.b12x_pcie_all_reduce import (\n"
        "                    B12xPcieAllReduce,\n"
        "                )\n"
        "\n"
        "                self.ca_comm = B12xPcieAllReduce(\n"
        "                    group=self.cpu_group,\n"
        "                    device_group=self.device_group,\n"
        "                    device=self.device,\n"
        "                )\n"
        "            else:\n"
        "                self.ca_comm = CustomAllreduce(\n"
        "                    group=self.cpu_group,\n"
        "                    device=self.device,\n"
        "                    symm_mem_enabled=(\n"
        "                        self.symm_mem_comm is not None\n"
        "                        and not self.symm_mem_comm.disabled\n"
        "                    ),\n"
        "                )\n",
    )
    replace_once(
        communicator,
        '        if self.ca_comm is not None and not self.ca_comm.disabled:\n'
        '            enabled_ar_backends.append("CUSTOM")\n',
        '        if self.ca_comm is not None and not self.ca_comm.disabled:\n'
        '            backend_name = getattr(self.ca_comm, "backend_name", None)\n'
        '            enabled_ar_backends.append(\n'
        '                backend_name() if backend_name is not None else "CUSTOM"\n'
        '            )\n',
    )
    replace_once(
        communicator,
        "    def destroy(self):\n"
        "        if self.pynccl_comm is not None:\n"
        "            self.pynccl_comm.destroy()\n"
        "            self.pynccl_comm = None\n"
        "        if self.ca_comm is not None:\n"
        "            self.ca_comm = None\n",
        "    def destroy(self):\n"
        "        # B12x close is coordinated through the still-live CPU group.\n"
        "        if self.ca_comm is not None:\n"
        "            self.ca_comm.close()\n"
        "            self.ca_comm = None\n"
        "        if self.pynccl_comm is not None:\n"
        "            self.pynccl_comm.destroy()\n"
        "            self.pynccl_comm = None\n",
    )

    parallel_state = root / "distributed/parallel_state.py"
    replace_once(
        parallel_state,
        "    def destroy(self):\n"
        "        if hasattr(self, \"device_group\"):\n"
        "            torch.distributed.destroy_process_group(self.device_group)\n"
        "            del self.device_group\n"
        "        if hasattr(self, \"cpu_group\"):\n"
        "            torch.distributed.destroy_process_group(self.cpu_group)\n"
        "            del self.cpu_group\n"
        "        if self.device_communicator is not None:\n"
        "            self.device_communicator.destroy()\n",
        "    def destroy(self):\n"
        "        # Communicators may own collectively-managed CUDA IPC mappings.\n"
        "        # Close them before destroying the exchange process groups.\n"
        "        if self.device_communicator is not None:\n"
        "            self.device_communicator.destroy()\n"
        "        if hasattr(self, \"device_group\"):\n"
        "            torch.distributed.destroy_process_group(self.device_group)\n"
        "            del self.device_group\n"
        "        if hasattr(self, \"cpu_group\"):\n"
        "            torch.distributed.destroy_process_group(self.cpu_group)\n"
        "            del self.cpu_group\n",
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} VLLM_PACKAGE_ROOT")
    port(Path(sys.argv[1]))
    print("B12x PCIe all-reduce port for GLM-5.3 applied")
