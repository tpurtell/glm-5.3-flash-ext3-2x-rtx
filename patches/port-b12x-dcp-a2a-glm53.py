#!/usr/bin/env python3
"""Route captured MLA DCP AG/RS through the qualified B12x PCIe channel."""

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
    dcp_utils = root / "v1/attention/ops/dcp_utils.py"
    replace_once(
        dcp_utils,
        "from vllm.v1.attention.ops.common import cp_lse_ag_out_ar, cp_lse_ag_out_rs\n",
        "from vllm.v1.attention.ops.common import (\n"
        "    cp_lse_ag_out_ar,\n"
        "    cp_lse_ag_out_rs,\n"
        "    mask_dcp_empty_shards_,\n"
        ")\n"
        "from vllm.distributed.device_communicators.b12x_dcp_a2a import (\n"
        "    get_b12x_mla_dcp_a2a,\n"
        ")\n",
    )
    replace_once(
        dcp_utils,
        "        self.padded_num_heads = padded_num_heads\n\n"
        "        self.combine = self._init_combine(\n",
        "        self.padded_num_heads = padded_num_heads\n"
        "        self.b12x_a2a = get_b12x_mla_dcp_a2a(\n"
        "            process_group=self.group.device_group,\n"
        "            device=self.device,\n"
        "            max_batch_size=self.max_num_tokens,\n"
        "            total_heads=num_heads * self.group.world_size,\n"
        "            output_head_dim=output_head_dim,\n"
        "            query_head_dim=query_head_dim,\n"
        "        )\n\n"
        "        self.combine = self._init_combine(\n",
    )
    old_combine_tail = '''        return functools.partial(
            combine_fn,
            cp_group=self.group,
            is_lse_base_on_e=is_lse_base_on_e,
        )

    def _init_query_gather(
'''
    new_combine_tail = '''        fallback = functools.partial(
            combine_fn,
            cp_group=self.group,
            is_lse_base_on_e=is_lse_base_on_e,
        )
        if self.b12x_a2a is None or use_pcp:
            return fallback

        def b12x_or_fallback(
            partial_output: torch.Tensor,
            partial_lse: torch.Tensor,
            *,
            seq_lens: torch.Tensor,
            query_start_loc: torch.Tensor,
        ) -> torch.Tensor:
            if not self.b12x_a2a.can_combine(partial_output, partial_lse):
                return fallback(
                    partial_output,
                    partial_lse,
                    seq_lens=seq_lens,
                    query_start_loc=query_start_loc,
                )
            partial_lse = partial_lse.contiguous()
            mask_dcp_empty_shards_(partial_lse, seq_lens, query_start_loc)
            return self.b12x_a2a.combine(
                partial_output,
                partial_lse,
                is_lse_base_on_e=is_lse_base_on_e,
            )

        return b12x_or_fallback

    def _init_query_gather(
'''
    replace_once(dcp_utils, old_combine_tail, new_combine_tail)
    replace_once(
        dcp_utils,
        "        if direct_workspace is not None:\n"
        "            logger.info_once(\"Using direct symmetric-memory DCP query gather for MLA.\")\n"
        "            return direct_workspace.gather\n"
        "        return self._gather_query\n",
        "        if direct_workspace is not None:\n"
        "            logger.info_once(\"Using direct symmetric-memory DCP query gather for MLA.\")\n"
        "            return direct_workspace.gather\n"
        "        if self.b12x_a2a is None:\n"
        "            return self._gather_query\n\n"
        "        def b12x_or_fallback(query: torch.Tensor) -> torch.Tensor:\n"
        "            if not self.b12x_a2a.can_gather(query):\n"
        "                return self._gather_query(query)\n"
        "            gathered = self.b12x_a2a.gather(query)\n"
        "            if self.padded_num_heads is not None:\n"
        "                gathered = reserve_query_head_storage(\n"
        "                    gathered, self.padded_num_heads\n"
        "                )\n"
        "            return gathered\n\n"
        "        return b12x_or_fallback\n",
    )

    adapter = root / "distributed/device_communicators/b12x_pcie_all_reduce.py"
    replace_once(
        adapter,
        "            with self.runtime.capture(stream=stream):\n"
        "                yield\n",
        "            from vllm.distributed.device_communicators.b12x_dcp_a2a import (\n"
        "                capture_b12x_mla_dcp_a2a,\n"
        "            )\n\n"
        "            with (\n"
        "                self.runtime.capture(stream=stream),\n"
        "                capture_b12x_mla_dcp_a2a(stream=stream),\n"
        "            ):\n"
        "                yield\n",
    )
    replace_once(
        adapter,
        "        self.runtime.close()\n"
        "        self._closed = True\n",
        "        from vllm.distributed.device_communicators.b12x_dcp_a2a import (\n"
        "            close_b12x_mla_dcp_a2a,\n"
        "        )\n\n"
        "        close_b12x_mla_dcp_a2a()\n"
        "        self.runtime.close()\n"
        "        self._closed = True\n",
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} VLLM_PACKAGE_ROOT")
    port(Path(sys.argv[1]))
    print("Captured B12x PCIe MLA DCP A2A port applied")
