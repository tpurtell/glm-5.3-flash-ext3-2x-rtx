#!/usr/bin/env python3
"""Profile and allocate reachable GLM sparse-attention bounds."""

import sys
from pathlib import Path


def replace(path: Path, old: str, new: str) -> None:
    source = path.read_text()
    if new in source:
        return
    if source.count(old) != 1:
        raise RuntimeError(f"{path}: sparse-memory source drift")
    source = source.replace(old, new)
    compile(source, str(path), "exec")
    path.write_text(source)


root = Path(sys.argv[1])
replace(
    root / "v1/attention/backends/mla/indexer.py",
    """    return max_model_len * 40
""",
    """    hf_config = vllm_config.model_config.hf_text_config
    if getattr(hf_config, "model_type", None) in {"glm5_next", "glm5_next_text"}:
        # This helper sizes both the live gather allocation and its chunk
        # planner. A scheduled batch cannot contain more full-length contexts
        # than max_num_seqs; retain the original 40-context upper bound.
        return max_model_len * min(
            40, vllm_config.scheduler_config.max_num_seqs
        )
    return max_model_len * 40
""",
)
replace(
    root / "model_executor/layers/attention/mla_attention.py",
    """            _ = torch.empty(
                (
                    self.chunked_prefill_workspace_size,
                    self.num_heads,
                    self.qk_nope_head_dim + self.v_head_dim,
                ),
                device=k_c_normed.device,
                dtype=k_c_normed.dtype,
            )
""",
    """            if self.attn_backend.get_name() == "B12X_MLA_SPARSE":
                # B12x never expands the complete context through kv_b_proj.
                # Its real caller-owned sparse workspaces were reserved at
                # construction, so borrow them during profiling as well.
                self.impl._borrow_workspaces()
            else:
                _ = torch.empty(
                    (
                        self.chunked_prefill_workspace_size,
                        self.num_heads,
                        self.qk_nope_head_dim + self.v_head_dim,
                    ),
                    device=k_c_normed.device,
                    dtype=k_c_normed.dtype,
                )
""",
)
replace(
    root / "v1/worker/gpu/model_runner.py",
    """        min_blocks = self.compilation_config.max_cudagraph_capture_size or 1
""",
    """        # Even a small capture ladder needs one KDA state per scheduler slot.
        min_blocks = max(
            self.compilation_config.max_cudagraph_capture_size or 1,
            self.scheduler_config.max_num_seqs,
        )
""",
)
print("GLM gather capacity and B12x sparse-only memory profiling applied")
