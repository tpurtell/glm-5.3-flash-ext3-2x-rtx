#!/usr/bin/env python3
"""Size graph metadata from GLM's physical selection buffer."""

import sys
from pathlib import Path


path = Path(sys.argv[1]) / "v1/attention/backends/mla/b12x_mla_sparse.py"
source = path.read_text()
old = "        self.topk_tokens = vllm_config.model_config.hf_config.index_topk\n"
new = old + '''        # GLM kpool appends trailing tokens and pads the shared index
        # buffer to 128 columns. The model/impl therefore owns the physical
        # width (2176 rather than config.index_topk=2048).
        context = vllm_config.compilation_config.static_forward_context
        widths = {
            int(layer.impl.topk_tokens)
            for name in layer_names
            if (layer := context.get(name)) is not None
            and hasattr(getattr(layer, "impl", None), "topk_tokens")
        }
        if len(widths) > 1:
            raise ValueError(f"MLA cache group has inconsistent index widths: {widths}")
        if widths:
            self.topk_tokens = widths.pop()
'''
if new not in source:
    if source.count(old) != 1:
        raise RuntimeError("sparse MLA metadata builder source drift")
    source = source.replace(old, new)
    compile(source, str(path), "exec")
    path.write_text(source)
print("GLM graph metadata uses the complete model-owned top-k buffer width")
