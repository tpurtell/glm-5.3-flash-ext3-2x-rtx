#!/usr/bin/env python3
"""Keep GLM's convolution verify width independent of compact state slots."""

import sys
from pathlib import Path


old = "            conv_mql = spec_state_indices_tensor.size(-1)\n"
new = (
    "            # Compact rollback stores one state slot, not one query token.\n"
    "            # The convolution must still process the full verify window.\n"
    "            conv_mql = (\n"
    "                self.num_spec + 1\n"
    "                if self.use_replayssm_spec\n"
    "                else spec_state_indices_tensor.size(-1)\n"
    "            )\n"
)
path = Path(sys.argv[1]) / "models/glm5next/nvidia/kda.py"
source = path.read_text()
if new not in source:
    if source.count(old) != 1:
        raise RuntimeError("GLM ReplaySSM convolution-width source drift")
    source = source.replace(old, new, 1)
    compile(source, str(path), "exec")
    path.write_text(source)
print("GLM ReplaySSM convolution uses K+1 tokens, not compact-state columns")
