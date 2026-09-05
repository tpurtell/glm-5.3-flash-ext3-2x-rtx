#!/usr/bin/env python3
"""Keep long-context prefill metadata scalar offsets shape-generic."""

import sys
from pathlib import Path


path = Path(sys.argv[1]) / "v1/attention/backends/mla/indexer.py"
source = path.read_text()
old = '''    @staticmethod
    @triton.jit
    def kernel(
'''
new = '''    @staticmethod
    @triton.jit(
        do_not_specialize=["query_slice_start", "query_slice_stop"],
        do_not_specialize_on_alignment=["query_slice_start", "query_slice_stop"],
    )
    def kernel(
'''
if new not in source:
    if source.count(old) != 1:
        raise RuntimeError("prefill metadata kernel source drift")
    source = source.replace(old, new)
    compile(source, str(path), "exec")
    path.write_text(source)

print("GLM prefill metadata slice offsets use one shape-generic Triton kernel")
