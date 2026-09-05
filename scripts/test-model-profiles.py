#!/usr/bin/env python3
"""CPU-only checks for target-specific context defaults and explicit overrides."""

import os
import subprocess
from pathlib import Path


def main():
    recipe = Path(__file__).resolve().parents[1]
    base = {k: v for k, v in os.environ.items() if k not in {
        "MODEL_PROFILE", "MODEL_ID", "MODEL_REVISION", "MAX_MODEL_LEN",
    }}
    cases = [
        ({}, "1048576"),
        ({"MODEL_PROFILE": "k3"}, "1048576"),
        ({"MODEL_PROFILE": "k325"}, "1048576"),
        ({"MODEL_PROFILE": "k4"}, "262144"),
        ({"MODEL_PROFILE": "k4", "MAX_MODEL_LEN": "131072"}, "131072"),
        ({"MODEL_PROFILE": "k325", "MAX_MODEL_LEN": "524288"}, "524288"),
        ({"MODEL_ID": "brandonmusic/GLM-5.3-Flash-EXL3-4bpw", "MODEL_REVISION": "test"}, "262144"),
        ({"MODEL_ID": "brandonmusic/GLM-5.3-Flash-tr3-4bpw", "MODEL_REVISION": "test"}, "262144"),
        ({"MODEL_ID": "example/custom", "MODEL_REVISION": "test"}, "1048576"),
        ({"MODEL_PROFILE": "k4", "MAX_MODEL_LEN": ""}, "262144"),
    ]
    for overrides, expected in cases:
        actual = subprocess.check_output([
            "bash", "-c",
            'source ./model-profiles.sh; resolve_glm53_model_profile && '
            'resolve_glm53_context_limit && printf "%s" "$MAX_MODEL_LEN"',
        ], cwd=recipe, env=base | overrides, text=True)
        assert actual == expected, (overrides, actual, expected)
    print(f"PASS: {len(cases)} model/context-profile cases; no Docker or GPU calls")


if __name__ == "__main__":
    main()
