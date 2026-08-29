#!/usr/bin/env python3
"""Port upstream vLLM DFlash2 onto the pinned GLM-5.3 runtime branch."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import tempfile
from pathlib import Path


UPSTREAM_COMMIT = "b389ac29465b33f9e9c534df221ea3c129e9793f"
UPSTREAM_PATCH_SHA256 = (
    "661897fd57ea70e302f06210944b4351680452d6544ec39a5a7129a36e07821e"
)

# These two tiny hunks conflict only because the pinned GLM branch has newer
# neighbouring registrations. Apply their exact upstream intent below.
MANUAL_PATHS = {
    "vllm/model_executor/models/registry.py",
    "vllm/v1/worker/gpu/spec_decode/__init__.py",
}


def _runtime_patch(raw_patch: str) -> str:
    kept: list[str] = []
    for block in raw_patch.split("diff --git ")[1:]:
        header = block.splitlines()[0]
        path = header.split()[0].removeprefix("a/")
        if path.startswith("vllm/") and path not in MANUAL_PATHS:
            kept.append("diff --git " + block)
    if not kept:
        raise RuntimeError("upstream DFlash2 patch contains no runtime hunks")
    return "".join(kept)


def _replace_once(path: Path, old: str, new: str, description: str) -> None:
    source = path.read_text(encoding="utf-8")
    if source.count(old) != 1:
        raise RuntimeError(
            f"{description}: expected exactly one insertion point in {path}"
        )
    path.write_text(source.replace(old, new), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("vllm_root", type=Path)
    parser.add_argument("upstream_patch", type=Path)
    args = parser.parse_args()

    root = args.vllm_root.resolve()
    raw_bytes = args.upstream_patch.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    if digest != UPSTREAM_PATCH_SHA256:
        raise RuntimeError(
            f"unexpected DFlash2 patch digest {digest}; "
            f"expected {UPSTREAM_PATCH_SHA256} for {UPSTREAM_COMMIT}"
        )

    with tempfile.NamedTemporaryFile("w", suffix=".patch") as filtered:
        filtered.write(_runtime_patch(raw_bytes.decode("utf-8")))
        filtered.flush()
        subprocess.run(
            ["patch", "--batch", "--forward", "-p1", "-i", filtered.name],
            cwd=root.parent,
            check=True,
        )

    registry = root / "model_executor/models/registry.py"
    registration = (
        '    "DFlashDraftModel": ("qwen3_dflash", "DFlashQwen3ForCausalLM"),\n'
    )
    _replace_once(
        registry,
        registration,
        registration
        + '    "DFlash2DraftModel": '
        '("qwen3_dflash2", "DFlash2Qwen3ForCausalLM"),\n',
        "register DFlash2 architecture",
    )

    spec_init = root / "v1/worker/gpu/spec_decode/__init__.py"
    dflash_branch = '    if speculative_config.method == "dflash":\n'
    _replace_once(
        spec_init,
        dflash_branch,
        dflash_branch
        + '        if "DFlash2DraftModel" in '
        "speculative_config.draft_model_config.architectures:\n"
        "            from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (\n"
        "                DFlash2Speculator,\n"
        "            )\n"
        "\n"
        "            return DFlash2Speculator(vllm_config, device)\n",
        "route DFlash2 to its V2 speculator",
    )

    required = (
        root / "model_executor/models/qwen3_dflash2.py",
        root / "v1/worker/gpu/spec_decode/dflash2/speculator.py",
    )
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"DFlash2 port did not create {path}")

    qwen3_dflash = (
        root / "model_executor/models/qwen3_dflash.py"
    ).read_text(encoding="utf-8")
    if "decoder_layer_cls = DFlashQwen3DecoderLayer" not in qwen3_dflash:
        raise RuntimeError("DFlash2 decoder-layer extension point is missing")
    if "self.decoder_layer_cls(" not in qwen3_dflash:
        raise RuntimeError(
            "DFlash2 decoder-layer extension point was regressed; see vLLM #53435"
        )

    config_source = (root / "config/vllm.py").read_text(encoding="utf-8")
    if "def _is_dflash2_draft" not in config_source:
        raise RuntimeError("DFlash2 did not install its V2-runner selection guard")

    rejects = sorted(root.parent.rglob("*.rej"))
    if rejects:
        raise RuntimeError(f"DFlash2 port left rejected hunks: {rejects}")

    print(f"ported upstream vLLM DFlash2 {UPSTREAM_COMMIT} onto GLM runtime")


if __name__ == "__main__":
    main()
