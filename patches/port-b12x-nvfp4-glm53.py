#!/usr/bin/env python3
"""Expose B12x's compact NVFP4 MLA cache through the GLM day-zero vLLM."""

from __future__ import annotations

import sys
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, found {count}: {old[:160]!r}")
    path.write_text(text.replace(old, new, 1))


def port(root: Path) -> None:
    # The transplanted B12x backend already owns both compact record kernels,
    # but Pydantic rejects the dtype before backend selection unless the newer
    # day-zero cache schema knows its public spelling.
    cache_config = root / "config/cache.py"
    replace_once(
        cache_config,
        '    "fp8_ds_mla",\n'
        '    "turboquant_k8v4",\n',
        '    "fp8_ds_mla",\n'
        '    "nvfp4_ds_mla",\n'
        '    "turboquant_k8v4",\n',
    )

    # The generic vLLM validator correctly rejects its unrelated NVFP4 cache
    # layouts for MLA, but this backend owns a distinct packed MLA record and
    # its matching reader/writer.  Admit only the explicit GLM+B12x spelling;
    # every other NVFP4/MLA combination keeps the fail-closed error.
    vllm_config = root / "config/vllm.py"
    replace_once(
        vllm_config,
        "        if (\n"
        '            self.cache_config.cache_dtype.startswith("nvfp4")\n'
        "            and self.model_config.use_mla\n"
        "        ):\n"
        "            raise ValueError(\n"
        '                "nvfp4 KV cache is not supported with MLA (Multi-head Latent "\n'
        '                "Attention) backends. Please use a different --kv-cache-dtype "\n'
        '                "(e.g., \'fp8\' or \'auto\') for MLA models such as DeepSeek."\n'
        "            )\n",
        "        backend = self.attention_config.backend\n"
        "        b12x_glm_nvfp4_mla = (\n"
        '            self.cache_config.cache_dtype == "nvfp4_ds_mla"\n'
        '            and getattr(backend, "name", None) == "B12X_MLA_SPARSE"\n'
        "        )\n"
        "        if (\n"
        '            self.cache_config.cache_dtype.startswith("nvfp4")\n'
        "            and self.model_config.use_mla\n"
        "            and not b12x_glm_nvfp4_mla\n"
        "        ):\n"
        "            raise ValueError(\n"
        '                "nvfp4 KV cache is not supported with MLA (Multi-head Latent "\n'
        '                "Attention) backends. Please use a different --kv-cache-dtype "\n'
        '                "(e.g., \'fp8\' or \'auto\') for MLA models such as DeepSeek."\n'
        "            )\n",
    )

    # NVFP4 MLA is packed bytes on the vLLM side. B12x interprets the E2M1
    # data and E4M3 scales inside its reader/writer kernels.
    torch_utils = root / "utils/torch_utils.py"
    replace_once(
        torch_utils,
        '    "fp8_ds_mla": torch.uint8,\n'
        '    "turboquant_k8v4": torch.uint8,\n',
        '    "fp8_ds_mla": torch.uint8,\n'
        '    "nvfp4_ds_mla": torch.uint8,\n'
        '    "turboquant_k8v4": torch.uint8,\n',
    )

    # Keep scheduler accounting identical to the backend's physical record:
    # 368 B/token for the GLM compact format (256B FP4 latent, 32B group
    # scales, 64B FP8 RoPE, one FP32 token scale in the reused pad), otherwise
    # the stock 432 B/token NVFP4 MLA record. This is the capacity-critical
    # piece that makes a 500K DCP2 profile possible.
    kv_interface = root / "v1/kv_cache_interface.py"
    replace_once(
        kv_interface,
        "import copy\n",
        "import copy\nimport os\n",
    )
    replace_once(
        kv_interface,
        "    def real_page_size_bytes(self) -> int:\n"
        '        if self.cache_dtype_str == "fp8_ds_mla":\n',
        "    def real_page_size_bytes(self) -> int:\n"
        '        if self.cache_dtype_str == "nvfp4_ds_mla":\n'
        '            record_bytes = 368 if os.getenv("KV_FP8_ROPE", "0") == "1" else 432\n'
        "            return self.storage_block_size * record_bytes\n"
        '        if self.cache_dtype_str == "fp8_ds_mla":\n',
    )

    # NVFP4 is a quantized cache, but its uint8 tensor is a packed record—not
    # a plain E4M3 tensor. Reinterpreting it as float8 corrupts its byte ABI.
    mla_attention = root / "model_executor/layers/attention/mla_attention.py"
    replace_once(
        mla_attention,
        '        if fp8_attention and self.kv_cache_dtype != "fp8_ds_mla":\n'
        "            kv_cache = kv_cache.view(current_platform.fp8_dtype())\n",
        "        if fp8_attention and self.kv_cache_dtype not in (\n"
        '            "fp8_ds_mla",\n'
        '            "nvfp4_ds_mla",\n'
        "        ):\n"
        "            kv_cache = kv_cache.view(current_platform.fp8_dtype())\n",
    )

    # The inherited compact-record gate predates GLM-5.3 and only recognizes
    # ``glm_moe_dsa``.  vLLM constructs this checkpoint twice: first as the
    # multimodal ``glm5_next`` wrapper, then as nested ``glm5_next_text``.
    # Both use the same 512-wide latent ABI and exact B12x record kernels.
    b12x_backend = root / "v1/attention/backends/mla/b12x_mla_sparse.py"
    replace_once(
        b12x_backend,
        '_IS_GLM_MOE_DSA_CACHE: bool | None = None\n\n\n'
        'def _is_glm_moe_dsa_model() -> bool:\n',
        '_IS_GLM_MOE_DSA_CACHE: bool | None = None\n'
        '_COMPACT_GLM_MLA_MODEL_TYPES = frozenset(\n'
        '    ("glm_moe_dsa", "glm5_next", "glm5_next_text")\n'
        ')\n\n\n'
        'def _is_glm_moe_dsa_model() -> bool:\n',
    )
    replace_once(
        b12x_backend,
        '    if model_type == "glm_moe_dsa":\n'
        '        _IS_GLM_MOE_DSA_CACHE = True\n'
        '        return True\n',
        '    if model_type in _COMPACT_GLM_MLA_MODEL_TYPES:\n'
        '        _IS_GLM_MOE_DSA_CACHE = True\n'
        '        return True\n',
    )
    replace_once(
        b12x_backend,
        '    result = model_type == "deepseek_mtp" and target_model_type == "glm_moe_dsa"\n',
        '    result = (\n'
        '        model_type == "deepseek_mtp"\n'
        '        and target_model_type in _COMPACT_GLM_MLA_MODEL_TYPES\n'
        '    )\n',
    )
    replace_once(
        b12x_backend,
        '                "model_type=glm_moe_dsa and its associated MTP draft"\n',
        '                "a qualified GLM MLA model type and its associated MTP draft"\n',
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} VLLM_PACKAGE_ROOT")
    port(Path(sys.argv[1]))
    print("B12x compact NVFP4 MLA cache port applied")
