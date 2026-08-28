# syntax=docker/dockerfile:1.7

# Use the user's qualified vLLM EXL3 implementation as a source artifact only.
# The final runtime remains the newer GLM-5.3 SM120 vLLM image.
ARG EXL3_SOURCE_IMAGE=ghcr.io/tpurtell/deepseek-v4-flash-0731-exl3-k2-spark@sha256:86c8c1054f9c24454949e37031ce6165c007963aa0c0ef30fa884f6d4170af32
ARG GLM_BASE_IMAGE=cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1@sha256:0bd709e80b8ff13ae5de8f7d7f708a499fade3a26970d56afb1be2ff3860fde5

FROM ${EXL3_SOURCE_IMAGE} AS exl3_source
FROM ${GLM_BASE_IMAGE}

ARG B12X_REPOSITORY=https://github.com/tpurtell/sparkinfer-glmrt
ARG B12X_COMMIT=988246c8b007c9c1c2006eb677f6fa4b26aeb561

SHELL ["/bin/bash", "-c"]

# Fetch an immutable snapshot of the user's current B12x fork. Keep the GLM
# base's Torch 2.13, CUDA 13, and CUTLASS DSL 4.6.2 stack intact; B12x is pure
# Python/CuTe DSL and compiles its selected SM120 specializations at runtime.
RUN B12X_REPOSITORY="${B12X_REPOSITORY}" B12X_COMMIT="${B12X_COMMIT}" \
    python3 - <<'PY'
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

repository = os.environ["B12X_REPOSITORY"].removesuffix(".git")
commit = os.environ["B12X_COMMIT"]
archive = Path("/tmp/b12x.tar.gz")
urllib.request.urlretrieve(f"{repository}/archive/{commit}.tar.gz", archive)
with tarfile.open(archive) as tar:
    tar.extractall("/tmp", filter="data")
sources = list(Path("/tmp").glob("sparkinfer-glmrt-*"))
if len(sources) != 1:
    raise RuntimeError(f"expected one B12x source tree, found: {sources}")
shutil.rmtree("/opt/b12x", ignore_errors=True)
shutil.move(sources[0], "/opt/b12x")
archive.unlink()
PY
RUN python3 -m pip install --no-cache-dir --no-deps -e /opt/b12x

# Carry only the proven EXL3 quantization implementation into the GLM vLLM
# tree, then adapt its narrow registration/model-recognition surface.
COPY --from=exl3_source \
    /opt/vllm/vllm/model_executor/layers/quantization/exl3.py \
    /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py
# Reuse the mature vLLM/B12x adapter from the same qualified source image. The
# adapter is ported below onto the newer GLM-5.3 vLLM APIs; all kernels still
# resolve from the current /opt/b12x checkout pinned above.
COPY --from=exl3_source \
    /opt/vllm/vllm/v1/attention/backends/mla/b12x_mla_sparse.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/b12x_mla_sparse.py
COPY --from=exl3_source \
    /opt/vllm/vllm/model_executor/layers/sparse_attn_indexer.py \
    /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer.py
COPY --from=exl3_source \
    /opt/vllm/vllm/model_executor/layers/mla_cache_format.py \
    /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mla_cache_format.py
COPY patches/port-exl3-glm53.py /tmp/port-exl3-glm53.py
COPY patches/port-exl3-mtp-glm53.py /tmp/port-exl3-mtp-glm53.py
COPY patches/port-b12x-glm53.py /tmp/port-b12x-glm53.py
COPY patches/port-b12x-kpool-glm53.py /tmp/port-b12x-kpool-glm53.py
COPY patches/b12x_dcp_topk.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/b12x_dcp_topk.py
COPY patches/port-b12x-dcp-glm53.py /tmp/port-b12x-dcp-glm53.py
COPY patches/port-b12x-dcp-owner-glm53.py /tmp/port-b12x-dcp-owner-glm53.py
COPY patches/port-b12x-nvfp4-glm53.py /tmp/port-b12x-nvfp4-glm53.py
COPY patches/port-b12x-glm-h64-query.py /tmp/port-b12x-glm-h64-query.py
COPY patches/port-b12x-mhc-glm53.py /tmp/port-b12x-mhc-glm53.py
COPY patches/port-glm53-sm12-stability.py /tmp/port-glm53-sm12-stability.py
COPY patches/b12x_pcie_all_reduce.py \
    /usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/b12x_pcie_all_reduce.py
COPY patches/port-b12x-pcie-glm53.py /tmp/port-b12x-pcie-glm53.py
COPY patches/b12x_dcp_a2a.py \
    /usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/b12x_dcp_a2a.py
COPY patches/port-b12x-dcp-a2a-glm53.py /tmp/port-b12x-dcp-a2a-glm53.py
COPY patches/vllm-replayssm-spec.patch /tmp/vllm-replayssm-spec.patch
COPY patches/vllm-dynamic-sd-cudagraph.patch /tmp/vllm-dynamic-sd-cudagraph.patch
COPY patches/port-replayssm-glm53.py /tmp/port-replayssm-glm53.py
COPY patches/port-glm53-mtp-prefix-cache.py /tmp/port-glm53-mtp-prefix-cache.py
COPY patches/adaptive_mtp.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/dynamic/adaptive_mtp.py
COPY patches/port-adaptive-mtp-glm53.py /tmp/port-adaptive-mtp-glm53.py
RUN python3 /tmp/port-exl3-glm53.py \
    /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-exl3-mtp-glm53.py \
    /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-b12x-glm53.py \
    /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-b12x-kpool-glm53.py \
    /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-b12x-dcp-glm53.py \
    /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-b12x-dcp-owner-glm53.py \
    /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-b12x-nvfp4-glm53.py \
    /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-b12x-glm-h64-query.py \
    /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-b12x-mhc-glm53.py \
    /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-glm53-sm12-stability.py \
    /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-b12x-pcie-glm53.py \
    /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-b12x-dcp-a2a-glm53.py \
    /usr/local/lib/python3.12/dist-packages/vllm

# Port the current upstream ReplaySSM series onto the exact day-zero vLLM
# commit, then add compact rollback for GLM's vector-gated KDA recurrence. The
# patch is intentionally applied after the local EXL3/B12x ports; this is the
# ordering qualified by the clean-image compatibility test.
RUN cd /usr/local/lib/python3.12/dist-packages \
 && patch --batch --forward -p1 < /tmp/vllm-replayssm-spec.patch \
 && patch --batch --forward -p1 < /tmp/vllm-dynamic-sd-cudagraph.patch \
 && python3 /tmp/port-replayssm-glm53.py \
    /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-glm53-mtp-prefix-cache.py \
    /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-adaptive-mtp-glm53.py \
    /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 -m compileall -q /usr/local/lib/python3.12/dist-packages/vllm

ENV PYTHONPATH=/opt/b12x:/usr/local/lib/python3.12/dist-packages \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_MODULE_LOADING=LAZY \
    VLLM_EXL3_TRELLIS_MIN_M=1 \
    VLLM_EXL3_TRELLIS_MAX_M=32 \
    VLLM_EXL3_TRELLIS_BLOCK_M=8 \
    VLLM_EXL3_PREFILL_TRELLIS=1 \
    VLLM_EXL3_PREFILL_BLOCK_M=64 \
    VLLM_EXL3_PREFILL_CAPACITY=1024 \
    VLLM_B12X_GLM_H64_QUERY_PROJ=auto \
    VLLM_USE_B12X_MHC=auto \
    VLLM_USE_B12X_SPARSE_INDEXER=1

# Build-time compatibility and scope probe. This proves that the newer vLLM
# imports the transplanted method, that the current B12x API is present, and
# that GLM enters B12x standard-fused mode without inventing FP8 base weights.
RUN python3 - <<'PY'
from pathlib import Path
from types import SimpleNamespace

import b12x
import cutlass
import torch
import vllm
from b12x.moe import fused_moe
from b12x.attention import nsa_indexer, sparse_mla
from b12x.gemm import mla_query_projection
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.exl3 import Exl3Config
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.model_executor.models.interfaces import supports_replayssm
from vllm.models.glm5next.nvidia.model import Glm5NextForConditionalGeneration
from vllm.third_party.flash_linear_attention.ops.kda_replayssm_spec_decode import (
    kda_replayssm_spec_decode,
    materialize_kda_replayssm_state,
)
from vllm.config.cache import CacheConfig
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.kv_cache_interface import MLAAttentionSpec
from vllm.v1.spec_decode.dynamic.adaptive_mtp import AdaptiveMTPController
from vllm.v1.attention.backends.mla.b12x_mla_sparse import B12xMLASparseBackend
from vllm.v1.attention.backends.registry import AttentionBackendEnum

def entry(name: str):
    return {
        "bits_per_weight": 4,
        "quant_format": "exl3",
        "stored_tensors": {
            f"{name}.suh": {},
            f"{name}.svh": {},
            f"{name}.trellis": {},
            f"{name}.mcg": {},
        },
    }

root = "model.language_model.layers.3.mlp.experts.0"
storage = {
    f"{root}.gate_proj": entry(f"{root}.gate_proj"),
    f"{root}.up_proj": entry(f"{root}.up_proj"),
    f"{root}.down_proj": entry(f"{root}.down_proj"),
}
config = Exl3Config(bits=4, codebook="mcg", tensor_storage=storage)
config._configure_standard_fused_moe(SimpleNamespace(model_type="glm5_next"))
config._configure_base_quantization(SimpleNamespace(model_type="glm5_next"))

assert get_quantization_config("exl3") is Exl3Config
assert supports_replayssm(Glm5NextForConditionalGeneration)
assert MambaStateShapeCalculator.replayssm_spec_ring_len(10, 5) == 16
assert MambaStateShapeCalculator.replayssm_spec_ring_len(16, 5) == 32
assert callable(kda_replayssm_spec_decode)
assert callable(materialize_kda_replayssm_state)
adaptive_probe = AdaptiveMTPController(max_depth=5, probe_interval=4)
assert adaptive_probe.select(["c1"], 1, 5) == 5
assert adaptive_probe.select(["c8"], 8, 1) == 1
assert config.standard_fused_moe
assert config._base_quant_config is None
assert config._moe_prefix_is_exl3(
    "language_model.model.layers.3.mlp.experts"
)
assert config._moe_prefix_is_exl3("model.layers.3.mlp.experts")
assert config._storage_entry(
    "model.layers.3.mtp_block.mlp.experts.0.gate_proj"
) is not None
assert config._moe_prefix_is_exl3(
    "model.layers.3.mtp_block.mlp.experts.routed_experts"
)
assert "is_standard_mtp" in Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
    "quantization/exl3.py"
).read_text()
assert "scheduler_config.max_num_seqs" in Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
    "quantization/exl3.py"
).read_text()
assert callable(fused_moe.plan_weights)
assert callable(fused_moe.prepare_weights)
assert callable(fused_moe.plan)
assert callable(fused_moe.bind)
assert callable(fused_moe.run)
assert callable(sparse_mla.plan)
assert callable(sparse_mla.bind)
assert callable(sparse_mla.run_decode)
assert callable(sparse_mla.run_extend)
assert callable(nsa_indexer.plan)
assert callable(nsa_indexer.index_topk_fp8)
assert callable(mla_query_projection.run_glm_h64_bf16)
from b12x.norm import mhc
assert callable(mhc.run_post_pre)
assert "nope" in __import__("inspect").signature(
    mla_query_projection.prewarm_glm_h64_bf16
).parameters
from b12x.comm.pcie import OneshotAllReducePool
from b12x.comm.pcie import DcpTopKOwnerExchange
from b12x.comm.pcie.pcie_dcp_a2a import PCIeDCPA2APool
from vllm.distributed.device_communicators.b12x_pcie_all_reduce import (
    B12xPcieAllReduce,
)
assert callable(OneshotAllReducePool.from_exchange_group)
assert callable(DcpTopKOwnerExchange.from_exchange_group)
assert callable(PCIeDCPA2APool.from_exchange_group)
assert B12xPcieAllReduce.backend_name(None) == "B12X_PCIE_ONESHOT"
assert B12xMLASparseBackend.get_supported_head_sizes() == [512, 576]
assert CacheConfig(cache_dtype="nvfp4_ds_mla").cache_dtype == "nvfp4_ds_mla"
assert STR_DTYPE_TO_TORCH_DTYPE["nvfp4_ds_mla"] is torch.uint8
assert B12xMLASparseBackend.get_kv_cache_shape(1, 64, 1, 576, "nvfp4_ds_mla") == (
    1,
    64,
    432,
)
assert MLAAttentionSpec(
    block_size=64,
    num_kv_heads=1,
    head_size=576,
    dtype=torch.uint8,
    cache_dtype_str="nvfp4_ds_mla",
).real_page_size_bytes == 64 * 432
assert AttentionBackendEnum.B12X_MLA_SPARSE.get_class() is B12xMLASparseBackend
assert "output_physical_slots" in __import__("inspect").signature(
    SparseAttnIndexer.__init__
).parameters
kpool_source = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
    "sparse_attn_indexer_kpool.py"
).read_text()
assert "Using B12x fused paged score+top-k for GLM kpool decode." in kpool_source
assert "Using B12x PCIe DCP owner top-k exchange" in __import__(
    "inspect"
).getsource(__import__(
    "vllm.model_executor.layers.sparse_attn_indexer",
    fromlist=["_get_b12x_dcp_topk_owner_exchange"],
))
config_source = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/config/vllm.py"
).read_text()
assert "b12x_glm_nvfp4_mla" in config_source
assert "and not b12x_glm_nvfp4_mla" in config_source
mhc_source = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mhc.py"
).read_text()
assert "Using B12x fused GLM H4096 mHC post+pre for decode M=1." in mhc_source
assert Path(b12x.__file__).is_relative_to(Path("/opt/b12x")), b12x.__file__
assert torch.__version__.startswith("2.13."), torch.__version__
assert vllm.__version__ == "0.1.dev20051+g487ecf187", vllm.__version__
assert cutlass.__version__ == "4.6.2", cutlass.__version__
print("GLM-5.3 vLLM + EXL3 + B12x compatibility probe passed")
PY

LABEL org.opencontainers.image.source="https://github.com/tpurtell/glm-5.3-flash-ext3-4-bit-2x-rtx" \
      org.opencontainers.image.description="GLM-5.3 Flash EXL3 on 2x SM120: compact NVFP4 MLA, DCP2, and B12x PCIe kernels" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.tpurtell.b12x.source="https://github.com/tpurtell/sparkinfer-glmrt" \
      io.tpurtell.glm-base.digest="sha256:0bd709e80b8ff13ae5de8f7d7f708a499fade3a26970d56afb1be2ff3860fde5" \
      io.tpurtell.exl3-source.digest="sha256:86c8c1054f9c24454949e37031ce6165c007963aa0c0ef30fa884f6d4170af32" \
      io.tpurtell.exl3-vllm.commit="30038602b71395f481ef4a6edfe4fcf8551d9c15" \
      io.tpurtell.b12x.commit="988246c8b007c9c1c2006eb677f6fa4b26aeb561"

EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=5s --start-period=30m --retries=5 \
  CMD ["python3", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=3).read()"]

# Inherit the GLM base image's normal ["vllm", "serve"] entrypoint.
