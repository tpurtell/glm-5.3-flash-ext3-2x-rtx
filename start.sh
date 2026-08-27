#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ID="${MODEL_ID:-brandonmusic/GLM-5.3-Flash-EXL3-4bpw}"
MODEL_REVISION="${MODEL_REVISION:-4739eb1bcfd478e8a32da6358908567bc3a9ac51}"
HF_CACHE_DIR="${HF_HOME:-${HOME}/.cache/huggingface}"
MODEL_CACHE_NAME="models--${MODEL_ID//\//--}"
MODEL_REPO_DIR="${MODEL_REPO_DIR:-${HF_CACHE_DIR}/hub/${MODEL_CACHE_NAME}}"
MODEL_DIR="${MODEL_REPO_DIR}/snapshots/${MODEL_REVISION}"
IMAGE="${IMAGE:-ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-glm53-flash-exl3-b12x-vllm}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL_ID}}"
PORT="${PORT:-8001}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
DECODE_CONTEXT_PARALLEL_SIZE="${DECODE_CONTEXT_PARALLEL_SIZE:-2}"
DCP_COMM_BACKEND="${DCP_COMM_BACKEND:-ag_rs}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-500000}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
if [[ -v KV_CACHE_PROFILE ]]; then
  KV_CACHE_PROFILE="${KV_CACHE_PROFILE}"
elif [[ "${KV_CACHE_DTYPE:-}" == fp8_ds_mla ]]; then
  # Preserve the original KV_CACHE_DTYPE override as a convenient shorthand.
  KV_CACHE_PROFILE=fp8
else
  KV_CACHE_PROFILE=nvfp4
fi
case "${KV_CACHE_PROFILE}" in
  nvfp4)
    PROFILE_KV_CACHE_DTYPE=nvfp4_ds_mla
    PROFILE_GPU_MEMORY_UTILIZATION=0.965
    ;;
  fp8)
    PROFILE_KV_CACHE_DTYPE=fp8_ds_mla
    PROFILE_GPU_MEMORY_UTILIZATION=0.970
    ;;
  *)
    echo "KV_CACHE_PROFILE must be nvfp4 or fp8; got: ${KV_CACHE_PROFILE}" >&2
    exit 2
    ;;
esac
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-${PROFILE_KV_CACHE_DTYPE}}"
if [[ "${KV_CACHE_DTYPE}" != "${PROFILE_KV_CACHE_DTYPE}" ]]; then
  echo "KV_CACHE_PROFILE=${KV_CACHE_PROFILE} requires KV_CACHE_DTYPE=${PROFILE_KV_CACHE_DTYPE}; got ${KV_CACHE_DTYPE}" >&2
  exit 2
fi
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-${PROFILE_GPU_MEMORY_UTILIZATION}}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-B12X_MLA_SPARSE}"
USE_B12X_SPARSE_INDEXER="${USE_B12X_SPARSE_INDEXER:-1}"
USE_B12X_KPOOL_INDEXER="${USE_B12X_KPOOL_INDEXER:-1}"
ENABLE_PCIE_ALLREDUCE="${ENABLE_PCIE_ALLREDUCE:-1}"
PCIE_ALLREDUCE_BACKEND="${PCIE_ALLREDUCE_BACKEND:-b12x}"
PCIE_ONESHOT_MAX_SIZE="${PCIE_ONESHOT_MAX_SIZE:-384KB}"
CACHE_DIR="${CACHE_DIR:-${SCRIPT_DIR}/.cache/vllm}"
GPU_REQUEST="${GPU_REQUEST:-all}"
VLLM_NVFP4_MLA_SCALES_FILE="${VLLM_NVFP4_MLA_SCALES_FILE:-}"

if [[ "${KV_CACHE_DTYPE}" == nvfp4_ds_mla ]]; then
  KV_FP8_ROPE="${KV_FP8_ROPE:-1}"
  if [[ -n "${VLLM_NVFP4_MLA_SCALES_FILE}" ]]; then
    VLLM_NVFP4_MLA_DYNAMIC_SCALE="${VLLM_NVFP4_MLA_DYNAMIC_SCALE:-0}"
  else
    VLLM_NVFP4_MLA_DYNAMIC_SCALE="${VLLM_NVFP4_MLA_DYNAMIC_SCALE:-1}"
  fi
else
  KV_FP8_ROPE="${KV_FP8_ROPE:-0}"
  VLLM_NVFP4_MLA_DYNAMIC_SCALE="${VLLM_NVFP4_MLA_DYNAMIC_SCALE:-0}"
fi

if [[ $# -ne 0 ]]; then
  echo "Use environment variables for overrides; positional arguments are not supported." >&2
  exit 1
fi
if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  echo "Pinned HF snapshot is missing: ${MODEL_DIR}" >&2
  echo "Run ${SCRIPT_DIR}/download.sh first." >&2
  exit 1
fi
if [[ "$(find "${MODEL_DIR}" -maxdepth 1 \( -type l -o -type f \) -name 'model-*-of-00120.safetensors' | wc -l)" -ne 120 ]]; then
  echo "Pinned HF snapshot is incomplete (expected 120 weight shards): ${MODEL_DIR}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_DIR}/quantization_config.json" ]]; then
  echo "EXL3 tensor manifest is missing: ${MODEL_DIR}/quantization_config.json" >&2
  exit 1
fi
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  printf 'Pulling %s...\n' "${IMAGE}"
  docker pull "${IMAGE}"
fi
if docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  if [[ "$(docker inspect --format '{{.State.Running}}' "${CONTAINER_NAME}")" == true ]]; then
    echo "Container is already running: ${CONTAINER_NAME}" >&2
    exit 1
  fi
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

mkdir -p "${CACHE_DIR}"

CUSTOM_ALL_REDUCE_ARGS=()
if [[ "${ENABLE_PCIE_ALLREDUCE}" == 0 ]]; then
  CUSTOM_ALL_REDUCE_ARGS+=(--disable-custom-all-reduce)
fi

# Mount the complete HF repository rather than only its symlinked snapshot so
# vLLM can resolve every shard into the repository's blobs directory.
docker run --detach \
  --name "${CONTAINER_NAME}" \
  --restart unless-stopped \
  --init \
  --gpus "${GPU_REQUEST}" \
  --ipc=host \
  --shm-size 32g \
  --publish "${PORT}:8001" \
  --env HF_HUB_OFFLINE=1 \
  --env CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}" \
  --env TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-0}" \
  --env VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}" \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env NCCL_DEBUG="${NCCL_DEBUG:-WARN}" \
  --env VLLM_ENABLE_PCIE_ALLREDUCE="${ENABLE_PCIE_ALLREDUCE}" \
  --env VLLM_PCIE_ALLREDUCE_BACKEND="${PCIE_ALLREDUCE_BACKEND}" \
  --env VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE="${PCIE_ONESHOT_MAX_SIZE}" \
  --env VLLM_B12X_PCIE_EAGER="${VLLM_B12X_PCIE_EAGER:-0}" \
  --env VLLM_B12X_DCP_A2A="${VLLM_B12X_DCP_A2A:-1}" \
  --env VLLM_USE_B12X_SPARSE_INDEXER="${USE_B12X_SPARSE_INDEXER}" \
  --env VLLM_USE_B12X_KPOOL_INDEXER="${USE_B12X_KPOOL_INDEXER}" \
  --env VLLM_DCP_GLOBAL_TOPK="${VLLM_DCP_GLOBAL_TOPK:-1}" \
  --env VLLM_DCP_QUERY_SPLIT="${VLLM_DCP_QUERY_SPLIT:-0}" \
  --env VLLM_DCP_TOPK_OWNER_MERGE="${VLLM_DCP_TOPK_OWNER_MERGE:-1}" \
  --env VLLM_B12X_DCP_TOPK_OWNER_EXCHANGE="${VLLM_B12X_DCP_TOPK_OWNER_EXCHANGE:-1}" \
  --env VLLM_B12X_DCP_TOPK_MIN_ROWS="${VLLM_B12X_DCP_TOPK_MIN_ROWS:-128}" \
  --env VLLM_B12X_DCP_TOPK_MAX_ROWS="${VLLM_B12X_DCP_TOPK_MAX_ROWS:-${MAX_NUM_BATCHED_TOKENS}}" \
  --env KV_FP8_ROPE="${KV_FP8_ROPE}" \
  --env VLLM_NVFP4_MLA_DYNAMIC_SCALE="${VLLM_NVFP4_MLA_DYNAMIC_SCALE}" \
  --env VLLM_NVFP4_MLA_SCALES_FILE="${VLLM_NVFP4_MLA_SCALES_FILE}" \
  --env VLLM_EXL3_TRELLIS_MIN_M="${VLLM_EXL3_TRELLIS_MIN_M:-1}" \
  --env VLLM_EXL3_TRELLIS_MAX_M="${VLLM_EXL3_TRELLIS_MAX_M:-32}" \
  --env VLLM_EXL3_TRELLIS_BLOCK_M="${VLLM_EXL3_TRELLIS_BLOCK_M:-8}" \
  --env VLLM_EXL3_PREFILL_TRELLIS="${VLLM_EXL3_PREFILL_TRELLIS:-1}" \
  --env VLLM_EXL3_PREFILL_BLOCK_M="${VLLM_EXL3_PREFILL_BLOCK_M:-64}" \
  --env B12X_EXL3_BF16_EPILOGUE="${B12X_EXL3_BF16_EPILOGUE:-1}" \
  --env B12X_EXL3_BF16_GEMV="${B12X_EXL3_BF16_GEMV:-1}" \
  --env VLLM_B12X_GLM_H64_QUERY_PROJ="${VLLM_B12X_GLM_H64_QUERY_PROJ:-auto}" \
  --env VLLM_USE_B12X_MHC="${VLLM_USE_B12X_MHC:-auto}" \
  --volume "${MODEL_REPO_DIR}:/model-repo:ro" \
  --volume "${CACHE_DIR}:/root/.cache" \
  "${IMAGE}" \
  "/model-repo/snapshots/${MODEL_REVISION}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 0.0.0.0 \
  --port 8001 \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --decode-context-parallel-size "${DECODE_CONTEXT_PARALLEL_SIZE}" \
  --dcp-comm-backend "${DCP_COMM_BACKEND}" \
  "${CUSTOM_ALL_REDUCE_ARGS[@]}" \
  --attention-backend "${ATTENTION_BACKEND}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --enable-prefix-caching \
  --no-enable-flashinfer-autotune \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45

printf 'Started %s on http://127.0.0.1:%s/v1. Initial B12x/CuTe compilation can take several minutes.\n' \
  "${CONTAINER_NAME}" "${PORT}"
printf 'Profile: %s, KV cache: %s, max length: %s, max sequences: %s, GPU memory utilization: %s\n' \
  "${KV_CACHE_PROFILE}" "${KV_CACHE_DTYPE}" "${MAX_MODEL_LEN}" "${MAX_NUM_SEQS}" \
  "${GPU_MEMORY_UTILIZATION}"
printf 'Follow startup: docker logs -f %s\n' "${CONTAINER_NAME}"
