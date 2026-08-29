#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ID="${MODEL_ID:-wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3-v1}"
MODEL_REVISION="${MODEL_REVISION:-4967efc3cf349681f208f217e29d29c11c9b0c45}"
HF_CACHE_DIR="${HF_HOME:-${HOME}/.cache/huggingface}"
MODEL_CACHE_NAME="models--${MODEL_ID//\//--}"
MODEL_REPO_DIR="${MODEL_REPO_DIR:-${HF_CACHE_DIR}/hub/${MODEL_CACHE_NAME}}"
MODEL_DIR_OVERRIDE="${MODEL_DIR_OVERRIDE:-}"
if [[ -n "${MODEL_DIR_OVERRIDE}" ]]; then
  MODEL_DIR="$(realpath -e -- "${MODEL_DIR_OVERRIDE}")"
  MODEL_MOUNT_SOURCE="${MODEL_DIR}"
  MODEL_MOUNT_TARGET=/model
  MODEL_CONTAINER_DIR=/model
else
  MODEL_DIR="${MODEL_REPO_DIR}/snapshots/${MODEL_REVISION}"
  MODEL_MOUNT_SOURCE="${MODEL_REPO_DIR}"
  MODEL_MOUNT_TARGET=/model-repo
  MODEL_CONTAINER_DIR="/model-repo/snapshots/${MODEL_REVISION}"
fi
IMAGE="${IMAGE:-ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-glm53-flash-exl3-b12x-vllm}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL_ID}}"
PORT="${PORT:-8001}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
DECODE_CONTEXT_PARALLEL_SIZE="${DECODE_CONTEXT_PARALLEL_SIZE:-2}"
DCP_COMM_BACKEND="${DCP_COMM_BACKEND:-ag_rs}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MTP_TOKENS="${MTP_TOKENS:-5}"
ADAPTIVE_MTP="${ADAPTIVE_MTP:-1}"
ADAPTIVE_MTP_MIN_DEPTH="${ADAPTIVE_MTP_MIN_DEPTH:-1}"
MTP_BATCH_SCHEDULE="${MTP_BATCH_SCHEDULE:-}"
USE_REPLAYSSM="${USE_REPLAYSSM:-1}"
REPLAYSSM_BUFFER_LEN="${REPLAYSSM_BUFFER_LEN:-10}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-1}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-}"
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
    PROFILE_GPU_MEMORY_UTILIZATION=0.950
    PROFILE_MAX_MODEL_LEN=500000
    PROFILE_MAX_NUM_BATCHED_TOKENS=2048
    ;;
  fp8)
    PROFILE_KV_CACHE_DTYPE=fp8_ds_mla
    PROFILE_GPU_MEMORY_UTILIZATION=0.950
    PROFILE_MAX_MODEL_LEN=500000
    PROFILE_MAX_NUM_BATCHED_TOKENS=2048
    ;;
  *)
    echo "KV_CACHE_PROFILE must be nvfp4 or fp8; got: ${KV_CACHE_PROFILE}" >&2
    exit 2
    ;;
esac
MAX_MODEL_LEN="${MAX_MODEL_LEN:-${PROFILE_MAX_MODEL_LEN}}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-${PROFILE_MAX_NUM_BATCHED_TOKENS}}"
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
if [[ ! "${MTP_TOKENS}" =~ ^[0-9]+$ ]]; then
  echo "MTP_TOKENS must be a non-negative integer; got: ${MTP_TOKENS}" >&2
  exit 2
fi
if [[ "${ADAPTIVE_MTP}" != 0 && "${ADAPTIVE_MTP}" != 1 ]]; then
  echo "ADAPTIVE_MTP must be 0 or 1; got: ${ADAPTIVE_MTP}" >&2
  exit 2
fi
if [[ "${ADAPTIVE_MTP}" == 1 && "${MTP_TOKENS}" == 0 ]]; then
  echo "ADAPTIVE_MTP=1 requires MTP_TOKENS to be positive" >&2
  exit 2
fi
if [[ "${ADAPTIVE_MTP}" == 1 ]]; then
  if [[ ! "${ADAPTIVE_MTP_MIN_DEPTH}" =~ ^[0-9]+$ ]] || \
     (( ADAPTIVE_MTP_MIN_DEPTH > MTP_TOKENS )); then
    echo "ADAPTIVE_MTP_MIN_DEPTH must be between 0 and MTP_TOKENS; got: ${ADAPTIVE_MTP_MIN_DEPTH}" >&2
    exit 2
  fi
fi
if [[ "${ADAPTIVE_MTP}" == 1 && -z "${MTP_BATCH_SCHEDULE}" ]]; then
  # Include every K=0..5 in graph preparation. The schedule supplies only
  # initial priors; request-local feedback takes over after initialization,
  # and the production K1 floor clamps the high-load K0 prior by default.
  MTP_BATCH_SCHEDULE='[[1,1,5],[2,2,4],[3,3,3],[4,4,2],[5,8,1],[9,16,0]]'
fi
if [[ "${USE_REPLAYSSM}" != 0 && "${USE_REPLAYSSM}" != 1 ]]; then
  echo "USE_REPLAYSSM must be 0 or 1; got: ${USE_REPLAYSSM}" >&2
  exit 2
fi
if [[ ! "${REPLAYSSM_BUFFER_LEN}" =~ ^[1-9][0-9]*$ ]]; then
  echo "REPLAYSSM_BUFFER_LEN must be a positive integer; got: ${REPLAYSSM_BUFFER_LEN}" >&2
  exit 2
fi
if [[ ! "${MAX_NUM_SEQS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_NUM_SEQS must be a positive integer; got: ${MAX_NUM_SEQS}" >&2
  exit 2
fi
if [[ -z "${MAX_CUDAGRAPH_CAPTURE_SIZE}" ]]; then
  if [[ "${ADAPTIVE_MTP}" == 1 ]]; then
    # Capturing the full C16 x (K5 + target) shape costs enough graph memory
    # to push the 500K NVFP4 profile just below its advertised KV capacity.
    # K4/K5 at C16 is a losing regime on this hardware anyway; the controller
    # normally contracts it, and vLLM safely falls back above this ceiling.
    # 64 still captures C16/K3 and every C1..C8/K5 execution shape.
    MAX_CUDAGRAPH_CAPTURE_SIZE=64
  else
    MAX_CUDAGRAPH_CAPTURE_SIZE=$((MAX_NUM_SEQS * (MTP_TOKENS + 1)))
  fi
fi
if [[ ! "${MAX_CUDAGRAPH_CAPTURE_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_CUDAGRAPH_CAPTURE_SIZE must be a positive integer; got: ${MAX_CUDAGRAPH_CAPTURE_SIZE}" >&2
  exit 2
fi
if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  echo "Pinned HF snapshot is missing: ${MODEL_DIR}" >&2
  if [[ -z "${MODEL_DIR_OVERRIDE}" ]]; then
    echo "Run ${SCRIPT_DIR}/download.sh first." >&2
  fi
  exit 1
fi
python3 - "${MODEL_DIR}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
index_path = root / "model.safetensors.index.json"
try:
    index = json.loads(index_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"Invalid checkpoint index {index_path}: {exc}")
weight_map = index.get("weight_map")
if not isinstance(weight_map, dict) or not weight_map:
    raise SystemExit(f"Checkpoint index has no weight_map: {index_path}")
shards = sorted(set(weight_map.values()))
for name in shards:
    relative = pathlib.PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or not name:
        raise SystemExit(f"Unsafe shard path in checkpoint index: {name!r}")
    if not (root / relative).is_file():
        raise SystemExit(f"Checkpoint shard is missing: {root / relative}")
print(f"Validated {len(shards)} checkpoint shards")
PY
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

SPECULATIVE_ARGS=()
if (( MTP_TOKENS > 0 )); then
  SPECULATIVE_CONFIG="{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}"
  if [[ -n "${MTP_BATCH_SCHEDULE}" ]]; then
    SPECULATIVE_CONFIG="$(python3 -c '
import json, sys
tokens = int(sys.argv[1])
schedule = json.loads(sys.argv[2])
print(json.dumps({
    "method": "mtp",
    "num_speculative_tokens": tokens,
    "num_speculative_tokens_per_batch_size": schedule,
}, separators=(",", ":")))
' "${MTP_TOKENS}" "${MTP_BATCH_SCHEDULE}")"
  fi
  SPECULATIVE_ARGS+=(
    --speculative-config
    "${SPECULATIVE_CONFIG}"
  )
  if [[ "${USE_REPLAYSSM}" == 1 ]]; then
    SPECULATIVE_ARGS+=(
      --use-replayssm
      --replayssm-buffer-len "${REPLAYSSM_BUFFER_LEN}"
    )
  fi
fi

MODEL_MODE_ARGS=()
if [[ "${LANGUAGE_MODEL_ONLY}" == 1 ]]; then
  MODEL_MODE_ARGS+=(--language-model-only)
elif [[ "${LANGUAGE_MODEL_ONLY}" != 0 ]]; then
  echo "LANGUAGE_MODEL_ONLY must be 0 or 1; got: ${LANGUAGE_MODEL_ONLY}" >&2
  exit 2
fi

CACHE_MODE_ARGS=()
if [[ "${ENABLE_PREFIX_CACHING}" == 1 ]]; then
  CACHE_MODE_ARGS+=(--enable-prefix-caching --mamba-cache-mode align)
elif [[ "${ENABLE_PREFIX_CACHING}" == 0 ]]; then
  CACHE_MODE_ARGS+=(--no-enable-prefix-caching --mamba-cache-mode none)
else
  echo "ENABLE_PREFIX_CACHING must be 0 or 1; got: ${ENABLE_PREFIX_CACHING}" >&2
  exit 2
fi

# A normal HF snapshot needs its complete repository mount so blob symlinks
# resolve. MODEL_DIR_OVERRIDE mounts a self-contained local checkpoint directly.
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
  --env VLLM_ADAPTIVE_MTP="${ADAPTIVE_MTP}" \
  --env VLLM_ADAPTIVE_MTP_HISTORY="${VLLM_ADAPTIVE_MTP_HISTORY:-16}" \
  --env VLLM_ADAPTIVE_MTP_MIN_DEPTH="${ADAPTIVE_MTP_MIN_DEPTH}" \
  --env VLLM_ADAPTIVE_MTP_DECISION_WINDOW="${VLLM_ADAPTIVE_MTP_DECISION_WINDOW:-8}" \
  --env VLLM_ADAPTIVE_MTP_PROBE_INTERVAL="${VLLM_ADAPTIVE_MTP_PROBE_INTERVAL:-32}" \
  --env VLLM_ADAPTIVE_MTP_PROBE_INTERVAL_MAX="${VLLM_ADAPTIVE_MTP_PROBE_INTERVAL_MAX:-256}" \
  --env VLLM_ADAPTIVE_MTP_LOAD_THRESHOLDS="${VLLM_ADAPTIVE_MTP_LOAD_THRESHOLDS:-0.28,0.45,0.55,0.75,0.90}" \
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
  --env VLLM_EXL3_PREFILL_CAPACITY="${VLLM_EXL3_PREFILL_CAPACITY:-1024}" \
  --env B12X_EXL3_BF16_EPILOGUE="${B12X_EXL3_BF16_EPILOGUE:-1}" \
  --env B12X_EXL3_BF16_GEMV="${B12X_EXL3_BF16_GEMV:-1}" \
  --env VLLM_B12X_GLM_H64_QUERY_PROJ="${VLLM_B12X_GLM_H64_QUERY_PROJ:-auto}" \
  --env VLLM_USE_B12X_MHC="${VLLM_USE_B12X_MHC:-auto}" \
  --volume "${MODEL_MOUNT_SOURCE}:${MODEL_MOUNT_TARGET}:ro" \
  --volume "${CACHE_DIR}:/root/.cache" \
  "${IMAGE}" \
  "${MODEL_CONTAINER_DIR}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 0.0.0.0 \
  --port 8001 \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --decode-context-parallel-size "${DECODE_CONTEXT_PARALLEL_SIZE}" \
  --dcp-comm-backend "${DCP_COMM_BACKEND}" \
  "${CUSTOM_ALL_REDUCE_ARGS[@]}" \
  "${SPECULATIVE_ARGS[@]}" \
  "${MODEL_MODE_ARGS[@]}" \
  "${CACHE_MODE_ARGS[@]}" \
  --attention-backend "${ATTENTION_BACKEND}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-cudagraph-capture-size "${MAX_CUDAGRAPH_CAPTURE_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --no-enable-flashinfer-autotune \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45

printf 'Started %s on http://127.0.0.1:%s/v1. Initial B12x/CuTe compilation can take several minutes.\n' \
  "${CONTAINER_NAME}" "${PORT}"
printf 'Profile: %s, KV cache: %s, max length: %s, max sequences: %s, GPU memory utilization: %s\n' \
  "${KV_CACHE_PROFILE}" "${KV_CACHE_DTYPE}" "${MAX_MODEL_LEN}" "${MAX_NUM_SEQS}" \
  "${GPU_MEMORY_UTILIZATION}"
if (( MTP_TOKENS > 0 )); then
  if [[ "${USE_REPLAYSSM}" == 1 ]]; then
    printf 'MTP: %s tokens with compact KDA ReplaySSM (buffer %s)\n' \
      "${MTP_TOKENS}" "${REPLAYSSM_BUFFER_LEN}"
  else
    printf 'MTP: %s tokens with baseline full-state rollback\n' "${MTP_TOKENS}"
  fi
  if [[ "${ADAPTIVE_MTP}" == 1 ]]; then
    printf 'MTP policy: request-local K=%s..%s estimates; arithmetic-mean batch K\n' \
      "${ADAPTIVE_MTP_MIN_DEPTH}" \
      "${MTP_TOKENS}"
  elif [[ -n "${MTP_BATCH_SCHEDULE}" ]]; then
    printf 'MTP policy: batch schedule %s\n' "${MTP_BATCH_SCHEDULE}"
  else
    printf 'MTP policy: static K=%s\n' "${MTP_TOKENS}"
  fi
else
  printf 'MTP: disabled\n'
fi
printf 'Follow startup: docker logs -f %s\n' "${CONTAINER_NAME}"
