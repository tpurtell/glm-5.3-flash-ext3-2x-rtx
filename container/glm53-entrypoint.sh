#!/usr/bin/env bash
set -euo pipefail

READY_FILE=/tmp/glm53-release-ready
rm -f "${READY_FILE}"

vllm serve "$@" &
server_pid=$!

forward_term() {
  kill -TERM "${server_pid}" 2>/dev/null || true
}
trap forward_term TERM INT

if [[ "${GLM53_STARTUP_WARMUP:-1}" == 1 ]]; then
  set +e
  python3 /usr/local/bin/glm53-release-warmup.py \
    --server-pid "${server_pid}" \
    --base-url http://127.0.0.1:8001
  warmup_status=$?
  set -e
  if [[ ${warmup_status} -ne 0 ]]; then
    printf 'GLM release startup warmup failed with status %s\n' \
      "${warmup_status}" >&2
    forward_term
    wait "${server_pid}" 2>/dev/null || true
    exit "${warmup_status}"
  fi
elif [[ "${GLM53_STARTUP_WARMUP}" != 0 ]]; then
  echo "GLM53_STARTUP_WARMUP must be 0 or 1" >&2
  forward_term
  wait "${server_pid}" 2>/dev/null || true
  exit 2
fi

touch "${READY_FILE}"
printf 'GLM release startup warmup complete; container is ready.\n'

set +e
wait "${server_pid}"
server_status=$?
set -e
exit "${server_status}"
