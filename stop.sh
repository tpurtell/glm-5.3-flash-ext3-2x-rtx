#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-glm53-flash-exl3-b12x-vllm}"
STOP_TIMEOUT="${STOP_TIMEOUT:-60}"

if ! docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  printf 'Container does not exist: %s\n' "${CONTAINER_NAME}"
  exit 0
fi
if [[ "$(docker inspect --format '{{.State.Running}}' "${CONTAINER_NAME}")" != true ]]; then
  printf 'Container is already stopped: %s\n' "${CONTAINER_NAME}"
  exit 0
fi
docker stop --time "${STOP_TIMEOUT}" "${CONTAINER_NAME}" >/dev/null
printf 'Stopped %s (container retained; start.sh replaces it on the next launch).\n' \
  "${CONTAINER_NAME}"
