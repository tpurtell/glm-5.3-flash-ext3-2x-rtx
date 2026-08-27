#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx:local}"
VCS_REVISION="${VCS_REVISION:-$(git -C "${SCRIPT_DIR}" rev-parse HEAD 2>/dev/null || printf unknown)}"

docker build --progress=plain \
  --label "org.opencontainers.image.revision=${VCS_REVISION}" \
  --tag "${IMAGE}" \
  "${SCRIPT_DIR}"
docker image inspect "${IMAGE}" --format 'Built {{.Id}} ({{.Size}} bytes)'
