#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx:local}"
VCS_REVISION="${VCS_REVISION:-$(git -C "${SCRIPT_DIR}" rev-parse HEAD 2>/dev/null || printf unknown)}"
SOURCE_URL="${SOURCE_URL:-https://github.com/tpurtell/glm-5.3-flash-ext3-4-bit-2x-rtx}"
IMAGE_VERSION="${IMAGE_VERSION:-${IMAGE##*:}}"

docker build --progress=plain \
  --label "org.opencontainers.image.revision=${VCS_REVISION}" \
  --label "org.opencontainers.image.source=${SOURCE_URL}" \
  --label "org.opencontainers.image.version=${IMAGE_VERSION}" \
  --label "org.opencontainers.image.title=GLM-5.3 Flash EXL3 4bpw for 2x RTX PRO 6000" \
  --label "org.opencontainers.image.description=vLLM GLM-5.3 Flash EXL3 with B12x, DCP2, compact MLA, and adaptive MTP" \
  --tag "${IMAGE}" \
  "${SCRIPT_DIR}"
docker image inspect "${IMAGE}" --format 'Built {{.Id}} ({{.Size}} bytes)'
