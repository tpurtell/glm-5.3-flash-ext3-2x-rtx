#!/usr/bin/env bash

# Resolve one immutable target checkpoint for both download.sh and start.sh.
# Callers may either choose a named profile or override MODEL_ID and
# MODEL_REVISION together. Keeping the pair atomic prevents a repository
# override from accidentally inheriting another checkpoint's revision.
resolve_glm53_model_profile() {
  local model_id_set=0
  local model_revision_set=0
  [[ -v MODEL_ID ]] && model_id_set=1
  [[ -v MODEL_REVISION ]] && model_revision_set=1

  if (( model_id_set != model_revision_set )); then
    echo "MODEL_ID and MODEL_REVISION must be overridden together." >&2
    return 2
  fi

  if (( model_id_set )); then
    if [[ -z "${MODEL_ID}" || -z "${MODEL_REVISION}" ]]; then
      echo "MODEL_ID and MODEL_REVISION overrides must be non-empty." >&2
      return 2
    fi
    MODEL_PROFILE="${MODEL_PROFILE:-custom}"
    return
  fi

  MODEL_PROFILE="${MODEL_PROFILE:-k3}"
  case "${MODEL_PROFILE}" in
    k3)
      MODEL_ID=wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3-v1
      MODEL_REVISION=1e4abd26e4e1e8d58d81fbd557d6c4099352fe63
      ;;
    k4)
      MODEL_ID=brandonmusic/GLM-5.3-Flash-tr3-4bpw
      MODEL_REVISION=aba59d2175e1ee2887ae0ae1300ba848b1deed84
      ;;
    custom)
      echo "MODEL_PROFILE=custom requires MODEL_ID and MODEL_REVISION." >&2
      return 2
      ;;
    *)
      echo "MODEL_PROFILE must be k3, k4, or custom; got: ${MODEL_PROFILE}" >&2
      return 2
      ;;
  esac
}
