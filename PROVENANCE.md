# Provenance

This image intentionally composes pinned artifacts rather than pretending one upstream tree already contains every required path.

| Component | Immutable source |
|---|---|
| Model | `brandonmusic/GLM-5.3-Flash-EXL3-4bpw@4739eb1bcfd478e8a32da6358908567bc3a9ac51` |
| GLM/vLLM base | `cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1@sha256:0bd709e80b8ff13ae5de8f7d7f708a499fade3a26970d56afb1be2ff3860fde5` |
| vLLM in base | `0.1.dev20051+g487ecf187` |
| EXL3 source image | `ghcr.io/tpurtell/deepseek-v4-flash-0731-exl3-k2-spark@sha256:86c8c1054f9c24454949e37031ce6165c007963aa0c0ef30fa884f6d4170af32` |
| EXL3 vLLM fork commit | `30038602b71395f481ef4a6edfe4fcf8551d9c15` |
| B12x fork | `tpurtell/sparkinfer-glmrt@988246c8b007c9c1c2006eb677f6fa4b26aeb561` |
| ReplaySSM base | vLLM PRs `#48792`, `#49847`, and `#49887`, ported onto the pinned vLLM commit |
| Runtime stack | Torch 2.13, CUDA 13, CUTLASS DSL 4.6.2 |

The Docker build copies only the qualified EXL3/vLLM adapter files from the source image, fetches the pinned B12x fork, applies the upstream ReplaySSM series, then layers the narrow GLM/EXL3/B12x ports in `patches/`. The local ReplaySSM extension adds GLM's vector-gated KDA recurrence, power-of-two speculative ring sizing, non-contiguous projection strides, aligned-prefix state materialization, and GLM MTP cache-group scoping. A build-time probe checks model recognition, EXL3 registration, ReplaySSM sizing/imports, B12x APIs, cache layouts, head geometry, and exact runtime versions.

## Upstream audit

The release was checked against vLLM's open GLM-5.3 support PR and its commits through `142062f13d16`, the open KDA host-sync fix, the open DCP correctness work, and the active compact-state/speculative-decoding work on 2026-08-28. Upstream's generic compact state-pool PR was still open/conflicted and disabled Mamba prefix caching; the GDN/Mamba2 speculative ReplaySSM PRs did not implement GLM's vector-gated KDA. This recipe ports that work instead of waiting, while retaining aligned prefix caching. Later GLM PR commits were merge-regression, ROCm, or multimodal changes; no newer CUDA NoPE sparse-MLA implementation superseded this B12x path.

Useful references:

- https://github.com/vllm-project/vllm/pull/53906
- https://github.com/vllm-project/vllm/pull/51540
- https://github.com/vllm-project/vllm/pull/50005
- https://github.com/vllm-project/vllm/issues/53963
- https://github.com/vllm-project/vllm/pull/37429
- https://github.com/vllm-project/vllm/pull/48792
- https://github.com/vllm-project/vllm/pull/49847
- https://github.com/vllm-project/vllm/pull/49887
- https://github.com/vllm-project/vllm/issues/46187
- https://github.com/vllm-project/vllm/issues/47572
- https://github.com/MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark

The Mia recipe targets two networked SM121 DGX Sparks. This recipe borrows relevant SM12x stability lessons but uses local PCIe collectives for two workstation GPUs.
