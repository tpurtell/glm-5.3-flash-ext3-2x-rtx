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
| Runtime stack | Torch 2.13, CUDA 13, CUTLASS DSL 4.6.2 |

The Docker build copies only the qualified EXL3/vLLM adapter files from the source image, fetches the pinned B12x fork, then applies the narrow port scripts in `patches/`. A build-time probe checks model recognition, EXL3 registration, B12x APIs, cache layouts, head geometry, and exact runtime versions.

## Upstream audit

The release was checked against vLLM's open GLM-5.3 support PR and its commits through `142062f13d16`, the open KDA host-sync fix, and the open DCP correctness work on 2026-08-28. Later GLM PR commits were merge-regression, ROCm, or multimodal changes; no newer CUDA NoPE sparse-MLA implementation superseded this B12x path. The local port includes the applicable host-chunk and DCP correctness repairs plus bounds/tail/SM12 stability fixes.

Useful references:

- https://github.com/vllm-project/vllm/pull/53906
- https://github.com/vllm-project/vllm/pull/51540
- https://github.com/vllm-project/vllm/pull/50005
- https://github.com/vllm-project/vllm/issues/53963
- https://github.com/MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark

The Mia recipe targets two networked SM121 DGX Sparks. This recipe borrows relevant SM12x stability lessons but uses local PCIe collectives for two workstation GPUs.
