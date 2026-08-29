# Provenance

This recipe consumes a finished Hugging Face model and composes pinned runtime artifacts. It contains no calibration corpus, quantization runner, writer checkpoints, or intermediate quant files.

## Immutable inputs

| Component | Immutable source |
|---|---|
| Served model | `wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3-v1@8c02fb69a03ef86f0d1f9f0b607002c46102538c` |
| Model source | `zai-org/GLM-5.3-Flash-BF16@f12e0fe1f6b2ea274c11a569582edfd99d993c5e` |
| GPTQModel writer fork | `tpurtell/GPTQModel@a64900815b30ef01c2221b2788701a7986e50491` |
| GLM/vLLM base | `cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1@sha256:0bd709e80b8ff13ae5de8f7d7f708a499fade3a26970d56afb1be2ff3860fde5` |
| vLLM in base | `0.1.dev20051+g487ecf187` |
| EXL3 runtime source image | `ghcr.io/tpurtell/deepseek-v4-flash-0731-exl3-k2-spark@sha256:86c8c1054f9c24454949e37031ce6165c007963aa0c0ef30fa884f6d4170af32` |
| EXL3 vLLM fork commit | `30038602b71395f481ef4a6edfe4fcf8551d9c15` |
| B12x fork | `tpurtell/sparkinfer-glmrt@988246c8b007c9c1c2006eb677f6fa4b26aeb561` |
| ReplaySSM base | vLLM PRs `#48792`, `#49847`, and `#49887`, ported onto the pinned vLLM commit |
| Dynamic-MTP graph fix | vLLM PR `#49652`, ported onto the pinned vLLM commit |
| Runtime stack | Torch 2.13, CUDA 13, CUTLASS DSL 4.6.2 |

## Model identity

The public model repository contains 33 files and 137,128,503,398 bytes at the pinned revision. Its model index SHA-256 is `d751549235ef63d1954be328754e001c8e488795ed4c2ef6d5b0e4a2dc08f0dc`; the complete local-file census SHA-256 is `5685313044e9d2d9e108f482b8788a2ca808f2820f1e0846504a4de6ed3be471`; and the final release-validation SHA-256 is `625dcdfc8a031506d1406804cc72b7373c600567e076220b756f9504bc0fd284`.

Revision `8c02fb6…` is a config-only repair over the originally qualified
`4967efc…` revision. It removes the duplicate 32.8 MB EXL3 `tensor_storage`
map from `config.json`; the complete map remains in the external quantization
manifests, and all 16 weight shards are unchanged.

Before upload, that exact model index passed ordinary vLLM generation and five representative tool-call scenarios in the same runtime image used for qualification. After upload, the public revision was verified file-for-file. The full 69-case evaluation in this repository then exercised the published model ID through the final adaptive-MTP launcher.

The GPTQModel commit is listed only to establish how the published artifact was written. Its mixed-checkpoint saver materializes EXL3 shells while streaming native tensors under their original keys, avoiding the accidental duplicate BF16 materialization found during export. That implementation does not ship in this serving image.

## Runtime composition

The Docker build copies only the qualified EXL3 loader/adapter files from the source image, fetches the pinned B12x fork, applies the upstream ReplaySSM series and dynamic-MTP graph fix, and then layers the narrow GLM/EXL3/B12x ports under `patches/`.

The local runtime work adds:

- GLM K3 routed-expert and MTP mappings for the mixed checkpoint.
- Vector-gated KDA ReplaySSM, physical power-of-two ring sizing, explicit non-contiguous projection strides, aligned-prefix materialization, and target/draft cache-group scoping.
- Request-lifetime adaptive-MTP feedback with arithmetic-mean K selection for concurrent fused batches.
- ReplaySSM mixed prefill/decode CUDA-graph exclusion on top of the generic dynamic-MTP graph fix.
- The bounded EXL3 prefill arena and GLM-specific B12x sparse-MLA, K-pool, DCP2, PCIe collective, query-projection, mHC, NVFP4, and FP8 cache ports.

Build-time probes check GLM architecture recognition, K3 EXL3 registration, MTP mapping, ReplaySSM sizing/imports, adaptive policy imports, B12x APIs, compact cache layouts, head geometry, and exact runtime versions.

## Upstream audit

The release was checked against vLLM's open GLM-5.3 support PR and recent commits, the open KDA host-sync fix, DCP correctness work, compact state-pool work, and active speculative-decoding/ReplaySSM work through 2026-08-29. The generic compact state-pool work still did not provide GLM's vector-gated KDA path while retaining aligned prefix caching, so this pinned recipe carries the qualified port.

Useful references:

- https://github.com/vllm-project/vllm/pull/53906
- https://github.com/vllm-project/vllm/pull/51540
- https://github.com/vllm-project/vllm/pull/50005
- https://github.com/vllm-project/vllm/issues/53963
- https://github.com/vllm-project/vllm/pull/37429
- https://github.com/vllm-project/vllm/pull/48792
- https://github.com/vllm-project/vllm/pull/49847
- https://github.com/vllm-project/vllm/pull/49887
- https://github.com/vllm-project/vllm/pull/49652
- https://github.com/vllm-project/vllm/issues/51303
- https://github.com/vllm-project/vllm/issues/46295
- https://github.com/vllm-project/vllm/issues/48627
- https://github.com/vllm-project/vllm/issues/47602
- https://github.com/vllm-project/vllm/issues/48494
- https://github.com/vllm-project/vllm/issues/46187
- https://github.com/vllm-project/vllm/issues/47572
- https://github.com/MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark

The Mia recipe targets two networked SM121 DGX Sparks. This recipe borrows relevant SM12x stability lessons but qualifies local PCIe collectives for two workstation GPUs.
