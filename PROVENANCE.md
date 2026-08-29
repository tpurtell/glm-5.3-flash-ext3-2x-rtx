# Provenance

This recipe consumes finished Hugging Face target and draft checkpoints and composes pinned runtime artifacts. It contains no calibration corpus, quantization runner, writer checkpoint, or intermediate quant files.

## Immutable inputs

| Component | Immutable source |
|---|---|
| Served target | `wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3-v1@319d66a8b53092b491f698440ecea781e4ddd4e4` |
| Target source | `zai-org/GLM-5.3-Flash-BF16@f12e0fe1f6b2ea274c11a569582edfd99d993c5e` |
| DFlash2 draft | `incoai/GLM-5.3-Flash-DFlash2@dc77ff1c99eeb2df044ee3d4f0094eb033fee410` |
| GPTQModel writer fork | `tpurtell/GPTQModel@a64900815b30ef01c2221b2788701a7986e50491` |
| GLM/vLLM base | `cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1@sha256:0bd709e80b8ff13ae5de8f7d7f708a499fade3a26970d56afb1be2ff3860fde5` |
| vLLM in base | `0.1.dev20051+g487ecf187` |
| vLLM DFlash2 delta | `vllm-project/vllm@b389ac29465b33f9e9c534df221ea3c129e9793f` (PR `#52816`) |
| EXL3 runtime source image | `ghcr.io/tpurtell/deepseek-v4-flash-0731-exl3-k2-spark@sha256:86c8c1054f9c24454949e37031ce6165c007963aa0c0ef30fa884f6d4170af32` |
| EXL3 vLLM fork commit | `30038602b71395f481ef4a6edfe4fcf8551d9c15` |
| B12x fork | `tpurtell/sparkinfer-glmrt@988246c8b007c9c1c2006eb677f6fa4b26aeb561` |
| ReplaySSM base | vLLM PRs `#48792`, `#49847`, and `#49887`, ported onto the pinned vLLM commit |
| Dynamic-MTP graph fix | vLLM PR `#49652`, ported onto the pinned vLLM commit |
| Runtime stack | Torch 2.13, CUDA 13, CUTLASS DSL 4.6.2 |

The DFlash2 checkpoint is a 1B-parameter BF16 draft model and is not a standalone language model. Inco AI publishes it under CC BY-NC-ND 4.0 for research and evaluation; commercial use requires separate licensing. The target model and base-image licenses also apply independently of this Apache-2.0 recipe.

## Target identity and metadata repair

Revision `319d66a…` is public, ungated, and contains the same 16 target weight shards as the previously qualified compact-config revision. It adds/restores the official multimodal processor, tokenizer configuration, generation configuration, and Z.ai chat template required for image inputs. No quant weights were downloaded or regenerated to perform that repair; the Hugging Face cache was relinked to the already present immutable blobs.

Release metadata hashes:

| File | SHA-256 |
|---|---|
| `config.json` | `e445e0443e7fce59943297323ff388e72c973f0e0a0f43b881be5ba55765e572` |
| `chat_template.jinja` | `34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891` |
| `processor_config.json` | `aae38374c94b08cc9b0547c6e64f05b951bd9735cea571c6988f5ed552bed3ed` |
| `tokenizer_config.json` | `98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc` |
| `generation_config.json` | `230c30609ecbbb9e6583bedde8e7bdda0c6eb8fe5fad0eaeb3d1b293d751cb4f` |

The compact `config.json` deliberately does not embed the duplicate 32.8 MB EXL3 `tensor_storage` map. The complete tensor map remains in the external quantization manifests. The launcher validates the index, every referenced shard, the EXL3 manifest, official processor metadata, and DFlash2 architecture before starting Docker.

The GPTQModel commit is listed only to establish how the public target was written. Its mixed-checkpoint saver materializes EXL3 shells while streaming native tensors under their original keys, avoiding duplicate BF16 materialization. GPTQModel does not ship in the serving image.

## Runtime composition

The Docker build copies only the qualified EXL3 loader/adapter files from the source image, fetches the pinned B12x fork, applies the existing ReplaySSM/dynamic-MTP series, ports the immutable upstream DFlash2 commit, then applies the narrow GLM/DFlash/DCP corrections under `patches/`.

The DFlash2 release work adds:

- GLM-5.3 target support for EAGLE3 hidden-state taps after mHC and the decoder-layer indirection expected by the upstream DFlash implementation.
- An independent hybrid KV group for the dense DFlash draft rather than forcing it into the target model's MLA/recurrent cache groups.
- Per-attention DCP scope: target attention remains DCP2, while the draft model, draft metadata, draft forward state, and draft CUDA graphs are replicated DCP1 on both ranks.
- A 128-token allocation block for the replicated DFlash sliding cache instead of the inherited 16-token page, reducing long-request shared-pool block-ID overhead.
- Draft-scratch exclusion from target prefix hashes and compatibility logic for the replicated draft sliding group.

The pre-existing local work remains available:

- GLM K3 routed-expert/MTP mappings for the mixed EXL3 checkpoint.
- Vector-gated KDA ReplaySSM, compact state accounting, and request-lifetime adaptive MTP as an alternate speculative method.
- The bounded EXL3 prefill arena and GLM-specific B12x sparse-MLA, K-pool, DCP2, PCIe collective, query-projection, mHC, NVFP4, and FP8 cache ports.

Build-time probes check target/draft architecture recognition, the DFlash2 V2 speculator, GLM EAGLE3 support, EXL3 registration, ReplaySSM/adaptive policy imports, B12x APIs, compact cache layouts, head geometry, and exact runtime versions.

## Qualification evidence

The published target revision was exercised through the normal launcher rather than a local override. Release gates include:

- DFlash2 K3/K5/K7 tuning and a final K5 code-agent C1–C16 curve.
- The seven-type GLMRT content blend with 21/21 semantic contracts passing.
- FP8 prefill measurements at 2K, 8K, 32K, 64K, and 128K existing depth.
- A 16-image exact-order test plus server-side rejection of image 17.
- One cold exact 1,000,000-token request with six successful needles from 50K through 990K.

Raw machine-readable artifacts live under `benchmarks/`; [RESULTS.md](benchmarks/RESULTS.md) defines their timing and scoring methods.

## Upstream and nearby references

- https://github.com/vllm-project/vllm/pull/52816
- https://github.com/vllm-project/vllm/commit/b389ac29465b33f9e9c534df221ea3c129e9793f
- https://github.com/vllm-project/vllm/pull/53906
- https://github.com/vllm-project/vllm/pull/51540
- https://github.com/vllm-project/vllm/pull/50005
- https://github.com/vllm-project/vllm/issues/53963
- https://github.com/vllm-project/vllm/pull/48792
- https://github.com/vllm-project/vllm/pull/49847
- https://github.com/vllm-project/vllm/pull/49887
- https://github.com/vllm-project/vllm/pull/49652
- https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2
- https://github.com/MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark
- https://github.com/samuelcardillo/glm-5.3-flash-2x-rtx-pro-6000-blackwell

The Mia recipe targets two networked SM121 DGX Sparks; this recipe borrows relevant SM12x stability lessons but qualifies local PCIe collectives on two workstation GPUs. The Samuel Cardillo recipe helped flag DFlash as the likely source of the reported high decode result, which this release then tested directly on the EXL3 target.
