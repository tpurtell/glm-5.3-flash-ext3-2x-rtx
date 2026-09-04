# Provenance

This recipe consumes finished Hugging Face target and draft checkpoints and composes pinned runtime artifacts. It contains no calibration corpus, quantization runner, writer checkpoint, or intermediate quant files.

## Immutable inputs

| Component | Immutable source |
|---|---|
| Served target | `wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3-v1@1e4abd26e4e1e8d58d81fbd557d6c4099352fe63` |
| Supported uniform-K4 target | `brandonmusic/GLM-5.3-Flash-tr3-4bpw@aba59d2175e1ee2887ae0ae1300ba848b1deed84` |
| Target source | `zai-org/GLM-5.3-Flash-BF16@f12e0fe1f6b2ea274c11a569582edfd99d993c5e` |
| Corrected chat-template source | `zai-org/GLM-5.3-Flash-BF16@a5b45eb41df6402735dedc900be14a42e8d5e538` |
| DFlash2 draft | `incoai/GLM-5.3-Flash-DFlash2@bf582e4eacc1810f76656d1811693ff6c6737d2a` |
| GPTQModel quant writer | `tpurtell/GPTQModel@0565af7ce20a93df9bbc0e5563d7c6f60916f41a` |
| GPTQModel compact-config follow-up | `tpurtell/GPTQModel@a64900815b30ef01c2221b2788701a7986e50491` |
| K3.25 GPTQModel writer | `tpurtell/GPTQModel@a053382584fa58cba7bf212ef1b829d08b29b2c0` |
| GLM/vLLM base | `cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1@sha256:0bd709e80b8ff13ae5de8f7d7f708a499fade3a26970d56afb1be2ff3860fde5` |
| vLLM in base | `0.1.dev20051+g487ecf187` |
| vLLM DFlash2 delta | `vllm-project/vllm@b389ac29465b33f9e9c534df221ea3c129e9793f` (PR `#52816`) |
| EXL3 runtime source image | `ghcr.io/tpurtell/deepseek-v4-flash-0731-exl3-k2-spark@sha256:86c8c1054f9c24454949e37031ce6165c007963aa0c0ef30fa884f6d4170af32` |
| EXL3 vLLM fork commit | `30038602b71395f481ef4a6edfe4fcf8551d9c15` |
| B12x fork | `tpurtell/sparkinfer-glmrt@c90cd0080274b90da4721f8ab7536bca29cae720` |
| ReplaySSM base | vLLM PRs `#48792`, `#49847`, and `#49887`, ported onto the pinned vLLM commit |
| Dynamic-MTP graph fix | vLLM PR `#49652`, ported onto the pinned vLLM commit |
| Runtime stack | Torch 2.13, CUDA 13, CUTLASS DSL 4.6.2 |

## Published release artifacts

The public `v0.6.0` and `latest` container tags resolve to the same immutable
OCI index:

| Artifact | Immutable identity |
|---|---|
| Recipe image | `ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx:v0.6.0@sha256:fe249b88d091430d8a88cd987d087d556053f0f067a649f2e9ca95895129e82b` |
| Linux/amd64 manifest | `sha256:4858ebdbc1e313e7024242fc246c90ceb0754fec2f46a0b997c02a985f2d75f7` |
| Image config | `sha256:710550e1b5a6acbccc76010c4b33eb3776c326ac572385f588916c93046b849c` |
| Build/source revision | `d6fbe22f2c825974b4276cc308937b5b7e5f7fbe` |

An anonymous GHCR pull resolved that index successfully. Its OCI labels pin
the source revision above, B12x `611ffe8…`, DFlash2/vLLM `b389ac2…`, and
version `v0.6.0`. The Git release tag includes this post-build provenance and
clean-start evidence; the serving code in that tag is unchanged from the
embedded image source revision.

The DFlash2 checkpoint is a 1B-parameter BF16 draft model and is not a standalone language model. Inco AI publishes it under CC BY-NC-ND 4.0 for research and evaluation; commercial use requires separate licensing. The target model and base-image licenses also apply independently of this Apache-2.0 recipe.

## Target identity and metadata repair

Revision `1e4abd2…` is public and ungated. It descends from `319d66a…`, which
restored the official multimodal processor, tokenizer configuration, generation
configuration, and Z.ai chat template required for image inputs. The newer
revision changes only `chat_template.jinja`, copying the exact corrected file
from Z.ai revision `a5b45eb…`. All other path, Git object, size, and LFS
identities—including all 16 quant shards—are unchanged from the prior public
head. No quant weights were downloaded or regenerated; the local HF cache made
a new immutable snapshot by relinking the existing shard blobs.

Release metadata hashes:

| File | SHA-256 |
|---|---|
| `config.json` | `e445e0443e7fce59943297323ff388e72c973f0e0a0f43b881be5ba55765e572` |
| `chat_template.jinja` | `0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5` |
| `processor_config.json` | `aae38374c94b08cc9b0547c6e64f05b951bd9735cea571c6988f5ed552bed3ed` |
| `tokenizer_config.json` | `98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc` |
| `generation_config.json` | `230c30609ecbbb9e6583bedde8e7bdda0c6eb8fe5fad0eaeb3d1b293d751cb4f` |

The compact `config.json` deliberately does not embed the duplicate 32.8 MB EXL3 `tensor_storage` map. The complete tensor map remains in the external quantization manifests. The launcher validates the index, every referenced shard, the EXL3 manifest, official processor metadata, and DFlash2 architecture before starting Docker.

The unchanged compact config is 10,951 bytes, parses as ordinary JSON and
through `PretrainedConfig.get_config_dict`, and resolves through the release
vLLM path as `Glm5NextConfig` / `Glm5NextForConditionalGeneration`. On
2026-09-04, the public Hugging Face API reported head `1e4abd2…`,
`model_type=glm5_next`, `pipeline_tag=image-text-to-text`, and no model-card
parse error. The corrected template compiled and rendered through the release
container's `AutoProcessor`, covering ordinary chat, null assistant content,
out-of-order tool results, and invalid-ID fallback. The complete local snapshot
passed `hf cache verify` for 35 files. Machine-readable checks are the original
[`release config receipt`](benchmarks/v0.6.0-b12x-ep2/release-hf-config-parse.json)
and the [`template-sync receipt`](benchmarks/chat-template-sync-20260904.json).

The GPTQModel commits are listed only to establish how the public target was
written and packaged. Quantization ran at `0565af7…`; `a649008…` later kept the
large EXL3 tensor map in the external manifest rather than duplicating it into
`config.json`. The mixed-checkpoint saver materializes EXL3 shells while
streaming native tensors under their original keys, avoiding duplicate BF16
materialization. GPTQModel does not ship in the serving image.

## Runtime composition

The Docker build copies only the qualified EXL3 loader/adapter files from the source image, fetches the pinned B12x fork, applies the existing ReplaySSM/dynamic-MTP series, ports the immutable upstream DFlash2 commit, then applies the narrow GLM/DFlash/DCP corrections under `patches/`.

The DFlash2 release work adds:

- GLM-5.3 target support for EAGLE3 hidden-state taps after mHC and the decoder-layer indirection expected by the upstream DFlash implementation.
- An independent hybrid KV group for the dense DFlash draft rather than forcing it into the target model's MLA/recurrent cache groups.
- Per-attention DCP scope: target attention remains DCP2, while the draft model, draft metadata, draft forward state, and draft CUDA graphs are replicated DCP1 on both ranks.
- A 128-token allocation block for the replicated DFlash sliding cache instead of the inherited 16-token page, reducing long-request shared-pool block-ID overhead.
- Draft-scratch exclusion from target prefix hashes and compatibility logic for the replicated draft sliding group.

The v0.6 B12x/EP2 work adds:

- EXL3 global-to-local weight loading for GLM's 288 global experts and 144
  local experts per EP2 rank. Non-local checkpoint weights are skipped before
  materialization; global top-8 route IDs and weights remain ordered.
- B12x `ep_moe` plan/bind/run integration with fixed decode and prefill
  workspaces, replicated input, and vLLM-owned final TP reduction.
- The public B12x full-rotation MCG Trellis EP fix at `611ffe8`, including
  graph-stable FP16 rotation, route, output, and barrier arenas.
- Migration from the retired `nsa_indexer` API name to the current B12x
  `dsa_indexer` contract and current RTX PRO 6000 Blackwell policy data.
- GLM mHC shape admission during engine warmup plus a container entrypoint
  that gates health on raw-greedy C1, rendered-chat C1, long-prefill, and four
  sampled C16 passes. This moves every observed route, DFlash, sampler,
  sparse-indexer, and mHC first-use compilation before the ready marker.

The K3.25 runtime update adds a second, projection-native EXL3 path without
replacing the qualified uniform-K3 path:

- It derives each expert's integral gate/up/down K value from the external
  EXL3 tensor manifest. The checkpoint-level `3.25` value is descriptive;
  executable projections remain exactly K3 or K4.
- It prepares B12x `ProjectionTrellisTierWeights` using the live-shape
  `trellis_t256_proj` selector. Uniform K3 and K4 checkpoints continue through
  the fixed `b12x_trellis` layout and keep their existing kernel policy.
- Under EP2, vLLM composes the immutable 288-global-to-144-local expert map
  with B12x's local-expert-to-tier map once while loading each layer. Decode
  and prefill then bind one precomputed global-route-to-tier map with no live
  allocation or route reordering.
- B12x `c90cd008...` admits that larger route namespace, preserves the
  projection-mixed live tile selector, and fails closed on route-map dtype,
  extent, device, and contiguity drift. Its focused planning suite passed
  69 tests with two device-only skips; the neighboring Trellis/TP scratch
  suites passed 51 tests with 25 device-only skips.

Before GPU qualification, the composed image passed its build probe, the
projection port's complete idempotence pass, and a full-manifest regression
gate over the published K3 model: all 37,152 routed projections across layers
3--45 remained uniform K3 and selected the fixed-tier metadata path. These are
source/metadata gates, not substitutes for the required K3.25 GPU performance
and numerical qualification.

The pre-existing local work remains available:

- GLM K3 routed-expert/MTP mappings for the mixed EXL3 checkpoint.
- Vector-gated KDA ReplaySSM, compact state accounting, and request-lifetime adaptive MTP as an alternate speculative method.
- The bounded EXL3 prefill arena and GLM-specific B12x sparse-MLA, K-pool, DCP2, PCIe collective, query-projection, mHC, NVFP4, and FP8 cache ports.

Build-time probes check target/draft architecture recognition, the DFlash2 V2 speculator, GLM EAGLE3 support, EXL3 registration, uniform and per-projection Trellis plans, the EP route namespace, ReplaySSM/adaptive policy imports, B12x APIs, compact cache layouts, head geometry, and exact runtime versions.

Current recipe launcher/build hashes. The v0.6.1 target-metadata patch and the
subsequent DFlash2 checkpoint refresh change only host-side model revision
pins; neither requires a new serving image:

| File | SHA-256 |
|---|---|
| `Dockerfile` | `e99e89491f4b7a918c632b5737e275fb8aa1944dd51d3c4d618d205d5d3e791b` |
| `build.sh` | `853a2a90245bc599ab01277246362dcdb9c789875ef564fbdc83d5b1a753b7cd` |
| `download.sh` | `21fb60b08ed01342e0fa5a3077ca3e3a3ffd26336f629373127ddb00afc88ec7` |
| `start.sh` | `d1d697e7bfbb0915a2205af861135295c7e43db4753aee448972346c9a3c305e` |
| `container/glm53-entrypoint.sh` | `c1de8b073277b8edfb5c85c7c8d83ee511593d77d271c26a7ddf4d1e1c5abc8d` |
| `container/glm53-release-warmup.py` | `7324b284a838063f18d29c40dda5c2e329b3133493b76bb518006aec534c0b9d` |

Changed/new serving patches in v0.6:

| File | SHA-256 |
|---|---|
| `port-exl3-ep-glm53.py` | `301e3561ba9b571c1033e3cb756255043a9960be06e2d7594e62ca0a69916ff2` |
| `port-exl3-glm53.py` | `0bf64f04c247e63438401f4976dd3e53251196b5bd004f730c1bf5c6ad84e153` |
| `port-b12x-glm53.py` | `8000e531342bec42bda592c9b2d4dbec5b82891ce9ba31c75c5bf9cea19017a5` |
| `port-b12x-dcp-owner-glm53.py` | `0f5cfb1abd157550f296720a05abd36ce83f9600a0cebfba1ee5bbf90e7b8bb7` |
| `port-glm53-mhc-warmup.py` | `215738ed88f4dbd02b7db689ff1ddb4d246dd3baceb4d1b7358281f1e5722f78` |

All other serving patches are unchanged from v0.5 and remain pinned in Git.
The build executes them idempotently against the fixed base image and probes
the resulting source/API contracts before committing the image layer.

## Qualification evidence

The published target revision was exercised through the normal launcher rather than a local override. Release gates include:

- An isolated old-B12x/new-B12x/EP2 matrix, DCP on/off comparison, and PCIe
  collective microbench under the same 400 W/GPU cap.
- A final K5 code-agent C1–C16 curve, existing-depth decode through 128K, and
  cold prefill through 128K.
- The seven-type GLMRT content blend with 21/21 semantic contracts passing.
- Focused B12x EP map, rank-partial parity, real MCG K3 Trellis parity,
  allocation-free graph replay, and 64-bit high-page tests: 20/20 pass.
- Direct DFlash K5 mask-anchor, rejection rollback, sampling-position,
  request-map, and inert-padding semantics.
- Exact vision ordering for 1, 4, and 16 images plus server-side rejection of
  image 17.
- A thinking-enabled structured tool call with prompt, expected result, and
  actual `get_weather` call retained.
- An exact 128K prefix replay with a 114,688-token cache-hit delta.
- One cold exact 1,000,000-token request with six successful needles from 50K through 990K.
- A post-1M C16 soak and final-image readiness audit with 30 post-ready API
  requests and zero post-ready JIT warnings.
- A public, anonymous image pull followed by a clean-clone launch from an
  absent vLLM cache and a restart from that populated cache. Both became
  healthy only after the release-ready marker, passed 7/7 deterministic
  content contracts, and recorded zero post-ready JIT. The clean clone also
  verified all 35 target and four draft files at their pinned public HF
  revisions without redownloading the quant.

Raw machine-readable artifacts live under
`benchmarks/v0.6.0-b12x-ep2/`; [RESULTS.md](benchmarks/RESULTS.md) defines their
timing, scoring, power, and attribution methods.

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
- https://github.com/tpurtell/sparkinfer-glmrt
- https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2
- https://github.com/MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark
- https://github.com/samuelcardillo/glm-5.3-flash-2x-rtx-pro-6000-blackwell
- https://github.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw

The Mia recipe targets two networked SM121 DGX Sparks; this recipe borrows
relevant SM12x stability lessons but qualifies local PCIe collectives on two
workstation GPUs. The Samuel Cardillo recipe helped flag DFlash as the likely
source of the reported high decode result, which this release then tested
directly on the EXL3 target. Brandon's public recipe supplied the EP2/runtime
optimization lead for the controlled topology comparison. No private
route-128 source, binary, or behavior was used.
