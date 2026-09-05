# Provenance

This recipe consumes finished Hugging Face target and draft checkpoints and composes pinned runtime artifacts. It contains no calibration corpus, quantization runner, writer checkpoint, or intermediate quant files.

## Immutable inputs

| Component | Immutable source |
|---|---|
| Served target | `wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1@701cd7456c13d87bf0147ad946f828a999afb59c` |
| Supported uniform-K3 target | `wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3-v1@1e4abd26e4e1e8d58d81fbd557d6c4099352fe63` |
| Supported uniform-K4 target | `brandonmusic/GLM-5.3-Flash-tr3-4bpw@aba59d2175e1ee2887ae0ae1300ba848b1deed84` |
| Target source | `zai-org/GLM-5.3-Flash-BF16@f12e0fe1f6b2ea274c11a569582edfd99d993c5e` |
| Corrected chat-template source | `zai-org/GLM-5.3-Flash-BF16@a5b45eb41df6402735dedc900be14a42e8d5e538` |
| DFlash2 draft | `incoai/GLM-5.3-Flash-DFlash2@bf582e4eacc1810f76656d1811693ff6c6737d2a` |
| GPTQModel quant writer | `tpurtell/GPTQModel@0565af7ce20a93df9bbc0e5563d7c6f60916f41a` |
| GPTQModel compact-config follow-up | `tpurtell/GPTQModel@a64900815b30ef01c2221b2788701a7986e50491` |
| K3.25 GPTQModel writer | `tpurtell/GPTQModel@a053382584fa58cba7bf212ef1b829d08b29b2c0` |
| K3.25 readable-plan follow-up | `tpurtell/GPTQModel@0b6734a8cfeabecae78a7a82f7fc82ec97bddfc5` |
| GLM/vLLM base | `cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1@sha256:0bd709e80b8ff13ae5de8f7d7f708a499fade3a26970d56afb1be2ff3860fde5` |
| vLLM in base | `0.1.dev20051+g487ecf187` |
| vLLM DFlash2 delta | `vllm-project/vllm@b389ac29465b33f9e9c534df221ea3c129e9793f` (PR `#52816`) |
| EXL3 runtime source image | `ghcr.io/tpurtell/deepseek-v4-flash-0731-exl3-k2-spark@sha256:86c8c1054f9c24454949e37031ce6165c007963aa0c0ef30fa884f6d4170af32` |
| EXL3 vLLM fork commit | `30038602b71395f481ef4a6edfe4fcf8551d9c15` |
| B12x fork | `tpurtell/sparkinfer-glmrt@fe054789069579e19ae5ec21f880b397bcf6575b` |
| ReplaySSM base | vLLM PRs `#48792`, `#49847`, and `#49887`, ported onto the pinned vLLM commit |
| ReplaySSM mixed-graph repair | This repository's `c51c3856f7f8ba50af3b3a60ff48e7d6a1fa303c` |
| Dynamic-MTP graph fix | vLLM PR `#49652`, ported onto the pinned vLLM commit |
| Runtime stack | Torch 2.13, CUDA 13, CUTLASS DSL 4.6.2 |

## Published release artifacts

The public `v0.7.0` and `latest` container tags resolve to the same immutable
OCI index:

| Artifact | Immutable identity |
|---|---|
| Recipe image | `ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx:v0.7.0@sha256:48e254d94f58137c8707e6044cde4528c6af3fdd9702726b9b362e9b0e0b4629` |
| Linux/amd64 manifest | `sha256:5b0486d3ada90ee3c0d822baed55e1a4d65e06e6ecc78c66a40bd22bcfa9a891` |
| Image config | `sha256:507791943bfd239a34fea7ef276353b14a14c8d597a97aac982e7aed372be0d9` |
| Build/source revision | `92eec28c9ad4d681af5f4861b74811695bfbcfa1` |

The registry independently returned that index, manifest, and image config.
Its OCI labels pin the source revision above, B12x `fe05478…`,
DFlash2/vLLM `b389ac2…`, and version `v0.7.0`. The Git release tag includes
this post-build provenance; the serving code in that tag is unchanged from
the embedded image source revision.

The DFlash2 checkpoint is a 1B-parameter BF16 draft model and is not a standalone language model. Inco AI publishes it under CC BY-NC-ND 4.0 for research and evaluation; commercial use requires separate licensing. The target model and base-image licenses also apply independently of this Apache-2.0 recipe.

## Target identity and metadata repair

K3.25 revision `701cd74…` is public and ungated. It contains 18 EXL3 shards,
the external 32.8 MB tensor manifest, the official multimodal processor,
tokenizer/generation metadata, and Z.ai's corrected chat template. Its compact
`config.json` does not duplicate the tensor manifest. The normal Hugging Face
cache was installed from the materialized local files, then verified at the
public revision without redownloading the quant.

Release metadata hashes:

| File | SHA-256 |
|---|---|
| `config.json` | `6b477cfc1fbf8cdf3795c6389bc9712503e3f7c3889145036488ffab2b1a7781` |
| `chat_template.jinja` | `34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891` |
| `processor_config.json` | `aae38374c94b08cc9b0547c6e64f05b951bd9735cea571c6988f5ed552bed3ed` |
| `tokenizer_config.json` | `926e1d0692d9f46940311494bd6de97f208e195c9150883c163f16b30c868ff4` |
| `generation_config.json` | `a07de3408f578c6a7ca8a1646aa91a41df55d539349fda15fb8b611eb007e9b7` |

The compact `config.json` deliberately does not embed the duplicate 32.8 MB EXL3 `tensor_storage` map. The complete tensor map remains in the external quantization manifests. The launcher validates the index, every referenced shard, the EXL3 manifest, official processor metadata, and DFlash2 architecture before starting Docker.

The compact K3.25 config is 13,180 bytes, parses as ordinary JSON and through
the release vLLM configuration path as `Glm5NextConfig` /
`Glm5NextForConditionalGeneration`. The corrected template compiled and
rendered through the release container's `AutoProcessor`, covering ordinary
chat, null assistant content, out-of-order tool results, and invalid-ID
fallback. The complete local snapshot passed `hf cache verify`.

The GPTQModel commits are listed only to establish how the public targets were
written and packaged. The earlier K3 work ran at `0565af7…`; the K3.25 one-shot
run used `a053382…`, with the compact-config work from `a649008…` already in
its ancestry. `0b6734a…` is a post-run readability improvement for published
mixed-tier plans, not a silent weight rewrite. The saver materializes EXL3
shells while streaming native tensors under their original keys, avoiding
duplicate BF16 materialization. GPTQModel does not ship in the serving image.

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
- B12x `fe054789...` admits that larger route namespace, preserves the
  projection-mixed live tile selector, reduces sparse-index and scratch width,
  and fails closed on route-map dtype, extent, device, and contiguity drift.
  Its final CPU suite passed 74 tests, and the composed two-GPU release gate
  passed 47/47 mixed-Trellis, EP, attention, high-page, and PCIe tests.

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

### ReplaySSM mixed-batch repair and release gate

[Samuel Cardillo's corruption investigation](https://github.com/samuelcardillo/glm-5.3-flash-2x-rtx-pro-6000-blackwell/commit/1755c0f0c01b98463a7b87ab613a6c894b569298)
is a valuable downstream reproducer, but it used this recipe at `3bff1d5...`
(`v0.3.0`). That source predates `c51c385...`, which prevents compact
ReplaySSM from selecting vLLM's generic mixed prefill/decode CUDA graph. The
generic graph supplies synthetic rows whose shape can disagree with the live
GLM KDA prefill state; Samuel's fatal
`ReplaySSM prefill source/state row count mismatch` occurred under the same
rolling C4 mixed workload. Uniform decode graphs remain enabled.

The current port also includes the request-lifecycle corrections from vLLM
[#49847](https://github.com/vllm-project/vllm/pull/49847): draft-less rows stay
on ReplaySSM, a temporarily unscheduled request retains its pending GPU
acceptance and decode anchor, and preempted/recycled pages reset. vLLM
[#54103](https://github.com/vllm-project/vllm/pull/54103) documents a nearby
KDA concurrency hazard in which a strided `state_indices[:, 0]` was consumed
as contiguous. This CUDA port already passes that stride into its verify and
cursor kernels; the KDA prefix materializer now carries explicit strides for
all four row tensors as well and reports every row count on invariant failure.

The K3.25 release gate first compares the KDA prefix materializer with a Torch
reference while all four request-row inputs are deliberately non-contiguous.
It then runs `scripts/test-replayssm-stress.py` with ReplaySSM on:
exact 32,768-token prompts, C4, 40 thinking-off shared-prefix requests, 40
maximum-thinking shared-prefix requests, and 40 maximum-thinking unique-prefix
requests. It retains raw SSE, rejects repeated-subword streams, requires exact
needle retrieval with thinking off, checks server health after every phase,
and compares the same workload with full-state rollback. ReplaySSM and the
matched control each passed 120/120 requests with zero loops, errors, engine
deaths, or post-ready JIT; the strided CUDA materializer matched exactly with
maximum absolute error 0.0. ReplaySSM is therefore a qualified option. It is
not the DFlash default because the release workload measured higher C1 decode
with baseline rollback.

The newer FlashInfer work in vLLM
[#52928](https://github.com/vllm-project/vllm/pull/52928) is Mamba2-only and
currently requires `mamba_cache_mode=none`; the prefix-cache follow-up
[#54609](https://github.com/vllm-project/vllm/pull/54609) is still WIP. Neither
is a drop-in replacement for GLM-5.3's vector-gated KDA plus aligned prefix
cache, so this release keeps the narrow KDA port rather than importing an
unrelated backend.

Build-time probes check target/draft architecture recognition, the DFlash2 V2 speculator, GLM EAGLE3 support, EXL3 registration, uniform and per-projection Trellis plans, the EP route namespace, ReplaySSM/adaptive policy imports, B12x APIs, compact cache layouts, head geometry, and exact runtime versions.

Current v0.7 launcher/build hashes:

| File | SHA-256 |
|---|---|
| `Dockerfile` | `682fe68886e019edd9d6eff3ae7c239833e3f9fd067facd342ad50492cb7516d` |
| `build.sh` | `785958e0d667dfbd0141d59b4978d3909ed3c191f88b1daa7575c99badbd2452` |
| `download.sh` | `f952a707a412605ff99c3336d5e7040ee95a20797aca224efd045b273fa6e9a8` |
| `start.sh` | `fbd116da87ee01525f76a6016a30c4d9ef0f5aa8fb6188cbb5e6ea571aa88212` |
| `container/glm53-entrypoint.sh` | `c1de8b073277b8edfb5c85c7c8d83ee511593d77d271c26a7ddf4d1e1c5abc8d` |
| `container/glm53-release-warmup.py` | `dd4e3bc041f267d288748552248a412773a2e308691145a70616977a39558146` |

Changed/new serving patches in v0.7:

| File | SHA-256 |
|---|---|
| `port-exl3-glm53.py` | `4c306ddde7e44e5888b4b75d26046b22a3d6cf920dbc30001c8fb0574bfee3cc` |
| `port-exl3-projection-mixed-glm53.py` | `8be477cca588993040c879ca4f572f4c91af2b43fd58e68dbb355297e4391c00` |
| `port-exl3-prepared-dtype.py` | `0d77bdd107a385ac7df550d0141e18b9c86d9dbe4b0e8ed4906cd8d905f22973` |
| `port-b12x-glm-index-width.py` | `0ccce3229e38654f506fe3049293853812fe436c2012b86feabb16fe5427e9d8` |
| `port-glm-sparse-memory.py` | `df1cde80246045ae9ef8cb2e151a8679f52ac27f012347bc00a61b1c42ce4851` |
| `port-glm-prefill-jit.py` | `cd535cdc25f02fcf681c799c09a3275aeb7151c8fb7802b838b2e2a4cb550c98` |
| `port-glm-replayssm-conv-window.py` | `78620f40d5c99d75bcf1e43da8e4d5456e147fa91dbae4394a45c25ce004b944` |

The build executes every port idempotently against the fixed base image and
probes the resulting source/API contracts before committing the image layer.

## Qualification evidence

The published target revision was exercised through the normal launcher rather than a local override. Release gates include:

- A final K5 code-agent C1–C16 curve, existing-depth decode through 128K, cold
  prefill through 128K, and matched final-image uniform-K3 control.
- The seven-type GLMRT content blend with 21/21 semantic contracts passing.
- Focused B12x projection-mixed parity, EP map, rank-partial parity,
  allocation-free graph replay, sparse attention, high-page, and PCIe tests:
  47/47 pass on the target GPUs.
- Direct DFlash K5 mask-anchor, rejection rollback, sampling-position,
  request-map, and inert-padding semantics.
- Exact vision ordering for 1, 4, and 16 images plus server-side rejection of
  image 17.
- The exact 69-case tool suite at parallelism 8 with thinking enabled for both
  K3 and K3.25, including every prompt result and per-case comparison.
- An exact 128K prefix replay with a 114,688-token cache-hit delta.
- One cold exact 1,000,000-token request with six successful needles from 50K through 990K.
- A post-1M C16 soak and final-image readiness audit with 30 post-ready API
  requests and zero post-ready JIT warnings.
- ReplaySSM and full-state matched 120-request rolling stresses, the strided
  CUDA materializer, ReplaySSM 128K replay/1M needle, and an explicit measured
  reason for leaving baseline rollback as the DFlash default.

Raw v0.7 machine-readable artifacts live under `benchmarks/v0.7.0-k325/`;
[RESULTS.md](benchmarks/RESULTS.md) defines their timing, scoring, power, and
attribution methods. The v0.6 evidence remains intact in its historical
directory.

The 2026-09-06 tool-eval refresh additionally exercises all 88 standard and
Hard Mode cases using evaluator commit
`cf54b4bfe705f12f71e8866f10730572497c8105` (`2.6.1.dev45+gcf54b4bfe`).
Both targets run in the public v0.7.0 container with matching serving flags
and 400 W/GPU limits, default thinking, and evaluation parallelism 8.
All cases are graded; K3 scores 156/176 and K3.25 scores 164/176. The
original 69-case receipts remain intact, and the new runs, environment
captures, and comparison live under
[`tool-eval-20260906/`](benchmarks/v0.7.0-k325/tool-eval-20260906/).

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
- https://github.com/vllm-project/vllm/pull/52928
- https://github.com/vllm-project/vllm/pull/54103
- https://github.com/vllm-project/vllm/pull/54609
- https://github.com/samuelcardillo/glm-5.3-flash-2x-rtx-pro-6000-blackwell/commit/1755c0f0c01b98463a7b87ab613a6c894b569298
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
