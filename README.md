# GLM-5.3 Flash EXL3 + DFlash2 on 2× RTX PRO 6000

One million tokens, sixteen images, and a speculative drafter with somewhere useful to be.

This recipe serves [`wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3-v1`](https://huggingface.co/wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3-v1) with [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) on two PCIe-connected SM120 GPUs. It consumes the finished Hugging Face quant; quantization code and intermediate artifacts deliberately live elsewhere.

The default profile is DFlash2 K5, FP8 MLA cache, TP2+EP2, target DCP2, a replicated draft, 16 scheduler slots, vision for up to 16 images, a 1,048,576-token model limit, and the qualified B12x PCIe paths.

`v0.6.1` is a recipe/model-metadata patch that pins Z.ai's corrected chat
template. It intentionally reuses the `v0.6.0` container because no serving
runtime, kernel, or model tensor changed.

## The fun numbers

| DFlash2 + FP8 release highlight | Measured result |
|---|---:|
| Code-agent decode, C1 | **222.6 tok/s** |
| Code-agent decode, C16 | **1,067.2 tok/s aggregate / 66.7 per request** |
| 128K prefill, C1 | **4,270.1 prompt tok/s** |
| DFlash-aware FP8 KV capacity | **2,912,711 request-equivalent tokens** |
| Exact cold 1M six-needle retrieval | **6/6 pass, 260.8 s** |
| Vision contract | **16 images pass; image 17 rejected** |

All published performance measurements used a **400 W power limit per GPU**. The code-agent benchmark asks for a complete replacement of a buggy async Python task runner, emits 256 tokens per sequence, and sums each request's first-to-last-token pure-decode rate. Prefill/TTFT is excluded from decode; the conservative whole-batch window is retained in the detailed report. The 128K prefill figure includes server tokenization and time to first token. Full curves, variance, acceptance, target-pass efficiency, methodology, and raw receipts are in [the detailed results](benchmarks/RESULTS.md).

## Quick start

You need Linux/amd64, Docker with the NVIDIA Container Toolkit, two SM120 GPUs with about 96 GiB each, a CUDA 13-capable driver, about 150 GiB of model storage, and the Hugging Face `hf` CLI.

```bash
git clone https://github.com/tpurtell/glm-5.3-flash-ext3-4-bit-2x-rtx.git
cd glm-5.3-flash-ext3-4-bit-2x-rtx
./download.sh
./start.sh
docker logs -f glm53-flash-exl3-b12x-vllm
```

The OpenAI-compatible endpoint appears at `http://127.0.0.1:8001/v1`. Cold startup loads the target and 1B-parameter drafter, profiles the real memory budget, and compiles/captures the useful SM120 shapes. Docker reports the container healthy only after raw-greedy C1, rendered-chat C1, long-prefill, and four sampled C16 passes have exercised the release traffic shapes. The final gate was followed by 30 real API requests with zero post-ready kernel compilation.

```bash
curl -s http://127.0.0.1:8001/v1/models | jq
./stop.sh
```

To build locally instead of pulling the public GHCR image:

```bash
IMAGE=glm53-dflash2:local IMAGE_VERSION=local ./build.sh
IMAGE=glm53-dflash2:local ./start.sh
```

## Release defaults

| Knob | Default | Purpose |
|---|---:|---|
| `MODEL_PROFILE` | `k3` | Select the immutable target pair; K3.25 becomes the release default after its public revision is validated |
| `SPECULATIVE_METHOD` | `dflash2` | Use the DFlash2 block-diffusion drafter |
| `DFLASH_TOKENS` | `5` | Best measured agent-workload compromise |
| `DFLASH_KV_CACHE_DTYPE` | `bfloat16` | Quality-preserving cache for the small draft model |
| `KV_CACHE_PROFILE` | `fp8` | Quality-leaning FP8 target MLA cache |
| `ENABLE_EXPERT_PARALLEL` | `1` | Use the qualified TP2+EP2 B12x expert path |
| `DECODE_CONTEXT_PARALLEL_SIZE` | `2` | Shard target long-context cache and attention work |
| `MAX_MODEL_LEN` | `1048576` | GLM-5.3 Flash architecture limit; cold 1M qualified |
| `MAX_NUM_SEQS` | `16` | Qualified C16 decode fan-out |
| `MAX_NUM_BATCHED_TOKENS` | `2048` | Qualified prefill chunk size |
| `LIMIT_MM_PER_PROMPT` | `{"image":16}` | Enable and enforce the 16-image contract |
| `GPU_MEMORY_UTILIZATION` | `0.950` | Qualified ceiling; the launcher does not exceed it |
| `GLM53_STARTUP_WARMUP` | `1` | Gate container health on the qualified first-use kernel warmup |
| FlashInfer autotune | off | Avoid the unhelpful/unstable GLM SM12x tuning path |

`MAX_NUM_SEQS=16` is scheduler capacity, not a claim that sixteen 1M prompts fit at once. vLLM reports **2,912,711 full-request-equivalent tokens**, or **2.78×** the configured maximum request. This is the conservative heterogeneous-pool result after target MLA, recurrent state, and per-request DFlash scratch are all accounted for; it is the capacity figure used by the scheduler.

The local port gives the replicated DFlash attention group a 128-token allocation block instead of inheriting the target backend's generic 16-token page. That cuts its per-request block-ID tax from 257 to 33 while preserving the target's DCP2 cache geometry. EP2 and the updated runtime raise the scheduler result by 4.5% over v0.5 without changing FP8 precision.

## Profiles and controls

K5 remains the default because it reaches 222.6 tok/s at C1 while retaining 1,067.2 tok/s aggregate at C16 on the release workload. K3 remains an explicit throughput-oriented control from the previous tuning sweep; K7 over-drafted on this hardware.

```bash
MODEL_PROFILE=k4 ./download.sh         # Brandon's current uniform-K4 target
MODEL_PROFILE=k4 ./start.sh
DFLASH_TOKENS=3 ./start.sh             # loaded-throughput profile
DFLASH_TOKENS=7 ./start.sh             # accepted, but not the measured winner
SPECULATIVE_METHOD=none ./start.sh      # target-only control
SPECULATIVE_METHOD=mtp ./start.sh       # request-local adaptive MTP K1…K5
LANGUAGE_MODEL_ONLY=1 ./start.sh        # disable the vision path
KV_CACHE_PROFILE=nvfp4 ./start.sh       # optional lower-precision target cache
```

`MODEL_PROFILE=k3` and `MODEL_PROFILE=k4` pin both the repository and its
revision. For another checkpoint, set `MODEL_ID` and `MODEL_REVISION` together;
the launcher rejects half-overrides so it cannot combine one model with
another model's commit.

Current vLLM DFlash2 executes one fixed K for the active fused batch. It does not yet expose request-local, within-request K adaptation like this recipe's alternate MTP controller. DFlash acceptance is very workload-dependent: K5 won the code-agent balance, while K3 won the C16 tuning point. A real adaptive DFlash policy needs to control the block-diffusion proposal/selector inside a request; swapping profiles only between requests would miss the point.

The launcher pins target revision `1e4abd2…`, which carries Z.ai's corrected
GLM-5.3 chat template from upstream `a5b45eb…`. The patch fixes null assistant
content and tool-result reordering without changing tokenizer metadata, model
weights, or quantization tensors. The v0.6.0 release test used the preceding
metadata revision with the same weights: 16 generated numbered images returned
the exact ordered list `1…16`, while image 17 received HTTP 400. DFlash receives
text-side draft inputs for multimodal requests while the target model performs
the actual vision encoding and verification.

## What differs from stock vLLM

This is a pinned runtime composition, not a lucky pile of launcher flags:

- The upstream vLLM DFlash2 implementation from PR #52816, ported onto the GLM day-zero branch.
- GLM-5.3 EAGLE3/DFlash target taps after mHC, plus the GLM decoder-layer indirection needed by the drafter.
- An independent DFlash KV group: the target remains DCP2 while the small dense draft attention/cache is correctly replicated at DCP1 on both ranks.
- A 128-token replicated draft-cache allocation block and DFlash-aware target prefix hashes, avoiding waste and cross-group cache contamination.
- EXL3 fixed-tier K3/K4 and adjacent projection-mixed loading for GLM routed experts. The mixed path preserves each projection's integral K value, prepares native B12x tier descriptors, and composes the 288-global/144-local EP2 route map once at load time.
- B12x projection-mixed decode/prefill uses live-shape Trellis tile selection and allocation-free graph replay; uniform checkpoints retain the existing fixed-layout kernel path.
- Graph-stable MCG Trellis full-rotation scratch and route arenas, with numerical rank-partial parity and changed/empty-route CUDA graph replay gates.
- B12x sparse MLA, paged K-pool score/top-k, DCP2 global owner exchange, graph-admitted PCIe DCP A2A, PCIe one-shot TP all-reduce, GLM H64 query projection, and batch-1 mHC fusion.
- A release-ready entrypoint that prewarms GLM mHC, route, DFlash, sampler, long-prefill, and C16 specializations before exposing a healthy container.
- Existing compact ReplaySSM and request-local adaptive MTP remain available as an alternate speculative method.
- Correctness/performance admission gates keep eager or unprofitable shapes on vLLM/NCCL fallbacks.

The build probes the DFlash2 architecture and V2 speculator, GLM EAGLE3 support, EXL3 loader, independent draft cache, ReplaySSM/adaptive-MTP invariants, B12x APIs, and pinned versions. [PROVENANCE.md](PROVENANCE.md) records the immutable source chain and upstream references.

## Thank you

Huge thanks to **Brandon** for the quant work and the public EP2/runtime optimization lead behind this release's controlled comparison. No private route-128 implementation, binary, or behavior was copied. Thanks to **Inco AI / Z-Lab** for DFlash2, **Z.ai** for GLM-5.3 Flash, **MiaAI-Lab** for the nearby dual-DGX-Spark SM121 references, **cstechdev** for the GLM day-zero image, the **vLLM** and **B12x/SparkInfer** contributors, and the ExLlamaV3 authors whose trellis work underpins EXL3. Special thanks as well to Jared and the other GLM upstream contributors credited by Z.ai's vLLM work.

Recipe code is Apache-2.0. Model licenses still apply. In particular, the DFlash2 checkpoint is published under **CC BY-NC-ND 4.0 for research and evaluation**; contact Inco AI for commercial licensing.
