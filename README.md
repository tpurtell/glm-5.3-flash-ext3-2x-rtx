# GLM-5.3 Flash EXL3 K3.25 + DFlash2 on 2× RTX PRO 6000

One million tokens, sixteen images, and a speculative drafter with somewhere useful to be.

This recipe serves [`wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1`](https://huggingface.co/wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1) with [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) on two PCIe-connected SM120 GPUs. Its routed experts use exact K3/K4 projections with a layerwise **3:5:8 gate:up:down** K4 budget; the finished Hugging Face quant is consumed as-is, and quantization machinery stays out of this recipe.

The default profile is DFlash2 K5, FP8 MLA cache, TP2+EP2, target DCP2, a replicated draft, 16 scheduler slots, vision for up to 16 images, a 1,048,576-token model limit, baseline state rollback, and the qualified B12x PCIe paths. Compact ReplaySSM is a qualified capacity/C16 option, but it is not the C1-oriented default.

`v0.7.0` adds the K3.25 target, projection-mixed Trellis kernels, current B12x,
large-cache index fixes, and a repaired DFlash2/ReplaySSM path. Uniform K3 and
K4 remain selectable and retain their fixed-tier runtime path.

## The fun numbers

| DFlash2 + FP8 release highlight | Measured result |
|---|---:|
| Code-agent decode, C1 | **213.0 tok/s** |
| Code-agent decode, C16 | **832.2 tok/s aggregate / 52.0 per request** |
| 128K prefill, C1 | **4,572.0 prompt tok/s** |
| DFlash-aware FP8 KV capacity | **2,758,919 request-equivalent tokens** |
| Exact cold 1M six-needle retrieval | **6/6 pass, 263.9 s** |
| Vision contract | **16 images pass; image 17 rejected** |

All published performance measurements used a **400 W power limit per GPU**. The code-agent benchmark asks for a complete replacement of a buggy async Python task runner, emits 256 tokens per sequence, and sums each request's first-to-last-token pure-decode rate. Prefill/TTFT is excluded from decode; the conservative whole-batch window is retained in the detailed report. The 128K prefill figure includes server tokenization and time to first token. Full curves, variance, acceptance, target-pass efficiency, methodology, and raw receipts are in [the detailed results](benchmarks/RESULTS.md).

On the exact same final image and flags, uniform K3 measured 215.4 tok/s at C1
and K3.25 measured 213.0 (**−1.1%**); K3.25 was **+2.4%** at 128K prefill.
The mixed kernel is not free under saturation: C16 was 832.2 versus 1,119.5
tok/s for uniform K3. The detailed profiler attribution is published too—no
benchmark hide-and-seek.

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
| `MODEL_PROFILE` | `k325` | Use the public projection-mixed K3.25 target; `k3` and `k4` remain supported |
| `SPECULATIVE_METHOD` | `dflash2` | Use the DFlash2 block-diffusion drafter |
| `DFLASH_TOKENS` | `5` | Best measured agent-workload compromise |
| `DFLASH_KV_CACHE_DTYPE` | `bfloat16` | Quality-preserving cache for the small draft model |
| `KV_CACHE_PROFILE` | `fp8` | Quality-leaning FP8 target MLA cache |
| `ENABLE_EXPERT_PARALLEL` | `1` | Use the qualified TP2+EP2 B12x expert path |
| `DECODE_CONTEXT_PARALLEL_SIZE` | `2` | Shard target long-context cache and attention work |
| `MAX_MODEL_LEN` | `1048576`; K4: `262144` | K3/K3.25 retain the 1M limit; Brandon's larger K4 defaults to 256K |
| `MAX_NUM_SEQS` | `16` | Qualified C16 decode fan-out |
| `MAX_NUM_BATCHED_TOKENS` | `2048` | Qualified prefill chunk size |
| `LIMIT_MM_PER_PROMPT` | `{"image":16}` | Enable and enforce the 16-image contract |
| `USE_REPLAYSSM` | `0` with DFlash2 | Opt into compact KDA rollback with `1`; it adds capacity and C16 throughput but costs C1 here |
| `GPU_MEMORY_UTILIZATION` | `0.950` | Qualified ceiling; the launcher does not exceed it |
| `GLM53_STARTUP_WARMUP` | `1` | Gate container health on the qualified first-use kernel warmup |
| FlashInfer autotune | off | Avoid the unhelpful/unstable GLM SM12x tuning path |

`MAX_NUM_SEQS=16` is scheduler capacity, not a claim that sixteen 1M prompts fit at once. vLLM reports **2,758,919 full-request-equivalent tokens**, or **2.63×** the configured maximum request. This is the conservative heterogeneous-pool result after target MLA, recurrent state, and per-request DFlash scratch are all accounted for; it is the capacity figure used by the scheduler. `USE_REPLAYSSM=1` raises this to **2,940,699 tokens (2.80×)**.

The local port gives the replicated DFlash attention group a 128-token allocation block instead of inheriting the target backend's generic 16-token page. That cuts its per-request block-ID tax from 257 to 33 while preserving the target's DCP2 cache geometry.

## Profiles and controls

K5 remains the DFlash default because it gave the best code-agent balance in the qualified tuning sweep. K3 remains an explicit loaded-throughput control; K7 over-drafted on this hardware.

```bash
MODEL_PROFILE=k4 ./download.sh         # Brandon's current uniform-K4 target
MODEL_PROFILE=k4 ./start.sh
MODEL_PROFILE=k3 ./start.sh             # this recipe's uniform-K3 control
DFLASH_TOKENS=3 ./start.sh             # loaded-throughput profile
DFLASH_TOKENS=7 ./start.sh             # accepted, but not the measured winner
SPECULATIVE_METHOD=none ./start.sh      # target-only control
SPECULATIVE_METHOD=mtp ./start.sh       # request-local adaptive MTP K1…K5
LANGUAGE_MODEL_ONLY=1 ./start.sh        # disable the vision path
KV_CACHE_PROFILE=nvfp4 ./start.sh       # optional lower-precision target cache
USE_REPLAYSSM=1 ./start.sh              # compact rollback: more cache/C16, less C1 here
```

`MODEL_PROFILE=k3` and `MODEL_PROFILE=k4` pin both the repository and its
revision. For another checkpoint, set `MODEL_ID` and `MODEL_REVISION` together;
the launcher rejects half-overrides so it cannot combine one model with
another model's commit.

Brandon's K4 defaults to **262,144 tokens**, including explicit overrides
using either his `GLM-5.3-Flash-tr3-4bpw` or original
`GLM-5.3-Flash-EXL3-4bpw` repository. Set `MAX_MODEL_LEN` explicitly to change
that limit. The original K4 checkpoint loaded on v0.7.0 but vLLM rejected
the 1M default: it needed 5.97 GiB of KV memory per GPU and had 3.56 GiB
available at 0.95 utilization. The 256K default is a capacity-based launcher
setting, not a newly qualified 256K needle result. K3/K3.25 defaults are
unchanged.

Current vLLM DFlash2 executes one fixed K for the active fused batch. It does not yet expose request-local, within-request K adaptation like this recipe's alternate MTP controller. DFlash acceptance is very workload-dependent: K5 won the code-agent balance, while K3 won the C16 tuning point. A real adaptive DFlash policy needs to control the block-diffusion proposal/selector inside a request; swapping profiles only between requests would miss the point.

The launcher pins K3.25 revision `0490d2f…`, with Z.ai's corrected
GLM-5.3 chat template and official multimodal processor metadata. The
2026-09-06 template-only fix replaces the older template accidentally shipped
in `701cd74…`; weights and the v0.7.0 image are unchanged. Existing benchmark
numbers, including tool-call scores, predate this fix and have not been rerun.
Run `git pull` and `./download.sh` to refresh metadata while reusing cached
weights; the next `./start.sh` uses the corrected revision. See the
[template sync receipt](benchmarks/chat-template-sync-k325-20260906.json).

In the original release vision test, sixteen
generated numbered images returned the exact ordered list `1…16`, while image
17 received HTTP 400. DFlash receives text-side draft inputs for multimodal
requests while the target model performs the actual vision encoding and
verification.

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
- Compact ReplaySSM works with DFlash2 as well as the alternate request-local adaptive-MTP mode. Its gate includes an exact strided KDA-state CUDA check and matched ReplaySSM/full-state 120-request 32K/C4 rolling-batch stresses derived from Samuel Cardillo's downstream report. ReplaySSM passed 120/120 with zero loops or engine errors and raises capacity by 6.6%; baseline rollback remains the DFlash default because it is faster at C1.
- Correctness/performance admission gates keep eager or unprofitable shapes on vLLM/NCCL fallbacks.

The build probes the DFlash2 architecture and V2 speculator, GLM EAGLE3 support, EXL3 loader, independent draft cache, ReplaySSM/adaptive-MTP invariants, B12x APIs, and pinned versions. [PROVENANCE.md](PROVENANCE.md) records the immutable source chain and upstream references.

## Thank you

Huge thanks to **Brandon** for his K4 quant, reproducible recipe, and public EP2/runtime optimization lead behind this release's controlled comparison. No private route-128 implementation, binary, or behavior was copied. Thanks to **Inco AI / Z-Lab** for DFlash2, **Z.ai** for GLM-5.3 Flash, **MiaAI-Lab** for the nearby dual-DGX-Spark SM121 references, **cstechdev** for the GLM day-zero image, the **vLLM** and **B12x/SparkInfer** contributors, and the ExLlamaV3 authors whose trellis work underpins EXL3. Special thanks as well to Jared and the other GLM upstream contributors credited by Z.ai's vLLM work.

Thanks to **Samuel Cardillo** for publishing the downstream 32K/C4 ReplaySSM corruption reproducer that became this release's rolling-batch regression gate.

Recipe code is Apache-2.0. Model licenses still apply. In particular, the DFlash2 checkpoint is published under **CC BY-NC-ND 4.0 for research and evaluation**; contact Inco AI for commercial licensing.
