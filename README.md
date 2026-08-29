# GLM-5.3 Flash EXL3 K3 on 2× RTX PRO 6000

Half a million tokens, two Blackwells, and very little patience for generic kernels.

This recipe serves [`wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3-v1`](https://huggingface.co/wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3-v1) with a patched vLLM runtime on two PCIe-connected SM120 GPUs. It only consumes the finished Hugging Face quant; quantization code and intermediate artifacts do not live in this repository.

The default is the fun profile: NVFP4 MLA cache, DCP2, request-local adaptive MTP K1…K5, compact ReplaySSM rollback, 16 scheduler slots, a 500,000-token model limit, and every B12x path that won qualification on this machine.

## The good bits

| NVFP4 release highlight | Measured result |
|---|---:|
| 128K prefill, C1 | **4,415.5 prompt tok/s** |
| Decode with adaptive MTP, C1 / C16 aggregate | **106.9 / 598.7 tok/s** |
| Decode with MTP off, C1 / C16 aggregate | **71.4 / 561.3 tok/s** |
| KV cache pool | **7,200,000 tokens** |
| Exact uncached 128K retrieval | **pass** |
| Full tool-use suite | **88/100**, 122/138 points, 0 transport errors |

These are steady-state medians. Prefill includes tokenization and time to first token; decode excludes TTFT and emits 128 tokens per sequence. C16 is aggregate throughput, not per-request throughput. See [the detailed results](benchmarks/RESULTS.md) for FP8, every prompt depth, C1–C16 curves, ranges, controls, tool-eval misses, and raw reports.

## Quick start

You need Linux/amd64, Docker with the NVIDIA Container Toolkit, two SM120 GPUs with about 96 GiB each, a CUDA 13-capable driver, about 145 GiB of free disk for the 127.30 GiB checkpoint, and the Hugging Face `hf` CLI.

```bash
git clone https://github.com/tpurtell/glm-5.3-flash-ext3-4-bit-2x-rtx.git
cd glm-5.3-flash-ext3-4-bit-2x-rtx
./download.sh
./start.sh
docker logs -f glm53-flash-exl3-b12x-vllm
```

The OpenAI-compatible endpoint appears at `http://127.0.0.1:8001/v1`. Cold startup loads 16 shards, profiles the real memory budget, then compiles and captures the useful B12x shapes.

```bash
curl -s http://127.0.0.1:8001/v1/models | jq
./stop.sh
```

To build locally instead of pulling the public GHCR package:

```bash
IMAGE=glm53-exl3-k3-b12x:local ./build.sh
IMAGE=glm53-exl3-k3-b12x:local ./start.sh
```

## Defaults

| Knob | Default | Purpose |
|---|---:|---|
| `KV_CACHE_PROFILE` | `nvfp4` | B12x compact 368-byte NoPE MLA records |
| `DECODE_CONTEXT_PARALLEL_SIZE` | `2` | DCP2 shards long-context cache and attention work |
| `MAX_MODEL_LEN` | `500000` | Qualified with both cache profiles |
| `MAX_NUM_SEQS` | `16` | Qualified C16 decode fan-out |
| `MAX_NUM_BATCHED_TOKENS` | `2048` | Good prefill throughput without wasting the memory pool |
| `MTP_TOKENS` | `5` | Maximum adaptive draft depth |
| `ADAPTIVE_MTP` | `1` | Re-estimate K throughout each request |
| `ADAPTIVE_MTP_MIN_DEPTH` | `1` | The measured floor while MTP remains resident |
| `USE_REPLAYSSM` | `1` | Compact KDA rollback instead of full state copies |
| `REPLAYSSM_BUFFER_LEN` | `10` | Fits the speculative history in a 16-slot physical ring |
| `GPU_MEMORY_UTILIZATION` | `0.950` | Qualified ceiling; the launcher never sneaks above it |
| FlashInfer autotune | off | Avoids the unstable/unhelpful GLM SM12x tuning path |

`MAX_NUM_SEQS=16` is scheduler concurrency, not a promise that sixteen 500K prompts fit simultaneously. The default pool holds 14.40 request-equivalents at 500K; shorter agent turns can use all 16 slots.

Adaptive MTP varies K *inside* a request. Each live request predicts its preferred depth from recent verification outcomes. Since vLLM drafts a fused batch together, the executed K is the half-up rounded arithmetic mean: predictions K5, K3, and K2 execute at `(5 + 3 + 2) / 3 → K3`. Finished requests do not leak policy state.

The controls remain explicit:

```bash
MTP_TOKENS=0 ./start.sh                 # no speculative decoding
ADAPTIVE_MTP=0 MTP_TOKENS=5 ./start.sh  # fixed K5
ADAPTIVE_MTP_MIN_DEPTH=0 ./start.sh      # permit K0 probes
USE_REPLAYSSM=0 MTP_TOKENS=5 ./start.sh # baseline full-state rollback
ENABLE_PREFIX_CACHING=0 ./start.sh       # mamba cache mode none
```

For the quality-leaning cache profile:

```bash
KV_CACHE_PROFILE=fp8 ./start.sh
```

FP8 uses the same qualified 500K/2K/C16 geometry and allocates 4,553,846 cache tokens. It stores a 656-byte MLA record versus NVFP4's 368 bytes. NVFP4 therefore supplies 1.58× the measured pool and slightly faster long prefill; FP8 is retained as the less aggressive cache-quantization option. NVFP4 accepts a calibrated scales file through `VLLM_NVFP4_MLA_SCALES_FILE`; without one, startup discovers scales dynamically.

## What differs from stock vLLM

This image is a deliberately pinned runtime composition, not a few launcher flags:

- EXL3 K3 mixed-checkpoint loading for GLM routed experts and the MTP block, plus B12x decode and bounded tiled-prefill trellis paths.
- GLM vector-gated KDA ReplaySSM with correct checkpoint, replay, verify, rejection, flush, and aligned-prefix materialization.
- A power-of-two physical rollback ring and explicit projection strides; both fix real repetition/corruption failures in the initial port.
- GLM MTP cache-group scoping and compact state accounting instead of five extra full recurrent-state pages at MTP5.
- Request-local adaptive MTP with evidence epochs, conservative expansion, stale-feedback rejection, load-aware contraction, and arithmetic-mean batch execution.
- The upstream dynamic-MTP CUDA-graph fix plus a local exclusion for ReplaySSM mixed prefill/decode batches, which are not valid static graph shapes.
- A bounded 1,024-row EXL3 prefill arena. Exact tiling cut its persistent allocation from 1,822.3 to 310.1 MiB per GPU.
- B12x sparse MLA, paged K-pool score/top-k, DCP2 global owner exchange, graph-admitted PCIe DCP A2A, PCIe one-shot TP all-reduce, GLM H64 query projection, and the batch-1 mHC fusion.
- Correctness/performance admission gates: eager or large collective shapes fall back to vLLM/NCCL when the B12x path does not win.

The image keeps the day-zero GLM base's Torch 2.13/CUDA 13 stack and applies these ports reproducibly at build time. The build probes the K3 loader, MTP mapping, ReplaySSM invariants, adaptive policy, compact cache layouts, B12x APIs, and pinned versions. [PROVENANCE.md](PROVENANCE.md) contains the immutable source chain and upstream references.

## K4 remains an option

The previous [`brandonmusic/GLM-5.3-Flash-EXL3-4bpw`](https://huggingface.co/brandonmusic/GLM-5.3-Flash-EXL3-4bpw) release remains usable for a quality/size trade-off. Its comparable tool suite scored 124/138 versus K3's 122/138, while K3 cuts the checkpoint to 127.30 GiB and unlocks the much larger cache pools above. The old K4 raw reports remain in `benchmarks/`; release `v0.3.0` preserves its original defaults.

```bash
MODEL_ID=brandonmusic/GLM-5.3-Flash-EXL3-4bpw \
MODEL_REVISION=4739eb1bcfd478e8a32da6358908567bc3a9ac51 ./download.sh
MODEL_ID=brandonmusic/GLM-5.3-Flash-EXL3-4bpw \
MODEL_REVISION=4739eb1bcfd478e8a32da6358908567bc3a9ac51 ./start.sh
```

## Thank you

Huge thanks to **Brandon** for the quant work this release builds on. Thanks to **Z.ai** for GLM-5.3 Flash, **MiaAI-Lab** for the nearby dual-DGX-Spark SM121 reference and its SM12x lessons, **cstechdev** for the GLM day-zero image, the **vLLM** and **B12x/SparkInfer** contributors, and the ExLlamaV3 authors whose trellis work underpins EXL3. Special thanks as well to Jared and the other GLM upstream contributors credited by Z.ai's vLLM work.

Apache-2.0. Model and base-image licenses still apply.
