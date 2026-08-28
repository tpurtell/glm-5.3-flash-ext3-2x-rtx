# GLM-5.3 Flash EXL3 on 2× RTX PRO 6000

Half a million tokens, two Blackwells, zero patience for generic kernels.

This recipe serves [Brandon's GLM-5.3 Flash EXL3 4 bpw quant](https://huggingface.co/brandonmusic/GLM-5.3-Flash-EXL3-4bpw) with vLLM on two PCIe-connected SM120 GPUs. The default is the fun profile: MTP5 with compact KDA rollback, NVFP4 MLA cache, DCP2, 16 scheduler slots, a 500,000-token model limit, and every B12x path that won its hardware qualification.

## The good bits

| Highlight | Measured result |
|---|---:|
| 128K prefill, C1 | **4,270 prompt tok/s** median (29.97 s TTFT) |
| Decode, C1 | **98.4 tok/s MTP5** / **71.0 tok/s MTP-off** |
| Decode, C16 aggregate | **276.5 tok/s MTP5** / **543.3 tok/s MTP-off** |
| Release KV pool | **537,500 tokens** at the safe 0.950 memory ceiling |

These are steady-state NVFP4 medians of three fixed-workload runs. Prefill is client-observed request-to-first-token time and includes server tokenization; decode excludes TTFT and emits 128 tokens per sequence. MTP loves this free-form workload at C1 and loses badly once C16 is saturated—acceptance and workload matter. The full NVFP4/FP8 prefill curves, ranges, and raw reports are in [benchmarks/RESULTS.md](benchmarks/RESULTS.md).

## Quick start

You need Linux/amd64, Docker with the NVIDIA Container Toolkit, two SM120 GPUs with about 96 GiB each, a CUDA 13-capable driver, roughly 165 GiB for the model, and the Hugging Face `hf` CLI.

```bash
git clone https://github.com/tpurtell/glm-5.3-flash-ext3-4-bit-2x-rtx.git
cd glm-5.3-flash-ext3-4-bit-2x-rtx
./download.sh
./start.sh
docker logs -f glm53-flash-exl3-b12x-vllm
```

The OpenAI-compatible endpoint appears at `http://127.0.0.1:8001/v1`. Cold startup takes a few minutes because the image loads 120 shards, profiles memory, and compiles/captures the useful B12x shapes.

```bash
curl -s http://127.0.0.1:8001/v1/models | jq
./stop.sh
```

To build locally instead of pulling GHCR:

```bash
IMAGE=glm53-exl3-b12x:local ./build.sh
IMAGE=glm53-exl3-b12x:local ./start.sh
```

## The defaults

| Knob | Default | Why |
|---|---:|---|
| `KV_CACHE_PROFILE` | `nvfp4` | B12x compact 368-byte NoPE MLA records |
| `DECODE_CONTEXT_PARALLEL_SIZE` | `2` | DCP2 shards long-context cache and attention work |
| `MAX_MODEL_LEN` | `500000` | Qualified at an exact 499K-token prompt |
| `MAX_NUM_SEQS` | `16` | Qualified with 16 simultaneous requests |
| `MAX_NUM_BATCHED_TOKENS` | `2048` | Leaves enough activation headroom for the 500K pool |
| `MTP_TOKENS` | `5` | Best default for structured and agent-shaped output |
| `USE_REPLAYSSM` | `1` | Compact KDA rollback instead of five full state pages |
| `REPLAYSSM_BUFFER_LEN` | `10` | Largest logical history that still uses a 16-slot ring |
| `GPU_MEMORY_UTILIZATION` | `0.950` | Measured release ceiling; no sneaky 0.96+ foot-gun |
| FlashInfer autotune | off | Avoids the unhelpful/unstable GLM SM12x tuning path |

`MAX_NUM_SEQS=16` is scheduler concurrency, not a promise that sixteen 500K prompts fit simultaneously. The measured MTP5/NVFP4 pool is 537,500 logical tokens, or 1.075× one 500K request. ReplaySSM is the reason MTP does not eat five extra full KDA state pages per live sequence.

MTP and cache behavior remain explicit knobs:

```bash
MTP_TOKENS=0 ./start.sh                 # no speculative decoding
USE_REPLAYSSM=0 MTP_TOKENS=5 ./start.sh # baseline full-state rollback
ENABLE_PREFIX_CACHING=0 ./start.sh      # mamba cache mode none
```

Want the quality-leaning cache instead? FP8 MLA is qualified too:

```bash
KV_CACHE_PROFILE=fp8 ./start.sh
```

That selects `fp8_ds_mla` at the same 0.950 memory ceiling, plus its qualified 360K model limit and 1K scheduler cap. FP8 stores a larger 656-byte MLA record than NVFP4's 368-byte record. The 500K launch geometry failed and estimated a 365,568-token ceiling; profiling at the qualified 360K geometry allocated 514,285 tokens. Override any launcher setting with an environment variable. NVFP4 also accepts a calibrated scales file through `VLLM_NVFP4_MLA_SCALES_FILE`; otherwise it performs dynamic scale discovery during startup.

## What this fork changes

This is well past stock vLLM. The image pins the day-zero GLM runtime, applies the active upstream ReplaySSM series, and then carries the GLM-specific work needed to make it real:

- A vector-gated KDA ReplaySSM speculative kernel for GLM, with checkpoint, replay, verify, rejection, flush, and aligned-prefix materialization paths.
- A physical power-of-two rollback ring. The inherited helper returned a logical 22-slot size while CUDA kernels used bitmask indexing; the port rounds physical storage correctly.
- Explicit input/output strides for GLM's non-contiguous fused projection views. Without this, MTP appeared to run but produced corrupted repetition and near-zero acceptance.
- GLM MTP cache-group scoping, so the mixed target/draft MLA group gets EAGLE semantics while target-only KDA groups retain aligned checkpoints.
- Compact KDA cache accounting: one recurrent checkpoint plus BF16 input and FP32 gate history instead of five extra full recurrent-state pages at MTP5.
- A bounded 1,024-row B12x EXL3 prefill arena. Scheduler prefills are tiled through it exactly, cutting the persistent arena from 1,822.3 to 310.1 MiB per GPU. The 2,048-token scheduler cap also trims peak activation memory enough for 500K at 0.95.
- The earlier EXL3 GLM/MTP weight mapping, B12x sparse MLA/indexer, NVFP4 cache, DCP2 global owner exchange, PCIe DCP A2A, PCIe TP all-reduce, H64 query projection, and mHC ports.

The Docker build probes these imports and invariants instead of assuming that a patch applied cleanly means it works. [PROVENANCE.md](PROVENANCE.md) pins every source and names the upstream PRs behind the port.

## What is actually accelerated

The launcher enables B12x EXL3 K4 MoE for decode and tiled prefill, native rope-free sparse MLA, compact NVFP4/FP8 MLA cache, fused paged K-pool score/top-k, DCP2 global top-k owner exchange, graph-captured PCIe MLA DCP all-to-all, PCIe one-shot TP all-reduce through 384 KiB, GLM H64 query projection, and the exact batch-1 mHC fusion. CUDA graphs cover the MTP-expanded batches through 96 rows.

The KDA ReplaySSM kernel checkpoints one recurrent state and keeps a compact power-of-two ring of inputs and vector gates. At C16/MTP5 it raised the measured pool from 51,200 to 107,054 tokens under the original 8K profiling setup; the bounded EXL3 prefill arena plus the 2K release batch cap then made the full 500K launch fit at 0.95. Aligned prefix caching stays enabled. With DCP2 plus MTP, a reusable prefix first appears above 30,720 tokens because the final 15,360-token physical MLA block is deliberately recomputed.

The custom collectives are deliberately admitted only where they won and are correct. Eager/large-shape DCP A2A falls back to vLLM/NCCL. Query splitting stays off because it conflicts with exact sharded global top-k ownership. This is a fast recipe, not a bag of unconditional flags.

See [benchmarks/RESULTS.md](benchmarks/RESULTS.md) for the receipts and [PROVENANCE.md](PROVENANCE.md) for the immutable source chain.

## Thank you

Huge thanks to **Brandon** for doing the EXL3 quant. Thanks also to **Z.ai** for GLM-5.3 Flash, **MiaAI-Lab** for publishing the nearby dual-DGX-Spark SM121 recipe and its SM12x fixes, the **vLLM** and **B12x/SparkInfer** contributors, **cstechdev** for the GLM day-zero base, and the ExLlamaV3 authors whose trellis work underpins EXL3. Special thanks to Jared and the other GLM upstream contributors called out by Z.ai's vLLM work.

Apache-2.0. Model and base-image licenses still apply.
