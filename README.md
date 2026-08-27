# GLM-5.3 Flash EXL3 on 2× RTX PRO 6000

Half a million tokens, two Blackwells, zero patience for generic kernels.

This recipe serves [Brandon's GLM-5.3 Flash EXL3 4 bpw quant](https://huggingface.co/brandonmusic/GLM-5.3-Flash-EXL3-4bpw) with vLLM on two PCIe-connected SM120 GPUs. The default is the fun profile: compact NVFP4 MLA cache, DCP2, 16 scheduler slots, a 500,000-token model limit, and every B12x path that won its hardware qualification.

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
| `MAX_NUM_BATCHED_TOKENS` | `8192` | Good prefill throughput without bloating warmup memory |
| `GPU_MEMORY_UTILIZATION` | `0.965` | Safe measured NVFP4 release setting |
| FlashInfer autotune | off | Avoids the unhelpful/unstable GLM SM12x tuning path |

`MAX_NUM_SEQS=16` is scheduler concurrency, not a promise that sixteen 500K prompts fit simultaneously. The measured NVFP4 pool is 600,000 logical tokens, or 1.20× one 500K request.

Want the quality-leaning cache instead? FP8 MLA is fully qualified too:

```bash
KV_CACHE_PROFILE=fp8 ./start.sh
```

That selects `fp8_ds_mla` and a measured-safe `GPU_MEMORY_UTILIZATION=0.970`; it held 523,529 tokens and passed the 500K configuration. Override any launcher setting with an environment variable. NVFP4 also accepts a calibrated scales file through `VLLM_NVFP4_MLA_SCALES_FILE`; otherwise it performs dynamic scale discovery during startup.

## What is actually accelerated

The launcher enables B12x EXL3 K4 MoE for decode and prefill, native rope-free sparse MLA, compact NVFP4/FP8 MLA cache, fused paged K-pool score/top-k, DCP2 global top-k owner exchange, graph-captured PCIe MLA DCP all-to-all, PCIe one-shot TP all-reduce through 384 KiB, GLM H64 query projection, and the exact batch-1 mHC fusion. CUDA graphs cover batches 1, 2, 4, 8, 16, 24, and 32.

The custom collectives are deliberately admitted only where they won and are correct. Eager/large-shape DCP A2A falls back to vLLM/NCCL. Query splitting stays off because it conflicts with exact sharded global top-k ownership. This is a fast recipe, not a bag of unconditional flags.

See [benchmarks/RESULTS.md](benchmarks/RESULTS.md) for the receipts and [PROVENANCE.md](PROVENANCE.md) for the immutable source chain.

## Thank you

Huge thanks to **Brandon** for doing the EXL3 quant. Thanks also to **Z.ai** for GLM-5.3 Flash, **MiaAI-Lab** for publishing the nearby dual-DGX-Spark SM121 recipe and its SM12x fixes, the **vLLM** and **B12x/SparkInfer** contributors, **cstechdev** for the GLM day-zero base, and the ExLlamaV3 authors whose trellis work underpins EXL3. Special thanks to Jared and the other GLM upstream contributors called out by Z.ai's vLLM work.

Apache-2.0. Model and base-image licenses still apply.
