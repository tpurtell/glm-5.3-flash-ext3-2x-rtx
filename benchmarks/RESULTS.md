# DFlash2 + FP8 qualification results

Measured 2026-08-29 on two NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPUs (SM120, 97,887 MiB each), driver 595.71.05. The final profile used [`wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3-v1@319d66a…`](https://huggingface.co/wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3-v1/tree/319d66a8b53092b491f698440ecea781e4ddd4e4), [`incoai/GLM-5.3-Flash-DFlash2@dc77ff1…`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/tree/dc77ff1c99eeb2df044ee3d4f0094eb033fee410), target FP8 MLA, target DCP2, replicated BF16 DFlash cache, DFlash K5, C16 scheduler capacity, a 1,048,576-token limit, a 2,048-token batch cap, and GPU memory utilization 0.950.

## Headline code-agent decode

The prompt asks the model to return a complete, precisely typed replacement for a buggy async Python task runner. Each sequence emits exactly 256 tokens at temperature 0.2. Timing starts on the first streamed token and ends on the last, so TTFT/prefill is excluded. Every point had two warmups and three measured runs; throughput is aggregate across the active requests.

| Concurrency | Median tok/s | Three-run range | Median draft acceptance |
|---:|---:|---:|---:|
| C1 | **199.1** | 189.3–199.4 | 68.28% |
| C2 | 311.0 | 306.4–313.9 | 65.67% |
| C4 | 541.4 | 539.8–550.6 | 66.75% |
| C8 | 542.5 | 249.5–570.4 | 67.25% |
| C16 | **800.9** | 741.8–810.3 | 66.24% |

The C8 range is real: its first measured pass encountered late first-seen TileLang/Triton/CuTe compilation and fell to 249.5 tok/s. It is retained rather than silently discarded. A separate final K5/block-128 C1+C16 validation measured **199.8 / 838.3 tok/s** medians.

Raw reports: [final C1–C16 curve](dflash2-fp8-code-agent-final.json) and [final block-128 validation](dflash2-fp8-code-agent-k5-block128-validation.json).

## DFlash depth tuning

The same code-agent workload was swept at K3, K5, and K7. These tuning runs preceded the final draft-cache block-size optimization, so use them to compare K policy rather than as the release headline.

| DFlash depth | C1 median | C16 aggregate median | C1 acceptance | C16 acceptance | Reading |
|---:|---:|---:|---:|---:|---|
| K3 | 170.7 | **846.5** | 81.53% | 80.72% | Best loaded-throughput profile |
| K5 | **195.8** | 753.0 | 67.46% | 66.18% | Best latency/throughput compromise |
| K7 | 193.0 | 491.4 | 50.77% | 53.44% | Too much verification work here |

K5 therefore ships as the agent-oriented default; `DFLASH_TOKENS=3` is the explicit high-concurrency alternative. The upstream vLLM DFlash2 path is fixed-K within a request. The recipe's adaptive controller applies to native MTP, not DFlash2, because genuine DFlash adaptation must alter proposal depth inside the block-diffusion/selector loop rather than merely pick a launcher profile.

Raw reports: [K3](dflash2-fp8-code-agent-k3-tuning.json), [K5](dflash2-fp8-code-agent-k5-tuning.json), and [K7](dflash2-fp8-code-agent-c1-c16.json).

## Prefill by existing depth

Each point is C1 with an exact-length unique prompt. Prefix caching cannot reuse an earlier block. Timing spans the client request through the first streamed token, including server tokenization and the one-token handoff. Every depth was warmed independently and then measured three times.

| Prompt tokens | Median effective prompt tok/s | Median TTFT |
|---:|---:|---:|
| 2,048 | 4,067.7 | 0.503 s |
| 8,192 | 4,518.0 | 1.813 s |
| 32,768 | 4,480.6 | 7.313 s |
| 65,536 | 4,428.7 | 14.798 s |
| 128,000 | **4,382.1** | **29.210 s** |

Raw three-run data: [FP8 prefill curve](dflash2-fp8-prefill-c1.json).

## Standard seven-workload GLMRT blend

This is the usual seven-type blend: code, math/reasoning, creative prose, a short greeting, exposition, structured JSON, and Traditional Chinese. Each case ran three times at C1. The aggregate is total decode tokens divided by aggregate pure-decode wall time, not an average of the seven medians.

| Workload | Median tok/s | Median acceptance | Contract |
|---|---:|---:|---:|
| Code | 220.0 | 78.78% | pass 3/3 |
| Math | 216.7 | 78.33% | pass 3/3 |
| Fable | 102.8 | 25.11% | pass 3/3 |
| Hello | 157.0 | 53.33% | pass 3/3 |
| Topic | 131.4 | 37.60% | pass 3/3 |
| Structured JSON | 197.8 | 68.57% | pass 3/3 |
| Multilingual | 107.7 | 26.92% | pass 3/3 |
| **Weighted aggregate** | **132.1** | **38.99%** | **pass 21/21** |

GLM wrapped the exact JSON object in a `json` fence in all three structured-output runs. The validator treats that as a presentation quirk rather than a semantic failure, and the raw previews make the decision auditable.

Raw report: [DFlash2 FP8 seven-workload blend](dflash2-fp8-glmrt-seven-blend-final.json).

## Cache capacity: two numbers, two meanings

The final engine starts with 15.88 GiB of available cache memory per GPU and 598 shared block IDs.

| Capacity view | Result | Meaning |
|---|---:|---|
| Physical target reservoir | **4,286,464 token slots** | 598 blocks × 3,584 target tokens/rank × DCP2 |
| vLLM full-request equivalent | **2,786,881 tokens** | Shared-pool capacity after recurrent and per-request DFlash scratch accounting |
| Max-length concurrency | **2.66×** | vLLM metric divided by 1,048,576 |

The physical figure is the apples-to-apples comparison with the prior approximately 4.55M FP8 target reservoir. The smaller number is not a second physical measurement: vLLM converts a heterogeneous shared KV pool into complete request capacity, and every DFlash request needs scratch blocks in addition to target cache blocks.

The first DFlash port inherited a 16-token allocation page for replicated draft attention, so a long request consumed 257 draft block IDs. The final port uses a qualified 128-token draft page and needs 33, lifting vLLM's request-equivalent estimate from roughly 1.16M to 2.79M without changing FP8 target precision.

## Exact cold 1M retrieval and stability

The final K5/block-128 candidate received one exact **1,000,000-token** uncached prompt containing six unique key/value needles after token offsets 50K, 250K, 500K, 750K, 950K, and 990K. It recovered all six values, produced 128 completion tokens, and remained healthy.

| Check | Result |
|---|---:|
| Retrieval | **6/6 pass** |
| Prompt construction | 3.983 s |
| Server request | **266.773 s** |
| Prompt SHA-256 | `14d5a1dccd2e5f3369741ed764e26491a3751d26393c084e97d73741c1ae6106` |

This was a cold unique cache salt on the final release configuration. The server logs also confirmed the B12x PCIe DCP owner/top-k exchange on the long prefill.

Raw report: [final 1M six-needle test](dflash2-fp8-1m-multi-needle-final.json).

## Vision contract

The target uses the restored official Z.ai multimodal processor/template. A single request containing 16 generated images labeled 1 through 16 returned the exact ordered list. A request with image 17 was rejected by the configured server-side limit.

| Check | Result |
|---|---:|
| 16-image request | **pass**, HTTP 200, exact `1…16` |
| Prompt size | 4,168 tokens |
| 17-image request | **pass**, HTTP 400: at most 16 images |

The target model performs vision encoding and token verification. The DFlash2 draft consumes text-side inputs for multimodal requests; vLLM logs that behavior explicitly.

Raw report: [16-image boundary test](dflash2-fp8-vision.json).

## Historical profiles

The earlier NVFP4/FP8 adaptive-MTP and no-MTP artifacts remain in this directory for reproducibility. They were not rerun or mixed into the DFlash2 headline. Git tag `v0.4.0` preserves the previous default recipe and its original report narrative.
