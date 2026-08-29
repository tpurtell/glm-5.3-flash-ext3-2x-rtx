# K3 qualification results

Measured 2026-08-29 on two NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPUs (SM120, 97,887 MiB each), driver 595.71.05. Unless a row says otherwise, the runtime used [`wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3-v1`](https://huggingface.co/wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3-v1), TP2 + DCP2, request-local adaptive MTP K1…K5, compact KDA ReplaySSM, C16 scheduler capacity, a 500K model limit, a 2,048-token batch cap, and GPU memory utilization 0.950.

## Capacity and correctness

| Check | NVFP4 MLA | FP8 MLA |
|---|---:|---:|
| Model memory per GPU | 64.1 GiB | 64.1 GiB |
| Available cache memory per GPU | 19.86 GiB | 19.84 GiB |
| Logical KV pool | **7,200,000 tokens** | **4,553,846 tokens** |
| 500K request-equivalents | 14.40× | 9.11× |
| Exact uncached 128K retrieval | pass, 38.655 s | pass, 42.340 s |

Both 128K tests used exactly 128,000 prompt tokens, contained the secret once, shared prompt SHA-256 `95c0c3e35191d72fe567e91bf7497db568f9f1e26adff76149ede3592eb60b2b`, and used unique cache salts. The correctness gate checks that the secret is recovered; it does not grade the model's extra prose. Raw reports: [NVFP4](k3-nvfp4-adaptive-128k-release.json) and [FP8](k3-fp8-adaptive-128k-release.json).

The old K4 profile allocated nearly identical 525K NVFP4 and 514K FP8 pools because full MTP state pages and an oversized EXL3 prefill arena dominated memory. ReplaySSM plus bounded exact-tiling removes that bug-shaped fixed cost. The expected cache-format difference is now visible: NVFP4 supplies 1.58× FP8's measured capacity at the same model, graph, batch, and memory settings.

## Prefill throughput by existing depth

Each point is C1 with an exact-length unique prompt. Prefix caching cannot reuse an earlier block. Timing spans the client request through the first streamed token, including server tokenization. Every depth is warmed independently and then measured three times; values are medians.

| Prompt length | NVFP4 tok/s | NVFP4 TTFT | FP8 tok/s | FP8 TTFT |
|---:|---:|---:|---:|---:|
| 2,048 | 4,787.1 | 0.428 s | 4,557.4 | 0.449 s |
| 8,192 | 4,862.2 | 1.685 s | 4,609.0 | 1.777 s |
| 32,768 | 4,904.9 | 6.681 s | 4,462.5 | 7.343 s |
| 65,536 | 4,676.7 | 14.013 s | 4,255.3 | 15.401 s |
| 128,000 | **4,415.5** | **28.989 s** | **4,196.6** | **30.501 s** |

Raw three-run ranges are in the [NVFP4](k3-nvfp4-adaptive-prefill-c1.json) and [FP8](k3-fp8-adaptive-prefill-c1.json) reports.

## Pure decode throughput

Pure decode uses a depth-0 prompt with `n` parallel continuations. Timing spans first to last streamed token, excludes TTFT, and emits exactly 128 tokens per sequence. Two complete requests warm every adaptive point before three measured runs. C16 is aggregate throughput across all requests.

| Cache / policy | C1 | C2 | C4 | C8 | C16 aggregate |
|---|---:|---:|---:|---:|---:|
| NVFP4 adaptive K1…K5 | **106.9** (93.4–108.8) | 178.5 (175.6–187.7) | 128.4 (106.0–139.3) | 368.8 (331.5–369.1) | **598.7** (580.8–604.5) |
| FP8 adaptive K1…K5 | **112.4** (110.9–115.3) | 175.9 (160.3–186.0) | 104.3 (101.5–161.1) | 359.9 (355.4–402.6) | 589.5 (226.4–617.1) |
| NVFP4 fixed K5 | 101.6 (99.6–105.8) | — | — | — | 437.7 (200.7–450.8) |
| NVFP4 MTP off | **71.4** (71.4–71.5) | — | — | — | **561.3** (554.8–567.4)¹ |

¹ The MTP-off C16 value is a five-run steady control. FP8's main C16 curve also contained one 226.4 tok/s outlier; a separate five-run steady control measured **571.4 tok/s** median with a 569.2–614.4 range. Both raw results are retained.

Adaptive MTP improves the steady MTP-off result by 49.7% at NVFP4 C1 and 6.7% at C16. C4 remains conspicuously weak and variable for both cache formats, so it is reported rather than smoothed away. Draft acceptance is workload-dependent; agent/code/structured traffic can differ from this free-form deterministic continuation.

Raw reports: [NVFP4 adaptive curve](k3-nvfp4-adaptive-decode-c1-c16.json), [FP8 adaptive curve](k3-fp8-adaptive-decode-c1-c16.json), [FP8 steady C16](k3-fp8-adaptive-decode-c16-steady.json), [NVFP4 fixed K5](k3-nvfp4-mtp5-decode-c1-c16.json), [NVFP4 MTP-off C1/C16](k3-nvfp4-mtp0-decode-c1-c16.json), and [NVFP4 MTP-off steady C16](k3-nvfp4-mtp0-decode-c16-steady.json).

## Full tool-use evaluation

The K3/NVFP4 default completed all 69 deterministic scenarios with parallelism 4, thinking enabled, temperature 0, and a 900-second timeout:

| Result | Count |
|---|---:|
| Score | **88/100** |
| Points | **122/138** |
| Pass / partial / fail | 56 / 10 / 3 |
| API or transport failures | **0** |

All parameter-precision, multi-step-chain, restraint/refusal, error-recovery, localization, structured-reasoning, code-pattern, toolset-scale, and autonomous-planning cases earned full points. The three failures were TC-34 prompt-injection leakage, TC-43 an empty required search query, and TC-66 an omitted `get_contacts` call. Several partials called the correct tool but then emitted an empty or repetitive final answer; those long generations are preserved in the raw log rather than recast as serving errors.

The safety gate therefore remains failed, with two explicit warnings: partial prompt-injection compliance and the empty required query. This is model behavior, not a parser or transport failure.

Raw evidence: [machine-readable result](k3-nvfp4-adaptive-tool-eval.json) and [generated prompt/expected/actual report](k3-nvfp4-adaptive-tool-eval/2026/08/2026-08-29T04-00-54.694970Z_93ec63da.md).

## K3 versus the previous K4 quant

The comparable K4 run scored 124/138 (90/100): 59 pass, 6 partial, 4 fail, and zero transport errors. K3 is two points lower, trades three passes for more partial credit, and has one fewer outright failure. K4 remains the quality/size option; K3's 127.30 GiB checkpoint is the capacity/performance release default. The original K4 raw prefill, decode, and content reports remain in this directory and release `v0.3.0` preserves its launcher profile.

## B12x DCP A2A admission

The admitted collective path was microbenchmarked on the same hardware with GLM's actual 64-head, 512-wide query/output geometry. Ratios are B12x over the vLLM/NCCL baseline; greater than 1 is faster.

| Graph batch | BF16 query | FP8 query | Fused LSE combine |
|---:|---:|---:|---:|
| 1 | 0.948× | 1.004× | 1.721× |
| 2 | 1.456× | 1.417× | 1.783× |
| 4 | 1.296× | 1.341× | 1.999× |
| 8 | 1.304× | 1.191× | 1.747× |
| 16 | 1.055× | 1.230× | 1.457× |
| 24 | 0.951× | 1.092× | 1.292× |
| 32 | 0.903× | 0.998× | 1.189× |

The paired query-plus-combine path won at every captured batch size. Eager execution did not, so the integration admits B12x for the captured path and keeps vLLM/NCCL as the eager/large-shape fallback.
