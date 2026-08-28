# Qualification results

Measured 2026-08-28 on two NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPUs (SM120, 97,887 MiB each), driver 595.71.05. Release defaults were request-local adaptive MTP K1…K5 with compact KDA ReplaySSM, NVFP4 MLA, TP2 + DCP2, concurrency 16, a 500K max length, a 2,048-token batch cap, and GPU memory utilization 0.950.

## Capacity and correctness

| Check | Result |
|---|---|
| Release KV cache allocation | 525,000 logical tokens; 1.05× one 500K request |
| FP8 profile allocation | 514,285 logical tokens; 1.43× one 360K request |
| MTP-off NVFP4 allocation | 1,560,000 logical tokens at the same 500K/0.950 settings |
| Compact rollback gain | 51,200 → 107,054 tokens (2.09×) under the same 8K-profile control |
| Exact 128,000-token prompt | pass; needle recovered; 92 generated tokens; 34.114 s |
| 500K configuration | pass at 0.950; one full-length request fits with 25,000 tokens spare |
| Aligned prefix reuse | pass; repeated 45,006-token prompt reused 15,360 tokens |
| Concurrency | C16 decode completed 16 × 128 output tokens in every measured run |
| Seven content types | 6/7 pass in 12.337 s; JSON miss was correct JSON wrapped in a fence |
| Tool-use suite | 90/100; 124/138 points; 59 pass, 6 partial, 4 fail; 0 transport errors |

The tool suite used 69 scenarios, 52 tools, parallelism 4, thinking enabled, and a 900-second timeout. Its two safety-gate warnings were partial prompt-injection leakage (without executing it) and an empty required web-search query. These are model-behavior results, not hidden as serving failures.

The strict JSON prompt and expected values were simple: `src/cache.rs`, `replace`, lines 41–47, plus a non-empty rationale. The model returned exactly those values but enclosed the object in a `json` fence, so the strict bare-JSON contract correctly remains a miss. Repeating the request with OpenAI `response_format={"type":"json_object"}` returned bare parseable JSON. This is a presentation-level model miss, not a transport, quantization, or parser failure.

Raw content results are in [seven-content-types-adaptive-mtp.jsonl](seven-content-types-adaptive-mtp.jsonl). The fixed-MTP5 control is retained in [seven-content-types-mtp5-replayssm.jsonl](seven-content-types-mtp5-replayssm.jsonl).

## Prefill throughput

Each point is C1 with an exact-length, unique prompt, so aligned prefix caching cannot reuse an earlier block. Timing spans client request to first streamed token, including server tokenization and the one-token handoff. Every depth is warmed separately, then measured three times. Rates and TTFTs below are medians.

| Prompt length | NVFP4 tok/s | NVFP4 TTFT | FP8 tok/s | FP8 TTFT |
|---:|---:|---:|---:|---:|
| 2,048 | 4,494 | 0.456 s | 4,214 | 0.486 s |
| 8,192 | 4,593 | 1.784 s | 4,294 | 1.908 s |
| 32,768 | 4,641 | 7.060 s | 4,155 | 7.887 s |
| 65,536 | 4,366 | 15.009 s | 4,078 | 16.069 s |
| 128,000 | 4,270 | 29.974 s | 4,045 | 31.643 s |

NVFP4 used the release 500K/2K profile; FP8 used its qualified 360K/1K profile. Both used MTP5, DCP2, C16 scheduler capacity, and memory utilization 0.950. Raw reports: [NVFP4 prefill](nvfp4-mtp5-prefill-c1.json) and [FP8 prefill](fp8-mtp5-prefill-c1.json).

## Pure decode throughput

Pure decode uses one depth-0 prompt with `n` parallel continuations. Timing spans first to last streamed token, excluding TTFT; every sequence emits exactly 128 tokens. Three full 128-token requests warm each adaptive point first, covering lazy JIT and variable-K shapes, and one sampling seed is fixed across repeats so MTP acceptance is comparable. Values are three-run medians with observed ranges.

| Cache / policy | C1 | C2 | C4 | C8 | C16 |
|---|---:|---:|---:|---:|---:|
| NVFP4 adaptive K1…K5 | **106.7** (104.3–110.9) | 154.7 (153.2–160.4) | 132.7 (115.7–163.6) | **345.4** (341.2–351.4) | **335.8** (300.1–338.9) |
| NVFP4 fixed K5 | 98.4 (96.5–104.5) | — | — | — | 276.5 (267.9–284.9) |
| NVFP4 fixed K1 | 79.1 (77.5–81.4) | — | — | — | 335.8 (326.5–342.2) |
| NVFP4 MTP-off | 71.0 (71.0–71.1) | — | — | — | 543.3 (526.1–547.4) |
| FP8 fixed K5 | 96.8 (94.9–102.8) | — | — | — | 174.8 (161.6–179.4) |

On this unconstrained free-form continuation, adaptive MTP improves fixed K5 by 8.5% at C1 and 21.5% at C16. It converges close to the measured K1 ceiling under load while retaining K5 when C1 acceptance pays for it. C4 remains conspicuously weak and variable; the raw point is included rather than smoothed away. Agent, code, and structured workloads can accept deeper drafts, so MTP performance remains workload-sensitive. FP8 changes the target trajectory enough to change draft acceptance too, so its decode result is not expected to scale only with cache bandwidth.

Raw reports: [NVFP4 adaptive curve](nvfp4-mtp-adaptive-request-mean-decode-c1-c16.json), [NVFP4 fixed K1](nvfp4-mtp-runtime-k1-decode-c1-c16.json), [NVFP4 fixed K5](nvfp4-mtp5-decode-c1-c16.json), [NVFP4 MTP-off](nvfp4-mtp0-decode-c1-c16.json), and [FP8 fixed K5](fp8-mtp5-decode-c1-c16.json). Reproduce them with `scripts/benchmark-prefill.py` and `scripts/benchmark-decode.py`.

## B12x DCP A2A admission

Actual GLM geometry was 64 total heads, query width 512, output width 512. Ratios below are B12x over the vLLM/NCCL baseline; greater than 1 is faster.

| Graph batch | BF16 query | FP8 query | Fused LSE combine |
|---:|---:|---:|---:|
| 1 | 0.948× | 1.004× | 1.721× |
| 2 | 1.456× | 1.417× | 1.783× |
| 4 | 1.296× | 1.341× | 1.999× |
| 8 | 1.304× | 1.191× | 1.747× |
| 16 | 1.055× | 1.230× | 1.457× |
| 24 | 0.951× | 1.092× | 1.292× |
| 32 | 0.903× | 0.998× | 1.189× |

The paired query-plus-combine path won at every captured batch size for both query formats. Eager execution did not, so the integration uses B12x only in graph capture and keeps the vLLM/NCCL eager fallback.
