# Qualification results

Measured 2026-08-28 on two NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPUs (SM120, 97,887 MiB each), driver 595.71.05. Runtime defaults were NVFP4 MLA, TP2 + DCP2, concurrency 16, 500K max length, and an 8,192-token batch cap.

## Capacity and correctness

| Check | Result |
|---|---|
| KV cache allocation | 600,000 logical tokens; 1.20× one 500K request |
| Exact 128,000-token prompt | pass; 18 generated tokens; 24.580 s |
| Near-limit 499,000-token prompt | pass; 64 generated tokens; 80.099 s |
| Concurrency | 16/16 simultaneous 8-token chats returned HTTP 200; 1.701 s wall time |
| FP8 MLA option | 523,529-token pool at utilization 0.970; 500K configuration passed |
| Seven content types | 6/7 pass in 18.264 s; JSON miss was valid JSON wrapped in a fence |
| Tool-use suite | 90/100; 124/138 points; 59 pass, 6 partial, 4 fail; 0 transport errors |

The tool suite used 69 scenarios, 52 tools, parallelism 4, thinking enabled, and a 900-second timeout. Its two safety-gate warnings were partial prompt-injection leakage (without executing it) and an empty required web-search query. These are model-behavior results, not hidden as serving failures.

The seven-content raw summary is in [seven-content-types.jsonl](seven-content-types.jsonl).

## Throughput

Each row is the mean of three `pp2048 tg128` runs at concurrency 16.

| Existing depth | Prompt tok/s | Generation tok/s | TTFT | Total |
|---:|---:|---:|---:|---:|
| 0 | 2,791 | 147.3 | 7,668 ms | 12,437 ms |
| 4,096 | 4,202 | 80.3 | 12,906 ms | 20,012 ms |
| 8,192 | 4,451 | 53.5 | 19,236 ms | 30,188 ms |

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
