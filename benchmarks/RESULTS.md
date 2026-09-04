# v0.6.0 B12x EP2 + DFlash2 FP8 qualification

Measured 2026-08-31 through 2026-09-01 on two PCIe-connected NVIDIA RTX PRO
6000 Blackwell Workstation Edition GPUs (SM120, 97,887 MiB each), driver
595.71.05. Every comparison and release measurement used an explicitly
verified **400 W power limit per GPU**. The user-selected 400 W condition
supersedes the 450 W value in the original test plan.

The release profile is the public target
[`wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3-v1@319d66a…`](https://huggingface.co/wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3-v1/tree/319d66a8b53092b491f698440ecea781e4ddd4e4),
the public draft
[`incoai/GLM-5.3-Flash-DFlash2@dc77ff1…`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/tree/dc77ff1c99eeb2df044ee3d4f0094eb033fee410),
B12x `611ffe8712e40e9ed0110e3cfb1d0b7f4580e631`, TP2+EP2, target
DCP2 AG/RS, graph-captured B12x DCP substitution, FP8 target MLA, replicated
BF16 DFlash cache, DFlash2 K5, 16 scheduler slots, 2,048 batched tokens,
1,048,576 maximum length, and GPU memory utilization 0.950.

The v0.6.1 recipe now pins metadata-only target revision `1e4abd2…`, which
syncs Z.ai's corrected chat template while retaining identical model and
quantization objects. These performance and quality receipts remain labeled
with the exact `319d66a…` revision on which they ran; they were not rerun or
silently relabeled for a template-only patch.

## What changed and what won

The only new material serving flag relative to v0.5 is
`--enable-expert-parallel`. The underlying work is larger: the EXL3 loader now
maps 288 global experts to 144 local slots per EP rank, and the updated B12x
runtime owns the current EP plan, route arenas, MCG Trellis preparation,
full-rotation graph scratch, and SM120 policy.

The clean B-to-C comparison isolates EP2 while holding the new B12x, model,
DFlash, cache, DCP, scheduler, and power constant:

| Measurement | New B12x TP2 | New B12x TP2+EP2 | EP2 delta |
|---|---:|---:|---:|
| Code-agent C1 | 207.2 | 208.6 | +0.7% |
| Code-agent C16 aggregate | 814.1 | 1,004.9 | **+23.4%** |
| 32K prefill | 4,723.7 | 4,728.0 | +0.1% |
| 64K prefill | 4,573.0 | 4,538.2 | -0.8% |
| FP8 request-equivalent capacity | 2,884,749 | 2,912,711 | +1.0% |

EP2 is therefore a loaded-decode win, not the explanation for a separate 6K+
prefill report. Profiling found no route-packing or collective pathology, so a
private-style route-128 specialization was neither copied nor independently
implemented. Public M=128 cannot be selected by the audited W4A16 kernel at
this GLM geometry; M=64 remains the honest qualified prefill path. The full
A–F controls, variance, profiler interpretation, DCP microbench, and PCIe
all-reduce decision are in [the isolated matrix](v0.6.0-b12x-ep2/MATRIX.md).

## Headline code-agent decode

The prompt requests a complete, precisely typed replacement for a buggy async
Python task runner. Each sequence emits exactly 256 tokens at temperature 0.2
and fixed seed. Each point had two warmups and five measured runs. Every
request is timed independently from its first streamed token to its last;
prefill and TTFT are excluded, and concurrency throughput is the sum of those
per-request rates.

| Concurrency | Median pure decode tok/s | Five-run range | Per request | Draft acceptance | Committed tokens / target pass |
|---:|---:|---:|---:|---:|---:|
| C1 | **222.6** | 201.8–227.6 | 222.6 | 69.47% | 4.47 |
| C2 | 355.6 | 326.8–365.3 | 177.8 | 66.00% | 4.30 |
| C4 | 563.4 | 541.7–585.6 | 140.9 | 66.28% | 4.31 |
| C8 | 777.8 | 745.1–816.8 | 97.2 | 66.84% | 4.34 |
| C16 | **1,067.2** | 1,044.5–1,073.8 | **66.7** | 66.15% | 4.31 |

The acceptance-dependent C1 spread is real; lower-throughput samples accepted
fewer drafts, while TTFT stayed near 100 ms. The old whole-batch timing window
would report 908.6 tok/s at C16. It is retained in the raw receipt, but is not
called pure decode because staggered admissions otherwise charge one
request's prefill to another request's generation.

The final dev11 image then ran five immediate C16 samples with **zero client
warmup** after Docker first became healthy: median 1,175.4 tok/s, range
1,096.5–1,198.8, median acceptance 66.49%. Thirty subsequent API requests
covering C16, deterministic content, rendered chat, tools, and vision produced
zero post-ready JIT warnings. This is a lifecycle gate, not substituted into
the same-run C1–C16 headline curve.

The published `v0.6.0` image was then pulled anonymously and launched from a
clean public-repository clone. A genuinely absent vLLM cache reached the
release-ready marker in 426.3 seconds; a restart against the populated cache
did so in 362.5 seconds. TileLang still forms its process-local mHC modules on
restart, so the cached boot is intentionally not described as instant. Both
boots stayed unready until warmup completed, passed 7/7 post-ready content
contracts, and recorded zero post-ready JIT. The public image index is
`sha256:fe249b88d091430d8a88cd987d087d556053f0f067a649f2e9ca95895129e82b`.

Raw receipts: [release curve](v0.6.0-b12x-ep2/release-code-agent-curve.json),
[telemetry](v0.6.0-b12x-ep2/release-code-agent-curve.telemetry.json),
[post-ready C16](v0.6.0-b12x-ep2/release-dev11-c16-post-ready.json), and
[startup/JIT audit](v0.6.0-b12x-ep2/release-dev11-startup-jit-audit.json).
Public artifact receipts: [summary](v0.6.0-b12x-ep2/release-public-artifact-verification.json),
[fresh-cache audit](v0.6.0-b12x-ep2/release-public-fresh-startup-jit-audit.json),
[fresh-cache content](v0.6.0-b12x-ep2/release-public-fresh-content.jsonl),
[existing-cache audit](v0.6.0-b12x-ep2/release-public-existing-startup-jit-audit.json),
and [existing-cache content](v0.6.0-b12x-ep2/release-public-existing-content.jsonl).

### DFlash verification accounting

vLLM's accepted-token counter counts accepted draft tokens and excludes the
one target/bonus token produced by each verification pass. At fixed K5,
effective committed tokens per target pass are therefore
`1 + 5 × draft_acceptance`. Rejected verification work is the complement of
the acceptance rate: 30.53% at C1 and 33.85% at C16 in the headline curve.
Every recorded draft count is exactly divisible by five. The v2 benchmark
schema now records vLLM's `num_drafts` counter directly as
`target_verification_passes`; the retained v1 receipts are exactly derivable
because this release uses fixed K5.

## Decode after existing context

These C1 points use the same code-agent prompt and output budget after unique
existing depths. The reported time remains first-to-last-token decode only.

| Existing depth | Median tok/s | Three-run range | Median acceptance |
|---:|---:|---:|---:|
| 0 | 215.2 | 211.8–215.7 | 69.66% |
| 8K | 187.8 | 184.8–229.3 | 61.27% |
| 32K | 203.4 | 184.0–206.8 | 68.97% |
| 64K | 201.6 | 187.6–212.3 | 66.78% |
| 128K | 208.6 | 196.5–211.4 | 68.28% |

The sparse MLA path avoids a monotonic decode collapse with context depth;
most spread follows DFlash acceptance rather than attention depth. Raw report:
[code-agent depth curve](v0.6.0-b12x-ep2/release-code-agent-depth.json).

## Cold prefill curve

Each point is a unique exact-length C1 prompt. Prefix caching cannot reuse an
earlier block. Timing covers the client request through first token, including
server tokenization and the one-token handoff. Every point was warmed
independently and measured three times.

| Prompt tokens | Median prompt tok/s | Three-run range | Median TTFT |
|---:|---:|---:|---:|
| 8,192 | 4,480.8 | 4,477.6–4,495.0 | 1.828 s |
| 16,384 | 4,312.5 | 4,295.1–4,369.2 | 3.799 s |
| 32,768 | 4,298.1 | 4,296.3–4,315.3 | 7.624 s |
| 65,536 | 4,298.0 | 4,293.7–4,304.9 | 15.248 s |
| 128,000 | **4,270.1** | 4,268.6–4,279.2 | **29.976 s** |

The new 128K result is 2.6% below v0.5's 4,382.1 tok/s, while the controlled
TP2-to-EP2 comparison is flat. In other words, the release chooses a large
C16 decode gain with a small broader-runtime prefill regression; it does not
misattribute prefill to EP2. Raw report:
[prefill curve](v0.6.0-b12x-ep2/release-prefill-curve.json).

## Standard seven-workload GLMRT blend

The standard blend covers code, math/reasoning, creative prose, a greeting,
exposition, structured JSON, and Traditional Chinese. Each case ran three
times at C1; the aggregate is total decode tokens divided by total pure-decode
time.

| Workload | Median tok/s | Median acceptance | Contract |
|---|---:|---:|---:|
| Code | 238.2 | 79.25% | pass 3/3 |
| Math | 222.5 | 72.31% | pass 3/3 |
| Fable | 115.3 | 26.88% | pass 3/3 |
| Hello | 173.3 | 53.33% | pass 3/3 |
| Topic | 142.5 | 37.85% | pass 3/3 |
| Structured JSON | 218.2 | 68.57% | pass 3/3 |
| Multilingual | 116.1 | 27.10% | pass 3/3 |
| **Weighted aggregate** | **145.5** | **40.33%** | **pass 21/21** |

The blend commits 3.02 tokens per target verification pass at K5. GLM wraps
the exact JSON object in a `json` fence; the validator treats that as a
presentation quirk and retains the raw preview. The v0.5 and v0.6 fixed
temperature-zero runs both pass 21/21 semantic contracts. Greeting and JSON
outputs are stable; unconstrained prose is semantically equivalent rather than
byte-identical. Raw report:
[seven-case blend](v0.6.0-b12x-ep2/release-glmrt-seven-blend.json).

## FP8 cache capacity and memory

vLLM reports **2,912,711 full-request-equivalent tokens**, or **2.78×** the
1,048,576-token request maximum. This heterogeneous-pool scheduler figure
accounts for target MLA, recurrent state, and per-request DFlash scratch. It
improves 4.5% over v0.5's 2,786,881-token result.

At startup, the limiting rank reports 16.59 GiB available for KV cache. The
engine's 0.950 budget is 90.22 GiB/GPU; its log separates roughly 70.04 GiB of
weights/non-Torch allocations, 3.60 GiB peak activation, 0.23–0.24 GiB actual
CUDA-graph pool, and 16.59–16.61 GiB current KV cache. C16 is the qualified
scheduler fan-out; 2.78× maximum-length capacity does not imply sixteen
simultaneous 1M prompts.

## Correctness, graph replay, and long context

- **MoE and graph parity:** 20/20 focused B12x tests pass. They cover both EP
  ranks and placement modes, 288/144 maps and boundary IDs, finite rank
  partials against full W4A16 and real MCG K3 references, >0.999 Trellis
  cosine, changed and empty routes under graph replay, and allocation-free
  prefill replay. [JUnit receipt](v0.6.0-b12x-ep2/release-b12x-focused-tests.xml)
- **64-bit page offsets:** two CUDA tests force live page offsets beyond the
  32-bit boundary in the GLM cache writer/reader and fused indexer. Both pass
  in the same JUnit receipt.
- **DFlash mask semantics:** the real K5 mask token, decode and chunked-prefill
  anchors, rejection rollback, sample positions, request mapping, and inert
  padding all pass. [Mask receipt](v0.6.0-b12x-ep2/release-dflash2-mask-semantics.json)
- **Content:** seven deterministic full-model cases pass on the final image.
  [Receipt](v0.6.0-b12x-ep2/release-dev11-post-ready-content.jsonl)
- **Tool call with thinking enabled:** prompt “What's the weather like in
  Berlin right now?” expects `get_weather` with Berlin only; the engine emits
  `get_weather {"location":"Berlin"}` and scores 100/100.
  [Expected/actual report](v0.6.0-b12x-ep2/release-dev11-post-ready-tool-run/2026/09/2026-09-01T01-56-18.066797Z_7ced63aa.md)
- **Vision:** exact ordered identification passes for 1, 4, and 16 images;
  image 17 receives HTTP 400. [Receipt](v0.6.0-b12x-ep2/release-dev11-post-ready-vision.json)
- **Prefix replay:** an exact 128,000-token needle passes twice. The second
  request records 114,688 cache-hit tokens and falls from 30.639 s to 3.735 s.
  [Receipt](v0.6.0-b12x-ep2/release-prefix-replay-128k.json)
- **One million tokens:** one cold, uniquely salted 1,000,000-token request
  retrieves six of six needles at 5%, 25%, 50%, 75%, 95%, and 99%. TTFT is
  258.203 s, the remaining stream is 2.628 s, and total request time is
  **260.831 s**. [Receipt](v0.6.0-b12x-ep2/release-1m-multi-needle.json)
- **Post-1M stability:** a C16 soak remained healthy. Its first sample exposed
  a previously lazy GLM mHC TileLang specialization; the final entrypoint now
  prewarms GLM mHC plus all subsequently observed route, DFlash, sampling,
  long-prefill, and C16 shapes. Final dev11 then served the post-ready suite
  with zero JIT.

## Power, clocks, and observer effects

The 1M endpoint snapshot caught both GPUs still busy at P1: 97%/96% SM
utilization, 393.41/399.73 W draw, 2,700/2,302 MHz SM clocks, 83/90 °C, and the
400 W configured limit. GPU 1 reported software power-cap activity; neither
GPU reported hardware slowdown, hardware thermal slowdown, power-brake
slowdown, or software thermal slowdown. At 1,000,000 / 258.203 = 3,872.9
prompt tok/s, that endpoint is approximately 4.88 prompt tok/s per observed
watt across the two GPUs.

Continuous `nvidia-smi` observation perturbed this host's PCIe/driver path and
could split C16 admissions. Published timing therefore uses detailed snapshots
only before and after the workload. Cap-normalized efficiencies are 0.278 C1
decode tok/s/W, 1.334 C16 aggregate tok/s/W, and 5.338 128K prefill tok/s/W at
the configured 800 W total ceiling. These conservative cap-normalized values
are not mislabeled as simultaneous power samples. The intrusive traces remain
in the matrix directory as diagnostic evidence, not headline performance.

## Historical results

The older NVFP4, adaptive-MTP, no-MTP, and v0.5 DFlash receipts remain under
`benchmarks/` for reproducibility. They were not rerun or relabeled as v0.6
results. Git tag `v0.5.0` preserves the previous report narrative; the new
raw evidence is isolated under `benchmarks/v0.6.0-b12x-ep2/`.
