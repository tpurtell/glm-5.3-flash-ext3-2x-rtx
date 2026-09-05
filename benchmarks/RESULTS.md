# v0.7.0 projection-mixed K3.25 + DFlash2 FP8 qualification

Measured 2026-09-05 through 2026-09-06 on two PCIe-connected NVIDIA RTX PRO
6000 Blackwell Workstation Edition GPUs (SM120, 97,887 MiB each), driver
595.71.05. Every published point was captured with an explicitly verified
**400 W power limit per GPU**.

The release target is
[`wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1@701cd74…`](https://huggingface.co/wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1/tree/701cd7456c13d87bf0147ad946f828a999afb59c).
Its target experts promote exactly 9,072 of 36,288 projections from K3 to K4:
1,701 gate, 2,835 up, and 4,536 down—the requested **3:5:8** allocation.
All 864 MTP projections remain straight K3. Packed weights occupy 136.16 GiB
across 18 shards; native attention, shared-expert, router, vision, embedding,
and normalization tensors are retained.

The runtime pairs that target with
[`incoai/GLM-5.3-Flash-DFlash2@bf582e4…`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/tree/bf582e4eacc1810f76656d1811693ff6c6737d2a),
B12x `fe054789…`, TP2+EP2, target DCP2 AG/RS, FP8 target MLA, replicated BF16
draft cache, DFlash2 K5, 16 scheduler slots, 2,048 batched tokens, 1,048,576
maximum length, baseline full-state rollback, and memory utilization 0.950.

## Headline code-agent decode

The workload asks for a complete typed replacement of a buggy async Python
task runner. Each sequence emits 256 tokens at temperature 0.2 with a fixed
seed. Every point has two warmups and five measured runs. Pure decode times
each request from its first streamed token through its last; TTFT/prefill is
excluded, and concurrent throughput is the sum of per-request rates.

| Concurrency | Median pure decode tok/s | Five-run range | Per request | Draft acceptance | Committed / target pass |
|---:|---:|---:|---:|---:|---:|
| C1 | **213.0** | 206.3–229.2 | 213.0 | 67.93% | 4.40 |
| C2 | 336.6 | 316.2–379.7 | 168.3 | 69.22% | 4.46 |
| C4 | 480.4 | 462.4–524.0 | 120.1 | 67.26% | 4.36 |
| C8 | 533.6 | 519.4–541.1 | 66.7 | 66.89% | 4.34 |
| C16 | **832.2** | 828.9–894.5 | **52.0** | 68.46% | 4.42 |

K3.25 C1 is 4.3% below the prior v0.6 K3 headline of 222.6 tok/s and remains
inside the release's 5% C1 budget. The raw report retains the conservative
whole-batch window separately; it is not mislabeled as pure decode.

## Same-image uniform-K3 control

The final runtime then loaded the published uniform-K3 model with the exact
same DFlash/cache/topology/rollback flags. It passed 7/7 content contracts and
recorded no post-ready JIT.

| Measurement | Uniform K3 | K3.25 | K3.25 delta |
|---|---:|---:|---:|
| Code-agent C1 | 215.4 | 213.0 | **−1.1%** |
| Code-agent C16 aggregate | 1,119.5 | 832.2 | −25.7% |
| 32K prefill | 4,538.3 | 4,593.4 | **+1.2%** |
| 128K prefill | 4,467.0 | 4,572.0 | **+2.4%** |

The fixed-tier path therefore did not regress, and projection mixing is
effectively free at the C1 agent target while improving prefill slightly. The
saturated C16 loss is real. A matched torch profile attributes it to the
projection-mixed Trellis kernel: rank 0 spent 3,267.9 ms across 2,814 launches
versus 2,009.0 ms across 2,940 for K3; the dominant mixed shape used 241
registers and 69.6 KiB shared memory versus 141 registers and 35.8 KiB.
Attention and PCIe collective time remained comparable. v0.7 ships the honest
result instead of hiding that remaining mixed-kernel optimization opportunity.

## Decode after existing context

These are C1 code-agent requests after unique existing context. Timing still
excludes prefill and TTFT.

| Existing depth | Median tok/s | Three-run range | Median acceptance |
|---:|---:|---:|---:|
| 0 | 209.7 | 190.2–221.1 | 67.93% |
| 8K | 204.2 | 197.0–214.8 | 67.93% |
| 32K | 203.8 | 194.2–208.0 | 68.62% |
| 64K | 213.0 | 192.1–213.0 | 71.07% |
| 128K | 190.5 | 181.9–218.6 | 61.59% |

## Cold FP8 prefill

Each point is a uniquely salted exact-length C1 prompt. Timing includes server
tokenization through the first generated token. Each length was warmed and
then measured three times.

| Prompt tokens | Median prompt tok/s | Median TTFT |
|---:|---:|---:|
| 8,192 | 4,562.8 | 1.795 s |
| 16,384 | 4,533.7 | 3.614 s |
| 32,768 | 4,593.4 | 7.134 s |
| 65,536 | 4,590.4 | 14.277 s |
| 128,000 | **4,572.0** | **27.997 s** |

## Standard seven-workload GLMRT blend

Code, math/reasoning, creative prose, greeting, exposition, structured JSON,
and Traditional Chinese each ran three times at C1. Aggregate throughput is
total decode tokens divided by total pure-decode time.

| Workload | Median tok/s | Median acceptance | Contract |
|---|---:|---:|---:|
| Code | 228.8 | 74.00% | pass |
| Math | 241.4 | 80.00% | pass |
| Fable | 106.7 | 23.15% | pass |
| Hello | 168.9 | 53.33% | pass |
| Topic | 144.1 | 38.42% | pass |
| Structured JSON | 187.0 | 57.65% | pass |
| Multilingual | 117.0 | 27.06% | pass |
| **Weighted aggregate** | **145.7** | **39.19%** | **pass 21/21** |

The exact JSON object is accepted whether bare or wrapped in a `json` fence;
that is presentation, not malformed structured content.

## Full tool-call comparison: 88 cases, including Hard Mode

The refreshed run on **2026-09-06 (Asia/Taipei)** uses
`tool-eval-bench 2.6.1.dev45+gcf54b4bfe` at commit
`cf54b4bfe705f12f71e8866f10730572497c8105`, with `--hardmode` to include
**all 88 available cases: TC-01 through TC-88**. Both models use the public
v0.7.0 image with identical DFlash2 K5, FP8 MLA, TP2/EP2/DCP2, baseline
rollback, and 400 W/GPU settings. Thinking is enabled, temperature is 0,
evaluation parallelism is 8, max turns is 8, timeout is 900 seconds, and the
reference date remains `2026-09-04`.

| Scope | Uniform K3 | K3.25 |
|---|---:|---:|
| Standard 69 cases | 124/138 (**89.9%**) | 127/138 (**92.0%**) |
| Hard Mode, 19 cases | 32/38 (**84.2%**) | 37/38 (**97.4%**) |
| **All 88 cases** | **156/176 (88.6%)** | **164/176 (93.2%)** |

The benchmark rounds the combined scores to **89 and 93**. K3 recorded
73 pass / 10 partial / 5 fail; K3.25 recorded 78 pass / 8 partial / 2 fail.
All 88 cases were graded for each model, with no excluded infrastructure
failures and zero post-ready JIT warnings. This is **one run per model**;
the two subset rows are extracted from those runs, not separate trials.

K3.25 scored higher on TC-40/50/53/58/60/80/85/88, lower on TC-21/61, and
matched the other 78 verdicts. Both failed TC-43 (empty required search
parameter) and TC-51 (invalid, duplicate, or unintended lunch notification).
On Hard Mode, K3.25 passed 18/19 and received partial credit on TC-85;
K3 passed 16/19 and failed TC-80/85/88. Both passed the new concurrency and
pagination cases TC-86/87; K3.25 also passed TC-88's reasoning continuity
test, where K3's first two replies contained tool calls instead of the
required 20-digit answers.

Full evidence: [per-case comparison](v0.7.0-k325/tool-eval-20260906/comparison.json),
[K3 prompts and outputs](v0.7.0-k325/tool-eval-20260906/k3-runs/2026/09/2026-09-05T22-30-44.312928Z_ee526f0c.md),
[K3.25 prompts and outputs](v0.7.0-k325/tool-eval-20260906/k325-runs/2026/09/2026-09-05T22-19-14.605024Z_243726eb.md),
and [all 19 Hard Mode verdicts plus reproduction instructions](v0.7.0-k325/tool-eval-20260906/RUN.md).

The evaluator update includes fixes to date handling, tool-call ordering,
clarification/refusal scoring, and schema compliance, plus reasoning replay
changes. Differences from the older scores therefore do not isolate a
change in model quality. API-assisted cases such as `tool_choice` and
structured output use the same vLLM features for both targets; comparison
with hosted providers requires matching that API support.

### Original release run: 69 cases

The original comparison used `tool-eval-bench 2.3.2.dev3+g5df1e9e0c`.
Both targets ran in the final image with thinking enabled, temperature 0, and
parallelism 8. The benchmark saw no API/runtime errors. K3 scored 88 with
57 pass / 7 partial / 5 fail; K3.25 scored 86 with 55 pass / 8 partial / 6
fail: **121/138 versus 118/138 raw points**. K3.25 improved TC-33,
regressed TC-40/68/69, and matched the other 65
cases. Category deltas were +2 points in Safety & Boundaries, −2 in Toolset
Scale, and −3 in Structured Output; all other categories matched. The
[complete per-case comparison](v0.7.0-k325/tool-eval/comparison.json) and both
raw runs are published, not just the aggregate score.

## ReplaySSM: fixed and qualified, optional by evidence

The launcher previously forwarded `--use-replayssm` only for native MTP, so
selecting it with DFlash2 did nothing. v0.7 wires compact KDA rollback into
DFlash2 and adds the GLM convolution-window and strided materialization fixes.

| DFlash2 K5 measurement | Baseline rollback | ReplaySSM | Delta |
|---|---:|---:|---:|
| Code-agent C1 | 213.0 | 206.9 | −2.8% |
| Code-agent C16 | 832.2 | 856.4 | +2.9% |
| FP8 schedulable tokens | 2,758,919 | 2,940,699 | **+6.6%** |
| 128K prefill | 4,572.0 | 4,441.5 | −2.9% |
| 1M six-needle | 6/6 | 6/6 | equal |

The CUDA materializer matched its Torch reference exactly with deliberately
strided request rows. ReplaySSM and full-state rollback then each passed 120
of 120 rolling C4 requests: thinking off/max, shared/unique 32K prefixes, zero
loops, zero request errors, zero engine deaths, and zero post-ready JIT.
ReplaySSM's broader optional suite passed vision16, 128K prefix replay, and the
1M needle; its semantic blend was 6/7 because one fable had 172 words against
a requested 140–170. It is therefore a transparent capacity/C16 option via
`USE_REPLAYSSM=1`, while baseline rollback remains the C1-oriented default.

## Correctness and long context

- **B12x GPU gate:** 47/47 focused tests pass on both cards. Coverage includes
  mixed K3/K4 parity and graph replay, 288→144 EP maps, changed/empty routes,
  2,051-row allocation-free prefill, 64-bit high-page access, DCP A2A/top-k,
  and fused PCIe all-reduce.
- **Content:** the release default passes all seven deterministic content
  contracts; the seven-type measured blend passes 21/21.
- **Vision:** exact ordered identification passes for 1, 4, and 16 images;
  image 17 receives HTTP 400.
- **Prefix replay:** the exact 128K needle passes twice. The second request
  records 114,688 cache-hit tokens and falls from 29.010 s to 3.357 s.
- **One million tokens:** a cold 1,000,000-token request retrieves six of six
  needles at 5%, 25%, 50%, 75%, 95%, and 99%. TTFT is 261.299 s and total
  request time is 263.905 s.
- **Lifecycle:** the default suite, post-1M C16 soak, K3 control, and ReplaySSM
  A/B all record zero kernel compilation after Docker's ready marker.

Machine-readable evidence is under [`v0.7.0-k325/`](v0.7.0-k325/):
`release-default/` holds the complete headline suite, `k3-control/` the
same-image comparison, `tool-eval/` the original 69-case run,
`tool-eval-20260906/` the updated 88-case run, `b12x-gpu/` the kernel gate,
and `replayssm-qualification/` plus `replayssm-option/` the matched rollback
evidence. Historical NVFP4, MTP, adaptive-MTP, and K3 DFlash measurements were
not rerun or relabeled.

---

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
