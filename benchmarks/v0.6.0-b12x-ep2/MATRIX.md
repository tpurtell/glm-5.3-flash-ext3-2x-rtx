# v0.6.0 isolated B12x / EP2 matrix

All measurements in this directory used the same two RTX PRO 6000 Blackwell
GPUs at an explicitly verified **400 W limit per GPU**. Unless a row says
otherwise, the model, FP8 MLA cache, DFlash2 K5 draft, DCP2 AG/RS topology,
1,048,576-token limit, C16 scheduler capacity, 2,048-token batch cap, and
0.950 memory utilization are unchanged.

## Results so far

| Candidate | 32K prefill | 64K prefill | Code-agent C1 | Code-agent C16 aggregate | DFlash acceptance C1 / C16 | Request-equivalent FP8 tokens | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| A: v0.5.0, old B12x, TP2 | 4,805.8 | 4,610.7 | 197.1* | 783.2* | not captured in the A quick screen | 2,786,881 | Baseline |
| B: new B12x, TP2 | 4,723.7 | 4,573.0 | 207.2 | 814.1 | 67.1% / 66.2% | 2,884,749 | Upstream-only control |
| C: new B12x, TP2+EP2, B12x captured DCP | **4,728.0** | **4,538.2** | 208.6 | 1,004.9 | 65.0% / 65.7% | 2,912,711 | **Selected release profile** |
| E: C with captured B12x DCP disabled | 4,527.4 | 4,379.2 | **213.4** | **1,022.1** | 71.1% / 65.3% | **2,922,031** | Control; not selected |

Rates are tokens/s and medians. Candidate B's repeated C16 range is
791.2--859.7 tok/s; candidate C's range is 967.1--1,057.8 tok/s. The ranges
are wider than 3%, but they do not overlap, both retain roughly 66% DFlash
acceptance, and neither run reported hardware thermal or hardware slowdown.
Candidate C improves the repeated B median by 23.4% at C16 while C1 changes by
only 0.7%. Prefill is unchanged within measurement noise, so EP2 does **not**
explain the separate 6K+ prefill report.

`*` Candidate A's early decode screen used the fixed generic continuation
prompt, not the release's code-agent prompt, so those two cells are provenance
rather than a headline old/new claim. B and C were rerun with the exact
code-agent prompt, 256 output tokens per sequence, fixed seed and temperature,
two warmups, five measured runs, and Prometheus acceptance-counter deltas.

The generic continuation is strongly workload-sensitive: C measured 618.9
tok/s at C16 and 135.2 tok/s at C1 there, versus B's 777.6 and 214.7. That
result is retained, not relabeled as a code-agent benchmark. It is also why the
release must state the headline workload rather than presenting one decode
number as universal.

## What EP2 changes

vLLM changes MoE parallel geometry from 288 experts at an intermediate width
of 1,024 per TP rank to 144 local experts at the full width of 2,048 per EP
rank. Both topologies retain one final reduction over the physical TP group;
CUDA traces prove that captured reductions select B12x PCIe one-shot when the
shape is eligible and fall through to PyNCCL above its size ceiling.

The hot paired C16 traces do not show route packing as the bottleneck. Median
B12x MoE kernel events were about 38--42 us per rank under EP2 versus 45--46 us
without EP; mapped packing was about 2.0 us versus 1.5 us. EP2's code-agent win
therefore comes from the local-expert/full-width execution geometry and its
interaction with that workload, not a copied route-128 kernel. Trace files and
the profiler summaries are under `profiles/C-ep2/` and `profiles/B-no-ep/`.

## Candidate D: prefill block M=128

This is not a valid configuration of the audited public kernel. B12x's routed
W4A16 block contract supports M in `(8, 16, 32, 48, 64)`. A compile-only probe
temporarily admitted 128 to the public generic selector at GLM's exact EP
geometry (H=4096, I=2048, E=144, top-k=8, 188 SMs, 101,376-byte opt-in shared
memory) and failed with:

```text
ValueError: no valid W4A16 tile config for M/N/K=1024/4096/4096,
moe_block_size=128
```

Thus merely setting `VLLM_EXL3_PREFILL_BLOCK_M=128` would be misleading: the
current EP bridge does not pass that unsupported value, and the honest generic
kernel cannot fit it. Supporting M=128 requires an independently designed
specialization. No private route-128 implementation, binary, or behavior was
copied. Since public EP2 already wins the release workload and M=128 cannot be
formed from the upstream kernel, the release keeps the qualified M=64 path.

## Candidate E: captured B12x DCP versus plain AG/RS

The command-line topology remains vLLM DCP2 with `dcp_comm_backend=ag_rs` in
both arms. `VLLM_B12X_DCP_A2A=1` is a narrower optimization: it replaces the
eligible graph-captured GLM query gather and LSE/output combine with B12x's
PCIe pool, while eager and unsupported shapes retain vLLM's normal path.

The clean, fully warmed DCP-off receipt measured 1,022.1 tok/s at C16
(993.9--1,033.6) and 213.4 tok/s at C1 (210.1--217.3), versus 1,004.9
(967.1--1,057.8) and 208.6 (204.9--223.7) with B12x enabled. Those decode
ranges overlap, and acceptance varies by prompt sampling, so this is a tie
rather than a claimed DCP-off win. Two earlier off receipts contain explicit
JIT-monitor warnings for unseen route/tail kernels and are retained as
contaminated warmup evidence, not folded into the median.

At 32K/64K, the B12x-on run measured 4,728.0/4,538.2 tok/s versus
4,527.4/4,379.2 off. The off run was hotter (GPU 1 reached 92 C versus 89 C)
and its median clock was lower, so the full 3.5--4.4% difference is not
attributed solely to DCP. Neither arm reported hardware or software thermal
slowdown. The capacity cost of the captured pool is only 9,320 request tokens
(0.32%).

The isolated exact-geometry CUDA-graph receipt resolves the path itself. For
batches 1--32, B12x's LSE/output combine is 1.20--2.00x faster than NCCL;
FP8 query gather is 1.00--1.46x, except for no meaningful win at batch 32;
BF16 query gather crosses over by batch 2 and is 1.07--1.41x through batch 16.
The adapter invokes this pool only for captured eligible shapes, which is
where these gains apply. With tied end-to-end decode, slightly better observed
prefill, negligible capacity cost, and graph-level wins, the release keeps
`VLLM_B12X_DCP_A2A=1`.

Raw receipts: `E-dcp-off-code-agent-steady.json`,
`E-dcp-off-prefill-32k-64k.json`, and
`E-dcp-collective-microbench.json` (plus their telemetry sidecars).

## Candidate F: TP-group PCIe all-reduce

CUDA traces already prove that the production TP group dispatches eligible
captured reductions through `B12X_PCIE_ONESHOT` and falls through to PyNCCL
above the configured ceiling. The exact H=4096 standalone control validates
lossless results, alternating graph replay, and an 1.09--1.16x B12x advantage
at the production graph sizes of 8--256 KiB (the 384 KiB serving ceiling lies
between the tested 32- and 64-row points). Eager B12x is intentionally not
enabled: PyNCCL wins those eager samples. Because the hot full-model traces
show no unexplained collective spike, the conditional second full-model boot
was not justified. The selected setting remains graph-only B12x one-shot with
PyNCCL fallback. Raw receipt: `F-pcie-allreduce-microbench.json`.

## Correctness and dispatch gates already passed

- Seven deterministic full-model content cases pass on C.
- The retained structured tool-call case produces the expected `get_weather`
  call for Berlin with thinking enabled.
- A 16-image request passes and image 17 is rejected with HTTP 400.
- CUDA graph capture/replay completed with fixed EP rotation, route, output,
  and barrier workspaces; no Torch/CPU/generic MoE fallback is reachable.
- The serving log records 288 global / 144 local experts and live B12x
  replicated-input EP dispatch.

The isolated matrix therefore selects candidate C: new audited B12x, TP2+EP2,
M=64 routed prefill, captured B12x DCP collectives, and graph-only B12x
one-shot TP reduction with PyNCCL fallback. Full qualification is performed on
that exact profile.

## Qualification follow-up: scheduler cap and telemetry observer

The first full C16 curve exposed intermittent five-second splits between
groups of continuations. This was not an EP, DCP, or one-shot regression: the
same behavior appeared in the no-EP and DCP-off controls. Raising
`max_num_batched_tokens` from 2,048 to 4,096 did not remove it and reduced
request-equivalent FP8 capacity from 2,912,711 to 2,464,357 tokens (15.4%), so
that arm was rejected.

The actual perturbation was continuous NVML observation. Both a freshly
spawned full `nvidia-smi` query each second and persistent `nvidia-smi dmon`
could hold this driver's path long enough to split admissions. Ten equivalent
C16 samples with no observer were 1,062.8--1,195.6 pure-decode tok/s with no
split; ten with boundary-only snapshots were likewise clean. The benchmark
helper now takes detailed power/clock/throttle snapshots only before and after
timed performance. Continuous dmon remains available as an explicitly
intrusive, separate PCIe/power characterization whose TPS is not published.

The old metric also mixed staggered sequence prefill into C16 decode by timing
from the first sequence's first token to the last sequence's last token. Final
reports now time each sequence from its own first to last token, sum those
per-request rates for aggregate pure decode, and retain the conservative batch
window separately. This implements the release requirement that headline
decode exclude prefill/TTFT instead of merely renaming a mixed window.
