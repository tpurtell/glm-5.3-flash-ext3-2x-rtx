# B12x update audit for v0.6.0

This audit compares the v0.5.0 pin
`988246c8b007c9c1c2006eb677f6fa4b26aeb561` with the reviewed public fork
commit `611ffe8712e40e9ed0110e3cfb1d0b7f4580e631`. The old pin is an ancestor of
the new pin; the exact range contains 145 commits. The last two commits add
focused GLM EP2 mapping/repeated-route tests and the full-rotation Trellis EP
fix described below. The runtime update through
`4f2a7841cd931cba4f591fdf385e01c267518f9a` is an upstream merge plus reviewed
fixes and RTX PRO 6000 Blackwell policy data.

No source, binary, or behavior was copied from Brandon's private route-128
container. His public recipe is used only as the lead motivating the isolated
TP2 versus TP2+EP2 comparison.

## Relevant changes

| Capability | Representative commit | Applicability and integration status |
|---|---|---|
| Projection-mixed EXL3 K3/K4/K5 routed experts | `9ae32e2` | The runtime can now execute mixed projection rates, but this model is uniformly K3: all 37,152 EXL3 tensor-storage records report 3 bits and all 288 experts have 129 records. The mixed-rate path is therefore available but is not credited with a gain for this quant. |
| High-SM W4A16 routing | `2edb44a` | Directly applicable to the two 188-SM RTX PRO 6000 Blackwell GPUs; selected through B12x policy rather than duplicated in the vLLM adapter. Runtime dispatch still requires matrix evidence. |
| Routed-row-scaled packing | `3b5f371` | Directly applicable to sparse top-8 MoE traffic and especially important after EP2 filters global routes to 144 local experts. |
| Fixed prefill-tail route arenas | `56ab5f4` | Directly applicable to variable prefill tails and CUDA graph replay. |
| Preallocated route histograms | `85d3681` | Directly applicable; removes capture-time/replay allocation from route histogram handling. |
| Mapped W4A16 route namespaces | `3a6a204` | Directly applicable to global-to-local EP expert maps. |
| GLM-5.3 sparse MLA contract | `f7c7fd9` | Directly applicable to the existing B12x sparse MLA bridge and FP8 MLA cache profile. |
| GLM C4 indexer reuse | `903667d` | Directly applicable; the public module was also renamed from `nsa_indexer` to `dsa_indexer`, so the recipe bridge and build probe were updated. |
| GLM pooled selector | `0ca533f` | Directly applicable to pooled sparse selection; the existing vLLM bridge remains the owner of runtime metadata and cache topology. |
| GLM KDA and mHC contracts | `a01f94c` | Directly applicable to the recurrent/KDA and mHC paths already enabled by the recipe. |
| Topology-scoped fused PCIe all-reduce | `ce7f622` | Potentially applicable to this PCIe-only dual-GPU host. It is retained as an isolated matrix arm and is not enabled together with contradictory disabled/custom-reduce settings. |
| RTX PRO 6000 Blackwell profiles | `59944c5` | Directly applicable to this exact GPU family. Runtime policy resolution and dispatch receipts are required before performance attribution. |
| Finalized MCG Trellis tensors and FP16 rotation scratch | `20319ba`, `4f2a784` | Directly applicable to EXL3 Trellis preparation and full-rotation correctness. |
| Full-rotation Trellis EP graph execution | `611ffe8` | Fixes the audited EP API so BF16 activations use the intended FP16-internal rotation payload, graph-stable rotation/barrier buffers, and mapped-direct GEMM scratch without capture-time allocation. |

The requested H64 NoPE query projection is present through B12x's public
`mla_query_projection.run_glm_h64_bf16` interface and remains wired by the
recipe. The isolated matrix retains vLLM's AG/RS topology and selects B12x's
eligible graph-captured DCP substitution; eager and unsupported shapes keep
the ordinary vLLM path.

## EP2 ownership and adapter work

B12x already exposed the `ep_moe` planning/execution API at the old pin. The
blocking issue was the EXL3 vLLM adapter: its custom weight loader bypassed
vLLM's routed-expert loader and therefore could not safely translate global
expert IDs to rank-local slots. Simply removing its EP guard would have loaded
global experts 144--287 into nonexistent or incorrect local slots on rank 1.

The v0.6 adapter now:

- consumes vLLM's static `int32` global-to-local expert map;
- validates complete one-to-one coverage of all 144 local slots;
- maps or skips checkpoint weights during loading rather than after loading;
- retains global top-8 IDs and weights, allowing B12x to filter rank-local
  routes without changing their order;
- plans and binds B12x `ep_moe` with fixed decode/prefill output workspaces;
- leaves the final cross-rank reduction to vLLM's MoE runner; and
- fails closed for EPLB, rank-sliced checkpoint metadata, mixed-bitrate EP,
  non-BF16 activations, or a configuration that permits a generic fallback.

Focused evidence at this pin:

- 14 CPU/API cases passed for linear and round-robin placement, both ranks,
  GLM's 288-global/144-local geometry, local slot boundaries 127, 128, and
  143, fixed scratch capacity, and the full-rotation Trellis arena contract.
- 3 CUDA API cases passed on an RTX PRO 6000 Blackwell: ordinary rank partials
  reduce to the full W4A16 reference; graph replay observes changed and
  empty-local routes; and real MCG K3 Trellis rank partials match the full TP
  reference above 0.999 cosine and replay under a CUDA graph.
- Three focused production-attention cases also pass: allocation-free 2,051-row
  GLM prefill replay and two independent 64-bit high-page offset tests, for 20
  focused tests total. The exact test names and timings are retained in
  `release-b12x-focused-tests.xml`.

The focused gates now have matching full-serving evidence. TP2+EP2 loaded all
288 global / 144 local experts per rank, logged live replicated-input B12x EP
dispatch, completed graph capture/replay, passed seven deterministic content
contracts, emitted the expected Berlin `get_weather` tool call, accepted 16
images while rejecting 17, and retained DFlash2 acceptance. The controlled
matrix selects EP2 at 1,004.9 code-agent tok/s for C16 versus the repeated
814.1 tok/s no-EP control, with C1 effectively tied. Full qualification on
the selected profile then reached 222.6 tok/s at C1, 1,067.2 tok/s aggregate
at C16, 4,270.1 prompt tok/s at 128K, 2,912,711 request-equivalent FP8 cache
tokens, and six-of-six retrieval in one exact 1M-token request. The final
startup-gated image handled 30 post-ready API requests with zero post-ready
JIT warnings.

## Patch redundancy review

The new B12x revision provides the kernels, planners, policies, and contracts;
it does not provide this base image's GLM-5.3/vLLM wiring. The recipe's vLLM
bridge patches therefore remain necessary at this stage. The recipe bridge
adopts the canonical `b12x_trellis`/MCG preparation contract and migrates all
stale copied `nsa_indexer` imports to the current `dsa_indexer` module. The
Trellis EP rotation/scratch fix lives in B12x itself rather than being
duplicated in the vLLM adapter.

No bridge patch is removed on source similarity alone. A patch may be removed
only after the no-EP and EP candidates prove equivalent dispatch, correctness,
graph replay, and output behavior with that patch absent. This keeps the audit
from turning an upstream symbol match into an unverified runtime claim.
