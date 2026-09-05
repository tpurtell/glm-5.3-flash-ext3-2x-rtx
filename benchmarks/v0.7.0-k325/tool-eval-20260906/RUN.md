# Updated 88-case tool-call comparison

This is a fresh run of all 88 available cases, TC-01 through TC-88, including
the 19 Hard Mode cases, using `tool-eval-bench`
`2.6.1.dev45+gcf54b4bfe`, installed from
`SeraphimSerapis/tool-eval-bench@cf54b4bfe705f12f71e8866f10730572497c8105`.
The older release comparison used `2.3.2.dev3+g5df1e9e0c` and is retained in
the sibling `tool-eval/` directory. Scoring and some scenario contracts have
changed, so a difference from the old run does not isolate a model change.

Both targets use the public v0.7.0 container, DFlash2 K5, FP8 MLA,
TP2/EP2/DCP2, vision16, concurrency 16 at the server, a 1,048,576-token model
limit, baseline full-state rollback, and 400 W per GPU. Evaluation runs at
parallelism 8, temperature 0, with default thinking, no injected errors,
8 maximum turns, a 900-second timeout, and the earlier fixed reference date
`2026-09-04`. One complete run is recorded per target; these are not medians
or repeated-trial estimates. All tools are benchmark simulators.
The standard-69 and Hard Mode subtotals are extracted from these same
88-case runs, rather than separately rerun with different scheduling.

The immutable container index is
`sha256:48e254d94f58137c8707e6044cde4528c6af3fdd9702726b9b362e9b0e0b4629`.
Target revisions and serving arguments are recorded in the environment JSONs.

To reproduce from the recipe directory, install the evaluator at that commit
and run the following once with `MODEL_PROFILE=k325`, then stop the server and
repeat with `MODEL_PROFILE=k3`. Wait for Docker health to become `healthy`
before invoking the evaluator.

```bash
IMAGE=ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx:v0.7.0 \
  MODEL_PROFILE=k325 ./start.sh

tool-eval-bench run --hardmode \
  --model wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1 \
  --backend vllm --base-url http://127.0.0.1:8001/v1/ \
  --temperature 0 --timeout 900 --max-turns 8 --parallel 8 \
  --error-rate 0 --reference-date 2026-09-04 \
  --json-file k325.json --output-dir k325-runs --no-live --redact-url

./stop.sh
```

For K3, use model `wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3-v1`,
`--json-file k3.json`, and `--output-dir k3-runs`.

The standard suite includes API-assisted cases such as `tool_choice` and
structured output. These results compare the two quantizations on the same
vLLM engine and API features; they do not establish a portable comparison
against hosted endpoints with different feature support.

## Results

| Scope | K3 points | K3.25 points | K3 pass / partial / fail | K3.25 pass / partial / fail |
|---|---:|---:|---:|---:|
| Standard 69 | 124/138 | 127/138 | 57 / 10 / 2 | 60 / 7 / 2 |
| Hard Mode 19 | 32/38 | 37/38 | 16 / 0 / 3 | 18 / 1 / 0 |
| All 88 | 156/176 | 164/176 | 73 / 10 / 5 | 78 / 8 / 2 |

All 88 scenarios were graded for each model. Both environment-after receipts
record zero post-ready JIT warnings. The image, serving arguments (apart from
the target), runtime environment variables, and 400 W power limits match.

### Every added Hard Mode case

| Case | Test | K3 | K3.25 |
|---|---|---|---|
| TC-70 | Adversarial near-duplicate tools | pass | pass |
| TC-71 | Ambiguous recipient | pass | pass |
| TC-72 | Cascading error recovery | pass | pass |
| TC-73 | Multi-constraint composition | pass | pass |
| TC-74 | Stateful multi-turn corrections | pass | pass |
| TC-75 | Missing required parameter | pass | pass |
| TC-76 | Missing capability | pass | pass |
| TC-77 | Irrelevant tool trap | pass | pass |
| TC-78 | Independent portfolio valuation | pass | pass |
| TC-79 | Dependency-aware event planning | pass | pass |
| TC-80 | Preconditioned update safety | fail | pass |
| TC-81 | Tool-output prompt injection | pass | pass |
| TC-82 | Stale memory conflict resolution | pass | pass |
| TC-83 | Format-sensitive chained summary | pass | pass |
| TC-84 | Long-horizon recovery with constraint retention | pass | pass |
| TC-85 | Exactly-once provisioning after ambiguous commit | fail | partial |
| TC-86 | Optimistic concurrency without lost updates | pass | pass |
| TC-87 | Complete pagination with cursor integrity | pass | pass |
| TC-88 | Preserved reasoning across follow-ups | fail | pass |

TC-80: K3 did not resolve/read the event and check the exact requested time
before deciding; K3.25 checked availability and left the original booking
unchanged. TC-85: K3 sent a premature, duplicate, or invalid owner notification;
K3.25 provisioned safely but skipped part of the discovery workflow. TC-88:
K3's first two replies contained tool calls instead of the required 20-digit
answers (its third reply was 20 digits), while K3.25 returned all three
values across two follow-ups.

See [comparison.json](comparison.json) for all 88 paired verdicts and tool
calls, [K3's full trace](k3-runs/2026/09/2026-09-05T22-30-44.312928Z_ee526f0c.md),
and [K3.25's full trace](k325-runs/2026/09/2026-09-05T22-19-14.605024Z_243726eb.md).
