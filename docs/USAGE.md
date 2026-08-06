# Usage note and model card — agentic-v1

**One document, not two.** This is the usage note *and* the model card. If the adapter is
ever uploaded to a Hub repository, this same text goes in its card; nothing here is
duplicated into a second file that can drift.

**Dated 2026-08-06.** No registered capability gate has been evaluated, so nothing in this
file is a performance claim. Read it with [docs/RESULTS.md](RESULTS.md) (what is and is not
measured) and [docs/SERVING.md](SERVING.md) (the only documented way to run it).

---

## 1. Read this before you use it — the minimum disclosure

The following wording is required and is quoted verbatim from the shippable-artifact
ruling of council round 8. It is hard-wrapped to 88 columns; wrapping moves no word.

> This release is a tool-loop configuration for a synthetic five-tool suite, not a
> standalone autonomous system. It requires the frozen `p2_plan_state_act` system prompt,
> client-side tool execution, thinking disabled, and the documented tool schemas and
> remediation protocol. At the recorded 13-shard distill snapshot, certified recovery was
> 1,302/1,752 faulted rollouts (74.3%). These faults were deliberately injected under a
> specific contract: transient, rate-limit, and malformed errors expose a recovery token
> that must be returned in the remedial call; rate-limit recovery must occur on a later
> assistant decision; wrong-unit recovery uses a corrected conversion target and no token.
> This is not evidence of 74.3% recovery from arbitrary API, network, or infrastructure
> failures. Fulfillment fell from 94.3–97.8% at H4/H8 to 59/1,152 (5.1%) at H14 in the
> same snapshot. H14/H20 are outside the supported reliability envelope and were
> measured-only cells. These are descriptive distill/dev findings, not held-out registered
> claim results. The RS-SFT adapter, if supplied, is experimental and is not claimed to
> outperform the base-plus-prompt configuration unless separately reported evidence
> supports that conclusion.

**Maintenance rule for that paragraph:** the snapshot figures are replaced by final
sealed-corpus counts when the rejection-sampling stage closes, and are **never silently
mixed**. The two snapshots that exist today, with their shard counts, are in
[docs/RESULTS.md](RESULTS.md) §10.

**Descriptive only; not a preregistered claim:** every number in this file is a distill- or
dev-split observation with its exact numerator and denominator. Each carries no registered
decision threshold, does not change or replace any original gate, and must not be read as
a confirmatory claim about training efficacy or general agentic capability.

---

## 2. The deep-horizon collapse — the single most important thing to know

In the recorded 13-shard distillation snapshot, certified success in the
`fulfillment-h14` cell was **59/1,152 (5.1%)**, against **1,509/1,600 (94.3%)** at H8 and
**1,174/1,200 (97.8%)** at H4. Recomputed over the 15 shards sealed as of
`2026-08-06T17:42:45Z` the same H14 cell is **70/1,600 (4.4%)**.

Consequences for a user:

- **The supported reliability envelope stops at H8.** H12, H14 and H20 exist in the suite
  as descriptive, measured-only cells. They are not a supported operating region, and no
  gate is computed on them.
- Do not read the cliff as *caused by horizon alone*. The deep cells raise state depth,
  budget pressure and the number of irreversible commitments together with the required
  call count. The measurement is solid; the single-cause explanation is not established.
- Do not read it as *every deep task fails*: 59 rollouts were certified.
- Do not assume an adapter fixes it. No adapter exists yet, and a corpus built by
  rejection sampling keeps the successes by construction, so the training data is
  systematically thin exactly where the model is weakest.
- If your workload needs more than eight causally dependent tool calls in one episode,
  **this configuration is the wrong tool** and nothing measured here says otherwise.

---

## 3. What the recovery number is, and is not

**1,302/1,752 (74.3%)** fault-assigned rollouts met the registered recovery predicate at
the 13-shard snapshot. That predicate is narrow and mechanical, not a general notion of
robustness. In `agentlab.suite.verify`, the event that *establishes* the post-fault
canonical result must itself satisfy the contract:

| fault class | what certified recovery requires |
|---|---|
| `transient` | the exact emitted `recovery_token` on the same stripped call identity |
| `malformed` | the same, on the same call identity |
| `rate_limit` | the same token **and** a strictly later assistant decision |
| `wrong_unit` | a later `unit_convert` explicitly requesting the original target unit; **no token** |
| ambiguous malformed mutation | a token-bearing idempotent replay with `replay=True`; a status query is **not** certified recovery |

So the faults were **deliberately injected under a contract the tools implement**: the
error envelope hands the model a token, and returning that token in the right place is what
counts. Real-world API errors, network partitions, arbitrary exceptions, and tools that do
not implement this remediation contract are **not** covered by that number, and the number
must not be quoted as robustness against them.

It is also a **mixture over the distillation cells that happened to be sealed**, not a
property of the model alone: per-cell it was 575/600 at H4, 714/800 at H8 and 13/352 at
H14, and adding two further H14 shards moved the pooled figure to **1,313/2,200 (59.7%)**
with no change to the model, the prompt or the contract. Quote the cells, or quote the
pooled figure with its shard count — never the pooled figure alone.

Finally, it is **not** the primary estimand. The registered primary conditions on the
common-clean subset `C`, which this corpus never constructs, and it is a TP−BP paired
contrast, which this corpus cannot compute because no trained arm exists.

---

## 4. The exact serving contract

[docs/SERVING.md](SERVING.md) is the single documented path and governs; this is the
contract in one place so that a user cannot miss it.

| Client model ID | Weights | Required system prompt | Status |
|---|---|---|---|
| `Qwen/Qwen3.5-4B` | Base revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | Exact bytes of `prompts/agentic/p2_plan_state_act.txt` | Always shipped; default |
| `trained` | Base plus validated `out/multiface/rssft-lora` | The same frozen prompt | Shipped only if training completes and validates; experimental |

Base only:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0 \
EXPECT_GPU=A5000 \
AGENTIC_RUN_ID=agentic-v1 \
PORT=8000 \
bash scripts/serve.sh Qwen/Qwen3.5-4B
```

Once the adapter validates, the same one command serves both IDs:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0 \
EXPECT_GPU=A5000 \
AGENTIC_RUN_ID=agentic-v1 \
PORT=8000 \
bash scripts/serve.sh Qwen/Qwen3.5-4B \
  --enable-lora \
  --lora-modules trained=out/multiface/rssft-lora \
  --max-lora-rank 32
```

Non-negotiable parts of the contract:

1. **The system prompt is not injected by the server.** Every request must begin with
   `{"role": "system", "content": "<exact contents of p2_plan_state_act.txt>"}`. The
   raw-file SHA-256 is
   `5facfd02997dae6985ff3cdcfda67fa83c0b6765fb5ca9658f46261aec18971d`. A bare request to
   `Qwen/Qwen3.5-4B` without that system message **is not the shipped configuration** and
   must not be presented as one.
2. **The client executes the tools.** vLLM does not. The client supplies the OpenAI
   `tools` array, runs the function, appends the assistant call and the tool result, and
   asks for the next decision. The schemas come from
   `agentlab.suite.runtime.tool_schemas_for_family(family)` — the one model-visible tool
   surface — including the optional `recovery_token` argument declared on every tool of
   every family.
3. **Thinking is disabled and multimodal input is rejected**, both explicitly:
   `--default-chat-template-kwargs '{"enable_thinking":false}'` and
   `--limit-mm-per-prompt '{"image":0,"video":0}'`, plus `enable_thinking: false` in every
   request. This checkpoint defaults thinking **on** and is natively multimodal, so
   without those the served policy is not the measured one. **Text only.**
4. **Parsers**: `--reasoning-parser qwen3`, `--enable-auto-tool-choice`,
   `--tool-call-parser qwen3_coder`. Qwen3.5 emits XML-like tool calls; a JSON-only parser
   silently sees zero calls, which reads as "the model never uses tools". Do not rely on
   `reasoning_content` — thinking is off.
5. **Engine settings** (read from `configs/multifaceted.yaml` `engine:`, the only copy):
   `bfloat16`, `max_model_len=8192`, `gpu_memory_utilization=0.80`, `max_num_seqs=8`,
   `max_num_batched_tokens=8192`, `tensor_parallel_size=1`, no `--enforce-eager`.
6. **Request shape for behaviour comparable to the shipping smoke/evaluation**:
   `temperature 0.0`, `top_p 1.0`, `seed 2786983945`, `max_tokens 1024`,
   `chat_template_kwargs {"enable_thinking": false}`.
7. **The five-tool cap is the whole world**: `kb_lookup`, `unit_convert`, `calculator`,
   `warehouse_query`, `warehouse_update`. Only `fulfillment` is offered the two warehouse
   operations. There is no browser, shell, filesystem, network or arbitrary-API
   competence, and none is measured.
8. **Tested on one RTX A5000 (24 GB)**, 25,282,805,760 CUDA-visible bytes, driver
   610.43.02, CUDA runtime 13.0, under vLLM 0.25.1. `scripts/serve.sh` refuses an
   unspecified device, refuses any device order but `PCI_BUS_ID`, and runs the registered
   hardware veto before vLLM starts. No other hardware has been tested.
9. **Do not serve an adapter merely because `adapter_config.json` exists.** Run the
   repository's own validators first — view building calls `require_accepted_corpus`, SFT
   validates the views chain, checkpoint locking validates the training manifest and the
   checkpoint tree digest — and refuse an incomplete one.
10. **While rejection sampling holds the card, the exclusivity check refuses a second
    engine, and that refusal is correct.** Serve after the stage finishes.

---

## 5. Licence, base model, and what the artifact actually is

| item | value |
|---|---|
| This repository | Apache License 2.0 (`LICENSE`) |
| Base model | `Qwen/Qwen3.5-4B`, Hub revision **`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`** |
| Base-model licence | `apache-2.0` — see the caveat below |
| Base-model bytes | 11 snapshot files, 9,342,816,694 bytes total, 2 weight shards; every file SHA-256'd in `env/model_revision.json` (manifest digest `ede202f39571374b8cda3c0ad9a72d006419d05936a98ebaa8724a60fec83cf6`) |
| Frozen system prompt | `prompts/agentic/p2_plan_state_act.txt`, raw SHA-256 `5facfd02997dae6985ff3cdcfda67fa83c0b6765fb5ca9658f46261aec18971d` |
| Adapter (if it exists) | LoRA, **rank 32**, alpha 64, dropout 0.05, RS-SFT on verified views, completion-only loss (assistant turns only); served under the alias `trained`; **experimental** |
| GRPO | **not run** — disposition `GRPO_NOT_RUN_HARDWARE_INFEASIBLE`; the variance probe is `NOT_EVALUATED_HARDWARE_SHORT_CIRCUIT` |
| Modality | **text only**; image and video parts are rejected by the server |

**Licence caveat, stated because the record states it.** The `apache-2.0` identifier is
recorded in `env/model_revision.json` as *observed from the Hub model-card metadata for
`Qwen/Qwen3.5-4B` on the recording date*. It is **not verified from the cached bytes**: the
snapshot carries no `LICENSE` file. Check the upstream repository yourself before
redistributing weights. This repository's own Apache-2.0 licence covers the code, the
prompts, the suite and the documentation, not the upstream weights.

**Revision-pinning caveat.** `env/model_revision.json` is an *additive, dated apparatus
record written after the preregistration was finalized*. The study registered the base
model by **name only**, with no revision and no digest; this record recovers and hashes
what the local cache had already resolved so the subject stops being mutable. It amends
nothing and must never be presented as though the revision had been pinned before the
study began. It is also **not yet enforced by the loaders**: `src/agentlab/env.py`,
`multidistill.py`, `sft.py`, `scripts/serve.sh`, `scripts/token_census.py` and
`scripts/preflight_dev.py` still call `from_pretrained` without `revision=`. Until that
propagation lands, offline pinning is an **operator obligation** enforced by that record
plus `python scripts/record_model_revision.py verify`.

---

## 6. What this release does not claim

- No held-out capability result exists. No registered gate has been evaluated. There is no
  study-level winner.
- The adapter, if supplied, is **an alternative, not "better."** It is not claimed to beat
  the base model under the frozen prompt unless separately reported evidence says so.
- No general agentic competence: no browser, shell, web, filesystem or arbitrary-API
  ability is measured or implied.
- No reliable execution beyond H8.
- No recovery from network outages, arbitrary exceptions, or tools that do not implement
  this repository's remediation contract.
- The demo (`scripts/demo_agentic.py`) is **five synthetic dev demonstrations**, not a
  benchmark and not held-out evidence. If all five happen to pass, that is not a 100%
  rate.
- Infrastructure failures are reported separately from model failures, never folded into a
  capability rate.

---

## 7. Cross-references

- Results, gate templates and the training-side observations: [docs/RESULTS.md](RESULTS.md).
- The one documented serving path and the demo: [docs/SERVING.md](SERVING.md).
- Reporting rules for the saturated endpoints: [docs/INTERPRETATION.md](INTERPRETATION.md).
- What is deliberately not run, and why it is not a deviation:
  [docs/AMENDED_REPLICATION_NOT_RUN.md](AMENDED_REPLICATION_NOT_RUN.md).
- Registered protocol (hash-pinned, unedited): `docs/AGENTIC_PROTOCOL.md`.
- Clean-clone reproduction: [docs/REPRODUCE.md](REPRODUCE.md).
