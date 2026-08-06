# Serving the shippable configurations

This is the whole shipping surface: **one script**, `scripts/serve.sh`, and
**one demo**, `scripts/demo_agentic.py`. There is no hosted service, no UI, no
merged checkpoint, no Docker image and no model registry, and none is planned.

What "shipping a working agentic model" means here:

- a reproducible local OpenAI-compatible endpoint serves the base model under
  the frozen winning prompt;
- if RS-SFT finishes and its training manifest/checkpoint digest validates, the
  same endpoint also serves the LoRA under a clearly experimental alias;
- a fixed, unfiltered demo exercises the real tool loop, including one injected
  failure.

The base configuration is already the default working artifact. **The adapter is
an alternative, not "better,"** unless later measurements support that
statement.

> Read this with the status block in [README.md](../README.md) and with
> [docs/INTERPRETATION.md](INTERPRETATION.md). No registered capability gate has
> been evaluated, so nothing on this page is a performance claim.

## The two configurations

| Client model ID | Weights | Required system prompt | Status |
|---|---|---|---|
| `Qwen/Qwen3.5-4B` | Base revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | Exact bytes of `prompts/agentic/p2_plan_state_act.txt` | Always shipped; default |
| `trained` | Base plus validated `out/multiface/rssft-lora` | The same frozen prompt | Shipped only if training completes and validates; experimental |

The frozen prompt is [`prompts/agentic/p2_plan_state_act.txt`](../prompts/agentic/p2_plan_state_act.txt),
raw-file SHA-256 `5facfd02997dae6985ff3cdcfda67fa83c0b6765fb5ca9658f46261aec18971d`.

## Base-only command

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0 \
EXPECT_GPU=A5000 \
AGENTIC_RUN_ID=agentic-v1 \
PORT=8000 \
bash scripts/serve.sh Qwen/Qwen3.5-4B
```

## Base **and** adapter, once the adapter validates

`out/multiface/rssft-lora` does not exist yet. This is the command for when it
does; it needs no edit to this page and no second script:

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

`scripts/serve.sh` refuses `--lora-modules` whose path does not exist, refuses
`--enable-lora` without a module (the server would serve base weights under a
trained label), refuses a second alias, and refuses a `--max-lora-rank` that is
not the registered rank 32. It accepts no other trailing flag, because an engine
setting an operator can override is an engine the recorded fingerprint does not
describe.

Before serving an adapter, run the repository's own validators once and refuse
an incomplete one: view building calls `require_accepted_corpus`, SFT validates
the views chain, and checkpoint locking validates the training manifest and the
checkpoint tree digest. **Do not serve an adapter merely because
`adapter_config.json` exists.**

## The registered engine contract this expands to

Every setting is read from `configs/multifaceted.yaml` `engine:` through
`agentlab.suite.configio.engine_contract()` — the one copy every stage reads —
so this page cannot drift from the served engine:

```text
--reasoning-parser qwen3
--enable-auto-tool-choice
--tool-call-parser qwen3_coder
--default-chat-template-kwargs '{"enable_thinking":false}'
--limit-mm-per-prompt '{"image":0,"video":0}'
```

The remaining effective settings are `bfloat16`, `max_model_len=8192`, GPU
utilization `0.80`, eight sequences, 8,192 batched tokens, tensor parallelism 1,
and no `--enforce-eager`.

Thinking is DISABLED and multimodal input is REJECTED — refused, not merely
unused. Both are load-bearing: this checkpoint defaults thinking ON and is
natively multimodal, so without those two flags the served policy differs from
the trained one and an image part would contribute an episode no registered
claim describes.

`CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=0` and `EXPECT_GPU=A5000`
are not decoration. The script refuses an unspecified device rather than
defaulting to GPU 0, refuses any device order but `PCI_BUS_ID`, and runs the
registered hardware veto (PCI order, registered index, registered card,
exclusive free memory of at least 23,500 MiB, the run's UUID binding) before
vLLM starts. It then captures a producer session manifest and `exec`s vLLM, so
the pid in that manifest is the pid that owns the card.

**While `distill` (rejection sampling) is running on the pinned A5000 the
exclusivity check will refuse to start a second engine, and that refusal is
correct.** Serve after the stage finishes.

## For behaviour comparable to the shipping smoke/evaluation

Each request also sends:

```json
{
  "temperature": 0.0,
  "top_p": 1.0,
  "seed": 2786983945,
  "max_tokens": 1024,
  "chat_template_kwargs": {"enable_thinking": false}
}
```

The prompt is not injected by vLLM. Every request must begin with:

```json
{"role": "system", "content": "<exact contents of p2_plan_state_act.txt>"}
```

A bare request to `Qwen/Qwen3.5-4B` without that system message is not the
shipped BP configuration and should not be presented as one.

## What an OpenAI-compatible client sees

With the adapter loaded, `GET /v1/models` exposes the base model ID and
`trained`. Requests use:

- `model="Qwen/Qwen3.5-4B"` for BP.
- `model="trained"` for the LoRA configuration.
- A standard OpenAI `tools` array supplied by the client.
- The same system prompt for both.

Responses contain standard structured calls:

```json
{
  "content": null,
  "tool_calls": [{
    "id": "...",
    "type": "function",
    "function": {
      "name": "warehouse_query",
      "arguments": "{\"sku\":\"...\"}"
    }
  }]
}
```

`function.arguments` is a JSON string on the wire. The `qwen3_coder` parser
converts Qwen's native XML-like output into that structure. The client executes
the function, appends the assistant call and tool result, and requests the next
decision. **vLLM does not execute tools itself.** The final message is ordinary
content ending in the prompt's `ANSWER: <value>` form.

The `qwen3` reasoning parser separates reasoning from answer content if any is
emitted, but thinking is explicitly disabled and users should not rely on
`reasoning_content`. The repository's wire-format behaviour is documented in the
evaluator ([`agentlab/suite/evaluate.py`](../src/agentlab/suite/evaluate.py),
`make_http_chat`).

The tool schemas the client must supply are not hand-written either: they come
from `agentlab.suite.runtime.tool_schemas_for_family(family)`, which is the one
model-visible tool surface — the canonical three tools, plus the two warehouse
tools for `fulfillment`, plus the optional `recovery_token` argument declared on
every one of them.

## The demo

Start the server with the base-only command above, then, in another shell:

```bash
PYTHONPATH=src .venv/bin/python scripts/demo_agentic.py
```

Side by side with the adapter, once it exists and the server registered it:

```bash
PYTHONPATH=src .venv/bin/python scripts/demo_agentic.py \
  --model Qwen/Qwen3.5-4B --model trained
```

The demo runs a five-episode panel whose task IDs are fixed in the source before
the server is contacted, one of them under a real injected `rate_limit` fault,
and prints every tool call, every fault envelope, the recovery attempt and the
verifier's verdict for each. It selects nothing by outcome, rerolls nothing, and
prints what it does and does not demonstrate as its first output. It needs no
GPU of its own: it is a pure HTTP client against the server you started.
