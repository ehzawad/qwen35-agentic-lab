#!/usr/bin/env python
"""Re-measure the observation and view size ceilings under the UNIFIED contract.

`scenario.tool_output_max_tokens` was 208, measured against the SMALLER tokenless
payloads: no recovery token, no remediation text, no receipt line. The registered
model-visible tool result now carries all three, so every size cap that is
denominated in "what the model reads" has to be re-measured before it can be
trusted, and any cap the measurement requires has to move.

What is measured, with the Qwen3.5-4B tokenizer and the repo's Transformers:

  tool_result   the exact model-visible tool-message bytes of every dispatch --
                envelope plus receipt line -- over every committed train/dev
                split, every family/horizon cell, the clean case, every eligible
                fault class, the same-decision rate-limit repeat and the
                ambiguous malformed mutation.
  rendered      the FULL rendered chat-template prefix of the terminal decision,
                for every episode above crossed with all eight preregistered
                prompt candidates. This is what `acceptance.max_view_tokens` and
                `sft.max_length` are denominated in, and it is measured on the
                rendered text rather than on a JSON envelope, because the
                template's own tokens are part of the length that rejects a view.

Every observation string is tokenized once (deduplicated by SHA-256) -- the
envelope shapes repeat across tasks, so the census is exhaustive over episodes
without tokenizing the same bytes 200,000 times.

The artifact records the tokenizer file hashes, the Transformers version, the
environment-contract digest, per-stratum maxima and the offending task IDs, and
its own SHA-256 goes into the preregistration amendment. Nothing here is a gate:
it is the measurement a cap has to follow.

    PYTHONPATH=src .venv/bin/python scripts/token_census.py \\
        [--splits distill oracle_sft grpo_train dev] \\
        [--out out/agentic/token_census.json]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_SPLITS = ("distill", "oracle_sft", "grpo_train", "dev")
BASE_MODEL = "Qwen/Qwen3.5-4B"

# Every registered fault class, plus the two behaviours that only appear as a
# SECOND observation: a rate-limit repeat inside the same assistant decision, and
# the ambiguous malformed mutation whose repair is a token-bearing replay.
VARIANTS = ("clean", "transient", "rate_limit", "rate_limit_same_decision",
            "malformed", "malformed_ambiguous", "wrong_unit")


def tokenizer_fingerprint(model: str) -> dict:
    """Which tokenizer produced these counts, hashed file by file."""
    import transformers
    from transformers.utils import cached_file

    files = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json",
                 "merges.txt", "chat_template.jinja", "special_tokens_map.json"):
        try:
            path = cached_file(model, name, _raise_exceptions_for_missing_entries=False)
        except Exception:
            path = None
        if path and os.path.exists(path):
            files[name] = hashlib.sha256(
                pathlib.Path(path).read_bytes()).hexdigest()
    return {"model": model, "transformers": transformers.__version__,
            "files_sha256": files}


def load_tokenizer(model: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model)


def prompt_candidates() -> list[dict]:
    from agentlab.prompt_control import candidate_text, candidates

    return [{"id": c["id"], "text": candidate_text(c)} for c in candidates()]


# ---------------------------------------------------------------------------
# episode construction: one scripted, fully-recovered episode per variant
# ---------------------------------------------------------------------------

def _variant_faults(bundle, variant: str):
    """-> list[FaultSpec] for this variant, or None when it does not apply."""
    from agentlab.suite.faults import wrong_unit_candidates
    from agentlab.suite.schema import FaultSpec

    nodes = bundle.nodes
    if variant == "clean":
        return []
    if variant == "wrong_unit":
        targets = [n for n in nodes if n.tool == "unit_convert"]
        if not targets:
            return None
        node = targets[-1]
        cands = wrong_unit_candidates(str(node.args["to_unit"]))
        if not cands:
            return None
        return [FaultSpec("wrong_unit", node.node_id, {"wrong_unit": cands[0]})]
    if variant == "malformed_ambiguous":
        targets = [n for n in nodes if n.mutating]
        if not targets:
            return None
        node = targets[0]
        return [FaultSpec("malformed", node.node_id,
                          {"ambiguous_mutation": True, "line": 1})]
    if variant == "malformed":
        node = next((n for n in nodes if not n.mutating), nodes[-1])
        return [FaultSpec("malformed", node.node_id, {})]
    if variant in ("rate_limit", "rate_limit_same_decision"):
        return [FaultSpec("rate_limit", nodes[-1].node_id,
                          {"retry_after_turns": 1})]
    if variant == "transient":
        return [FaultSpec("transient", nodes[-1].node_id, {})]
    raise ValueError(variant)


def run_variant(bundle, variant: str, secret: bytes):
    """-> (transcript messages, list of model-visible tool strings) or None.

    The episode is driven the way the registered contract is satisfied: the token
    is echoed on the reissued call, and the wrong-unit trap is repaired by
    re-requesting the original target unit. `rate_limit_same_decision`
    additionally issues the token-bearing retry INSIDE the faulted decision first,
    so the `rate_limit_active` envelope is measured too.
    """
    import dataclasses

    from agentlab.chat import assistant_tool_message
    from agentlab.suite.contract import budgets_for
    from agentlab.suite.faults import TOKEN_ARG
    from agentlab.suite.runtime import (EpisodeRuntime, parse_observation,
                                        recovery_token_in)

    faults = _variant_faults(bundle, variant)
    if faults is None:
        return None
    budgets = budgets_for(bundle.spec.horizon, "clean" if not faults else "faulted")
    spec = dataclasses.replace(bundle.spec, faults=faults,
                               max_decisions=budgets["max_decisions"],
                               max_calls=budgets["max_calls"])
    rt = EpisodeRuntime(spec, bundle.kb, bundle.nodes, secret=secret)
    messages = [{"role": "system", "content": ""},
                {"role": "user", "content": spec.prompt}]
    observations: list[str] = []

    def call(node, args):
        rt.begin_decision()
        messages.append(assistant_tool_message(
            "", [{"name": node.tool, "arguments": args}]))
        text = rt.dispatch(node.tool, args)
        messages.append({"role": "tool", "name": node.tool, "content": text})
        observations.append(text)
        return text

    for node in bundle.nodes:
        token = None
        for attempt in range(4):
            args = dict(node.args)
            if token is not None:
                args[TOKEN_ARG] = token
            text = call(node, args)
            new_token = recovery_token_in(text)
            if (variant == "rate_limit_same_decision" and new_token is not None
                    and attempt == 0):
                # the same-decision token-bearing retry: still limited
                same = dict(node.args, **{TOKEN_ARG: new_token})
                messages.append(assistant_tool_message(
                    "", [{"name": node.tool, "arguments": same}]))
                extra = rt.dispatch(node.tool, same)
                messages.append({"role": "tool", "name": node.tool,
                                 "content": extra})
                observations.append(extra)
            token = new_token
            if token is not None:
                continue
            body = next((o for o in parse_observation(text)["objects"]
                         if isinstance(o, dict)), None)
            if body is None or not body.get("ok"):
                break
            if (node.tool == "unit_convert"
                    and str(body.get("unit", "")).lower()
                    != str(node.args["to_unit"]).lower()):
                continue
            break
    messages.append({"role": "assistant",
                     "content": f"Done.\nANSWER: \\boxed{{{spec.answer}}}"})
    return messages, observations


# ---------------------------------------------------------------------------
# the census
# ---------------------------------------------------------------------------

class Census:
    """Deduplicated token counting with per-stratum maxima and offenders."""

    def __init__(self, tok) -> None:
        self.tok = tok
        self._cache: dict[str, int] = {}
        self.strata: dict[tuple, dict] = {}

    def count(self, text: str) -> int:
        return self.count_many([text])[0]

    def count_many(self, texts: list[str]) -> list[int]:
        """Batched tokenization; identical strings are counted once per run.

        The Rust tokenizer is far faster on a batch than on a loop of singles, and
        the envelope shapes repeat heavily across tasks, so a census that is
        exhaustive over episodes stays affordable.
        """
        keys = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
        todo = {k: t for k, t in zip(keys, texts) if k not in self._cache}
        if todo:
            items = list(todo.items())
            enc = self.tok([t for _k, t in items], add_special_tokens=False)
            for (k, _t), ids in zip(items, enc["input_ids"]):
                self._cache[k] = len(ids)
        return [self._cache[k] for k in keys]

    def record(self, stratum: tuple, text: str, task_id: str,
               n: int | None = None) -> int:
        n = self.count(text) if n is None else n
        row = self.strata.setdefault(stratum, {"max_tokens": 0, "max_chars": 0,
                                               "n": 0, "task_id": None,
                                               "chars_task_id": None})
        row["n"] += 1
        if n > row["max_tokens"]:
            row["max_tokens"] = n
            row["task_id"] = task_id
        if len(text) > row["max_chars"]:
            row["max_chars"] = len(text)
            row["chars_task_id"] = task_id
        return n

    def rollup(self, keys) -> dict:
        out: dict[str, dict] = {}
        for stratum, row in self.strata.items():
            label = "|".join(str(s) for s in stratum)
            out[label] = dict(row)
        return out


def census(splits, out_path: pathlib.Path, *, model: str = BASE_MODEL,
           limit_per_split: int = 0) -> dict:
    from agentlab.suite.contract import environment_contract_sha256
    from agentlab.suite.generate import load_bundles
    from agentlab.suite.runtime import tool_schemas_for_family

    t0 = time.time()
    tok = load_tokenizer(model)
    fp = tokenizer_fingerprint(model)
    prompts = prompt_candidates()
    secret = bytes.fromhex("a5" * 32)  # a census is not a run; pin the secret
    obs = Census(tok)
    views = Census(tok)
    schema_tokens = {}
    episodes = 0
    skipped: dict[str, int] = collections.Counter()

    data_dir = str(REPO / "data" / "suite" / "v1")
    for split in splits:
        bundles = load_bundles(data_dir, split)
        if limit_per_split:
            by_cell: dict = {}
            kept = []
            for b in bundles:
                cell = (b.spec.family, b.spec.horizon)
                if by_cell.get(cell, 0) < limit_per_split:
                    by_cell[cell] = by_cell.get(cell, 0) + 1
                    kept.append(b)
            bundles = kept
        for family in {b.spec.family for b in bundles}:
            schemas = tool_schemas_for_family(family)
            schema_tokens[family] = obs.count(json.dumps(schemas, sort_keys=True))
        schema_cache = {f: tool_schemas_for_family(f)
                        for f in {b.spec.family for b in bundles}}
        for b in bundles:
            cell = f"{b.spec.family}-h{b.spec.horizon}"
            for variant in VARIANTS:
                got = run_variant(b, variant, secret)
                if got is None:
                    skipped[f"{cell}|{variant}"] += 1
                    continue
                messages, observations = got
                episodes += 1
                counts = obs.count_many(observations)
                for text, n in zip(observations, counts):
                    obs.record((split, cell, variant), text, b.spec.task_id, n)
                # The terminal view: the whole transcript up to (excluding) the
                # final assistant decision is the PROMPT an SFT row carries, and
                # the completion is the terminal decision -- together they are the
                # longest view of the trajectory, which is what the view budget
                # rejects on.
                body = messages[1:]
                rendered = [
                    tok.apply_chat_template(
                        [{"role": "system", "content": cand["text"]}] + body,
                        tools=schema_cache[b.spec.family],
                        tokenize=False, add_generation_prompt=False,
                        enable_thinking=False)
                    for cand in prompts]
                vcounts = views.count_many(rendered)
                for cand, text, n in zip(prompts, rendered, vcounts):
                    views.record((split, cell, variant, cand["id"]), text,
                                 b.spec.task_id, n)

    def worst(strata):
        best = {"max_tokens": 0, "max_chars": 0, "stratum": None, "task_id": None}
        for stratum, row in strata.items():
            if row["max_tokens"] > best["max_tokens"]:
                best = {"max_tokens": row["max_tokens"],
                        "max_chars": row["max_chars"],
                        "stratum": "|".join(str(s) for s in stratum),
                        "task_id": row["task_id"]}
        return best

    worst_chars_obs = max(
        ((row["max_chars"], "|".join(str(s) for s in st), row["chars_task_id"])
         for st, row in obs.strata.items()), default=(0, None, None))

    report = {
        "kind": "token_census",
        "schema_version": 1,
        "measured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment_contract_sha256": environment_contract_sha256(),
        "tokenizer": fp,
        "splits": list(splits),
        "limit_per_split_per_cell": limit_per_split or None,
        "variants": list(VARIANTS),
        "prompt_candidates": [c["id"] for c in prompts],
        "episodes_measured": episodes,
        "distinct_strings_tokenized": len(obs._cache) + len(views._cache),
        "skipped_inapplicable_variants": dict(sorted(skipped.items())),
        "tool_schema_tokens": schema_tokens,
        "tool_result": {
            "worst": worst(obs.strata),
            "worst_chars": {"chars": worst_chars_obs[0],
                            "stratum": worst_chars_obs[1],
                            "task_id": worst_chars_obs[2]},
            "observations_measured": sum(r["n"] for r in obs.strata.values()),
            "per_stratum": obs.rollup(None),
        },
        "rendered_terminal_view": {
            "worst": worst(views.strata),
            "views_measured": sum(r["n"] for r in views.strata.values()),
            "per_stratum": views.rollup(None),
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(report, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False) + "\n"
    out_path.write_text(body, encoding="utf-8")
    report["artifact_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    (out_path.parent / (out_path.name + ".sha256")).write_text(
        f"{report['artifact_sha256']}  {out_path.name}\n", encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    ap.add_argument("--out", default="out/agentic/token_census.json")
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--limit-per-split", type=int, default=0,
                    help="tasks per cell per split (0 = every committed task)")
    args = ap.parse_args()
    report = census([s for s in args.splits], REPO / args.out, model=args.model,
                    limit_per_split=args.limit_per_split)
    print(json.dumps({
        "artifact": args.out,
        "artifact_sha256": report["artifact_sha256"],
        "environment_contract_sha256": report["environment_contract_sha256"],
        "episodes_measured": report["episodes_measured"],
        "tool_result_max_tokens": report["tool_result"]["worst"],
        "tool_result_max_chars": report["tool_result"]["worst_chars"],
        "rendered_terminal_view_max_tokens": report["rendered_terminal_view"]["worst"],
        "elapsed_s": report["elapsed_s"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
