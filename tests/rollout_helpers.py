"""A scripted CPU policy that drives the real rollout engine, no model involved.

`agentlab.multidistill.RolloutEngine` takes `render_fn` and `generate_fn`, so a
test can substitute a deterministic policy for vLLM and exercise the whole
training-path stack -- canonical runtime, fault injection, strict verifier,
replay parity, SFT views -- on CPU.

The policy below re-derives oracle progress from the TRANSCRIPT ALONE (parse each
tool result, decide whether it was usable, advance or retry). That is a
deliberately independent implementation of the "usable observation" rule: if it
agreed with the runtime only because it shared its code, the parity tests would
be circular.
"""

from __future__ import annotations

import json

from agentlab.suite.faults import MALFORMED_LITERAL


def render_messages(messages, schemas):
    """The engine only forwards what render returns, so forward the messages."""
    return messages


class OraclePolicy:
    """Walks each task's oracle plan, recovering from every scheduled fault.

    Recovery actions are exactly the four the fault contract accepts:
      transient / rate_limit -> re-issue the same call on the next decision
      malformed              -> re-issue the same call (idempotent for mutations)
      wrong_unit             -> re-issue the conversion for the requested unit
    """

    def __init__(self, bundles, *, terminal_text=None, break_at=None,
                 skip_node=None, extra_call=None):
        self.by_prompt = {b.spec.prompt: b for b in bundles}
        self.terminal_text = terminal_text
        self.break_at = break_at        # stop calling tools after N credited nodes
        self.skip_node = skip_node      # omit this node index entirely
        self.extra_call = extra_call    # (after_n, tool, args) harmless extra call

    # -- transcript reading ---------------------------------------------------

    @staticmethod
    def _usable(result_text: str, requested_unit: str | None) -> bool:
        if result_text.startswith(MALFORMED_LITERAL):
            return False
        try:
            obj = json.loads(result_text)
        except json.JSONDecodeError:
            return False
        if not obj.get("ok"):
            return False
        if requested_unit is not None:
            return str(obj.get("unit", "")).strip().lower() == requested_unit
        return True

    def _is_extra(self, call: dict) -> bool:
        if self.extra_call is None:
            return False
        _n, tool, args = self.extra_call
        return call["name"] == tool and call["arguments"] == args

    def _progress(self, messages) -> int:
        """Credited nodes = usable tool results, counted off the transcript.

        A deliberately-injected harmless extra call is excluded: it earns no
        oracle progress in the runtime either, so counting it here would make the
        policy skip a required node.
        """
        done = 0
        pending = None          # None = no call awaiting its result
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                call = msg["tool_calls"][0]["function"]
                if self._is_extra(call):
                    pending = None
                    continue
                unit = None
                if call["name"] == "unit_convert":
                    unit = str(call["arguments"].get("to_unit", "")).strip().lower()
                pending = (unit,)
            elif msg.get("role") == "tool" and pending is not None:
                if self._usable(str(msg.get("content", "")), pending[0]):
                    done += 1
                pending = None
        return done

    # -- the policy ------------------------------------------------------------

    def __call__(self, prompts):
        return [self._act(messages) for messages in prompts]

    def _act(self, messages):
        bundle = self.by_prompt[messages[1]["content"]]
        nodes = bundle.nodes
        done = self._progress(messages)
        if self.extra_call is not None and done == self.extra_call[0]:
            _n, tool, args = self.extra_call
            if not any(m.get("role") == "assistant" and m.get("tool_calls")
                       and m["tool_calls"][0]["function"]["name"] == tool
                       and m["tool_calls"][0]["function"]["arguments"] == args
                       for m in messages):
                return self._call(tool, args)
        index = done
        if self.skip_node is not None and index == self.skip_node:
            index += 1
        if (self.break_at is not None and done >= self.break_at) or index >= len(nodes):
            answer = bundle.spec.answer
            text = (self.terminal_text if self.terminal_text is not None
                    else f"Done. The answer is \\boxed{{{answer}}}")
            return text, "stop"
        node = nodes[index]
        return self._call(node.tool, dict(node.args))

    @staticmethod
    def _call(tool: str, args: dict):
        payload = json.dumps({"name": tool, "arguments": args}, sort_keys=True)
        return f"<tool_call>\n{payload}\n</tool_call>", "stop"


def run_engine(bundles, policy=None, cfg=None, generations=1, variants=("canonical",)):
    """Roll every bundle once (or `generations` times) through the real engine."""
    from agentlab.multidistill import RolloutEngine
    from agentlab.suite.configio import load_config

    cfg = cfg or load_config()
    policy = policy or OraclePolicy(bundles)
    engine = RolloutEngine(cfg, render_messages, policy)
    convos = engine.rollouts_for(bundles, k_override=generations, variants=variants)
    return engine.run(convos, verbose=False)


def token_counter_stub(limit_map=None):
    """A cheap token counter for view construction: words, or a scripted map."""
    def count(prompt_msgs, completion_msgs, tools):
        if limit_map is not None:
            key = len(prompt_msgs)
            if key in limit_map:
                return limit_map[key]
        text = json.dumps(list(prompt_msgs) + list(completion_msgs), default=str)
        return len(text.split())
    return count
