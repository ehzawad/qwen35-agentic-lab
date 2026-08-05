"""S10: no train/dev/eval overlap in task IDs, KB namespaces/keys, or
template hashes -- and the checker actually catches each kind."""

from agentic_helpers import chain_spec

from agentlab.suite.splits import check_split_leakage


def _mk(split, ns, offset, n=5):
    return [chain_spec(offset + i, split=split, ns=ns) for i in range(n)]


def test_disjoint_splits_are_clean():
    splits = {"train": _mk("train", "train-a", 0),
              "dev": _mk("dev", "dev-a", 100),
              "eval": _mk("eval", "eval-a", 200)}
    assert check_split_leakage(splits) == []


def test_shared_task_id_is_flagged():
    train = _mk("train", "train-a", 0)
    ev = _mk("eval", "eval-a", 100)
    ev[0]["task_id"] = train[0]["task_id"]
    kinds = {v["kind"] for v in check_split_leakage({"train": train, "eval": ev})}
    assert "task_id" in kinds


def test_shared_kb_namespace_and_keys_are_flagged():
    train = _mk("train", "shared-ns", 0)
    ev = _mk("eval", "shared-ns", 0)  # same namespace AND overlapping keys
    kinds = {v["kind"] for v in check_split_leakage({"train": train, "eval": ev})}
    assert "kb_namespace" in kinds
    assert "kb_key" in kinds


def test_shared_template_hash_is_flagged():
    train = _mk("train", "train-a", 0)
    ev = _mk("eval", "eval-a", 100)
    ev[1]["template_hash"] = train[1]["template_hash"]
    kinds = {v["kind"] for v in check_split_leakage({"train": train, "eval": ev})}
    assert "template_hash" in kinds


def test_shared_graph_signature_is_flagged():
    train = _mk("train", "train-a", 0)
    ev = _mk("eval", "eval-a", 100)
    train[0]["graph_signature"] = ev[0]["graph_signature"] = "iso-XYZ"
    kinds = {v["kind"] for v in check_split_leakage({"train": train, "eval": ev})}
    assert "graph_signature" in kinds


def test_violation_reports_carry_counts_and_examples():
    train = _mk("train", "ns-a", 0)
    ev = _mk("eval", "ns-a", 0)
    violations = check_split_leakage({"train": train, "eval": ev})
    assert all(v["count"] >= 1 and v["examples"] for v in violations)
