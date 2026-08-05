"""No key, terminal token, task ID, or template ever crosses split groups."""

from __future__ import annotations

from agentlab.suite.generate import (SPLIT_KIND, SPLIT_SEED_KEY, SPLITS,
                                     build_split)
from agentlab.suite.schema import TEMPLATE_RANGES
from agentlab.suite.splits import check_split_leakage

from .conftest import SEEDS, SUITE

# The group of every split comes from SPLIT_KIND, not from a second table: a
# copied mapping is how a newly added claim split silently ends up ungrouped and
# therefore unchecked for leakage.
_GROUP = dict(SPLIT_KIND)


def _tiny_groups():
    groups = {"train": [], "dev": [], "eval": []}
    answers = {g: {"token": set(), "integer": set(), "token_list": []}
               for g in groups}
    for split in SPLITS:
        result = build_split(SUITE, split, SEEDS[SPLIT_SEED_KEY[split]], 2)
        for b in result["bundles"]:
            group = _GROUP[split]
            groups[group].append({"task_id": b.spec.task_id,
                                  "template_id": b.spec.template_id,
                                  "kb": b.kb})
            answers[group][b.spec.answer_kind].add(b.spec.answer)
            if b.spec.answer_kind == "token":
                answers[group]["token_list"].append(b.spec.answer)
    return groups, answers


def test_no_leakage_between_groups():
    groups, answers = _tiny_groups()
    assert check_split_leakage(groups) == []
    names = sorted(groups)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (answers[a]["token"] & answers[b]["token"])


def test_terminal_tokens_are_unique_but_integer_answers_may_collide():
    """The binding condition covers terminal TOKENS, not numeric results.

    A 128-bit code or completion token that recurred anywhere would be a
    memorisable secret, so tokens must be globally unique. typed_relay answers
    are bounded integers and collide across splits by the birthday paradox
    (measured over the committed sizes: 53 of 2,554 distinct integers between
    train and eval, zero token crossings). Flagging those as leakage would be a
    false alarm that masks real ones, so the validator must not.
    """
    _groups, answers = _tiny_groups()
    all_tokens = [t for a in answers.values() for t in a["token_list"]]
    assert len(all_tokens) == len(set(all_tokens))
    assert all_tokens, "the suite must contain token-answer tasks"
    assert any(a["integer"] for a in answers.values()), "and integer-answer tasks"


def test_task_ids_globally_unique():
    groups, _ = _tiny_groups()
    ids = [r["task_id"] for rows in groups.values() for r in rows]
    assert len(ids) == len(set(ids))


def test_template_ranges_disjoint_per_group():
    groups, _ = _tiny_groups()
    for group, rows in groups.items():
        allowed = set(TEMPLATE_RANGES[group])
        assert {r["template_id"] for r in rows} <= allowed
    assert not (set(TEMPLATE_RANGES["train"]) & set(TEMPLATE_RANGES["dev"]))
    assert not (set(TEMPLATE_RANGES["train"]) & set(TEMPLATE_RANGES["eval"]))
    assert not (set(TEMPLATE_RANGES["dev"]) & set(TEMPLATE_RANGES["eval"]))


def test_eval_inputs_never_in_train_inputs():
    """Binding validator condition 11, at test scale."""
    groups, _ = _tiny_groups()
    train_keys = {k for r in groups["train"] for k in r["kb"]}
    for group in ("dev", "eval"):
        held_keys = {k for r in groups[group] for k in r["kb"]}
        assert not (train_keys & held_keys)
