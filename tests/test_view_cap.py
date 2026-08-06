"""THE OVER-FULL SIDE of the registered SFT view range.

`views.expected_rows` is [5,000, 6,000] and it is now enforced on BOTH sides, so
a corpus with too MANY rows is a hard stop exactly like a short one. That side is
reachable from the registered minima alone: `totals.min_accepted` admits >= 1,350
accepted trajectories and the view grammar yields about 4-5 rows each, so
acceptance that overshoots its floors -- which nothing forbids and which good
model behaviour makes likely -- exceeds 6,000 and the corpus refuses.

The registered answer is a DETERMINISTIC, CONTENT-BLIND, SEED-KEYED per-stratum
cap: order each stratum by `sha256("view-cap-v1|<stratum>|<task_id>")`, keep the
first k. This module is the teeth for every word of that sentence:

  deterministic     same plan from any input order, and across repeats
  content-blind     scores, rewards, lengths and verdicts are not even read
  seed-keyed        the key is the registered literal, recomputed here
  per-stratum       every non-empty cell keeps trajectories, in proportion
  in range          the total lands inside [5,000, 6,000], and the GATE is
                    unchanged -- the cap is a mechanism, not a widened range

Which trajectory survives must never depend on how WELL it did. A corpus trimmed
by score would make the training set a function of the outcomes the study is
blind to; one trimmed by length would silently re-weight the horizon mixture.

Nothing here touches a GPU.
"""

from __future__ import annotations

import hashlib
import json
import random

import pytest

from agentlab import multidistill as md
from agentlab.suite import configio
from agentlab.suite import contract as contract_mod
from agentlab.suite import datasets as ds
from agentlab.suite.schema import extract_committed_answer

CFG = configio.load_config()
LO, HI = (int(x) for x in CFG["views"]["expected_rows"])
CELLS = ["lookup_chain-h2", "lookup_chain-h4", "lookup_chain-h8",
         "lookup_chain-h12", "typed_relay-h2", "typed_relay-h4",
         "typed_relay-h8", "typed_relay-h12", "fulfillment-h4",
         "fulfillment-h8", "fulfillment-h14", "fulfillment-h20"]


def _entries(per_cell: int, rows: int = 5, cells=None) -> list:
    """A synthetic build: `per_cell` trajectories of `rows` rows in every cell."""
    cells = cells or CELLS
    return [{"stratum": cell, "task_id": f"distill-{cell}-{i:04d}", "rows": rows}
            for cell in cells for i in range(per_cell)]


def _kept(plan: dict, entries: list) -> set:
    return {(entries[i]["stratum"], entries[i]["task_id"])
            for i in plan["kept_indexes"]}


# ---------------------------------------------------------------------------
# 1. the key
# ---------------------------------------------------------------------------

def test_the_cap_key_is_the_registered_literal():
    """Recomputed from the registered string, so the key cannot be re-derived.

    A different key string would silently produce a different corpus from the
    same accepted trajectories, which is the one thing a registered selection
    rule may not do.
    """
    expect = hashlib.sha256(
        b"view-cap-v1|lookup_chain-h4|distill-lookup_chain-h4-0007").hexdigest()
    assert ds.view_cap_key("lookup_chain-h4",
                           "distill-lookup_chain-h4-0007") == expect
    assert ds.VIEW_CAP_KEY_VERSION == "view-cap-v1"


# ---------------------------------------------------------------------------
# 2. inert below the ceiling
# ---------------------------------------------------------------------------

def test_the_cap_is_inert_below_the_ceiling():
    entries = _entries(50)                     # 12 x 50 x 5 = 3,000 rows
    plan = ds.plan_view_cap(entries, CFG)
    assert plan["applied"] is False
    assert plan["rows"] == plan["full_rows"] == 3000
    assert plan["trajectories_capped"] == 0
    assert len(plan["kept_indexes"]) == len(entries)


def test_a_short_corpus_is_never_padded_or_rescued_by_the_cap():
    """The under-full refusal is untouched: the cap drops rows, it never adds."""
    entries = _entries(10)                     # 600 rows, far below the floor
    plan = ds.plan_view_cap(entries, CFG)
    assert plan["applied"] is False and plan["rows"] == 600
    with pytest.raises(SystemExit) as exc:
        ds.require_expected_rows({"rows": 600, "terminal_weight": 0.6,
                                  "expected_rows": [LO, HI], "strata": {}}, CFG)
    assert "outside the registered range" in str(exc.value)


# ---------------------------------------------------------------------------
# 3. the ceiling: the total lands inside the registered range
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("per_cell,rows", [(120, 5), (200, 4), (150, 5), (400, 3),
                                           (113, 5)])
def test_the_capped_total_lands_inside_the_registered_range(per_cell, rows):
    entries = _entries(per_cell, rows)
    plan = ds.plan_view_cap(entries, CFG)
    assert plan["full_rows"] > HI, "this case must actually be over-full"
    assert plan["applied"] is True
    assert LO <= plan["rows"] <= HI, plan["rows"]
    # ... and the arithmetic reconciles: kept rows + capped rows = the full build
    capped_rows = plan["full_rows"] - plan["rows"]
    assert capped_rows == sum(
        entries[i]["rows"] for i in range(len(entries))
        if i not in set(plan["kept_indexes"]))
    assert plan["trajectories"] + plan["trajectories_capped"] == len(entries)


def test_the_registered_minima_case_is_the_one_that_forced_this_rule():
    """1,350 accepted trajectories at 4-5 rows each: the documented overshoot."""
    at_floor = ds.plan_view_cap(
        [{"stratum": CELLS[i % 12], "task_id": f"t{i:05d}", "rows": 4 + (i % 2)}
         for i in range(1350)], CFG)
    assert at_floor["full_rows"] > HI, at_floor["full_rows"]
    assert at_floor["applied"] is True and LO <= at_floor["rows"] <= HI


# ---------------------------------------------------------------------------
# 4. deterministic
# ---------------------------------------------------------------------------

def test_the_plan_is_identical_across_input_order_and_repeats():
    entries = _entries(150)
    first = ds.plan_view_cap(entries, CFG)
    again = ds.plan_view_cap(list(entries), CFG)
    shuffled = list(entries)
    random.Random(20260806).shuffle(shuffled)
    permuted = ds.plan_view_cap(shuffled, CFG)

    assert _kept(first, entries) == _kept(again, entries)
    assert _kept(first, entries) == _kept(permuted, shuffled)
    assert first["kept_task_ids_sha256"] == again["kept_task_ids_sha256"] \
        == permuted["kept_task_ids_sha256"]
    assert first["rows"] == permuted["rows"]


def test_the_kept_set_is_a_prefix_of_the_seed_keyed_order_in_every_stratum():
    """"Take the first k" literally -- and therefore never "take the k that fit".

    A scan that kept looking for a smaller trajectory after the first miss would
    be selection by LENGTH, which is an outcome-adjacent property of the
    trajectory (a shorter successful transcript is a different kind of episode).
    """
    entries = _entries(150)
    plan = ds.plan_view_cap(entries, CFG)
    kept = _kept(plan, entries)
    for cell in CELLS:
        order = sorted((e for e in entries if e["stratum"] == cell),
                       key=lambda e: (ds.view_cap_key(cell, e["task_id"]),
                                      e["task_id"]))
        flags = [(cell, e["task_id"]) in kept for e in order]
        k = sum(flags)
        assert flags == [True] * k + [False] * (len(flags) - k), cell
        assert k >= 1, cell


# ---------------------------------------------------------------------------
# 5. content-blind: no outcome may reach the decision
# ---------------------------------------------------------------------------

def test_scores_rewards_lengths_and_verdicts_do_not_change_the_plan():
    """Every outcome field is attached, then inverted. The plan must not move."""
    entries = _entries(150)
    base = ds.plan_view_cap(entries, CFG)

    loud = []
    for i, e in enumerate(entries):
        loud.append(dict(e, score=1.0 if i % 2 else 0.0, reward=float(i),
                         certified_success=bool(i % 3),
                         n_calls=20 - (i % 7), wall_s=float(i),
                         fault_class="transient" if i % 4 else None,
                         verdict={"certified_success": bool(i % 5)}))
    with_outcomes = ds.plan_view_cap(loud, CFG)
    inverted = ds.plan_view_cap(
        [dict(e, score=1.0 - e["score"], reward=-e["reward"],
              certified_success=not e["certified_success"]) for e in loud], CFG)

    assert _kept(base, entries) == _kept(with_outcomes, loud)
    assert _kept(base, entries) == _kept(inverted, loud)
    assert base["kept_task_ids_sha256"] == with_outcomes["kept_task_ids_sha256"]
    assert base["ranked_by"].startswith("sha256(view-cap-v1|stratum|task_id)")
    assert base["outcome_blind"] is True


def test_the_plan_does_not_depend_on_which_trajectory_is_longest():
    """Row counts decide only HOW MANY fit, never WHICH ones are preferred.

    Same multiset of row counts, permuted across the task ids: the kept COUNT per
    stratum may shift by a trajectory (the prefix fills differently), but the
    order the prefix walks must be untouched -- so the kept set is still the
    prefix of the same seed-keyed order.
    """
    entries = _entries(150, 4)
    varied = [dict(e, rows=4 + (i % 3)) for i, e in enumerate(entries)]
    plan = ds.plan_view_cap(varied, CFG)
    kept = _kept(plan, varied)
    for cell in CELLS:
        order = sorted((e for e in varied if e["stratum"] == cell),
                       key=lambda e: (ds.view_cap_key(cell, e["task_id"]),
                                      e["task_id"]))
        flags = [(cell, e["task_id"]) in kept for e in order]
        k = sum(flags)
        assert flags == [True] * k + [False] * (len(flags) - k), cell


# ---------------------------------------------------------------------------
# 6. stratum balance
# ---------------------------------------------------------------------------

def test_every_stratum_keeps_its_share_of_the_corpus():
    """Each cell's share of the capped rows equals its share of the full rows.

    Whole trajectories are indivisible, so the tolerance is the largest
    trajectory a stratum has -- not a free parameter, and checked as such.
    """
    entries = [{"stratum": cell, "task_id": f"t-{cell}-{i:04d}",
                "rows": 4 + (i % 2)}
               for n, cell in enumerate(CELLS)
               for i in range(60 + 25 * n)]        # deliberately UNEVEN cells
    plan = ds.plan_view_cap(entries, CFG)
    assert plan["applied"] is True and LO <= plan["rows"] <= HI

    for cell, got in plan["per_stratum"].items():
        want = got["full_rows"] / plan["full_rows"] * plan["rows"]
        biggest = max(e["rows"] for e in entries if e["stratum"] == cell)
        assert abs(got["rows"] - want) <= biggest + 1, (cell, got, want)
        assert got["trajectories"] >= 1, cell


def test_a_single_trajectory_stratum_is_never_capped_out_of_existence():
    """The measured-only H14/H20 cells are exactly why the floor exists.

    A pure proportional share rounds a one-trajectory cell to nothing, and a
    vanished cell is a corpus that no longer covers the registered mixture.
    """
    entries = _entries(300, 5, cells=CELLS[:11])
    entries.append({"stratum": "fulfillment-h20", "task_id": "distill-h20-0001",
                    "rows": 5})
    plan = ds.plan_view_cap(entries, CFG)
    assert plan["per_stratum"]["fulfillment-h20"]["trajectories"] == 1
    assert plan["per_stratum"]["fulfillment-h20"]["rows"] == 5
    assert LO <= plan["rows"] <= HI


def test_a_view_plan_too_large_for_the_ceiling_refuses_rather_than_dropping_a_cell():
    """The one case the floor cannot satisfy: refuse, never widen, never drop."""
    huge = [{"stratum": cell, "task_id": f"t-{cell}", "rows": HI} for cell in CELLS]
    with pytest.raises(SystemExit) as exc:
        ds.plan_view_cap(huge, CFG)
    assert "above the registered ceiling" in str(exc.value)
    assert "Do not widen it" in str(exc.value)


# ---------------------------------------------------------------------------
# 7. the gate is unchanged
# ---------------------------------------------------------------------------

def test_the_row_range_gate_still_refuses_an_over_full_corpus():
    """The cap is a MECHANISM. The registered range did not move a row.

    A report above the ceiling still refuses, and the message says the cap did
    not run rather than offering to raise the ceiling.
    """
    report = {"rows": HI + 1, "terminal_weight": 0.6,
              "expected_rows": [LO, HI], "strata": {}}
    with pytest.raises(SystemExit) as exc:
        ds.require_expected_rows(report, CFG)
    message = str(exc.value)
    assert f"{HI + 1} rows is outside the registered range {LO}-{HI}" in message
    assert ds.VIEW_CAP_KEY_VERSION in message
    assert "Do not raise the ceiling" in message
    assert ds.require_expected_rows({**report, "rows": HI}, CFG)["ok"] is True


# ---------------------------------------------------------------------------
# 8. end to end through the real builder
# ---------------------------------------------------------------------------

def _synthetic_record(cell: str, index: int, *, faulted: bool) -> dict:
    """One accepted-shaped trajectory: a dependency-bearing call, then a commit.

    Cheap on purpose -- the cap arithmetic needs THOUSANDS of trajectories, and
    real rollouts at that scale would make this a different kind of test. The
    shapes that matter are real: a committed `ANSWER:` line the one shared
    grammar reads, a call whose arguments consume a prior tool value (so a pivot
    view is eligible), and a fault result with a later decision (so a recovery
    view is eligible).
    """
    family, horizon = cell.rsplit("-h", 1)
    answer = f"{index}.5"
    messages = [
        {"role": "system", "content": "You can call tools."},
        {"role": "user", "content": f"resolve {cell} {index}"},
        {"role": "assistant", "tool_calls": [
            {"type": "function",
             "function": {"name": "kb_lookup", "arguments": {"key": "KAAAA2222BBBB"}}}]},
        {"role": "tool", "name": "kb_lookup",
         "content": ('{"ok":false,"error":"transient_backend","recovery_token":'
                     '"' + "a" * 32 + '"}\nreceipt: r-1' if faulted
                     else '{"ok":true,"value":' + answer + '}\nreceipt: r-1')},
    ]
    fault = {}
    if faulted:
        messages.append({"role": "assistant", "tool_calls": [
            {"type": "function",
             "function": {"name": "kb_lookup",
                          "arguments": {"key": "KAAAA2222BBBB",
                                        "recovery_token": "a" * 32}}}]})
        messages.append({"role": "tool", "name": "kb_lookup",
                         "content": '{"ok":true,"value":' + answer + '}\nreceipt: r-2'})
        fault = {"fired": True, "result_msg_index": 3}
    messages.append({"role": "assistant", "tool_calls": [
        {"type": "function",
         "function": {"name": "unit_convert",
                      "arguments": {"value": float(answer), "to_unit": "kg"}}}]})
    messages.append({"role": "tool", "name": "unit_convert",
                     "content": '{"ok":true,"value":' + answer + ',"unit":"kg"}'})
    messages.append({"role": "assistant", "content": f"Done.\nANSWER: {answer}"})
    rec = {"task_id": f"distill-{cell}-{index:04d}", "family": family,
           "horizon": int(horizon), "messages": messages, "fault": fault,
           "fault_types": ["transient"] if faulted else [],
           "provenance": dict(md.cpu_provenance("test-view-cap", CFG))}
    rec[contract_mod.STAMP_FIELD] = contract_mod.environment_contract_sha256()
    return rec


@pytest.fixture(scope="module")
def over_full_corpus():
    from rollout_helpers import token_counter_stub

    # `% 5` and not `% 3`: with twelve cells a `% 3` schedule makes the fault a
    # property of the CELL, so four cells would carry every recovery view and
    # eight would carry none. The cap must be exercised on a realistic mixture.
    records = [_synthetic_record(CELLS[i % 12], i, faulted=(i % 5 == 0))
               for i in range(2000)]
    # the fixtures must actually be trainable, or the cap is being tested on
    # rejections rather than on eligible trajectories
    assert extract_committed_answer(records[0]["messages"][-1]["content"])
    rows, meta, report = ds.build_views(records, token_counter_stub(), CFG)
    # The records themselves are handed back: determinism means SAME INPUT ->
    # same output, and a rebuilt "identical" record is not the same input. Each
    # row id is keyed by its source trajectory's content digest (one task can
    # contribute several accepted trajectories, so a task-keyed id collides), and
    # that digest covers the rollout's provenance block, which carries the
    # producer's `timestamp_utc`. Constructing the fixtures a second time
    # therefore yields genuinely different artifacts. In production the digest is
    # taken over rollout rows read from disk, which do not move.
    return rows, meta, report, records


def test_build_views_caps_an_over_full_corpus_into_the_registered_range(
        over_full_corpus):
    rows, meta, report, _records = over_full_corpus
    cap = report["view_cap"]
    assert cap["applied"] is True, cap
    assert cap["full_rows"] > HI
    assert report["rows"] == len(rows) == len(meta) == cap["rows"]
    assert LO <= report["rows"] <= HI
    assert report["rows_in_expected_range"] is True
    # the gate the corpus would otherwise have failed now passes, unwidened
    assert ds.require_expected_rows(report, CFG)["ok"] is True
    assert report["terminal_weight"] >= CFG["views"]["terminal_weight_min"]


def test_the_census_reconciles_with_the_cap_receipt(over_full_corpus):
    _rows, _meta, report, _records = over_full_corpus
    cap = report["view_cap"]
    assert sum(c["rows"] for c in report["strata"].values()) == report["rows"]
    assert sum(c["trajectories_capped"] for c in report["strata"].values()) \
        == cap["trajectories_capped"]
    # a capped trajectory is NOT a dropped one: the shortfall estimator must not
    # see rows that were removed on purpose
    assert all(c["trajectories_dropped"] == 0
               for c in report["strata"].values()), report["strata"]
    for cell, got in cap["per_stratum"].items():
        assert report["strata"][cell]["rows"] == got["rows"]
        assert report["strata"][cell]["trajectories"] == got["trajectories"]


def test_the_capped_corpus_rebuilds_byte_identically(over_full_corpus):
    """Determinism where it is finally observable: the row-id digest."""
    from rollout_helpers import token_counter_stub

    _rows, _meta, report, records0 = over_full_corpus
    # THE SAME artifacts, in a different order -- not rebuilt ones. Each row id is
    # keyed by its source trajectory's content digest, because one task can
    # contribute several accepted trajectories (different rollout samples, and the
    # clean and faulted conditions of one scenario) that legitimately supervise the
    # same view kind at the same turn; a task-keyed id collided and the builder
    # rightly refused the corpus. That digest covers the rollout's provenance,
    # which carries the producer's `timestamp_utc`, so re-CONSTRUCTING the fixtures
    # would hand `build_views` genuinely different artifacts and test nothing.
    # Determinism is: same input, any order, same corpus.
    records = list(records0)
    random.Random(7).shuffle(records)
    rows2, meta2, report2 = ds.build_views(records, token_counter_stub(), CFG)
    assert report2["rows"] == report["rows"]
    assert report2["view_cap"]["kept_task_ids_sha256"] == \
        report["view_cap"]["kept_task_ids_sha256"]
    assert sorted(m["row_id"] for m in meta2) == \
        sorted(m["row_id"] for m in _meta)
    assert len(rows2) == len(_rows)
    assert report2["view_cap"]["per_stratum"] == report["view_cap"]["per_stratum"]


def test_the_receipt_is_json_serialisable_and_names_the_rule(over_full_corpus):
    _rows, _meta, report, _records = over_full_corpus
    text = json.dumps(report["view_cap"], indent=2)
    assert "view-cap-v1" in text
    assert "never score, reward, length or any outcome" in text
    assert "kept_indexes" not in report["view_cap"], \
        "the receipt carries the digest of the kept identities, not 5,000 indexes"
