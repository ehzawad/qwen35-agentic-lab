"""Multifaceted agentic suite v1: tasks, runtime, verification, measurement.

Task-suite core (env-architect spec; deterministic, CPU-only):

  schema.py    TaskSpec / OracleNode / FaultSpec / TraceEvent, canonical JSON
  rng.py       SHA-256 label streams + counter-mode CounterRNG (no `random`)
  kb.py        isolated per-episode KBs; misses return only no_entry
  faults.py    the four fault injectors (transient, malformed incl. the
               ambiguous post-mutation case, wrong-unit, rate-limit) and the
               logical decision clock
  runtime.py   EpisodeRuntime dispatch: per-episode KB view, state, fault
               schedule, oracle progress, trace events; scripted oracle agent
  verify.py    strict trace/state verifier, dependency edges across later
               assistant decisions, the three recovery denominators
  rewards.py   binary strict evaluation + bounded GRPO shaping
  generate.py  split orchestration, fault mixture, committed serialization,
               the split loader (`load_bundles`) and the certification-layer
               spec adapter (`certification_spec`)
  envs/        lookup_chain (H2-12), typed_relay (H2-12), fulfillment
               (H4-20, environment-scoped warehouse_query/warehouse_update)

There is exactly ONE environment stack. Every consumer -- rejection sampling
(agentlab.multidistill), the prompt tournament (agentlab.prompt_control), the
GRPO grip probe (agentlab.variance), the SFT view builder (datasets.py) and the
held-out certification runner (evaluate.py) -- executes calls through
runtime.EpisodeRuntime and scores them with verify.verify_episode. Tasks come
only from generate.load_bundles over the committed data; `runtime.replay_trace`
plus `runtime.verify_replay` are the machinery that proves a consumer's
trajectories replay to identical observation digests and oracle progress, and
tests/test_suite_reconciliation.py enforces it for all twelve cells.

Measurement side (gates spec):

  stats.py     Wilson, exact McNemar, Holm, template-clustered bootstrap
  splits.py    train/dev/test leakage checks (S10)

The global tool registry in agentlab.tools keeps only the canonical three;
the two warehouse tools exist solely as environment-scoped operations inside
fulfillment episodes.
"""
