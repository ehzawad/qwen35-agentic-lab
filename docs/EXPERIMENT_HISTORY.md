# Experiment history

Compact record of the round-1 GSM8K tool-loop study and the experiments excluded
from the shipped tree. Everything summarized here remains fully recoverable at the
pre-cleanup commit `9ebe44f6bd687e0a6a489ff6cfcd3770abd3b49f` (`git show <sha>:<path>`).

## Excluded experiments

Single-turn xlam SFT reduced held-out GSM8K accuracy to 1/20 and ran away on 19/20
episodes because its corpus contained tool calls but no post-tool termination turns —
a 16x degradation of the base model. The reward-model run ended near chance; its
adapter-only checkpoint also reinitializes the trained score head on reload, making
the saved artifact unsafe for reuse. DPO and BudgetEnv proved trainer plumbing only:
DPO had no quality evaluation, and BudgetEnv saturated at reward 1.0 with zero
variance, validating the interface rather than any learning. Their runnable stages
and raw artifacts were removed from current `main`; the original record remains in
commits `95b8326` and `3ab3201`.

## Retired-but-validated GSM8K results

Held-out GSM8K, n=200 per arm, paired by problem; harness sanity checks clean and
all five preregistered gates passed. Full evidence (eval JSONs, traces, corpus,
verdict, analyzer) lives at commit `9ebe44f6bd687e0a6a489ff6cfcd3770abd3b49f`.

| Arm | Accuracy | Statistics |
|---|---:|---|
| Base (default prompt) | 0.810 | reference arm |
| RS-SFT (rejection-sampled distillation) | 0.920 | vs base: paired McNemar p<0.001 |
| Concise-prompt control (zero training) | 0.900 | recovers 81.8% of the observed RS-SFT gain |
| RS-GRPO (GRPO after RS-SFT) | 0.930 | vs RS-SFT: +0.010, null (p=0.804) |

The controlling lesson: most of the measured "training gain" was elicitation of
latent behaviour, so every future capability claim requires a best-prompt base
control, and GRPO after successful SFT produced an informative null.

## Deferred — recorded so they are not rediscovered; neither is planned

- **RM-in-the-loop RLOO Goodhart experiment**: RLOO against a real 0.8B reward
  model, evaluated as a Goodhart curve. Blocked on a checkpoint fix — the RM must
  be merged via `merge_and_unload()` with a pre/post-merge logit-parity assertion
  (the adapter-only checkpoint silently reinitializes its score head on reload).
  TRL 1.9.2 has no PPOTrainer; RLOO would be the algorithm. DPO is not a dependency.
- **Inference-limits agenda, native context**: 262144-token native context costs
  ~8 GiB of KV cache on this model (32 KiB/token in bf16: 8 full-attention layers
  x 4 KV heads x head_dim 256 x K+V x 2 bytes).
- **Inference-limits agenda, YaRN extension**: a 1.01M-token batch-1 sequence needs
  ~30.87 GiB of paged hybrid cache and is predicted to fit this card at
  `gpu_memory_utilization` 0.89-0.90 (measured smoke at 0.85: weights 8.61 GiB,
  CUDA graphs 0.56 GiB, 29.34 GiB cache available).
