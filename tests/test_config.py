"""Config construction for every stage, plus reward-function behaviour.

This is the file that earns its keep. TRL moved `reward_weights` onto the config
and deleted `max_prompt_length` between versions; both were found only by
burning a GPU run to reach the traceback. Constructing every config here takes
under a second and fails on the same errors, so an upgrade that breaks the API
is caught before anything touches the card.
"""

from __future__ import annotations

import inspect

import pytest

trl = pytest.importorskip("trl")


class TestTrainerSignatures:
    """Pin the trainer/config split we actually depend on."""

    def test_reward_weights_is_a_config_field_not_a_trainer_kwarg(self):
        import dataclasses

        from trl import GRPOConfig, GRPOTrainer

        fields = {f.name for f in dataclasses.fields(GRPOConfig)}
        params = set(inspect.signature(GRPOTrainer.__init__).parameters)
        assert "reward_weights" in fields
        assert "reward_weights" not in params

    def test_tools_and_environment_factory_are_trainer_kwargs(self):
        from trl import GRPOTrainer

        params = set(inspect.signature(GRPOTrainer.__init__).parameters)
        assert {"tools", "environment_factory", "reward_funcs"} <= params

    def test_vllm_max_model_length_exists(self):
        # Without this the KV cache is sized for the model's native 262144
        # context and colocate mode refuses to start.
        import dataclasses

        from trl import GRPOConfig

        assert "vllm_max_model_length" in {f.name for f in dataclasses.fields(GRPOConfig)}

    def test_dpo_config_has_no_max_prompt_length(self):
        import dataclasses

        from trl import DPOConfig

        fields = {f.name for f in dataclasses.fields(DPOConfig)}
        assert "max_length" in fields
        assert "max_prompt_length" not in fields, "TRL re-added it; dpo.py can use it again"

    def test_peft_model_plus_peft_config_is_rejected_by_trl(self):
        # The constraint that forces policy_and_peft() to return peft_config=None
        # when continuing an adapter. If TRL ever relaxes it, we want to know.
        src = inspect.getsource(trl.GRPOTrainer.__init__)
        assert "PeftModel" in src and "peft_config" in src


class TestStageConfigs:
    """Every stage's config must construct with the exact args the code passes."""

    def test_sft_config(self):
        from trl import SFTConfig

        cfg = SFTConfig(
            output_dir="/tmp/_t", num_train_epochs=1.0, per_device_train_batch_size=4,
            gradient_accumulation_steps=4, learning_rate=1e-4, lr_scheduler_type="cosine",
            warmup_ratio=0.03, max_length=2048, bf16=True, gradient_checkpointing=True,
            logging_steps=10, eval_strategy="steps", eval_steps=100, save_strategy="epoch",
            save_total_limit=1, report_to=[], packing=False,
        )
        assert cfg.max_length == 2048

    def test_dpo_config(self):
        from trl import DPOConfig

        cfg = DPOConfig(
            output_dir="/tmp/_t", num_train_epochs=1.0, per_device_train_batch_size=2,
            gradient_accumulation_steps=8, learning_rate=5e-6, lr_scheduler_type="cosine",
            warmup_ratio=0.03, beta=0.1, max_length=1536, bf16=True,
            gradient_checkpointing=True, logging_steps=10, eval_strategy="steps",
            eval_steps=100, save_strategy="epoch", save_total_limit=1, report_to=[],
        )
        assert cfg.beta == 0.1

    def test_reward_config(self):
        from trl import RewardConfig

        cfg = RewardConfig(
            output_dir="/tmp/_t", num_train_epochs=1.0, per_device_train_batch_size=4,
            gradient_accumulation_steps=4, learning_rate=1e-5, max_length=1024, bf16=True,
            gradient_checkpointing=True, logging_steps=10, eval_strategy="steps",
            eval_steps=100, save_strategy="epoch", save_total_limit=1, report_to=[],
        )
        assert cfg.max_length == 1024

    @pytest.mark.parametrize("mode", ["tools", "env"])
    def test_grpo_config(self, mode):
        from trl import GRPOConfig

        cfg = GRPOConfig(
            output_dir="/tmp/_t", num_train_epochs=1.0, per_device_train_batch_size=8,
            gradient_accumulation_steps=4, num_generations=8, max_completion_length=1024,
            learning_rate=1e-6, lr_scheduler_type="constant_with_warmup", warmup_ratio=0.03,
            beta=0.0, loss_type="dapo", temperature=1.0, top_p=0.95, top_k=20,
            mask_truncated_completions=True, bf16=True, gradient_checkpointing=True,
            logging_steps=1, save_strategy="epoch", save_total_limit=1, report_to=[],
            use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=0.25,
            vllm_max_model_length=4096,
            reward_weights=[1.0, 0.3, 0.2] if mode == "tools" else None,
            max_tool_calling_iterations=4,
        )
        assert cfg.vllm_max_model_length == 4096

    def test_lora_config_excludes_the_vision_tower(self):
        from agentlab.peft_cfg import lora_config

        cfg = lora_config(r=32)
        assert cfg.r == 32 and cfg.lora_alpha == 64
        assert cfg.target_modules == "all-linear"
        assert any("vision" in m for m in cfg.exclude_modules)


class TestRewardFunctions:
    """Rewards must RANK rollouts correctly; the absolute values matter less."""

    @staticmethod
    def _rollout(answer=None, tool=None, tool_result=None):
        msgs = []
        if tool:
            msgs.append({"role": "assistant", "tool_calls": [
                {"type": "function", "function": {"name": tool, "arguments": {"expression": "1"}}}]})
            msgs.append({"role": "tool", "name": tool, "content": tool_result or "1"})
        msgs.append({"role": "assistant",
                     "content": f"\\boxed{{{answer}}}" if answer is not None else "no answer"})
        return msgs

    def test_correct_answer_scores_one(self):
        from agentlab.grpo import correctness_reward

        assert correctness_reward([self._rollout("250")], ["250"]) == [1.0]

    def test_wrong_answer_scores_zero(self):
        from agentlab.grpo import correctness_reward

        assert correctness_reward([self._rollout("999")], ["250"]) == [0.0]

    def test_numeric_equivalence_is_accepted(self):
        from agentlab.grpo import correctness_reward

        assert correctness_reward([self._rollout("250.0")], ["250"]) == [1.0]

    def test_missing_box_scores_zero(self):
        from agentlab.grpo import correctness_reward

        assert correctness_reward([self._rollout(None)], ["250"]) == [0.0]

    def test_tool_use_is_rewarded_and_tool_errors_penalised(self):
        from agentlab.grpo import tool_use_reward

        used = tool_use_reward([self._rollout("1", tool="calculator")])[0]
        none = tool_use_reward([self._rollout("1")])[0]
        errored = tool_use_reward([self._rollout("1", tool="calculator", tool_result="error: bad")])[0]
        assert used > none
        assert errored < used

    def test_format_reward_sign(self):
        from agentlab.grpo import format_reward

        assert format_reward([self._rollout("1")])[0] > 0
        assert format_reward([self._rollout(None)])[0] < 0

    def test_total_ordering_of_rollouts(self):
        # The property that actually matters: a clean correct rollout must beat a
        # correct-but-errored one, which must beat wrong, which must beat silent.
        from agentlab.grpo import correctness_reward, format_reward, tool_use_reward

        cases = {
            "clean": self._rollout("250", tool="calculator"),
            "errored": self._rollout("250", tool="calculator", tool_result="error: bad"),
            "wrong": self._rollout("999", tool="calculator"),
            "silent": self._rollout(None),
        }
        comps = list(cases.values())
        gt = ["250"] * len(comps)
        total = [
            1.0 * c + 0.3 * t + 0.2 * f
            for c, t, f in zip(
                correctness_reward(comps, gt), tool_use_reward(comps), format_reward(comps)
            )
        ]
        scored = dict(zip(cases, total))
        assert scored["clean"] > scored["errored"] > scored["wrong"] > scored["silent"]

    def test_rewards_accept_plain_string_completions(self):
        # Without tools TRL hands back strings, not message lists.
        from agentlab.grpo import correctness_reward

        assert correctness_reward([r"\boxed{7}"], ["7"]) == [1.0]


class TestBudgetEnv:
    def test_exact_hit_scores_one(self):
        from agentlab.grpo import BudgetEnv

        env = BudgetEnv()
        env.reset()
        env.spend(env.target)
        assert env.get_reward() == 1.0

    def test_reward_decays_with_distance(self):
        from agentlab.grpo import BudgetEnv

        env = BudgetEnv()
        env.reset()
        env.spend(env.target - 5)
        assert 0.0 < env.get_reward() < 1.0

    def test_spend_budget_is_enforced(self):
        from agentlab.grpo import BudgetEnv

        env = BudgetEnv()
        env.reset()
        for _ in range(5):
            env.spend(1)
        assert env.spend(1).startswith("error:")
