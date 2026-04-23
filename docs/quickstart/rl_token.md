# RL-Token

Two-phase TD3 online RL fine-tuning on a pretrained QwenOFT VLA — encoder pretrain + off-policy rollouts.

---

## Prerequisites

- A pretrained VLA checkpoint (see [Baseline VLA](baselineVLA.md)).
- LIBERO in a separate conda env; `.env` has `LIBERO_PYTHON` and `LIBERO_HOME`.
- **6 GPUs** (5 rollout + 1 train) for the default. Reduce `--num_envs_per_task` for smaller setups.

---

## Train

```bash
# default GPU layout: 0,2,3,4,5 rollout, 1 train
bash scripts/run_rl_scripts/run_action_token_5traj_alltasks.sh

# custom GPU layout (rollout0,rollout1,...,train)
bash scripts/run_rl_scripts/run_action_token_5traj_alltasks.sh "0,1,2,3,4,5"
```

Checkpoints: `results/action_token_training_TD3/<run>_<ts>/rl_offpolicy/checkpoints/rl_offpolicy_iter_NNNNN/`.

## Evaluate

```bash
bash scripts/run_rl_scripts/run_eval_action_token.sh \
    results/action_token_training_TD3/action_token_5traj_alltasks_release_0414_1727/rl_offpolicy
```

Results: `<RUN_DIR>/eval_rl_offpolicy_iter_<NNNNN>/summary.json`.

---

Full CLI reference, rollout/TD math, and release disclaimer: [`scripts/run_rl_scripts/README.md`](https://github.com/AlphaBrainGroup/AlphaBrain/blob/main/scripts/run_rl_scripts/README.md).
