# Training › Reinforcement Learning

Source path: `AlphaBrain/training/reinforcement_learning/`

Full implementation of VLA online RL training (RLT_a — RL Token). Paper: *RL Token: Bootstrapping Online RL with VLA Models* (Physical Intelligence).

Layout:

- **algos/RLT_a/** — encoder/decoder, actor-critic, trainer, fast rollout
- **common/** — rollout, replay buffer, checkpoint I/O
- **envs/** — LIBERO environment, persistent env pool, env workers
- **eval/** — LIBERO evaluation and shard aggregation
- **trainers/** — on-policy / off-policy / pretrain entrypoints

---

## Top-level re-exports

::: AlphaBrain.training.reinforcement_learning
    options:
      heading_level: 3
      show_submodules: false

---

## RLT_a algorithm

### Encoder / Decoder

::: AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_encoder_decoder
    options:
      heading_level: 4

### Actor / Critic

::: AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_actor_critic
    options:
      heading_level: 4

### Trainer (loss / update)

::: AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_trainer
    options:
      heading_level: 4

### Fast rollout

::: AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_rollout_fast
    options:
      heading_level: 4

---

## Shared components (common/)

### Rollout

::: AlphaBrain.training.reinforcement_learning.common.rollout
    options:
      heading_level: 4

### Replay buffer

::: AlphaBrain.training.reinforcement_learning.common.replay_buffer
    options:
      heading_level: 4

### Checkpoint I/O

::: AlphaBrain.training.reinforcement_learning.common.ckpt_io
    options:
      heading_level: 4

---

## Environments (envs/)

### LIBERO environment

::: AlphaBrain.training.reinforcement_learning.envs.libero_env
    options:
      heading_level: 4

### LIBERO environment workers

::: AlphaBrain.training.reinforcement_learning.envs.libero_env_worker
    options:
      heading_level: 4

::: AlphaBrain.training.reinforcement_learning.envs.libero_env_worker_fast
    options:
      heading_level: 4

### Persistent environment pool

::: AlphaBrain.training.reinforcement_learning.envs.persistent_env_pool
    options:
      heading_level: 4

---

## Evaluation (eval/)

::: AlphaBrain.training.reinforcement_learning.eval.eval_libero
    options:
      heading_level: 4

::: AlphaBrain.training.reinforcement_learning.eval.eval_helpers
    options:
      heading_level: 4

::: AlphaBrain.training.reinforcement_learning.eval.aggregate_shards
    options:
      heading_level: 4

---

## Training entrypoints (trainers/)

### Shared CLI arguments

::: AlphaBrain.training.reinforcement_learning.trainers.train_args
    options:
      heading_level: 4

### Main entrypoint

::: AlphaBrain.training.reinforcement_learning.trainers.train
    options:
      heading_level: 4

### Pretrain

::: AlphaBrain.training.reinforcement_learning.trainers.train_pretrain
    options:
      heading_level: 4

### On-policy RL

::: AlphaBrain.training.reinforcement_learning.trainers.train_rl_onpolicy
    options:
      heading_level: 4

### Off-policy RL

::: AlphaBrain.training.reinforcement_learning.trainers.train_rl_offpolicy
    options:
      heading_level: 4
