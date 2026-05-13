"""RLT_a — our variant of a frozen-VLA + bottleneck-encoder + TD3 pipeline.

Inspired by the RL Token paper (Physical Intelligence, 2026), but differs in
several design choices (encoder input uses action-query tokens rather than the
paper's full image embeddings, an extra 2048→256 linear bottleneck, pretrain
data is random-policy rollouts rather than task demonstrations, etc.). The
faithful paper-accurate implementation is still under test; this module is the
working training recipe we ship today.

Public API re-exported here so callers can write:

    from AlphaBrain.training.reinforcement_learning.algos.RLT_a import (
        ActionTokenActor, ActionTokenEncoderDecoder, action_token_td_critic_update, ...
    )

instead of the longer per-module path
``...algos.RLT_a.action_token_trainer`` / ``...algos.RLT_a.action_token_actor_critic`` / etc.
"""
from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_actor_critic import (
    ActionTokenActor,
    ActionTokenCritic,
    ActionTokenQCritic,
    soft_update_target,
)
from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_encoder_decoder import (
    ActionTokenDecoder,
    ActionTokenEncoder,
    ActionTokenEncoderDecoder,
)
from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_rollout_fast import (
    action_token_collect_group_steplock,
    action_token_collect_multitask_steplock,
)
from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_trainer import (
    BatchInferenceServer,
    ActionTokenEpisode,
    ActionTokenStepRecord,
    collect_observations_fast,
    compute_action_token_gae,
    extract_action_queries_dataset,
    extract_action_queries_from_obs,
    pretrain_encoder_step,
    push_episodes_to_buffer,
    action_token_collect_group,
    action_token_ppo_loss,
    action_token_td_actor_update,
    action_token_td_critic_update,
    action_token_td_update,
    vla_finetune_step,
)
