# VLA Online RL Training Module — RLT_a (our variant).
# Inspired by "RL Token: Bootstrapping Online RL with VLA Models" (Physical
# Intelligence, 2026) with several deviations; see
# algos/RLT_a/__init__.py for details. A faithful paper-accurate
# implementation is still under test.

from AlphaBrain.training.reinforcement_learning.envs.libero_env import LiberoEnv, get_suite_info
from AlphaBrain.training.reinforcement_learning.common.rollout import collect_group, Episode, StepRecord
from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_encoder_decoder import ActionTokenEncoder, ActionTokenDecoder, ActionTokenEncoderDecoder
from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_actor_critic import ActionTokenActor, ActionTokenCritic, ActionTokenQCritic, soft_update_target
from AlphaBrain.training.reinforcement_learning.common.replay_buffer import ReplayBuffer
from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_trainer import (
    collect_observations_fast, extract_action_queries_from_obs,
    action_token_collect_group, action_token_ppo_loss, action_token_td_update,
    action_token_td_critic_update, action_token_td_actor_update,
    push_episodes_to_buffer,
    ActionTokenEpisode, ActionTokenStepRecord,
)
