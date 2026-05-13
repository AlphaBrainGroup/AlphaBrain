"""
Fast ActionToken rollout — step-lock architecture.

Key design: all envs move in lockstep.
  1. Batch VLA forward for ALL active envs (one GPU call)
  2. Batch encoder + actor
  3. ALL envs execute chunk in parallel threads
  4. Collect results, repeat

No BatchInferenceServer needed. No async queuing. No batch fragmentation.

Speedup: ~50x vs original (env creation + batch fragmentation eliminated).
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np
import torch

from AlphaBrain.training.reinforcement_learning.envs.persistent_env_pool import PersistentEnvPool
from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_encoder_decoder import ActionTokenEncoderDecoder
from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_actor_critic import ActionTokenActor, ActionTokenCritic
from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_trainer import ActionTokenEpisode, ActionTokenStepRecord
from AlphaBrain.training.reinforcement_learning.common.rollout import _unnormalize, _postprocess_action, DUMMY_ACTION

logger = logging.getLogger(__name__)


def _run_vla_and_encode(
    frozen_vla,
    encoder,
    batch_images,
    batch_instrs,
    props_t,
    encoder_mode: str,
):
    """One fused VLA forward + RL-token encoding for the rollout step.

    Dispatches on ``encoder_mode`` and (for ``rlt``) on Qwen vs Pi05
    backbone. Returns ``(rl_tokens, vla_actions, action_queries)`` —
    ``action_queries`` is ``None`` for the Pi05 rlt path which has
    no in-stream action tokens.
    """
    if encoder_mode == "rlt":
        from AlphaBrain.training.reinforcement_learning.algos.RLT.pi05_inference import (
            run_rlt_inference,
        )
        rl_tokens, vla_actions = run_rlt_inference(
            frozen_vla, encoder, batch_images, batch_instrs, props_t,
        )
        # action_queries unused on the rlt path (Pi05 has none; Qwen
        # path discards them inside run_rlt_inference).
        return rl_tokens, vla_actions, None

    # action_token mode: encoder takes the action-query slice directly.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        action_queries, vla_actions = frozen_vla.get_vla_action(
            batch_images=batch_images, instructions=batch_instrs)
    rl_tokens = encoder.encode(action_queries)
    return rl_tokens, vla_actions, action_queries


def _env_step_chunk(env_pool, env_idx, action_chunk_unnorm, chunk_len, record_frames=False):
    """Execute chunk_len env steps in ONE pipe round-trip (8x fewer I/Os)."""
    actions = [_postprocess_action(action_chunk_unnorm[step]) for step in range(chunk_len)]
    try:
        obs, reward, done, steps_taken = env_pool.envs[env_idx].step_chunk(actions)
    except RuntimeError as e:
        print(f"  [WARNING] env {env_idx} step_chunk failed: {e}, marking as done", flush=True)
        # Return a fake "failed" result — episode will be marked as failure
        obs = {"primary_image": np.zeros((256,256,3), dtype=np.uint8),
               "wrist_image": np.zeros((256,256,3), dtype=np.uint8),
               "state": np.zeros(8, dtype=np.float32)}
        return obs, 0.0, True, 0, []
    return obs, reward, done, steps_taken, []


def _env_dummy_steps(env_pool, env_idx, n_steps):
    """Execute dummy actions (warmup). Returns final obs.

    On worker hang/crash, returns a zero-obs placeholder instead of raising —
    matches `_env_step_chunk`'s recovery so a single env's failure doesn't
    kill the entire rollout thread (the affected episode just becomes a
    silent failure for this iter; auto-reset will recover the env next iter).
    """
    obs = None
    for _ in range(n_steps):
        try:
            obs, _, _ = env_pool.step_env(env_idx, DUMMY_ACTION)
        except RuntimeError as e:
            print(f"  [WARNING] env {env_idx} dummy step failed: {e}", flush=True)
            return {"primary_image": np.zeros((256, 256, 3), dtype=np.uint8),
                    "wrist_image": np.zeros((256, 256, 3), dtype=np.uint8),
                    "state": np.zeros(8, dtype=np.float32)}
    return obs


@torch.no_grad()
def action_token_collect_group_steplock(
    env_pool: PersistentEnvPool,
    frozen_vla,
    encoder: ActionTokenEncoderDecoder,
    actor: ActionTokenActor,
    critic: ActionTokenCritic,
    suite_name: str,
    task_id: int,
    n_initial_states: int,
    action_norm_stats: dict,
    max_steps: int,
    chunk_len: int,
    G: int = 64,
    seed: int = 42,
    num_steps_wait: int = 10,
    device: str = "cuda",
    video_dir: Optional[str] = None,
    group_idx: int = 0,
    store_images: bool = False,
    group_size: int = 1,
    reward_coef: float = 1.0,
    actor_chunk_len: int = None,
    env_offset: int = 0,
    warmup_mode: bool = False,
    encoder_mode: str = "action_token",
) -> List[ActionTokenEpisode]:
    """
    Collect G episodes using step-lock architecture.

    All envs move in lockstep:
      1. Batch VLA forward (one GPU call for ALL active envs)
      2. Batch encoder + actor (or skip actor if warmup_mode)
      3. All envs execute chunk in parallel
      4. Repeat

    Args:
        env_offset: starting env index in the pool
        actor_chunk_len: if set, actor outputs shorter chunk than VLA
        warmup_mode: if True, use VLA actions directly (skip actor). For buffer pre-fill.
    """
    if actor_chunk_len is None:
        actor_chunk_len = chunk_len
    exec_chunk_len = actor_chunk_len  # how many steps to execute per chunk

    frozen_vla.eval()
    encoder.eval()
    actor.eval()

    # Assign initial states (same-state grouping)
    num_unique = max(1, G // group_size)
    _rng = np.random.RandomState(seed + group_idx)
    unique_states = _rng.randint(0, n_initial_states, size=num_unique)
    state_ids = np.repeat(unique_states, group_size)[:G]

    n_workers = min(G, len(env_pool))

    # ── Phase 1: Reset all envs in parallel ──
    from concurrent.futures import as_completed as _as_completed
    obs_list = [None] * G
    with ThreadPoolExecutor(max_workers=G) as _pool:
        _futs = {_pool.submit(env_pool.reset_env, env_offset + g, suite_name, task_id, int(state_ids[g]), seed + g): g for g in range(G)}
        for _f in _as_completed(_futs):
            obs_list[_futs[_f]] = _f.result()
    print(f"  reset done: {G} envs (parallel)", flush=True)

    task_descriptions = [env_pool.envs[env_offset + g].task_description for g in range(G)]

    # ── Phase 2: Warmup dummy steps (parallel) ──
    if num_steps_wait > 0:
        with ThreadPoolExecutor(max_workers=G) as _pool:
            _futs = {_pool.submit(_env_dummy_steps, env_pool, env_offset + g, num_steps_wait): g for g in range(G)}
            for _f in _as_completed(_futs):
                obs_list[_futs[_f]] = _f.result()

    # ── Phase 3: Step-lock main loop ──
    episodes = [ActionTokenEpisode(task_id=task_id, state_idx=int(state_ids[g])) for g in range(G)]
    active = [True] * G  # which envs are still running
    env_steps = [0] * G
    all_frames = [[] for _ in range(G)]  # video frames

    max_chunks = max_steps // exec_chunk_len + 1

    # Timing accumulators
    _t_vla_forward = 0.0
    _t_encoder_actor = 0.0
    _t_store_records = 0.0
    _t_unnormalize = 0.0
    _t_env_step = 0.0
    _t_total_chunks = 0
    _t_rollout_start = time.time()

    for chunk_idx in range(max_chunks):
        active_ids = [g for g in range(G) if active[g]]
        if not active_ids:
            break

        _t_chunk_start = time.time()

        # ── Step 1: Batch VLA forward for all active envs ──
        _t0 = time.time()
        batch_images = [[obs_list[g]["primary_image"], obs_list[g]["wrist_image"]] for g in active_ids]
        batch_instrs = [task_descriptions[g] for g in active_ids]
        batch_props = [np.array(obs_list[g]["state"], dtype=np.float32) for g in active_ids]

        print(f"  [VLA forward] batch={len(batch_images)}, active_envs={len(active_ids)}", flush=True)
        # Build props_t up-front so Pi05's fused forward can also use it for
        # diffusion conditioning (it needs `state`).
        props_t = torch.tensor(np.array(batch_props), dtype=torch.float32).to(device)
        rl_tokens, vla_actions, action_queries = _run_vla_and_encode(
            frozen_vla, encoder, batch_images, batch_instrs, props_t, encoder_mode,
        )
        torch.cuda.synchronize()
        _t1 = time.time()
        _t_vla_forward += _t1 - _t0
        print(f"  [VLA done] va={vla_actions.shape} time={_t1-_t0:.3f}s", flush=True)

        # ── Step 2: Batch actor ──
        _t0 = time.time()

        # Slice VLA actions for actor if actor uses shorter chunk
        if actor_chunk_len < vla_actions.size(1):
            vla_actions_for_actor = vla_actions[:, :actor_chunk_len, :]
        else:
            vla_actions_for_actor = vla_actions

        if warmup_mode:
            # Warmup: use VLA actions directly, skip actor (like BatchInferenceServer.warmup_mode)
            actions_t = vla_actions_for_actor
            log_probs = torch.zeros(len(active_ids), device=device)
            values = torch.zeros(len(active_ids), device=device)
        else:
            actions_t, log_probs = actor(rl_tokens, vla_actions_for_actor, props_t, deterministic=False)
            values = critic(rl_tokens)  # (N_active,)
        torch.cuda.synchronize()
        _t1 = time.time()
        _t_encoder_actor += _t1 - _t0

        # Convert to numpy
        actions_np = actions_t.cpu().numpy()  # (N_active, exec_chunk_len, action_dim)
        vla_actions_cpu = vla_actions_for_actor.cpu()

        # ── Store step records ──
        _t0 = time.time()
        for i, g in enumerate(active_ids):
            sr = ActionTokenStepRecord(
                rl_token=rl_tokens[i:i+1].cpu().squeeze(0),
                vla_action=vla_actions_cpu[i],
                action_taken=actions_t[i].detach().cpu(),
                old_log_prob=log_probs[i].item() if log_probs is not None else 0.0,
                value=values[i].item(),
                prop_state=torch.tensor(batch_props[i]),
                images=[obs_list[g]["primary_image"].copy(), obs_list[g]["wrist_image"].copy()] if store_images else None,
                instruction=task_descriptions[g] if store_images else None,
            )
            episodes[g].step_records.append(sr)
        _t1 = time.time()
        _t_store_records += _t1 - _t0

        # ── Step 3: Unnormalize actions ──
        _t0 = time.time()
        action_chunks_unnorm = []
        for i in range(len(active_ids)):
            action_chunks_unnorm.append(_unnormalize(actions_np[i], action_norm_stats))
        _t1 = time.time()
        _t_unnormalize += _t1 - _t0

        # ── Step 4: All envs execute chunk in parallel ──
        _t0 = time.time()
        record_video = video_dir is not None
        with ThreadPoolExecutor(max_workers=len(active_ids)) as _pool:
            _futs = {}
            for i, g in enumerate(active_ids):
                _futs[_pool.submit(
                    _env_step_chunk, env_pool, env_offset + g, action_chunks_unnorm[i],
                    exec_chunk_len, record_video
                )] = (i, g)
            for _f in _as_completed(_futs):
                i, g = _futs[_f]
                obs, reward, done, steps_taken, frames = _f.result()
                obs_list[g] = obs
                env_steps[g] += steps_taken
                if record_video:
                    all_frames[g].extend(frames)
                if done or env_steps[g] >= max_steps:
                    active[g] = False
                    ep = episodes[g]
                    ep.success = bool(done and reward > 0.5)
                    ep.reward = reward_coef if ep.success else 0.0
                    ep.done_cache_idx = steps_taken
                    ep.finish_step = len(ep.step_records)
                    ep.env_steps = env_steps[g]
        _t1 = time.time()
        _t_env_step += _t1 - _t0
        _t_total_chunks += 1

        print(
            f"[TIMING] chunk {chunk_idx}: active={len(active_ids)} | "
            f"vla={_t_vla_forward/_t_total_chunks:.3f}s  enc+act={_t_encoder_actor/_t_total_chunks:.3f}s  "
            f"store={_t_store_records/_t_total_chunks:.3f}s  unnorm={_t_unnormalize/_t_total_chunks:.3f}s  "
            f"env_step={_t_env_step/_t_total_chunks:.3f}s  "
            f"chunk_total={time.time()-_t_chunk_start:.3f}s"
        )

    # ── Timing summary ──
    _t_rollout_total = time.time() - _t_rollout_start
    if _t_total_chunks > 0:
        print(
            f"\n[TIMING SUMMARY] rollout group {group_idx} | G={G} | {_t_total_chunks} chunks | total={_t_rollout_total:.2f}s\n"
            f"  vla_forward:    {_t_vla_forward:.2f}s ({100*_t_vla_forward/_t_rollout_total:.1f}%)  avg={_t_vla_forward/_t_total_chunks:.3f}s/chunk\n"
            f"  encoder+actor:  {_t_encoder_actor:.2f}s ({100*_t_encoder_actor/_t_rollout_total:.1f}%)  avg={_t_encoder_actor/_t_total_chunks:.3f}s/chunk\n"
            f"  store_records:  {_t_store_records:.2f}s ({100*_t_store_records/_t_rollout_total:.1f}%)  avg={_t_store_records/_t_total_chunks:.3f}s/chunk\n"
            f"  unnormalize:    {_t_unnormalize:.2f}s ({100*_t_unnormalize/_t_rollout_total:.1f}%)  avg={_t_unnormalize/_t_total_chunks:.3f}s/chunk\n"
            f"  env_step:       {_t_env_step:.2f}s ({100*_t_env_step/_t_rollout_total:.1f}%)  avg={_t_env_step/_t_total_chunks:.3f}s/chunk\n"
            f"  other/overhead: {_t_rollout_total - _t_vla_forward - _t_encoder_actor - _t_store_records - _t_unnormalize - _t_env_step:.2f}s"
        )

    # ── Finalize episodes ──
    for g in range(G):
        ep = episodes[g]
        if ep.finish_step == 0:  # timeout, never set
            ep.finish_step = len(ep.step_records)
            ep.env_steps = env_steps[g]
            ep.reward = 0.0

        if all_frames[g] and video_dir is not None:
            from AlphaBrain.training.reinforcement_learning.common.rollout import _save_video
            os.makedirs(video_dir, exist_ok=True)
            status = "success" if ep.success else "fail"
            vpath = os.path.join(video_dir,
                                 f"g{group_idx:04d}_e{g:02d}_t{task_id}_s{int(state_ids[g]):02d}_{status}.mp4")
            ep.video_path = _save_video(all_frames[g], vpath)

    return episodes


@torch.no_grad()
def action_token_collect_multitask_steplock(
    env_pool: PersistentEnvPool,
    frozen_vla,
    encoder: ActionTokenEncoderDecoder,
    actor: ActionTokenActor,
    critic: ActionTokenCritic,
    suite_name: str,
    task_ids: List[int],
    n_initial_states: int,
    action_norm_stats: dict,
    max_steps: int,
    chunk_len: int,
    G_per_task: int = 8,
    seed: int = 42,
    num_steps_wait: int = 10,
    device: str = "cuda",
    group_idx: int = 0,
    store_images: bool = False,
    group_size: int = 1,
    reward_coef: float = 1.0,
    actor_chunk_len: int = None,
    warmup_mode: bool = False,
    encoder_mode: str = "action_token",
) -> List[ActionTokenEpisode]:
    """
    Collect episodes for MULTIPLE tasks on ONE GPU in a single step-lock loop.

    All tasks' envs are merged into one batch for VLA forward — no per-task
    threading, no CUDA concurrency issues, maximum GPU batch utilization.

    Args:
        task_ids: list of task IDs to run on this GPU
        G_per_task: episodes per task
    Returns:
        flat list of all episodes across all tasks
    """
    if actor_chunk_len is None:
        actor_chunk_len = chunk_len
    exec_chunk_len = actor_chunk_len

    frozen_vla.eval()
    encoder.eval()
    actor.eval()

    n_tasks = len(task_ids)
    total_G = G_per_task * n_tasks

    # Assign states per task
    _rng = np.random.RandomState(seed + group_idx)
    all_state_ids = []
    all_task_labels = []  # which task each episode belongs to
    for tid in task_ids:
        num_unique = max(1, G_per_task // group_size)
        states = _rng.randint(0, n_initial_states, size=num_unique)
        states = np.repeat(states, group_size)[:G_per_task]
        all_state_ids.extend(states)
        all_task_labels.extend([tid] * G_per_task)

    n_workers = min(total_G, len(env_pool))

    # ── Phase 1: Reset all envs in parallel ──
    from concurrent.futures import as_completed as _as_completed
    obs_list = [None] * total_G
    with ThreadPoolExecutor(max_workers=total_G) as _pool:
        _futs = {_pool.submit(env_pool.reset_env, g, suite_name, all_task_labels[g], int(all_state_ids[g]), seed + g): g for g in range(total_G)}
        for _f in _as_completed(_futs):
            obs_list[_futs[_f]] = _f.result()
    print(f"  reset done: {total_G} envs (parallel)", flush=True)

    task_descriptions = [env_pool.envs[g].task_description for g in range(total_G)]

    # ── Phase 2: Warmup (parallel) ──
    if num_steps_wait > 0:
        with ThreadPoolExecutor(max_workers=total_G) as _pool:
            _futs = {_pool.submit(_env_dummy_steps, env_pool, g, num_steps_wait): g for g in range(total_G)}
            for _f in _as_completed(_futs):
                obs_list[_futs[_f]] = _f.result()

    # ── Phase 3: Step-lock main loop (ALL tasks merged) ──
    episodes = [ActionTokenEpisode(task_id=all_task_labels[g], state_idx=int(all_state_ids[g]))
                for g in range(total_G)]
    active = [True] * total_G
    env_steps = [0] * total_G
    max_chunks = max_steps // exec_chunk_len + 1

    _t_vla = 0.0
    _t_env = 0.0
    _n_chunks = 0

    for chunk_idx in range(max_chunks):
        active_ids = [g for g in range(total_G) if active[g]]
        if not active_ids:
            break

        # ── ONE batched VLA forward for ALL active envs across ALL tasks ──
        _t0 = time.time()
        batch_images = [[obs_list[g]["primary_image"], obs_list[g]["wrist_image"]] for g in active_ids]
        batch_instrs = [task_descriptions[g] for g in active_ids]
        batch_props = [np.array(obs_list[g]["state"], dtype=np.float32) for g in active_ids]

        # Build props_t up-front so Pi05's fused forward can use it for
        # diffusion conditioning (state).
        props_t = torch.tensor(np.array(batch_props), dtype=torch.float32).to(device)
        rl_tokens, vla_actions, action_queries = _run_vla_and_encode(
            frozen_vla, encoder, batch_images, batch_instrs, props_t, encoder_mode,
        )

        if actor_chunk_len < vla_actions.size(1):
            vla_actions_for_actor = vla_actions[:, :actor_chunk_len, :]
        else:
            vla_actions_for_actor = vla_actions

        if warmup_mode:
            actions_t = vla_actions_for_actor
            log_probs = torch.zeros(len(active_ids), device=device)
            values = torch.zeros(len(active_ids), device=device)
        else:
            actions_t, log_probs = actor(rl_tokens, vla_actions_for_actor, props_t, deterministic=False)
            values = critic(rl_tokens)
        _t_vla += time.time() - _t0

        actions_np = actions_t.cpu().numpy()
        vla_actions_cpu = vla_actions_for_actor.cpu()

        # Store records
        for i, g in enumerate(active_ids):
            episodes[g].step_records.append(ActionTokenStepRecord(
                rl_token=rl_tokens[i:i+1].cpu().squeeze(0),
                vla_action=vla_actions_cpu[i],
                action_taken=actions_t[i].detach().cpu(),
                old_log_prob=log_probs[i].item() if log_probs is not None else 0.0,
                value=values[i].item(),
                prop_state=torch.tensor(batch_props[i]),
            ))

        # Unnormalize
        action_chunks_unnorm = [_unnormalize(actions_np[i], action_norm_stats) for i in range(len(active_ids))]

        # ── ALL envs execute chunk in parallel ──
        _t0 = time.time()
        with ThreadPoolExecutor(max_workers=len(active_ids)) as _pool:
            _futs = {}
            for i, g in enumerate(active_ids):
                _futs[_pool.submit(_env_step_chunk, env_pool, g, action_chunks_unnorm[i],
                                   exec_chunk_len, False)] = (i, g)
            for _f in _as_completed(_futs):
                i, g = _futs[_f]
                obs, reward, done, steps_taken, _ = _f.result()
                obs_list[g] = obs
                env_steps[g] += steps_taken
                if done or env_steps[g] >= max_steps:
                    active[g] = False
                    ep = episodes[g]
                    ep.success = bool(done and reward > 0.5)
                    ep.reward = reward_coef if ep.success else 0.0
                    ep.done_cache_idx = steps_taken
                    ep.finish_step = len(ep.step_records)
                    ep.env_steps = env_steps[g]
        _t_env += time.time() - _t0
        _n_chunks += 1

    # Finalize
    for g in range(total_G):
        ep = episodes[g]
        if ep.finish_step == 0:
            ep.finish_step = len(ep.step_records)
            ep.env_steps = env_steps[g]
            ep.reward = 0.0

    if _n_chunks > 0:
        print(f"[MULTITASK TIMING] {n_tasks} tasks × {G_per_task} eps = {total_G} total | "
                     f"{_n_chunks} chunks | vla={_t_vla:.1f}s env={_t_env:.1f}s "
                     f"total={_t_vla+_t_env:.1f}s")

    return episodes
