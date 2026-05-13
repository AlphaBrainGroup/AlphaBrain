"""
ActionToken Trainer: Two-phase training for the RLT_a variant.

Phase 1 — Encoder Pretraining:
  Freeze VLA, train encoder-decoder via reconstruction loss on rollout data.

Phase 2 — Actor-Critic RL:
  Freeze VLA, use pretrained encoder.
  Rollout with actor (action editing), update actor + critic via PPO-clip.
  Optional encoder fine-tuning with reconstruction regularization.
"""

from __future__ import annotations

import logging
import os
import queue
import time
import threading
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from AlphaBrain.training.reinforcement_learning.envs.libero_env import LiberoEnv
from AlphaBrain.training.reinforcement_learning.common.replay_buffer import ReplayBuffer
from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_encoder_decoder import ActionTokenEncoderDecoder
from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_actor_critic import ActionTokenActor, ActionTokenCritic, ActionTokenQCritic
from AlphaBrain.training.reinforcement_learning.common.rollout import _unnormalize, _postprocess_action, _save_video

logger = logging.getLogger(__name__)

DUMMY_ACTION = np.array([0.0] * 6 + [-1.0], dtype=np.float32)


# ------------------------------------------------------------------
# Batched inference server for rollout
# ------------------------------------------------------------------

class BatchInferenceServer:
    """
    Batched VLA + encoder + actor + critic inference for rollout.

    Problem with model_lock (old approach):
      - Each env thread acquires lock → VLA forward batch=1 → releases lock
      - GPU processes one observation at a time regardless of how many envs are running
      - With chunk_len=8, GPU is idle 7/8 of the time (CPU env.step dominates)

    Solution:
      - All env threads submit (img_pair, instruction) to a request queue and block
      - A single background thread drains the queue and runs ONE batched GPU forward
      - All waiting threads get results simultaneously → GPU utilization scales with num_envs

    Cross-task batching:
      - In multi-task mode, multiple tasks on the same GPU share ONE server
      - When task-0 envs and task-5 envs both need VLA inference, they batch together
      - Effective batch size = n_tasks_per_gpu × n_envs_per_task
    """

    def __init__(
        self,
        frozen_vla,
        encoder,
        actor,
        critic,
        device: str,
        max_batch_size: int = 64,
        batch_timeout_s: float = 0.005,
        actor_chunk_len: int = None,
        encoder_mode: str = "action_token",
    ):
        self.frozen_vla = frozen_vla
        self.encoder = encoder
        self.actor = actor
        self.critic = critic
        self.device = device
        self.max_batch_size = max_batch_size
        self.batch_timeout_s = batch_timeout_s
        # If actor uses shorter chunk than VLA, slice vla_actions accordingly
        self.actor_chunk_len = actor_chunk_len
        # Encoder input mode:
        #   "action_token": feed encoder the chunk_len action-query slice (default).
        #   "rlt":      feed encoder the full last_hidden image-token slice
        #                   (z_{1:M} from the RL Token reference, Eq. 1).
        #   Both produce a rl_token of shape (B, 1, encoder.hidden_dim).
        self.encoder_mode = encoder_mode

        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self.warmup_mode = False   # when True, return pure VLA actions (no actor)

    def start(self) -> "BatchInferenceServer":
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=10.0)

    def infer(self, img_pair: list, instruction: str,
              prop_state=None):
        """
        Called from env threads. Blocks until the batch result is ready.

        Args:
            img_pair: [primary_img, wrist_img]  (numpy arrays, already flipped)
            instruction: task language description
            prop_state: np.ndarray or torch.Tensor (prop_dim,) — proprioceptive state

        Returns:
            (rl_token_cpu, vla_action_cpu, action_cpu, log_prob_float, value_float)
            tensors are (1, ...) on CPU.
        """
        done = threading.Event()
        box: list = [None]
        self._q.put((img_pair, instruction, prop_state, done, box))
        done.wait()
        return box[0]

    def _loop(self):
        while not self._stop.is_set():
            # Wait for the first request
            reqs = []
            try:
                reqs.append(self._q.get(timeout=0.02))
            except queue.Empty:
                continue

            # Drain queue within batch_timeout_s for additional requests
            deadline = time.perf_counter() + self.batch_timeout_s
            while len(reqs) < self.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    reqs.append(self._q.get(timeout=remaining))
                except queue.Empty:
                    break

            # Single batched GPU forward for all pending requests
            batch_images = [r[0] for r in reqs]   # [[primary, wrist], ...]
            batch_instrs = [r[1] for r in reqs]
            batch_props  = [r[2] for r in reqs]   # prop_state per request (or None)

            with torch.no_grad():
                # Build props_t up-front so the Pi05 fused forward (which needs
                # state for diffusion conditioning) can also use it.
                if batch_props[0] is not None:
                    props_list = []
                    for p in batch_props:
                        if isinstance(p, torch.Tensor):
                            props_list.append(p.float())
                        else:
                            props_list.append(torch.tensor(p, dtype=torch.float32))
                    props_t = torch.stack(props_list).to(self.device)             # (B, prop_dim)
                else:
                    props_t = None

                if self.encoder_mode == "rlt":
                    # Pi05 (PaliGemmaPi05) takes a different inference path
                    # entirely (no in-stream action tokens; action chunk comes
                    # from the flow-matching head). Dispatch on framework type.
                    from AlphaBrain.training.reinforcement_learning.algos.RLT.pi05_inference import (
                        is_pi05, get_pi05_rl_state_and_action,
                    )
                    if is_pi05(self.frozen_vla):
                        rl_tokens, vla_actions = get_pi05_rl_state_and_action(
                            self.frozen_vla, self.encoder,
                            batch_images=batch_images,
                            instructions=batch_instrs,
                            batch_props=props_t,
                        )
                        # Pi05 has no separate action_queries concept; keep a
                        # placeholder so downstream code that may reference it
                        # doesn't NameError.
                        action_queries = None
                    else:
                        # Qwen rlt: one VLM forward gives full hidden +
                        # action_queries + vla_actions; compact image-token
                        # slice into z_{1:M} and feed RLT encoder.
                        from AlphaBrain.training.reinforcement_learning.algos.RLT import (
                            get_vla_hidden_states_and_action,
                            compact_by_mask,
                            pad_mask_from_attention,
                        )
                        last_hidden, encoder_mask, _action_mask, action_queries, vla_actions = \
                            get_vla_hidden_states_and_action(
                                self.frozen_vla,
                                batch_images=batch_images,
                                instructions=batch_instrs,
                                image_only=True,
                            )
                        dense, kp_mask = compact_by_mask(last_hidden, encoder_mask)
                        rl_tokens = self.encoder.encode(
                            dense.float(), key_padding_mask=kp_mask
                        )                                                              # (B, 1, H)
                else:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        action_queries, vla_actions = self.frozen_vla.get_vla_action(
                            batch_images=batch_images,
                            instructions=batch_instrs,
                        )
                    rl_tokens = self.encoder.encode(action_queries)                # (B, 1, D)

                B = rl_tokens.size(0)

                # Slice VLA actions to actor_chunk_len if specified
                C_actor = self.actor_chunk_len
                if C_actor is not None and C_actor < vla_actions.size(1):
                    vla_actions_for_actor = vla_actions[:, :C_actor, :]
                else:
                    vla_actions_for_actor = vla_actions

                if self.warmup_mode:
                    # VLA warmup: return VLA actions directly, no actor modification
                    actions = vla_actions_for_actor
                    log_probs = torch.zeros(B, device=self.device)
                    values = torch.zeros(B, device=self.device)
                else:
                    actions, log_probs = self.actor(
                        rl_tokens, vla_actions_for_actor, props_t, deterministic=False)
                    values = self.critic(rl_tokens)                               # (B,)

            # Dispatch results back to each waiting env thread
            for i, (_, _, _, done, box) in enumerate(reqs):
                box[0] = (
                    rl_tokens[i:i+1].cpu(),              # (1, 1, D)
                    vla_actions_for_actor[i:i+1].cpu(),   # (1, C_actor, A)
                    actions[i:i+1].cpu(),                  # (1, C_actor, A)
                    log_probs[i].item() if log_probs is not None else 0.0,
                    values[i].item(),
                )
                done.set()


# ------------------------------------------------------------------
# Data structures (same pattern as existing rollout.py)
# ------------------------------------------------------------------

@dataclass
class ActionTokenStepRecord:
    """One inference step during RLT_a rollout."""
    rl_token: torch.Tensor        # (1, D) detached cpu
    vla_action: torch.Tensor      # (chunk_len, action_dim) detached cpu
    action_taken: torch.Tensor    # (chunk_len, action_dim) detached cpu
    old_log_prob: float
    value: float = 0.0
    prop_state: Optional[torch.Tensor] = None  # (prop_dim,) proprioceptive at chunk start
    # Chunk subsampling: intermediate VLA inference results at stride positions [2,4,6]
    # Each element: (rl_token (1,D), vla_action (C,A), prop_state (prop_dim,))
    sub_tokens: list = field(default_factory=list)
    # Optional: store raw images for VLA full fine-tune (re-encode during training)
    images: Optional[list] = None         # [primary_img, wrist_img] numpy arrays
    instruction: Optional[str] = None


@dataclass
class ActionTokenEpisode:
    step_records: List[ActionTokenStepRecord] = field(default_factory=list)
    reward: float = 0.0
    task_id: int = 0
    success: bool = False
    finish_step: int = 0
    env_steps: int = 0          # total env.step() calls (excluding wait steps)
    done_cache_idx: int = -1    # cache_idx at termination (within last chunk); step = idx-1
    video_path: Optional[str] = None
    state_idx: int = -1


# ------------------------------------------------------------------
# Phase 1: Encoder pretraining via rollout data collection
# ------------------------------------------------------------------

def pretrain_encoder_step(
    frozen_vla,
    enc_dec: ActionTokenEncoderDecoder,
    batch_images: list,
    instructions: list,
    device: str = "cuda",
):
    """
    One pretraining step: VLA forward → encoder → decoder → reconstruction loss.

    Returns:
        recon_loss: scalar tensor (with grad on enc_dec params only)
    """
    with torch.no_grad():
        action_queries = frozen_vla.get_action_queries(
            batch_images=batch_images,
            instructions=instructions,
        )  # (B, chunk_len, H) on device

    _, recon_loss = enc_dec(action_queries)
    return recon_loss


def collect_observations_fast(
    suite_name: str,
    task_id: int,
    n_observations: int,
    steps_per_env: int = 20,
    num_envs: int = 4,
    n_initial_states: int = 50,
    libero_python: str = None,
    seed: int = 42,
) -> list:
    """
    Lightweight observation collection for encoder pretraining.

    Instead of running full episodes (300+ steps each, 99% CPU idle),
    just reset envs to diverse initial states and take a few random steps.
    Returns list of (images, instruction) tuples ready for VLA forward.

    Much faster than collect_group: no VLA forward during collection,
    no action caching, just random exploration for observation diversity.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    observations = []

    # Shared progress counter across threads
    obs_per_reset = 1 + steps_per_env
    n_resets = max(1, (n_observations + obs_per_reset - 1) // obs_per_reset)
    resets_per_env = (n_resets + num_envs - 1) // num_envs
    progress_lock = threading.Lock()
    progress = {"resets_done": 0, "obs_count": 0}
    log_interval = max(1, n_resets // 20)  # log every ~5%

    logger.info(f"  [collect_obs] Plan: {n_resets} env resets × {steps_per_env} steps/reset "
                f"= ~{n_resets * obs_per_reset} obs, using {num_envs} envs")

    def _collect_from_env(env_idx, states_to_visit):
        local_rng = np.random.RandomState(seed + env_idx * 10000)
        local_obs = []
        env = LiberoEnv(libero_python=libero_python)
        try:
            for s_idx in states_to_visit:
                obs = env.reset(
                    suite_name=suite_name,
                    task_id=task_id,
                    initial_state_idx=s_idx % n_initial_states,
                    seed=seed + env_idx * 1000 + s_idx,
                )
                task_desc = env.task_description
                local_obs.append((
                    [obs["primary_image"].copy(), obs["wrist_image"].copy()],
                    task_desc,
                ))
                for _ in range(steps_per_env):
                    random_action = local_rng.uniform(-1, 1, size=7).astype(np.float32)
                    random_action[6] = local_rng.choice([-1.0, 1.0])
                    obs, _, done = env.step(random_action)
                    local_obs.append((
                        [obs["primary_image"].copy(), obs["wrist_image"].copy()],
                        task_desc,
                    ))
                    if done:
                        break
                # Update shared progress
                with progress_lock:
                    progress["resets_done"] += 1
                    rd = progress["resets_done"]
                    if rd % log_interval == 0 or rd == n_resets:
                        est_obs = rd * obs_per_reset
                        logger.info(f"  [collect_obs] reset {rd}/{n_resets} "
                                    f"({rd * 100 // n_resets}%), ~{est_obs} obs collected")
        finally:
            env.close()
        return local_obs

    with ThreadPoolExecutor(max_workers=num_envs) as pool:
        futures = {}
        for e in range(num_envs):
            start_state = e * resets_per_env
            end_state = min(start_state + resets_per_env, n_resets)
            if start_state >= n_resets:
                break
            states = list(range(start_state, end_state))
            futures[pool.submit(_collect_from_env, e, states)] = e

        for fut in as_completed(futures):
            local_obs = fut.result()
            observations.extend(local_obs)

    # Trim to requested size
    if len(observations) > n_observations:
        np.random.shuffle(observations)
        observations = observations[:n_observations]

    logger.info(f"  [collect_obs] Collected {len(observations)} observations")
    return observations


def extract_action_queries_from_obs(
    frozen_vla,
    observations: list,
    batch_size: int = 16,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Batch-extract action_queries from observations via frozen VLA.

    Args:
        observations: list of (images, instruction) where images=[primary, wrist]

    Returns:
        all_queries: (N, chunk_len, H) tensor on device
    """
    N = len(observations)
    n_batches = (N + batch_size - 1) // batch_size
    all_queries = []
    for b_idx, start in enumerate(range(0, N, batch_size)):
        end = min(start + batch_size, N)
        batch_imgs = [observations[i][0] for i in range(start, end)]
        batch_instr = [observations[i][1] for i in range(start, end)]
        with torch.no_grad():
            aq = frozen_vla.get_action_queries(
                batch_images=batch_imgs,
                instructions=batch_instr,
            )  # (B, chunk_len, H)
        all_queries.append(aq)
        if (b_idx + 1) % max(1, n_batches // 10) == 0 or b_idx == n_batches - 1:
            logger.info(f"  [extract] batch {b_idx + 1}/{n_batches} "
                        f"({end}/{N} samples)")

    return torch.cat(all_queries, dim=0)  # (N, chunk_len, H)


def extract_action_queries_dataset(
    frozen_vla,
    episodes: list,
    batch_size: int = 16,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Batch-extract all action_queries from rollout episodes via frozen VLA.
    (Legacy: used when full episodes are already collected.)
    """
    obs_list = []
    for ep in episodes:
        for step in ep.step_records:
            obs_list.append(
                ([step.primary_image, step.wrist_image], step.instruction)
            )
    return extract_action_queries_from_obs(frozen_vla, obs_list, batch_size, device)


# ------------------------------------------------------------------
# Phase 2: RLT_a Rollout — frozen VLA + encoder + actor
# ------------------------------------------------------------------

def _action_token_rollout_one(
    env: LiberoEnv,
    batch_server: BatchInferenceServer,
    suite_name: str,
    task_id: int,
    state_idx: int,
    action_norm_stats: dict,
    max_steps: int,
    chunk_len: int,
    num_steps_wait: int,
    seed: int,
    record_video: bool,
    episode_idx: int,
    group_idx: int,
    video_dir: Optional[str],
    store_images: bool = False,
    reward_coef: float = 1.0,
) -> ActionTokenEpisode:
    """Run one episode with RLT_a actor via BatchInferenceServer.

    Chunk subsampling (paper): at stride-2 positions (2, 4, 6) within each chunk,
    call batch_server.infer() on the intermediate observation to get (rl_token, vla_action).
    These are stored in step_record.sub_tokens for push_episodes_to_buffer to use.
    """
    episode = ActionTokenEpisode(task_id=task_id, state_idx=state_idx)
    frames: List[np.ndarray] = [] if record_video else None

    obs = env.reset(
        suite_name=suite_name,
        task_id=task_id,
        initial_state_idx=state_idx,
        seed=seed,
    )
    task_description = env.task_description

    env_step = 0
    action_cache: Optional[np.ndarray] = None
    cache_idx = 0
    current_sr: Optional[ActionTokenStepRecord] = None

    # Stride positions within a chunk for subsampling (paper: stride=2)
    _sub_positions = set(range(2, chunk_len, 2))  # {2, 4, 6} for chunk_len=8

    while env_step < max_steps + num_steps_wait:
        if env_step < num_steps_wait:
            obs, _, done = env.step(DUMMY_ACTION)
            env_step += 1
            continue

        if action_cache is None or cache_idx >= chunk_len:
            # Submit to batch server — blocks until the batch is processed
            img_pair = [obs["primary_image"], obs["wrist_image"]]
            prop_np = np.array(obs["state"], dtype=np.float32)
            rl_token_cpu, vla_action_cpu, action_cpu, log_prob_val, value = \
                batch_server.infer(img_pair, task_description, prop_np)

            action_np = action_cpu[0].numpy()  # (chunk_len, action_dim)

            current_sr = ActionTokenStepRecord(
                rl_token=rl_token_cpu[0],       # (1, D)
                vla_action=vla_action_cpu[0],   # (chunk_len, A)
                action_taken=action_cpu[0],     # (chunk_len, A)
                old_log_prob=log_prob_val,
                value=value,
                prop_state=torch.tensor(prop_np),
                images=[obs["primary_image"].copy(), obs["wrist_image"].copy()] if store_images else None,
                instruction=task_description if store_images else None,
            )
            episode.step_records.append(current_sr)

            action_cache = _unnormalize(action_np, action_norm_stats)
            cache_idx = 0
            _chunk_count = len(episode.step_records)
            print(f"    [ep{episode_idx}] chunk {_chunk_count} infer done, env_step={env_step}", flush=True)

        env_action = _postprocess_action(action_cache[cache_idx])
        cache_idx += 1
        obs, reward, done = env.step(env_action)
        env_step += 1

        if frames is not None:
            frames.append(obs["primary_image"].copy())

        # Capture intermediate observations for chunk subsampling (stride=2)
        # cache_idx was just incremented, so its value equals the position AFTER this step
        if not done and cache_idx in _sub_positions and current_sr is not None:
            sub_img = [obs["primary_image"].copy(), obs["wrist_image"].copy()]
            sub_prop_np = np.array(obs["state"], dtype=np.float32)
            sub_tok_cpu, sub_vla_cpu, _, _, _ = \
                batch_server.infer(sub_img, task_description, sub_prop_np)
            current_sr.sub_tokens.append((
                sub_tok_cpu[0],               # rl_token (1, D)
                sub_vla_cpu[0],               # vla_action (C, A)
                torch.tensor(sub_prop_np),    # prop_state (prop_dim,)
            ))

        if done:
            episode.success = bool(reward > 0.5)
            episode.reward = reward_coef if episode.success else 0.0
            episode.done_cache_idx = cache_idx   # step that triggered done = cache_idx-1
            break

    # Timeout counts as failure
    if not episode.success:
        episode.reward = 0.0
        episode.done_cache_idx = cache_idx       # wherever we stopped in the last chunk

    episode.finish_step = len(episode.step_records)
    episode.env_steps = max(0, env_step - num_steps_wait)

    if frames and video_dir is not None:
        os.makedirs(video_dir, exist_ok=True)
        status = "success" if episode.success else "fail"
        vpath = os.path.join(
            video_dir,
            f"at_g{group_idx:02d}_t{task_id:02d}_ep{episode_idx:02d}_{status}.mp4",
        )
        episode.video_path = _save_video(frames, vpath)

    return episode


@torch.no_grad()
def action_token_collect_group(
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
    G: int = 8,
    libero_python: Optional[str] = None,
    seed: int = 42,
    num_steps_wait: int = 10,
    device: str = "cuda",
    video_dir: Optional[str] = None,
    num_envs: int = 4,
    group_idx: int = 0,
    batch_server: Optional[BatchInferenceServer] = None,
    store_images: bool = False,
    group_size: int = 1,
    reward_coef: float = 1.0,
) -> List[ActionTokenEpisode]:
    """
    Collect G episodes using RLT_a policy.

    Uses BatchInferenceServer for GPU inference: all num_envs env threads submit
    requests concurrently; a single background thread batches them into one GPU
    forward pass. This maximizes GPU utilization vs. the old model_lock (batch=1).

    Args:
        batch_server: shared server for this GPU (created by caller for cross-task
                      batching). If None, creates a local server for this call only.
        group_size: number of trajectories per initial state. G episodes are
                    split into G//group_size unique states, each repeated
                    group_size times. Default 1 = legacy behavior (no repeat).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    frozen_vla.eval()
    encoder.eval()
    actor.eval()
    critic.eval()

    # Use caller-provided server (shared across tasks on same GPU) or create locally
    _own_server = batch_server is None
    if _own_server:
        batch_server = BatchInferenceServer(
            frozen_vla=frozen_vla,
            encoder=encoder,
            actor=actor,
            critic=critic,
            device=device,
        ).start()

    n_workers = min(G, num_envs)
    # Each episode gets its own env (LiberoEnv is not thread-safe for reuse)
    envs = []
    for _ei in range(G):
        envs.append(LiberoEnv(libero_python=libero_python))
        if (_ei + 1) % 10 == 0 or _ei == G - 1:
            print(f"  envs created: {_ei+1}/{G}", flush=True)

    # Assign initial states: same-state grouping.
    # G episodes → G//group_size unique states, each repeated group_size times
    num_unique = max(1, G // group_size)
    _rng = np.random.RandomState(seed + group_idx)
    unique_states = _rng.randint(0, n_initial_states, size=num_unique)
    state_ids = np.repeat(unique_states, group_size)[:G]  # [s0,s0,s0,s0, s1,s1, ...]

    episodes = [None] * G
    try:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {}
            for g in range(G):
                fut = pool.submit(
                    _action_token_rollout_one,
                    env=envs[g],
                    batch_server=batch_server,
                    suite_name=suite_name,
                    task_id=task_id,
                    state_idx=int(state_ids[g]),
                    action_norm_stats=action_norm_stats,
                    max_steps=max_steps,
                    chunk_len=chunk_len,
                    num_steps_wait=num_steps_wait,
                    seed=seed + g,
                    record_video=(video_dir is not None),
                    episode_idx=g,
                    group_idx=group_idx,
                    video_dir=video_dir,
                    store_images=store_images,
                    reward_coef=reward_coef,
                )
                futures[fut] = g
            done_count = 0
            success_count = 0
            for fut in as_completed(futures):
                g_idx = futures[fut]
                try:
                    ep = fut.result()
                except Exception as _ep_err:
                    # Async mode spawns ~G fresh LIBERO subprocesses per iter;
                    # robosuite/MuJoCo EGL render-context init occasionally
                    # native-SEGVs the worker under that spawn pressure (the
                    # parent then sees "LIBERO worker exited unexpectedly").
                    # Losing one episode is fine; poisoning the whole batch
                    # via the rollout-thread's try/except (and the main loop
                    # via the poison pill) is not. Mark this slot as an empty
                    # failed episode — push_episodes_to_buffer skips
                    # finish_step==0 so the buffer stays clean.
                    print(f"  [rollout][dev={device}] ep g_idx={g_idx} FAILED "
                          f"({type(_ep_err).__name__}: {_ep_err}); "
                          f"dropping this episode.", flush=True)
                    ep = ActionTokenEpisode(
                        task_id=task_id, state_idx=int(state_ids[g_idx]))
                    ep.success = False
                    ep.reward = 0.0
                    ep.finish_step = 0
                    ep.env_steps = 0
                episodes[g_idx] = ep
                done_count += 1
                if ep.success:
                    success_count += 1
                print(f"  [rollout][dev={device}] ep {done_count}/{G} done "
                      f"({'SUCCESS' if ep.success else 'fail'}, "
                      f"{ep.env_steps} steps) "
                      f"[{success_count}/{done_count} success so far]", flush=True)
    finally:
        for env in envs:
            env.close()
        if _own_server:
            batch_server.stop()

    return episodes


# ------------------------------------------------------------------
# Phase 2: PPO-clip loss for RLT_a small networks
# ------------------------------------------------------------------

def compute_action_token_gae(
    episode: ActionTokenEpisode,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
):
    """
    Compute GAE advantages and returns for a single RLT_a episode.

    Returns:
        advantages: list of floats (len = finish_step)
        returns: list of floats (len = finish_step)
    """
    steps = episode.step_records
    n = episode.finish_step
    if n == 0:
        return [], []

    values = [s.value for s in steps[:n]]
    # Terminal value = 0 (episode ended)
    advantages = [0.0] * n
    returns = [0.0] * n

    # Sparse reward at last step only
    rewards = [0.0] * n
    rewards[-1] = episode.reward

    gae = 0.0
    for t in reversed(range(n)):
        next_value = values[t + 1] if t + 1 < n else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae
        returns[t] = gae + values[t]

    return advantages, returns


def action_token_ppo_loss(
    encoder: ActionTokenEncoderDecoder,
    actor: ActionTokenActor,
    critic: ActionTokenCritic,
    episodes: List[ActionTokenEpisode],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    recon_loss_coef: float = 0.0,
    frozen_vla=None,
    device: str = "cuda",
):
    """
    Compute PPO loss on a batch of RLT_a episodes.

    Only encoder + actor + critic have gradients.
    Optionally add reconstruction loss as regularizer.

    Returns:
        loss: scalar tensor
        stats: dict with training metrics
    """
    all_rl_tokens = []
    all_vla_actions = []
    all_actions_taken = []
    all_old_log_probs = []
    all_advantages = []
    all_returns = []
    all_old_values = []
    all_prop_states = []

    for ep in episodes:
        adv, ret = compute_action_token_gae(ep, gamma, gae_lambda)
        for t in range(ep.finish_step):
            step = ep.step_records[t]
            all_rl_tokens.append(step.rl_token)
            all_vla_actions.append(step.vla_action)
            all_actions_taken.append(step.action_taken)
            all_old_log_probs.append(step.old_log_prob)
            all_advantages.append(adv[t])
            all_returns.append(ret[t])
            all_old_values.append(step.value)
            prop = step.prop_state if step.prop_state is not None else torch.zeros(8)
            all_prop_states.append(prop)

    if not all_rl_tokens:
        return torch.tensor(0.0, device=device, requires_grad=True), {"n_steps": 0}

    # Stack to batched tensors
    rl_tokens = torch.stack(all_rl_tokens).to(device)          # (N, 1, D)
    vla_actions = torch.stack(all_vla_actions).to(device)      # (N, C, A)
    actions_taken = torch.stack(all_actions_taken).to(device)  # (N, C, A)
    old_lp = torch.tensor(all_old_log_probs, device=device)   # (N,)
    advantages = torch.tensor(all_advantages, device=device)   # (N,)
    returns = torch.tensor(all_returns, device=device)         # (N,)
    old_values = torch.tensor(all_old_values, device=device)   # (N,)
    prop_states = torch.stack(all_prop_states).to(device)      # (N, prop_dim)

    # Normalize advantages
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # New policy log probs
    new_lp = actor.log_prob_of(rl_tokens, vla_actions, actions_taken, prop_states)  # (N,)

    # PPO clipped policy loss
    ratio = torch.exp(new_lp - old_lp)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    pg_loss = -torch.min(surr1, surr2).mean()

    # Value loss (clipped)
    new_values = critic(rl_tokens)  # (N,)
    v_clipped = old_values + torch.clamp(new_values - old_values, -10.0, 10.0)
    vf_loss = torch.max(
        (new_values - returns) ** 2,
        (v_clipped - returns) ** 2,
    ).mean()

    loss = pg_loss + vf_coef * vf_loss

    # Optional reconstruction regularization
    recon_loss_val = 0.0
    if recon_loss_coef > 0.0 and rl_tokens.size(0) > 0:
        # Re-encode to get reconstruction loss (encoder must be in train mode)
        # Note: rl_tokens were computed with no_grad during rollout, so we need
        # to recompute through encoder for gradient flow
        # This requires action_queries which we don't have — skip if not available
        pass

    stats = {
        "pg_loss": pg_loss.item(),
        "vf_loss": vf_loss.item(),
        "loss": loss.item(),
        "ratio_mean": ratio.mean().item(),
        "ratio_max": ratio.max().item(),
        "ratio_min": ratio.min().item(),
        "clip_frac": ((ratio - 1.0).abs() > clip_eps).float().mean().item(),
        "n_steps": len(all_rl_tokens),
        "advantage_mean": advantages.mean().item(),
        "value_mean": new_values.mean().item(),
    }
    return loss, stats


# ------------------------------------------------------------------
# Off-policy: Push episodes to replay buffer
# ------------------------------------------------------------------

def push_episodes_to_buffer(
    episodes: List[ActionTokenEpisode],
    replay_buffer: ReplayBuffer,
    gamma_per_step: float = 0.99,
):
    """
    Convert episode step_records into (s, a, r, s', done) transitions and push
    them into the replay buffer.

    Chunk subsampling (paper, stride=2):
      For each executed chunk, we create transitions starting from positions
      [0, 2, 4, 6] within the chunk. Position-p transition:
        - state:      rl_token at obs[p] within this chunk
        - action:     action_taken[p:C] || next_chunk.action_taken[0:p]  (length-C cross-chunk slice)
        - next_state: rl_token at obs[p] within the NEXT chunk
        - done:       True only for terminal chunk

    Reward (paper Eq. 3): Q̂ = Σ_{t'=0}^{C-1} γ^{t'} r_{t+t'} + γ^C Q_next.
    For sparse reward, only the terminal chunk has non-zero reward.
    At stride position p in terminal chunk (done at step d):
        reward = γ^(d-p) * episode_reward   (discounted by steps from p to done)

    Reward scheme: success=1.0, failure=0.0 (paper scheme)
    """
    n_pushed = 0
    for ep in episodes:
        steps = ep.step_records
        n = ep.finish_step
        if n == 0:
            continue

        # Infer chunk_len from first step record
        chunk_len = steps[0].action_taken.shape[0]
        stride = 2
        stride_positions = list(range(0, chunk_len, stride))  # [0, 2, 4, 6] for C=8

        # Terminal step within last chunk (0-based): done_cache_idx was set after +1
        done_step = max(0, ep.done_cache_idx - 1) if ep.done_cache_idx >= 0 else chunk_len - 1

        for t in range(n):
            s = steps[t]
            is_last = (t == n - 1)
            done = is_last
            s_next = steps[t + 1] if not is_last else None

            for pos_idx, p in enumerate(stride_positions):
                # ── Terminal chunk: skip stride positions beyond the done step ──
                if is_last and p > done_step:
                    break

                # ── Current state at position p ──
                if p == 0:
                    rl_tok = s.rl_token
                    vla_act = s.vla_action
                    prop = s.prop_state
                else:
                    sub_idx = pos_idx - 1  # sub_tokens index (0→pos2, 1→pos4, 2→pos6)
                    if sub_idx >= len(s.sub_tokens):
                        # Episode ended before position p within this chunk; skip
                        break
                    rl_tok, vla_act, prop = s.sub_tokens[sub_idx]

                # ── Action: cross-chunk slice of length C ──
                if p == 0:
                    action = s.action_taken  # full chunk, no slicing needed
                else:
                    tail = s.action_taken[p:]           # (C-p, A)
                    if s_next is not None:
                        head = s_next.action_taken[:p]  # (p, A)
                    else:
                        # Terminal chunk: pad with zeros
                        head = torch.zeros(p, s.action_taken.shape[-1],
                                           dtype=s.action_taken.dtype)
                    action = torch.cat([tail, head], dim=0)  # (C, A)

                # ── Reward (paper Eq. 3): discounted within-chunk reward ──
                if is_last and ep.reward != 0.0:
                    # Sparse reward at terminal step: γ^(done_step - p) * R
                    stride_reward = (gamma_per_step ** (done_step - p)) * ep.reward
                else:
                    stride_reward = 0.0

                # ── Next state at position p ──
                if is_last:
                    next_rl_tok = torch.zeros_like(rl_tok)
                    next_vla_act = torch.zeros_like(vla_act)
                    next_prop = torch.zeros_like(prop) if prop is not None else None
                elif p == 0:
                    next_rl_tok = s_next.rl_token
                    next_vla_act = s_next.vla_action
                    next_prop = s_next.prop_state
                else:
                    sub_idx = pos_idx - 1
                    if sub_idx >= len(s_next.sub_tokens):
                        # Next chunk doesn't have sub_token at position p (ended early)
                        # Sub-positions p+2, p+4 also missing → stop this chunk's subsampling
                        break
                    next_rl_tok, next_vla_act, next_prop = s_next.sub_tokens[sub_idx]

                replay_buffer.push(
                    rl_token=rl_tok,
                    vla_action=vla_act,
                    action_taken=action,
                    reward=stride_reward,
                    next_rl_token=next_rl_tok,
                    next_vla_action=next_vla_act,
                    done=done,
                    task_id=ep.task_id,
                    prop_state=prop,
                    next_prop_state=next_prop,
                )
                n_pushed += 1
    return n_pushed


# ------------------------------------------------------------------
# VLA full fine-tune step (re-encode from images, gradients flow to VLA)
# ------------------------------------------------------------------

def vla_finetune_step(
    vla,
    encoder: ActionTokenEncoderDecoder,
    actor: ActionTokenActor,
    q_critic: ActionTokenQCritic,
    episodes: List[ActionTokenEpisode],
    beta: float = 0.1,
    device: str = "cuda",
    micro_batch: int = 4,
):
    """
    Full fine-tune: re-run VLA forward on stored images with gradients enabled.

    Gradient path: actor_loss → actor → rl_token → encoder → action_queries → VLA
    The critic is frozen during this step (only provides Q signal, no param update).

    Args:
        episodes: current iteration's episodes (must have .images stored)
        micro_batch: VLA forward batch size (controls GPU memory)
    """
    # Collect all (images, instruction, prop) from step records
    all_imgs, all_instrs, all_props = [], [], []
    for ep in episodes:
        for sr in ep.step_records:
            if sr.images is not None:
                all_imgs.append(sr.images)
                all_instrs.append(sr.instruction)
                all_props.append(sr.prop_state)

    if not all_imgs:
        return torch.tensor(0.0, device=device, requires_grad=True), {}

    # Freeze critic params (we only want gradients for VLA/encoder/actor)
    critic_was_training = q_critic.training
    for p in q_critic.parameters():
        p.requires_grad_(False)

    total_loss = 0.0
    total_q = 0.0
    total_bc = 0.0
    n_batches = 0

    for i in range(0, len(all_imgs), micro_batch):
        batch_imgs = all_imgs[i:i + micro_batch]
        batch_instr = all_instrs[i:i + micro_batch]
        batch_props = all_props[i:i + micro_batch]
        B = len(batch_imgs)

        # VLA forward WITH gradients (the whole point of full fine-tune)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            action_queries, vla_actions = vla.get_vla_action(
                batch_images=batch_imgs, instructions=batch_instr)
        # action_queries: (B, chunk_len, H) with grad through VLA

        rl_tokens = encoder.encode(action_queries)  # (B, 1, D) with grad

        props_t = torch.stack([p.float() for p in batch_props]).to(device) if batch_props[0] is not None else None
        actions, _ = actor(rl_tokens, vla_actions, props_t, deterministic=False)

        # Q provides gradient signal to actions and rl_tokens (but critic params frozen)
        q_val = q_critic.q1_forward(rl_tokens, actions, props_t)
        bc_penalty = ((actions - vla_actions) ** 2).sum(dim=(-2, -1)).mean()
        loss = -q_val.mean() + beta * bc_penalty
        loss.backward()

        total_loss += loss.item()
        total_q += q_val.mean().item()
        total_bc += bc_penalty.item()
        n_batches += 1

    # Restore critic
    for p in q_critic.parameters():
        p.requires_grad_(True)
    if critic_was_training:
        q_critic.train()

    n = max(n_batches, 1)
    stats = {
        "vla_loss": total_loss / n,
        "vla_q": total_q / n,
        "vla_bc": total_bc / n,
        "vla_n_samples": len(all_imgs),
    }
    return stats


# ------------------------------------------------------------------
# Off-policy TD update (RL Token paper style: TD3 twin-Q + DDPG actor)
# ------------------------------------------------------------------

def action_token_td_critic_update(
    actor: ActionTokenActor,
    q_critic: ActionTokenQCritic,
    target_q_critic: ActionTokenQCritic,
    replay_buffer: ReplayBuffer,
    batch_size: int = 256,
    gamma: float = 0.99,
    device: str = "cuda",
    target_noise_std: float = 0.2,
    target_noise_clip: float = 0.5,
    n_tasks: int = 0,
    target_actor: ActionTokenActor = None,
):
    """
    TD3-style twin-Q critic update from replay buffer (Eq. 3 in paper).

    Target: Q̂ = Σ γ^t' r_t' + γ^C * min(Q1', Q2')(s', a')
    where a' ~ π_target(·|s', ã') + clipped noise  (target policy smoothing)

    Args:
        n_tasks: if > 0, use per-task stratified sampling for balanced multi-task update.
        target_actor: Polyak-averaged actor for computing next actions (TD3).
                      Falls back to online actor if None.

    Returns:
        critic_loss: scalar tensor (with grad on q_critic params)
        stats: dict
    """
    if n_tasks > 0:
        rl_tok, vla_act, act_taken, rew, next_rl_tok, next_vla_act, done, prop, next_prop = \
            replay_buffer.sample_balanced(batch_size, n_tasks=n_tasks, device=device)
    else:
        rl_tok, vla_act, act_taken, rew, next_rl_tok, next_vla_act, done, prop, next_prop = \
            replay_buffer.sample(batch_size, device=device)

    # ── Target Q value (TD3: target policy smoothing + min of twin Q) ──
    with torch.no_grad():
        # Next action from target actor (TD3) + smoothing noise
        _actor_for_target = target_actor if target_actor is not None else actor
        next_action, _ = _actor_for_target(next_rl_tok, next_vla_act, next_prop, deterministic=True)
        # TD3 target policy smoothing: add clipped noise to target actions
        noise = torch.randn_like(next_action) * target_noise_std
        noise = noise.clamp(-target_noise_clip, target_noise_clip)
        next_action = (next_action + noise).clamp(-1.0, 1.0)

        # Target Q with min of twin Q (paper: Eq.3)
        tq1, tq2 = target_q_critic(next_rl_tok, next_action, next_prop)
        next_q = torch.min(tq1, tq2)  # (B,)
        target = rew + gamma * next_q * (1.0 - done)
        # Clip to theoretical upper bound (paper reward: success=1, so Q ≤ 1/(1-γ)).
        # Prevents bootstrap overestimation from runaway positive Q values.
        # Clip to theoretical upper bound: max_reward / (1 - gamma)
        # With reward_coef=5, gamma=0.99 → upper bound = 500. Use reward_coef as safe proxy.
        q_upper = max(1.0, rew.abs().max().item() * 2) if rew.numel() > 0 else 1.0
        target = target.clamp(max=q_upper)

    # ── Online Q loss ──
    q1, q2 = q_critic(rl_tok, act_taken, prop)
    critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

    stats = {
        "critic_loss": critic_loss.item(),
        "q1_mean": q1.mean().item(),
        "q2_mean": q2.mean().item(),
        "target_mean": target.mean().item(),
    }
    return critic_loss, stats


def action_token_td_actor_update(
    actor: ActionTokenActor,
    q_critic: ActionTokenQCritic,
    replay_buffer: ReplayBuffer,
    batch_size: int = 256,
    beta: float = 1.0,
    device: str = "cuda",
    n_tasks: int = 0,
):
    """
    DDPG-style actor update from the RL Token paper (Eq. 5):

    L_π(θ) = E[ -Q_ψ(x, a) + β ‖a - ã‖² ]

    where a ~ π_θ (stochastic rsample, paper Eq. 5). With fixed small std=0.1,
    the gradient direction is nearly identical to the deterministic mean, but we
    match the paper's formulation exactly.

    Args:
        n_tasks: if > 0, use per-task stratified sampling for balanced multi-task update.

    Returns:
        actor_loss: scalar tensor (with grad on actor params)
        stats: dict
    """
    if n_tasks > 0:
        rl_tok, vla_act, _, _, _, _, _, prop, _ = \
            replay_buffer.sample_balanced(batch_size, n_tasks=n_tasks, device=device)
    else:
        rl_tok, vla_act, _, _, _, _, _, prop, _ = \
            replay_buffer.sample(batch_size, device=device)

    # Paper Eq. 5: a ~ π_θ (stochastic rsample for correct gradient)
    action, _ = actor(rl_tok, vla_act, prop, deterministic=False)

    # Q-value of the sampled action (only Q1 for efficiency, as in TD3)
    q_val = q_critic.q1_forward(rl_tok, action, prop)  # (B,)

    # BC regularization: ‖a - ã‖² (anchor to VLA reference)
    bc_penalty = ((action - vla_act) ** 2).sum(dim=(-2, -1)).mean()  # scalar

    # Paper Eq. 5: minimize -Q + β * BC
    actor_loss = -q_val.mean() + beta * bc_penalty

    stats = {
        "actor_loss": actor_loss.item(),
        "q_actor_mean": q_val.mean().item(),
        "bc_penalty": bc_penalty.item(),
    }
    return actor_loss, stats


def action_token_td_update(
    actor: ActionTokenActor,
    critic,  # ActionTokenQCritic or legacy ActionTokenCritic
    replay_buffer: ReplayBuffer,
    batch_size: int = 256,
    gamma: float = 0.99,
    device: str = "cuda",
    target_critic=None,
    beta: float = 1.0,
    update_actor: bool = True,
    target_noise_std: float = 0.2,
    target_noise_clip: float = 0.5,
):
    """
    Combined TD3-style update step (backward compat wrapper).

    If critic is ActionTokenQCritic: uses the new TD3/DDPG-style from paper.
    If critic is ActionTokenCritic (legacy V(s)): falls back to old logic.

    Args:
        beta: BC regularization coefficient (paper Eq. 5)
        update_actor: If False, only update critic (TD3 delayed actor update)
        target_noise_std: Std of noise added to target policy actions
        target_noise_clip: Clip range for target policy noise

    Returns:
        loss: scalar tensor
        stats: dict
    """
    if isinstance(critic, ActionTokenQCritic):
        # ── New TD3-style from paper ──
        # Critic update
        critic_loss, c_stats = action_token_td_critic_update(
            actor=actor,
            q_critic=critic,
            target_q_critic=target_critic,
            replay_buffer=replay_buffer,
            batch_size=batch_size,
            gamma=gamma,
            device=device,
            target_noise_std=target_noise_std,
            target_noise_clip=target_noise_clip,
        )

        if update_actor:
            # Actor update (DDPG + BC regularization)
            actor_loss, a_stats = action_token_td_actor_update(
                actor=actor,
                q_critic=critic,
                replay_buffer=replay_buffer,
                batch_size=batch_size,
                beta=beta,
                device=device,
            )
            loss = critic_loss + actor_loss
            stats = {**c_stats, **a_stats, "td_loss": loss.item()}
        else:
            loss = critic_loss
            stats = {**c_stats, "actor_loss": 0.0, "td_loss": loss.item()}

        return loss, stats

    else:
        # ── Legacy V(s) path (backward compat for PPO-based code) ──
        rl_tok, vla_act, act_taken, rew, next_rl_tok, next_vla_act, done, prop, next_prop = \
            replay_buffer.sample(batch_size, device=device)

        with torch.no_grad():
            value_net = target_critic if target_critic is not None else critic
            next_value = value_net(next_rl_tok)  # (B,)
            target = rew + gamma * next_value * (1.0 - done)

        value = critic(rl_tok)
        critic_loss = F.mse_loss(value, target)

        advantage = (target - value).detach()
        if advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        log_prob = actor.log_prob_of(
            rl_tok.unsqueeze(1) if rl_tok.dim() == 2 else rl_tok,
            vla_act,
            act_taken,
            prop if hasattr(actor, 'prop_dim') and actor.prop_dim > 0 else None,
        )
        actor_loss = -(advantage * log_prob).mean()

        loss = actor_loss + 0.5 * critic_loss

        stats = {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "td_loss": loss.item(),
            "value_mean": value.mean().item(),
            "advantage_mean": advantage.mean().item(),
            "target_mean": target.mean().item(),
        }
        return loss, stats
