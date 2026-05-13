"""Phase 2 (production): off-policy TD3 with split rollout/training GPUs.

Architecture (single process, no Accelerate):
  - Rollout GPUs: each loads a frozen VLA copy, collects episodes in parallel
  - Train GPU:    runs actor-critic TD updates (backward pass)
  - Replay buffer: centralized on CPU, fed by all rollout GPUs
  - Eval:         distributed across rollout GPUs for speed
"""
import copy
import json
import logging
import os
import queue
import random
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import wandb
from accelerate.utils import set_seed

from AlphaBrain.model.framework.base_framework import BaseFramework
from AlphaBrain.training.reinforcement_learning.common.ckpt_io import save_rlt_checkpoint
from AlphaBrain.training.reinforcement_learning.eval.eval_helpers import _eval_deterministic_local
from AlphaBrain.training.reinforcement_learning.eval.eval_helpers_rlt import (
    _eval_deterministic_local_rlt,
)
from AlphaBrain.training.reinforcement_learning.envs.libero_env import MAX_STEPS, get_suite_info
from AlphaBrain.training.reinforcement_learning.common.replay_buffer import ReplayBuffer
from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_actor_critic import (
    ActionTokenActor,
    ActionTokenCritic,
    ActionTokenQCritic,
    soft_update_target,
)
from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_encoder_decoder import ActionTokenEncoderDecoder
from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_trainer import push_episodes_to_buffer

logger = logging.getLogger(__name__)


def run_rl_offpolicy(args):
    """Phase 2 off-policy variant: split rollout/training GPUs.

    Lets you freely assign GPUs, e.g.:
      --rollout_gpus 0,1,2,3,4  --train_gpu 5
    Train GPU can overlap with rollout GPUs (rollout and training are sequential).
    """
    set_seed(args.seed)

    # ── Parse GPU config ─────────────────────────────────
    if args.rollout_gpus is not None:
        rollout_gpu_ids = [int(g.strip()) for g in args.rollout_gpus.split(",")]
    else:
        rollout_gpu_ids = list(range(torch.cuda.device_count()))

    train_gpu_id = args.train_gpu if args.train_gpu is not None else rollout_gpu_ids[0]
    train_device = f"cuda:{train_gpu_id}"
    n_rollout_gpus = len(rollout_gpu_ids)

    logger.info(f"=== Off-Policy TD Mode (Split GPU) ===")
    logger.info(f"  Rollout GPUs: {rollout_gpu_ids}")
    logger.info(f"  Train GPU: {train_gpu_id}")

    # ── Load frozen VLA on each rollout GPU ──────────────
    vla_copies = {}
    # bf16 for both Qwen and Pi05: halves weight memory (~12→6 GB per copy)
    # and matches the standalone eval setting (use_bf16=true), which scores
    # 91%+ SR on the same Pi05 ckpts.
    _vla_dtype = torch.bfloat16

    for i, gpu_id in enumerate(rollout_gpu_ids):
        device = f"cuda:{gpu_id}"
        logger.info(f"  Loading frozen VLA on GPU {gpu_id} ({i+1}/{n_rollout_gpus})...")
        vla = BaseFramework.from_pretrained(args.ckpt_path)
        vla = vla.to(_vla_dtype).to(device).eval()
        for p in vla.parameters():
            p.requires_grad_(False)
        vla_copies[gpu_id] = vla

    # Load VLA on train GPU (for eval, and optionally for fine-tuning)
    if args.finetune_vla:
        # Full fine-tune: need a SEPARATE trainable VLA on train GPU
        # (the rollout loop may have already loaded a frozen copy — replace it)
        logger.info(f"  Loading TRAINABLE VLA on train GPU {train_gpu_id} (full fine-tune)...")
        vla = BaseFramework.from_pretrained(args.ckpt_path)
        vla = vla.to(_vla_dtype).to(train_device).train()
        if hasattr(vla, "qwen_vl_interface") and hasattr(vla.qwen_vl_interface, "model"):
            vla.qwen_vl_interface.model.gradient_checkpointing_enable()
        vla_copies[train_gpu_id] = vla
        logger.info(f"  Train GPU VLA: TRAINABLE ({sum(p.numel() for p in vla.parameters()) / 1e9:.2f}B params, gradient_checkpointing)")
    elif train_gpu_id not in vla_copies:
        logger.info(f"  Loading frozen VLA on train GPU {train_gpu_id}...")
        vla = BaseFramework.from_pretrained(args.ckpt_path)
        vla = vla.to(_vla_dtype).to(train_device).eval()
        for p in vla.parameters():
            p.requires_grad_(False)
        vla_copies[train_gpu_id] = vla

    # Backbone-agnostic metadata (Qwen vs Pi05): hidden_dim, action_norm_stats,
    # chunk_len, action_dim. See pi05_inference.resolve_vla_metadata.
    from AlphaBrain.training.reinforcement_learning.algos.RLT.pi05_inference import (
        is_pi05, resolve_vla_metadata,
    )
    ref_vla = vla_copies[rollout_gpu_ids[0]]
    hidden_dim, action_norm_stats, chunk_len, action_dim = resolve_vla_metadata(ref_vla)
    if is_pi05(ref_vla):
        logger.info("Pi05 detected: using identity action_norm_stats (Pi05 returns env-space actions)")

    # Actor chunk length: paper uses C < H (e.g. VLA H=50, actor C=10)
    # For LIBERO: VLA chunk=8, actor chunk=4 (re-plan every 4 steps)
    actor_chunk_len = args.actor_chunk_len if args.actor_chunk_len else chunk_len
    logger.info(f"VLA chunk_len={chunk_len}, actor_chunk_len={actor_chunk_len}")

    suite_info = get_suite_info(args.suite)
    n_tasks = suite_info["n_tasks"]
    max_steps = MAX_STEPS[args.suite]

    # ── Handle --task_ids: subset of tasks treated like --all_tasks ──
    if args.task_ids is not None:
        selected_task_ids = [int(x) for x in args.task_ids.split(",")]
        logger.info(f"--task_ids={args.task_ids}: training on tasks {selected_task_ids}")
        args.all_tasks = True
        # Remap: override n_tasks and suite_info so all_tasks branch iterates
        # only over the selected tasks. We patch range(n_tasks) → selected_task_ids
        # by storing the list and overriding the iteration below.
        args._selected_task_ids = selected_task_ids
    else:
        args._selected_task_ids = None

    # ── Create trainable modules on train_gpu ────────────
    encoder_mode = getattr(args, "encoder_mode", "action_token")
    if encoder_mode == "rlt":
        # RL Token reference track: z_rl kept at VLA hidden dim (no extra
        # bottleneck projection). --bottleneck_dim here is repurposed as
        # the encoder's hidden_dim and should equal the VLA's hidden_size
        # (2048 for Qwen2.5-VL-3B).
        from AlphaBrain.training.reinforcement_learning.algos.RLT import (
            RLTokenEncoderDecoder,
        )
        if args.bottleneck_dim != hidden_dim:
            logger.warning(
                f"  --bottleneck_dim={args.bottleneck_dim} != VLA hidden_dim={hidden_dim}; "
                f"RLT encoder uses VLA hidden dim. Overriding bottleneck_dim."
            )
            args.bottleneck_dim = hidden_dim
        enc_dec = RLTokenEncoderDecoder(
            hidden_dim=hidden_dim,
            num_heads=args.encoder_heads,
            encoder_layers=args.encoder_layers,
            decoder_layers=getattr(args, "decoder_layers", args.encoder_layers),
            max_len=getattr(args, "max_len", 4096),
        ).to(train_device)
        # rlt + steplock: now supported via the encoder_mode kwarg on
        # action_token_collect_*_steplock (we pass it through below). The
        # rollout-fast loop dispatches to the rlt encoder path internally.
    else:
        enc_dec = ActionTokenEncoderDecoder(
            input_dim=hidden_dim,
            bottleneck_dim=args.bottleneck_dim,
            chunk_len=chunk_len,
            num_heads=args.encoder_heads,
            encoder_layers=args.encoder_layers,
            decoder_layers=args.encoder_layers,
        ).to(train_device)

    if args.encoder_path:
        logger.info(f"  Loading pretrained encoder from {args.encoder_path}")
        state = torch.load(args.encoder_path, map_location=train_device)
        enc_dec.load_state_dict(state)

    if args.finetune_vla:
        # Full fine-tune: encoder trainable (updated via VLA fine-tune step)
        enc_dec.train()
        logger.info("  Encoder: TRAINABLE (full fine-tune mode)")
    else:
        # Freeze encoder for off-policy (buffer rl_tokens must stay valid)
        enc_dec.eval()
        for p in enc_dec.parameters():
            p.requires_grad_(False)

    actor = ActionTokenActor(
        bottleneck_dim=args.bottleneck_dim,
        action_dim=action_dim,
        chunk_len=actor_chunk_len,
        hidden_dim=args.actor_hidden_dim,
        ref_dropout=args.ref_dropout,
        fixed_std=args.fixed_std,
        prop_dim=8,  # paper: x = (z_rl, s_p) where s_p = eef_pos(3)+axisangle(3)+gripper(2)
    ).to(train_device)

    # Paper: Q(s, a) twin-Q critic (TD3 style)
    q_critic = ActionTokenQCritic(
        bottleneck_dim=args.bottleneck_dim,
        action_dim=action_dim,
        chunk_len=actor_chunk_len,
        hidden_dim=args.critic_hidden_dim,
        prop_dim=8,  # paper: x = (z_rl, s_p)
    ).to(train_device)

    # Target critic (Polyak-averaged copy of twin-Q)
    target_q_critic = copy.deepcopy(q_critic).to(train_device)
    target_q_critic.eval()
    for p in target_q_critic.parameters():
        p.requires_grad_(False)

    # Target actor (Polyak-averaged copy — TD3 uses this for next_action in target Q)
    target_actor = copy.deepcopy(actor).to(train_device)
    target_actor.eval()
    for p in target_actor.parameters():
        p.requires_grad_(False)

    # Also keep a lightweight V(s) critic for rollout (value estimate logging)
    # The rollout only needs a dummy critic for the episode data structure
    dummy_critic = ActionTokenCritic(
        bottleneck_dim=args.bottleneck_dim,
        hidden_dim=64,  # tiny, just for rollout value logging
    ).to(train_device)

    # ── Rollout module copies (one per rollout GPU, tiny ~9M) ──
    rollout_modules = {}
    for gpu_id in rollout_gpu_ids:
        device = f"cuda:{gpu_id}"
        r_enc = copy.deepcopy(enc_dec).to(device).eval()
        r_actor = copy.deepcopy(actor).to(device).eval()
        r_critic = copy.deepcopy(dummy_critic).to(device).eval()
        rollout_modules[gpu_id] = (r_enc, r_actor, r_critic)

    # ── Eval module copies — separate from rollout to allow async eval ──
    # (rollout_modules are read by the rollout thread; eval_modules are read by
    #  the async eval thread. Keeping them separate avoids races.)
    eval_modules = {}
    for gpu_id in rollout_gpu_ids:
        device = f"cuda:{gpu_id}"
        e_enc = copy.deepcopy(enc_dec).to(device).eval()
        e_actor = copy.deepcopy(actor).to(device).eval()
        for p in e_enc.parameters(): p.requires_grad_(False)
        for p in e_actor.parameters(): p.requires_grad_(False)
        eval_modules[gpu_id] = (e_enc, e_actor)

    # ── Per-GPU rollout infrastructure ──
    if args.use_steplock:
        # Step-lock mode: persistent env pools (no BatchInferenceServer needed)
        from AlphaBrain.training.reinforcement_learning.envs.persistent_env_pool import PersistentEnvPool
        from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_rollout_fast import (
            action_token_collect_group_steplock,
            action_token_collect_multitask_steplock,
        )
        rollout_servers = {}  # not used in steplock mode
        rollout_env_pools = {}
        # Compute tasks per GPU for env pool sizing
        gpu_task_map = {}
        if args.all_tasks:
            task_list_all = args._selected_task_ids if args._selected_task_ids else list(range(n_tasks))
            for t_idx in task_list_all:
                gid = rollout_gpu_ids[t_idx % n_rollout_gpus]
                gpu_task_map.setdefault(gid, []).append(t_idx)
        max_tasks_per_gpu = max(len(v) for v in gpu_task_map.values()) if gpu_task_map else 1
        # Pool sizing: num_envs_per_task × tasks_on_this_gpu
        # Decouples parallelism from G_per_task. If G_per_task > num_envs_per_task,
        # the rollout chunks into ceil(G_per_task / num_envs_per_task) sequential passes.
        envs_per_gpu = args.num_envs_per_task * max_tasks_per_gpu
        # Map logical GPU ID → physical GPU ID for MuJoCo EGL rendering
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        physical_gpus = [int(x) for x in cuda_devices.split(",") if x.strip()] if cuda_devices else list(range(8))
        for gpu_id in rollout_gpu_ids:
            physical_gpu = physical_gpus[gpu_id] if gpu_id < len(physical_gpus) else gpu_id
            pool = PersistentEnvPool(
                num_envs=envs_per_gpu,
                libero_python=os.environ.get("LIBERO_PYTHON"),
                egl_gpu_id=physical_gpu,
            )
            rollout_env_pools[gpu_id] = pool
        n_passes_per_iter = max(1, (args.G_per_task + args.num_envs_per_task - 1) // args.num_envs_per_task)
        logger.info(f"  Step-lock mode: {len(rollout_gpu_ids)} GPU × {envs_per_gpu} persistent envs "
                     f"({max_tasks_per_gpu} tasks/GPU × {args.num_envs_per_task} envs/task, "
                     f"G_per_task={args.G_per_task} → {n_passes_per_iter} passes/iter)")

        # No pre-warm needed — parallel reset in rollout handles MuJoCo init.
    else:
        # Async mode: BatchInferenceServer per GPU
        from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_trainer import BatchInferenceServer
        rollout_servers = {}
        rollout_env_pools = {}  # not used in async mode
        for gpu_id in rollout_gpu_ids:
            r_enc, r_actor, r_critic = rollout_modules[gpu_id]
            server = BatchInferenceServer(
                frozen_vla=vla_copies[gpu_id],
                encoder=r_enc,
                actor=r_actor,
                critic=r_critic,
                device=f"cuda:{gpu_id}",
                max_batch_size=args.num_envs * 4,
                actor_chunk_len=actor_chunk_len if actor_chunk_len != chunk_len else None,
                encoder_mode=encoder_mode,
            ).start()
            rollout_servers[gpu_id] = server
        logger.info(f"  Started BatchInferenceServer on GPU {gpu_id} "
                    f"(max_batch={args.num_envs * 4})")

    enc_params = sum(p.numel() for p in enc_dec.parameters())
    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in q_critic.parameters())
    vla_params = sum(p.numel() for p in ref_vla.parameters())
    logger.info(f"  Frozen VLA: {vla_params / 1e9:.2f}B params × {n_rollout_gpus} GPUs")
    logger.info(f"  Frozen encoder: {enc_params / 1e6:.2f}M params")
    logger.info(f"  Trainable: actor={actor_params / 1e6:.2f}M, critic={critic_params / 1e6:.2f}M")
    logger.info(f"  Replay buffer: capacity={args.buffer_capacity}, warmup={args.buffer_warmup}")
    logger.info(f"  TD updates: {args.td_updates_per_iter}/iter, batch={args.td_batch_size}, tau={args.tau}")
    logger.info(f"  Rollout: {n_rollout_gpus} GPUs × {args.num_envs} envs × "
                f"{args.G} episodes/GPU = {n_rollout_gpus * args.G} episodes/iter")

    # ── Separate optimizers for actor and critic (TD3 pattern) ──
    optimizer_critic = torch.optim.AdamW(
        q_critic.parameters(), lr=args.lr_critic,
        betas=(0.9, 0.95), weight_decay=1e-8)
    optimizer_actor = torch.optim.AdamW(
        actor.parameters(), lr=args.lr_actor,
        betas=(0.9, 0.95), weight_decay=1e-8)

    # ── VLA fine-tune optimizer (only when --finetune_vla) ──
    optimizer_vla = None
    if args.finetune_vla:
        train_vla = vla_copies[train_gpu_id]
        vla_params = [p for p in train_vla.parameters() if p.requires_grad]
        enc_params = [p for p in enc_dec.parameters() if p.requires_grad]
        optimizer_vla = torch.optim.AdamW(
            [{"params": vla_params, "lr": args.lr_vla},
             {"params": enc_params, "lr": args.lr_vla * 2}],
            betas=(0.9, 0.95), weight_decay=1e-8)
        n_vla_trainable = sum(p.numel() for p in vla_params)
        n_enc_trainable = sum(p.numel() for p in enc_params)
        logger.info(f"  VLA optimizer: {n_vla_trainable / 1e9:.2f}B VLA + "
                     f"{n_enc_trainable / 1e6:.2f}M encoder, lr={args.lr_vla}")

    # ── Replay buffer (centralized) ──────────────────────
    replay_buffer = ReplayBuffer(capacity=args.buffer_capacity)

    # ── WandB ─────────────────────────────────────────────
    if args.use_wandb:
        run_name = args.run_name or f"action_token_offpolicy_{args.suite}_task{args.task_id}"
        wandb.init(project=args.wandb_project, name=run_name,
                   config={**vars(args), "chunk_len": chunk_len,
                           "hidden_dim": hidden_dim, "action_dim": action_dim,
                           "n_rollout_gpus": n_rollout_gpus, "mode": "offpolicy_td"})

    video_dir = Path(args.output_dir) / "videos"
    metrics_history = []
    best_sr = 0.0
    best_eval_sr = 0.0
    running_sr = []
    total_env_steps = 0  # cumulative environment steps (sample steps)

    # ── Async rollout helper ────────────────────────────
    # Rollout runs on background threads (rollout GPUs) while TD updates
    # run on the main thread (train GPU). This matches the PI paper's
    # asynchronous rollout + learning design.

    buffer_lock = threading.Lock()
    # Bounded queue: rollout blocks on `put()` when main falls behind instead
    # of racing ahead indefinitely. Decouples rollout lifetime from
    # args.max_iter (see _rollout_thread_fn below).
    rollout_stats_queue = queue.Queue(maxsize=8)   # (episodes, iteration)
    _stop_rollout = threading.Event()
    _weight_sync_lock = threading.Lock()  # protects weight copy (non-blocking)

    def _sync_rollout_weights():
        """Copy latest actor/encoder weights to all rollout GPU copies (non-blocking)."""
        with _weight_sync_lock:
            enc_state_cpu = {k: v.cpu() for k, v in enc_dec.state_dict().items()}
            actor_state_cpu = {k: v.cpu() for k, v in actor.state_dict().items()}
            dummy_critic_state_cpu = {k: v.cpu() for k, v in dummy_critic.state_dict().items()}
            for gpu_id in rollout_gpu_ids:
                r_enc, r_actor, r_critic = rollout_modules[gpu_id]
                dev = f"cuda:{gpu_id}"
                r_enc.load_state_dict({k: v.to(dev) for k, v in enc_state_cpu.items()})
                r_actor.load_state_dict({k: v.to(dev) for k, v in actor_state_cpu.items()})
                r_critic.load_state_dict({k: v.to(dev) for k, v in dummy_critic_state_cpu.items()})
        return enc_state_cpu, actor_state_cpu

    _steplock_warmup = [True]  # shared flag: rollout thread reads, main thread sets False
    _rollout_go = threading.Event()  # clear = paused, set = running
    _rollout_go.set()  # start unpaused

    # ── Async eval state ──
    # Eval runs in a background thread so train + rollout don't block.
    # Results arrive in _eval_results_queue, drained each main iteration.
    _eval_results_queue = queue.Queue()
    _eval_thread_holder = [None]  # mutable holder so closures can update
    _eval_lock = threading.Lock()  # ensures only one eval at a time

    def _sync_eval_weights():
        """Copy latest train weights to eval modules (fast, ~10ms)."""
        with _weight_sync_lock:
            enc_state_cpu = {k: v.cpu() for k, v in enc_dec.state_dict().items()}
            actor_state_cpu = {k: v.cpu() for k, v in actor.state_dict().items()}
            for gpu_id in rollout_gpu_ids:
                e_enc, e_actor = eval_modules[gpu_id]
                dev = f"cuda:{gpu_id}"
                e_enc.load_state_dict({k: v.to(dev) for k, v in enc_state_cpu.items()})
                e_actor.load_state_dict({k: v.to(dev) for k, v in actor_state_cpu.items()})

    def _run_eval_inline(iteration, save_video):
        """Synchronous eval body — same logic as before but uses eval_modules.

        Returns dict with eval_sr, eval_result, per_task_eval_sr.
        Does NOT mutate outer state (best_eval_sr is updated in main thread).
        """
        per_task_eval_sr_local = {}
        eval_result_local = None
        # Route eval through the encoder-mode-matching helper. The default
        # helper's encoder.encode(action_queries) path is wrong for rlt
        # encoders (trained on compacted image hidden states) and silently
        # gives low SR even when rollout SR is high.
        _eval_fn = (_eval_deterministic_local_rlt
                    if encoder_mode == "rlt"
                    else _eval_deterministic_local)

        if args.all_tasks:
            # Multi-task eval
            eval_task_list = args._selected_task_ids if args._selected_task_ids else list(range(n_tasks))
            n_eval_tasks = len(eval_task_list)
            eval_n_per_task = max(1, args.eval_n_episodes // n_eval_tasks)
            total_eval_eps = eval_n_per_task * n_eval_tasks
            logger.info(f"[ASYNC EVAL @ iter {iteration}] Multi-task: {eval_n_per_task} eps/task × "
                         f"{n_eval_tasks} tasks = {total_eval_eps} episodes")

            eval_gpu_jobs = {gpu_id: [] for gpu_id in rollout_gpu_ids}
            job_idx = 0
            for task_id_eval in eval_task_list:
                for ep_idx in range(eval_n_per_task):
                    gpu_id = rollout_gpu_ids[job_idx % n_rollout_gpus]
                    eval_gpu_jobs[gpu_id].append((task_id_eval, ep_idx))
                    job_idx += 1

            eval_video_dir = (str(video_dir / f"eval_iter_{iteration:05d}") if save_video else None)

            all_eval_results = []
            with ThreadPoolExecutor(max_workers=n_rollout_gpus * 2) as pool:
                futures = {}
                for gpu_id, jobs in eval_gpu_jobs.items():
                    if not jobs:
                        continue
                    task_groups = defaultdict(list)
                    for tid, eidx in jobs:
                        task_groups[tid].append(eidx)
                    for tid, ep_indices in task_groups.items():
                        e_enc, e_actor = eval_modules[gpu_id]
                        task_vid_dir = (os.path.join(eval_video_dir, f"task_{tid}") if eval_video_dir else None)
                        fut = pool.submit(
                            _eval_fn,
                            frozen_vla=vla_copies[gpu_id],
                            encoder=e_enc,
                            actor=e_actor,
                            suite_name=args.suite,
                            task_id=tid,
                            action_norm_stats=action_norm_stats,
                            max_steps=max_steps,
                            chunk_len=actor_chunk_len,
                            episode_indices=ep_indices,
                            num_steps_wait=args.num_steps_wait,
                            seed=42,
                            device=f"cuda:{gpu_id}",
                            rank=gpu_id,
                            video_dir=task_vid_dir,
                        )
                        futures[fut] = (gpu_id, tid)
                for fut in as_completed(futures):
                    gpu_id, tid = futures[fut]
                    results = fut.result()
                    for ep_idx, state_idx, success in results:
                        all_eval_results.append((tid, ep_idx, state_idx, success))

            task_successes_map = defaultdict(list)
            for tid, _, _, success in all_eval_results:
                task_successes_map[tid].append(success)
            for tid in sorted(task_successes_map.keys()):
                v = task_successes_map[tid]
                task_sr = float(np.mean(v))
                per_task_eval_sr_local[tid] = task_sr
                logger.info(f"  [async eval] task {tid} ({suite_info['task_names'][tid][:40]}): "
                             f"SR={task_sr:.2%} ({sum(v)}/{len(v)})")

            all_success = [s for _, _, _, s in all_eval_results]
            eval_sr_local = float(np.mean(all_success)) if all_success else 0.0
            eval_result_local = {
                "eval_sr": eval_sr_local,
                "per_task": per_task_eval_sr_local,
                "n_episodes": len(all_success),
            }
        else:
            # Single-task eval
            task_id_eval = args.task_id if args.task_id >= 0 else 0
            n_eval = args.eval_n_episodes
            logger.info(f"[ASYNC EVAL @ iter {iteration}] Single-task: {n_eval} episodes")

            eval_video_dir = str(video_dir / f"eval_iter_{iteration:05d}") if save_video else None
            eval_assignments = {gpu_id: [] for gpu_id in rollout_gpu_ids}
            for ep_idx in range(n_eval):
                gpu_id = rollout_gpu_ids[ep_idx % n_rollout_gpus]
                eval_assignments[gpu_id].append(ep_idx)

            all_eval_results = []
            with ThreadPoolExecutor(max_workers=n_rollout_gpus) as pool:
                futures = {}
                for gpu_id, ep_indices in eval_assignments.items():
                    if not ep_indices:
                        continue
                    e_enc, e_actor = eval_modules[gpu_id]
                    fut = pool.submit(
                        _eval_fn,
                        frozen_vla=vla_copies[gpu_id],
                        encoder=e_enc,
                        actor=e_actor,
                        suite_name=args.suite,
                        task_id=task_id_eval,
                        action_norm_stats=action_norm_stats,
                        max_steps=max_steps,
                        chunk_len=actor_chunk_len,
                        episode_indices=ep_indices,
                        num_steps_wait=args.num_steps_wait,
                        seed=42,
                        device=f"cuda:{gpu_id}",
                        rank=gpu_id,
                        video_dir=eval_video_dir,
                    )
                    futures[fut] = gpu_id
                for fut in as_completed(futures):
                    all_eval_results.extend(fut.result())

            per_state = defaultdict(list)
            all_success = []
            for ep_idx, state_idx, success in all_eval_results:
                per_state[state_idx].append(success)
                all_success.append(success)

            eval_sr_local = float(np.mean(all_success)) if all_success else 0.0
            per_state_sr = {sid: float(np.mean(v)) for sid, v in sorted(per_state.items())}
            eval_result_local = {
                "eval_sr": eval_sr_local,
                "per_state": per_state_sr,
                "n_episodes": len(all_success),
            }

        return {
            "iteration": iteration,
            "eval_sr": eval_sr_local,
            "eval_result": eval_result_local,
            "per_task_eval_sr": per_task_eval_sr_local,
        }

    def _async_eval_fn(iteration, save_video):
        """Background thread target: run eval, push result to queue."""
        try:
            result = _run_eval_inline(iteration, save_video)
            _eval_results_queue.put(result)
            logger.info(f"[ASYNC EVAL @ iter {iteration}] done, SR={result['eval_sr']:.2%}")
        except Exception:
            logger.exception(f"[ASYNC EVAL @ iter {iteration}] crashed")
        finally:
            _eval_thread_holder[0] = None

    def _rollout_thread_fn(start_iter):
        """Background thread: continuously collects episodes and pushes to buffer.

        Runs until `_stop_rollout` is set — which happens only after the main
        loop has consumed its `args.max_iter` training tuples. Previously this
        was a `for it in range(..., max_iter + 1)` loop, which made rollout
        exit at iter==max_iter even when the slower main loop (async mode,
        large td_updates_per_iter × utd_ratio) hadn't yet drained the queue.
        If rollout then hit any late-iteration exception — e.g. a transient
        cuda error while contending with TD updates on a shared GPU — the
        poison-pill path broke the main loop early. Decoupling rollout's
        lifetime from max_iter removes that coupling: rollout keeps producing
        (subject to the bounded queue's backpressure) as long as training
        still wants data.
        """
        it = start_iter
        try:
          while not _stop_rollout.is_set():
            # Wait if paused (during eval)
            _rollout_go.wait()  # blocks until set
            if _stop_rollout.is_set():
                break

            # Build task list
            if args.all_tasks:
                task_list_all = args._selected_task_ids if args._selected_task_ids else list(range(n_tasks))
                gpu_task_assignments = {gpu_id: [] for gpu_id in rollout_gpu_ids}
                for t_idx in task_list_all:
                    gpu_id = rollout_gpu_ids[t_idx % n_rollout_gpus]
                    gpu_task_assignments[gpu_id].append(t_idx)
            else:
                task_id = args.task_id if args.task_id >= 0 else random.randint(0, n_tasks - 1)
                gpu_task_assignments = {gpu_id: [task_id] for gpu_id in rollout_gpu_ids}

            if args.use_steplock:
                # Step-lock: use plain threads (no nested ThreadPoolExecutor).
                # One thread per GPU, each runs action_token_collect_multitask_steplock.
                #
                # Auto-chunk: if G > num_envs, run ceil(G/num_envs) sequential passes per iter.
                # Each pass uses different seeds → different states/noise sampled.
                # This decouples parallelism (num_envs) from total ep/iter (G).
                n_passes = max(1, (args.G_per_task + args.num_envs_per_task - 1) // args.num_envs_per_task)
                G_per_pass = min(args.G_per_task, args.num_envs_per_task)

                all_eps = []
                per_task_sr = {}
                for pass_idx in range(n_passes):
                    gpu_results = {}
                    gpu_threads = []

                    def _run_gpu(gpu_id, task_list, pass_idx=pass_idx):
                        r_enc, r_actor, r_critic = rollout_modules[gpu_id]
                        group_seed = args.seed + it * 1000 + gpu_id * 100 + pass_idx * 50000
                        unique_group_idx = (it * n_passes + pass_idx) * n_rollout_gpus + gpu_id
                        if len(task_list) > 1:
                            eps = action_token_collect_multitask_steplock(
                                env_pool=rollout_env_pools[gpu_id],
                                frozen_vla=vla_copies[gpu_id],
                                encoder=r_enc, actor=r_actor, critic=r_critic,
                                suite_name=args.suite, task_ids=task_list,
                                n_initial_states=50, action_norm_stats=action_norm_stats,
                                max_steps=max_steps, chunk_len=chunk_len,
                                G_per_task=G_per_pass, seed=group_seed,
                                num_steps_wait=args.num_steps_wait,
                                device=f"cuda:{gpu_id}",
                                group_idx=unique_group_idx,
                                store_images=args.finetune_vla,
                                group_size=args.group_size, reward_coef=args.reward_coef,
                                actor_chunk_len=actor_chunk_len if actor_chunk_len != chunk_len else None,
                                warmup_mode=_steplock_warmup[0],
                                encoder_mode=encoder_mode,
                            )
                        else:
                            eps = action_token_collect_group_steplock(
                                env_pool=rollout_env_pools[gpu_id],
                                frozen_vla=vla_copies[gpu_id],
                                encoder=r_enc, actor=r_actor, critic=r_critic,
                                suite_name=args.suite, task_id=task_list[0],
                                n_initial_states=50, action_norm_stats=action_norm_stats,
                                max_steps=max_steps, chunk_len=chunk_len, G=G_per_pass,
                                seed=group_seed, num_steps_wait=args.num_steps_wait,
                                device=f"cuda:{gpu_id}",
                                group_idx=unique_group_idx,
                                store_images=args.finetune_vla,
                                group_size=args.group_size, reward_coef=args.reward_coef,
                                actor_chunk_len=actor_chunk_len if actor_chunk_len != chunk_len else None,
                                warmup_mode=_steplock_warmup[0],
                                encoder_mode=encoder_mode,
                            )
                        gpu_results[gpu_id] = (task_list, eps)

                    for gpu_id, task_list in gpu_task_assignments.items():
                        t = threading.Thread(target=_run_gpu, args=(gpu_id, task_list))
                        t.start()
                        gpu_threads.append(t)
                    for t in gpu_threads:
                        t.join()

                    for gpu_id, (tid_list, eps) in gpu_results.items():
                        all_eps.extend(eps)
                        for ep in eps:
                            per_task_sr.setdefault(ep.task_id, []).append(ep.success)
                        n_s = sum(1 for e in eps if e.success)
                        pass_str = f" pass {pass_idx+1}/{n_passes}" if n_passes > 1 else ""
                        logger.info(f"  [rollout iter {it}{pass_str}] GPU {gpu_id} tasks {tid_list}: "
                                    f"{len(eps)} eps, {n_s} success")
            else:
                # Async mode: use ThreadPoolExecutor
                from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_trainer import action_token_collect_group
                all_eps = []
                per_task_sr = {}
                futs = {}
                with ThreadPoolExecutor(max_workers=n_rollout_gpus * 2) as rollout_pool:
                    for gpu_id, task_list in gpu_task_assignments.items():
                        r_enc, r_actor, r_critic = rollout_modules[gpu_id]
                        for tid in task_list:
                            group_seed = args.seed + it * 1000 + gpu_id * 100 + tid * 10
                            fut = rollout_pool.submit(
                                action_token_collect_group,
                                frozen_vla=vla_copies[gpu_id],
                                encoder=r_enc, actor=r_actor, critic=r_critic,
                                suite_name=args.suite, task_id=tid,
                                n_initial_states=50, action_norm_stats=action_norm_stats,
                                max_steps=max_steps, chunk_len=actor_chunk_len, G=args.G,
                                libero_python=os.environ.get("LIBERO_PYTHON"),
                                seed=group_seed, num_steps_wait=args.num_steps_wait,
                                device=f"cuda:{gpu_id}",
                                num_envs=args.num_envs,
                                group_idx=it * n_tasks * n_rollout_gpus + gpu_id * n_tasks + tid,
                                batch_server=rollout_servers.get(gpu_id),
                                store_images=args.finetune_vla,
                                group_size=args.group_size, reward_coef=args.reward_coef,
                            )
                            futs[fut] = (gpu_id, tid)
                    for fut in as_completed(futs):
                        gpu_id, tid = futs[fut]
                        eps = fut.result()
                        all_eps.extend(eps)
                        n_s = sum(1 for e in eps if e.success)
                        per_task_sr.setdefault(tid, []).extend([e.success for e in eps])
                        logger.info(f"  [rollout iter {it}] GPU {gpu_id} task {tid}: "
                                    f"{len(eps)} eps, {n_s} success")

            if args.all_tasks:
                task_sr_str = " | ".join(
                    f"t{tid}={np.mean(v):.0%}" for tid, v in sorted(per_task_sr.items()))
                logger.info(f"  [rollout iter {it}] Per-task SR: {task_sr_str}")

            with buffer_lock:
                n_pushed = push_episodes_to_buffer(
                    all_eps, replay_buffer, gamma_per_step=args.gamma)

            # Backpressure: block on put but poll _stop_rollout so shutdown
            # doesn't deadlock here when the main loop has already exited and
            # stopped consuming.
            while not _stop_rollout.is_set():
                try:
                    rollout_stats_queue.put((all_eps, it, n_pushed), timeout=1.0)
                    break
                except queue.Full:
                    continue
            it += 1
        except Exception as e:
            import traceback
            logger.error(f"!!! Rollout thread CRASHED at iter {it}: {e}")
            logger.error(traceback.format_exc())
            rollout_stats_queue.put(None)

    # ── VLA Warmup (paper Sec. V): pre-fill buffer with pure VLA rollouts ──
    if args.warmup_iters > 0:
        logger.info(f"=== VLA Warmup: {args.warmup_iters} iters of pure VLA rollout ===")
        if not args.use_steplock:
            for gpu_id, server in rollout_servers.items():
                server.warmup_mode = True

    # ── Training loop (async rollout + TD updates) ────
    # Launch rollout in background
    rollout_thread = threading.Thread(
        target=_rollout_thread_fn, args=(1,), daemon=True)
    rollout_thread.start()
    logger.info("Started async rollout thread")

    td_global_step = 0
    last_sync_step = 0
    last_completed_iter = 0  # last training iter whose body fully ran; drives final-ckpt name
    sync_every_n_updates = 500  # sync weights to rollout every N TD3 updates

    from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_trainer import (
        action_token_td_actor_update,
        action_token_td_critic_update,
    )

    for iteration in range(1, args.max_iter + 1):
        # ── Drain all available rollout data (non-blocking after first) ────
        all_episodes = []
        # Block on first get (wait for rollout to produce data)
        result = rollout_stats_queue.get()
        if result is None:
            logger.error("Rollout thread crashed (poison pill). Stopping.")
            break
        eps_batch, rollout_iter, n_pushed = result
        all_episodes = list(eps_batch)

        rewards = np.array([ep.reward for ep in all_episodes])
        success_rate = float(np.mean(rewards > 0.5)) if len(rewards) > 0 else 0.0
        iter_env_steps = sum(ep.env_steps for ep in all_episodes)
        total_env_steps += iter_env_steps
        running_sr.append(success_rate)
        if len(running_sr) > 20:
            running_sr.pop(0)
        running_sr_avg = np.mean(running_sr)
        best_sr = max(best_sr, success_rate)

        per_task_rollout_sr = {}
        if args.all_tasks:
            _task_successes = defaultdict(list)
            for ep in all_episodes:
                _task_successes[ep.task_id].append(ep.success)
            per_task_rollout_sr = {tid: float(np.mean(v))
                                   for tid, v in sorted(_task_successes.items())}
            task_sr_str = " | ".join(f"t{tid}={sr:.0%}"
                                     for tid, sr in per_task_rollout_sr.items())
        else:
            task_sr_str = ""

        logger.info(f"{'='*60}")
        logger.info(f"[iter {iteration}/{args.max_iter}] Got {len(all_episodes)} episodes "
                     f"(rollout batch {rollout_iter}) | SR={success_rate:.2f} "
                     f"(best={best_sr:.2f}, avg={running_sr_avg:.2f}) "
                     f"| buffer={len(replay_buffer)}/{args.buffer_capacity} "
                     f"| total_env_steps={total_env_steps} | td_steps={td_global_step}")
        if task_sr_str:
            logger.info(f"  Per-task rollout SR: {task_sr_str}")

        # ── VLA warmup phase ──
        td_stats_list = []  # empty during warmup; filled during TD3
        if iteration <= args.warmup_iters:
            logger.info(f"[iter {iteration}] VLA warmup ({iteration}/{args.warmup_iters}), "
                         f"buffer={len(replay_buffer)} — skipping TD updates")
            # Don't `continue` — fall through to logging + wandb so metrics are tracked
        elif iteration == args.warmup_iters + 1:
            if args.use_steplock:
                _steplock_warmup[0] = False
            else:
                for gpu_id, server in rollout_servers.items():
                    server.warmup_mode = False
            _sync_rollout_weights()
            logger.info(f"=== VLA warmup done. Buffer pre-filled with {len(replay_buffer)} "
                         f"transitions. Starting TD3 training. ===")

        # ── Async TD3 updates: run UTD×new_data updates per new data batch ──
        # Paper Algorithm 1: TD updates run EVERY step (including warmup),
        # warmup only controls which action is used for rollout (VLA vs actor).
        if replay_buffer.is_ready(min_size=args.buffer_warmup):
            actor.train()
            q_critic.train()

            n_tasks_for_balance = len(args._selected_task_ids) if args._selected_task_ids else (n_tasks if args.all_tasks else 0)
            batch_sz = min(args.td_batch_size, len(replay_buffer))

            # UTD-based: n_updates = new_transitions × utd_ratio / batch_size
            n_new_transitions = n_pushed
            n_updates = max(1, int(n_new_transitions * args.utd_ratio / batch_sz))
            n_updates = min(n_updates, args.td_updates_per_iter)  # cap

            td_stats_list = []
            for td_step in range(n_updates):
                optimizer_critic.zero_grad()
                critic_loss, c_stats = action_token_td_critic_update(
                    actor=actor,
                    q_critic=q_critic,
                    target_q_critic=target_q_critic,
                    replay_buffer=replay_buffer,
                    batch_size=batch_sz,
                    gamma=args.gamma ** actor_chunk_len,
                    device=train_device,
                    target_noise_std=args.target_noise_std,
                    target_noise_clip=args.target_noise_clip,
                    n_tasks=n_tasks_for_balance,
                    target_actor=target_actor,
                )
                critic_loss.backward()
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(q_critic.parameters(), args.max_grad_norm)
                optimizer_critic.step()

                a_stats = {"actor_loss": 0.0, "q_actor_mean": 0.0, "bc_penalty": 0.0}
                if (td_step + 1) % args.actor_update_freq == 0:
                    optimizer_actor.zero_grad()
                    actor_loss, a_stats = action_token_td_actor_update(
                        actor=actor,
                        q_critic=q_critic,
                        replay_buffer=replay_buffer,
                        batch_size=batch_sz,
                        beta=args.beta,
                        device=train_device,
                        n_tasks=n_tasks_for_balance,
                    )
                    actor_loss.backward()
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
                    optimizer_actor.step()
                    soft_update_target(q_critic, target_q_critic, tau=args.tau)
                    soft_update_target(actor, target_actor, tau=args.tau)

                td_stats_list.append({**c_stats, **a_stats,
                                      "td_loss": c_stats["critic_loss"] + a_stats["actor_loss"]})
                td_global_step += 1

            avg_td = np.mean([s["td_loss"] for s in td_stats_list])
            avg_critic = np.mean([s["critic_loss"] for s in td_stats_list])
            avg_actor = np.mean([s["actor_loss"] for s in td_stats_list])
            avg_bc = np.mean([s.get("bc_penalty", 0.0) for s in td_stats_list])
            avg_q = np.mean([s.get("q1_mean", 0.0) for s in td_stats_list])
            logger.info(f"[iter {iteration}] TD3: {n_updates} updates (UTD={n_new_transitions}×{args.utd_ratio}/{batch_sz}→{n_updates}) "
                         f"critic={avg_critic:.4f} actor={avg_actor:.4f} "
                         f"bc={avg_bc:.4f} q_mean={avg_q:.4f}")

            # Sync weights to rollout periodically
            if td_global_step - last_sync_step >= sync_every_n_updates:
                _sync_rollout_weights()
                last_sync_step = td_global_step
                logger.info(f"  [sync] Weights synced to rollout (td_step={td_global_step})")

            # ── VLA fine-tune step ──
            if (args.finetune_vla and optimizer_vla is not None
                    and iteration % args.vla_update_freq == 0):
                from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_trainer import vla_finetune_step
                train_vla = vla_copies[train_gpu_id]
                train_vla.train()
                optimizer_vla.zero_grad()
                try:
                    vla_stats = vla_finetune_step(
                        vla=train_vla, encoder=enc_dec, actor=actor,
                        q_critic=q_critic, episodes=all_episodes,
                        beta=args.beta, device=train_device,
                        micro_batch=args.vla_micro_batch)
                    if args.max_grad_norm > 0:
                        all_vla_params = list(train_vla.parameters()) + list(enc_dec.parameters())
                        torch.nn.utils.clip_grad_norm_(all_vla_params, args.max_grad_norm)
                    optimizer_vla.step()
                    logger.info(f"[iter {iteration}] VLA fine-tune: loss={vla_stats.get('vla_loss', 0):.4f}")
                except torch.cuda.OutOfMemoryError:
                    logger.warning(f"[iter {iteration}] VLA fine-tune OOM — skipping")
                    optimizer_vla.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                finally:
                    train_vla.eval()
                    for ep in all_episodes:
                        for sr in ep.step_records:
                            sr.images = None; sr.instruction = None
                # Sync VLA weights
                train_vla = vla_copies[train_gpu_id]
                vla_state_cpu = {k: v.cpu() for k, v in train_vla.state_dict().items()}
                for gpu_id in rollout_gpu_ids:
                    if gpu_id != train_gpu_id:
                        vla_copies[gpu_id].load_state_dict(
                            {k: v.to(f"cuda:{gpu_id}") for k, v in vla_state_cpu.items()})
        else:
            logger.info(f"[iter {iteration}] Buffer warmup: {len(replay_buffer)}/{args.buffer_warmup} "
                         f"(waiting for more data)")

        # ── Async eval ──
        # Eval runs in a background thread; rollout + train do not block.
        # Results arrive in _eval_results_queue and are drained below.
        eval_sr = None
        eval_result = None
        per_task_eval_sr = {}
        do_eval = (args.eval_interval > 0
                   and (iteration == 1 or iteration % args.eval_interval == 0))
        if do_eval:
            save_video = (args.save_video_interval > 0 and
                          (iteration == 1 or iteration % args.save_video_interval == 0))
            with _eval_lock:
                prev = _eval_thread_holder[0]
                if prev is None or not prev.is_alive():
                    # Sync latest train weights to eval modules (~10ms)
                    _sync_eval_weights()
                    # Spawn async eval thread (non-blocking)
                    t = threading.Thread(
                        target=_async_eval_fn,
                        args=(iteration, save_video),
                        daemon=True,
                        name=f"async_eval_{iteration}",
                    )
                    _eval_thread_holder[0] = t
                    t.start()
                    logger.info(f"[iter {iteration}] Spawned async eval (rollout/train continue)")
                else:
                    logger.warning(f"[iter {iteration}] Skip eval — previous async eval still running")

        # ── Drain async eval results (non-blocking, every iter) ──
        while True:
            try:
                eval_data = _eval_results_queue.get_nowait()
            except queue.Empty:
                break
            from_iter = eval_data["iteration"]
            eval_sr = eval_data["eval_sr"]
            eval_result = eval_data["eval_result"]
            per_task_eval_sr = eval_data["per_task_eval_sr"]
            if eval_sr > best_eval_sr:
                best_eval_sr = eval_sr
            logger.info(f"[ASYNC RESULT] from iter {from_iter}: "
                         f"SR={eval_sr:.2%} (best_eval={best_eval_sr:.2%})")
            if eval_result and "per_state" in eval_result:
                for sid, sr in eval_result["per_state"].items():
                    logger.info(f"    state {sid}: {sr:.2%}")

        # ── Logging ───────────────────────────────────
        try:
            log_entry = {
                "iter": iteration,
                "total_env_steps": total_env_steps,
                "iter_env_steps": iter_env_steps,
                "success_rate": success_rate,
                "best_success_rate": best_sr,
                "running_avg_sr": running_sr_avg,
                "mean_reward": float(np.mean(rewards)) if len(rewards) > 0 else 0.0,
                "buffer_size": len(replay_buffer),
                "n_pushed": n_pushed,
            }
            if td_stats_list:
                avg_fn = lambda k: float(np.mean([s[k] for s in td_stats_list if k in s]))
                log_entry.update({
                    "td_loss": avg_fn("td_loss"),
                    "actor_loss": avg_fn("actor_loss"),
                    "critic_loss": avg_fn("critic_loss"),
                    "q1_mean": avg_fn("q1_mean"),
                    "q2_mean": avg_fn("q2_mean"),
                    "target_mean": avg_fn("target_mean"),
                    "bc_penalty": avg_fn("bc_penalty"),
                    "q_actor_mean": avg_fn("q_actor_mean"),
                })
            if eval_sr is not None:
                log_entry["eval_sr"] = eval_sr
                log_entry["best_eval_sr"] = best_eval_sr
            if per_task_rollout_sr:
                log_entry["per_task_rollout_sr"] = per_task_rollout_sr
            if per_task_eval_sr:
                log_entry["per_task_eval_sr"] = per_task_eval_sr
            metrics_history.append(log_entry)

            if args.use_wandb:
                wandb_log = {
                    "rollout/success_rate": success_rate,
                    "rollout/best_success_rate": best_sr,
                    "rollout/running_avg_sr": running_sr_avg,
                    "rollout/mean_reward": log_entry["mean_reward"],
                    "rollout/total_env_steps": total_env_steps,
                    "rollout/iter_env_steps": iter_env_steps,
                    "buffer/size": len(replay_buffer),
                    "buffer/pushed": n_pushed,
                }
                # Per-task rollout SR
                for tid, sr in per_task_rollout_sr.items():
                    wandb_log[f"rollout/task_{tid:02d}_sr"] = sr
                if td_stats_list:
                    wandb_log.update({
                        "train/td_loss": log_entry["td_loss"],
                        "train/actor_loss": log_entry["actor_loss"],
                        "train/critic_loss": log_entry["critic_loss"],
                        "train/q1_mean": log_entry["q1_mean"],
                        "train/q2_mean": log_entry["q2_mean"],
                        "train/target_mean": log_entry["target_mean"],
                        "train/bc_penalty": log_entry["bc_penalty"],
                        "train/q_actor_mean": log_entry["q_actor_mean"],
                        "train/actor_lr": optimizer_actor.param_groups[0]["lr"],
                        "train/n_updates": n_updates if td_stats_list else 0,
                    })
                if eval_sr is not None:
                    wandb_log["eval/success_rate"] = eval_sr
                    wandb_log["eval/best_success_rate"] = best_eval_sr
                    # Per-task eval SR
                    for tid, sr in per_task_eval_sr.items():
                        wandb_log[f"eval/task_{tid:02d}_sr"] = sr
                    if eval_result and "per_state" in eval_result:
                        for sid, sr in eval_result["per_state"].items():
                            wandb_log[f"eval/state_{sid:02d}"] = sr
                for ep in sorted(all_episodes, key=lambda e: -e.success):
                    if ep.video_path and os.path.exists(ep.video_path):
                        status = "success" if ep.success else "fail"
                        wandb_log[f"video/{status}"] = wandb.Video(
                            ep.video_path, fps=10, format="mp4")
                        break
                wandb.log(wandb_log, step=iteration)
                logger.info(f"[iter {iteration}] wandb.log OK (step={iteration})")
        except Exception as _log_err:
            logger.error(f"[iter {iteration}] LOGGING BLOCK EXCEPTION: {_log_err}")
            import traceback; traceback.print_exc()

        # ── 7. Checkpoint ────────────────────────────────
        if iteration % args.save_interval == 0:
            save_rlt_checkpoint(enc_dec, actor, q_critic,
                                iteration, args.output_dir, phase="rl_offpolicy")

        last_completed_iter = iteration

    # Stop rollout thread
    _stop_rollout.set()
    rollout_thread.join(timeout=10)
    logger.info("Rollout thread stopped")

    # Wait for any pending async eval to finish + drain final results
    last_eval = _eval_thread_holder[0]
    if last_eval is not None and last_eval.is_alive():
        logger.info("Waiting for final async eval to finish (max 600s)...")
        last_eval.join(timeout=600)
    while True:
        try:
            eval_data = _eval_results_queue.get_nowait()
        except queue.Empty:
            break
        from_iter = eval_data["iteration"]
        eval_sr_final = eval_data["eval_sr"]
        if eval_sr_final > best_eval_sr:
            best_eval_sr = eval_sr_final
        logger.info(f"[ASYNC RESULT @ shutdown] from iter {from_iter}: "
                     f"SR={eval_sr_final:.2%} (best_eval={best_eval_sr:.2%})")

    # Stop rollout infrastructure
    if args.use_steplock:
        for gpu_id, pool in rollout_env_pools.items():
            pool.close()
            logger.info(f"  Closed PersistentEnvPool on GPU {gpu_id}")
    else:
        for gpu_id, server in rollout_servers.items():
            server.stop()
            logger.info(f"  Stopped BatchInferenceServer on GPU {gpu_id}")

    # Final save — use the last actually-completed iter, not args.max_iter.
    # Matters when the loop bailed early (e.g. rollout poison pill): the ckpt
    # then accurately reflects the weights' true training iter instead of
    # silently naming itself iter_<max_iter>.
    save_rlt_checkpoint(enc_dec, actor, q_critic,
                        last_completed_iter, args.output_dir, phase="rl_offpolicy")
    metrics_path = Path(args.output_dir) / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics_history, f, indent=2)
    logger.info(f"Done. Metrics -> {metrics_path}")

    if args.use_wandb:
        wandb.finish()
