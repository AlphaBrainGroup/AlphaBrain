"""CLI args for RLT_a training (all three phases)."""
import argparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--phase",
        type=str,
        required=True,
        choices=["pretrain", "pretrain_rlt", "rl", "rl_offpolicy", "grpo", "vla_ppo"],
    )
    p.add_argument("--ckpt_path", type=str, required=True, help="SFT checkpoint path")
    p.add_argument("--encoder_path", type=str, default=None,
                   help="Pretrained encoder checkpoint (required for --phase rl)")
    p.add_argument("--output_dir", type=str, default="results/action_token_training")
    p.add_argument("--suite", type=str, default="libero_goal",
                   choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"])
    p.add_argument("--task_id", type=int, default=0,
                   help="Task to train on. -1 = random.")
    p.add_argument("--task_ids", type=str, default=None,
                   help="Comma-separated task IDs to train on (e.g. '0,1'). "
                        "Overrides --task_id. Works like --all_tasks but for a subset.")
    p.add_argument("--all_tasks", action="store_true",
                   help="Train on ALL tasks in the suite (overrides --task_id). "
                        "Each iteration collects episodes from every task.")

    # RLT_a architecture
    p.add_argument("--encoder_mode", type=str, default="action_token",
                   choices=["action_token", "rlt"],
                   help="Which encoder family to use in Phase 2. "
                        "'action_token' (default): pragmatic RLT_a "
                        "encoder that consumes the chunk_len action-query slice. "
                        "'rlt': reference RL Token encoder that consumes the "
                        "full VLM image-token sequence and keeps z_rl at the VLA "
                        "hidden dim — pair with --bottleneck_dim 2048.")
    p.add_argument("--bottleneck_dim", type=int, default=256)
    p.add_argument("--encoder_layers", type=int, default=2)
    p.add_argument("--encoder_heads", type=int, default=4)
    p.add_argument("--actor_chunk_len", type=int, default=None,
                   help="Actor/critic chunk length (default: same as VLA chunk_len). "
                        "Paper: VLA H=50, actor C=10. For LIBERO: VLA=8, actor=4.")
    p.add_argument("--actor_hidden_dim", type=int, default=256,
                   help="Paper: 256 for most tasks, 512 for hard tasks")
    p.add_argument("--critic_hidden_dim", type=int, default=256)
    p.add_argument("--ref_dropout", type=float, default=0.5,
                   help="Paper: 50% reference-action dropout")

    # Pretraining (Phase 1)
    p.add_argument("--pretrain_n_obs", type=int, default=2000,
                   help="Number of observations to collect for encoder pretraining")
    p.add_argument("--pretrain_steps_per_reset", type=int, default=20,
                   help="Random steps per env reset for observation diversity")
    p.add_argument("--pretrain_epochs", type=int, default=50,
                   help="Upper bound on dataset passes. See also --pretrain_max_steps.")
    p.add_argument("--pretrain_max_steps", type=int, default=0,
                   help="Early-stop the pretrain at N gradient steps (one optimizer.step = "
                        "one step). 0 disables the cap and lets pretrain_epochs run to completion. "
                        "When both are set, whichever hits first wins.")
    p.add_argument("--pretrain_lr", type=float, default=1e-4)
    p.add_argument("--pretrain_batch_size", type=int, default=32,
                   help="Batch size for encoder-decoder pretraining (tensor-level, no VLA forward)")
    p.add_argument("--vla_extract_batch_size", type=int, default=16,
                   help="Batch size for VLA action_queries extraction (one-time, memory-bound)")

    # RLT Phase 1 (paper-faithful pretraining)
    p.add_argument("--demo_config", type=str, default=None,
                   help="Path to a YAML with datasets.vla_data (LeRobot demo mixture) for "
                        "the RLT pretrain phase. Required when alpha_vla > 0; "
                        "otherwise the trainer falls back to random-rollout observations.")
    p.add_argument("--alpha_vla", type=float, default=0.0,
                   help="Weight on L_vla for joint VLA fine-tune during RLT pretrain "
                        "(reference Alg. 1: ϕ, θ_vla = argmin L_ro + α L_vla). "
                        "0 keeps the VLA frozen. Requires --demo_config.")
    p.add_argument("--decoder_layers", type=int, default=2,
                   help="RLT decoder transformer layers")
    p.add_argument("--max_len", type=int, default=4096,
                   help="Max VLA token sequence length the RLT decoder supports "
                        "(controls target-stream positional embedding size)")
    p.add_argument("--image_only", action="store_true", default=True,
                   help="Strict-reference mode: keep ONLY image-token positions "
                        "(matches Fig. 2 / footnote 1). Default True.")
    p.add_argument("--all_tokens", dest="image_only", action="store_false",
                   help="Relaxed mode: keep image + language (and optionally action) "
                        "tokens; mirrors Sec. IV-A's general claim. Subject to "
                        "--drop_action_tokens.")
    p.add_argument("--drop_action_tokens", action="store_true", default=True,
                   help="Only relevant when --all_tokens: drop action-placeholder "
                        "token positions from the encoder input (they are prediction "
                        "targets, not inputs).")
    p.add_argument("--keep_action_tokens", dest="drop_action_tokens",
                   action="store_false",
                   help="Only relevant when --all_tokens: keep action-placeholder "
                        "positions in the encoder input.")

    # RL training (Phase 2)
    # Naming: *_per_task is the canonical name (clearer for multi-task);
    # the old --G / --num_envs are aliases for backward compat with old scripts.
    p.add_argument("--G_per_task", "--G", dest="G_per_task", type=int, default=8,
                    help="Episodes per task per main_iter. "
                         "Single-task: total ep/iter = G_per_task. "
                         "Multi-task:  total ep/iter = G_per_task × n_tasks. "
                         "If G_per_task > num_envs_per_task, the rollout is auto-chunked into "
                         "ceil(G_per_task / num_envs_per_task) sequential passes.")
    p.add_argument("--group_size", type=int, default=1,
                   help="Trajectories per initial state. "
                        "G_per_task episodes split into G_per_task//group_size unique states, "
                        "each repeated group_size times. Default 1 = no repeat.")
    p.add_argument("--num_envs_per_task", "--num_envs", dest="num_envs_per_task", type=int, default=4,
                   help="Persistent env pool size PER TASK PER ROLLOUT GPU. "
                        "Total pool size per GPU = num_envs_per_task × max_tasks_per_gpu. "
                        "Total envs across all GPUs = num_envs_per_task × n_tasks (for balanced multi-task).")
    p.add_argument("--use_steplock", action="store_true",
                   help="Use step-lock rollout (persistent env pool + batched VLA inference). "
                        "~3x faster than default async rollout.")
    p.add_argument("--ppo_epochs", type=int, default=10,
                   help="PPO epochs per iteration (small net → more epochs)")
    p.add_argument("--max_iter", type=int, default=500)
    p.add_argument("--lr_actor", type=float, default=3e-4)
    p.add_argument("--lr_critic", type=float, default=3e-4)
    p.add_argument("--lr_encoder", type=float, default=1e-5,
                   help="Encoder LR during RL (0 = freeze encoder)")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--recon_loss_coef", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--num_steps_wait", type=int, default=10)

    # GRPO-specific (phase grpo)
    p.add_argument("--grpo_kl_coef", type=float, default=0.04,
                   help="KL-to-ref penalty coefficient (DeepSeek default 0.04)")
    p.add_argument("--ref_update_interval", type=int, default=0,
                   help="Refresh reference actor every N iters (0 = never; "
                        "keeps initial actor as fixed reference)")

    # Vanilla VLA+PPO (phase vla_ppo)
    # Note: --lr_vla is shared with the off-policy finetune path (defined below).
    p.add_argument("--micro_batch", type=int, default=4,
                   help="VLA re-forward mini-batch size during PPO update "
                        "(memory-bound; lower if OOM)")

    # Off-policy RL (Phase 2 offpolicy)
    p.add_argument("--buffer_capacity", type=int, default=100000,
                   help="Replay buffer capacity")
    p.add_argument("--buffer_warmup", type=int, default=512,
                   help="Minimum buffer size before starting TD updates")
    p.add_argument("--warmup_iters", type=int, default=5,
                   help="Iterations of pure VLA rollout (no actor) to pre-fill buffer (paper Sec. V)")
    p.add_argument("--td_updates_per_iter", type=int, default=64,
                   help="Max TD gradient steps per iteration (actual capped by UTD ratio)")
    p.add_argument("--utd_ratio", type=float, default=2.0,
                   help="Update-to-data ratio: max gradient steps = buffer_size * utd / batch_size")
    p.add_argument("--td_batch_size", type=int, default=256,
                   help="Batch size for sampling from replay buffer")
    p.add_argument("--tau", type=float, default=0.005,
                   help="Target critic soft update coefficient")
    # TD3/RL Token paper specific
    p.add_argument("--beta", type=float, default=1.0,
                   help="BC regularization coefficient (paper Eq. 5): β‖a - ã‖². "
                        "Use ≥1.0 for stable training; 0.1 is too weak and allows policy drift.")
    p.add_argument("--actor_update_freq", type=int, default=2,
                   help="Update actor every N critic updates (TD3 delayed policy, paper uses 2)")
    p.add_argument("--target_noise_std", type=float, default=0.2,
                   help="Std of noise added to target policy actions (TD3 smoothing)")
    p.add_argument("--target_noise_clip", type=float, default=0.5,
                   help="Clip range for target policy noise")
    p.add_argument("--reward_coef", type=float, default=1.0,
                   help="Reward multiplier for success")
    p.add_argument("--fixed_std", type=float, default=0.1,
                   help="Fixed Gaussian std for actor exploration (paper: small fixed std)")
    # Full VLA fine-tune (off-policy TD3 + VLA gradient via re-encoding)
    p.add_argument("--finetune_vla", action="store_true", default=False,
                   help="Unfreeze VLA and periodically update via re-encoding from images")
    p.add_argument("--lr_vla", type=float, default=5e-6,
                   help="VLA + encoder learning rate (separate from actor/critic)")
    p.add_argument("--vla_update_freq", type=int, default=1,
                   help="Do VLA fine-tune step every N iterations")
    p.add_argument("--vla_micro_batch", type=int, default=4,
                   help="Micro batch size for VLA forward during fine-tune (controls GPU mem)")

    # GPU split (off-policy only): separate rollout GPUs from training GPU
    p.add_argument("--rollout_gpus", type=str, default=None,
                   help="Comma-separated GPU IDs for rollout, e.g. '0,1,2,3,4'. "
                        "If None, uses all visible GPUs for rollout.")
    p.add_argument("--train_gpu", type=int, default=None,
                   help="GPU ID for actor-critic training. Can overlap with rollout_gpus. "
                        "If None, defaults to first rollout GPU.")

    # Eval
    p.add_argument("--eval_interval", type=int, default=10,
                   help="Run deterministic eval every N iterations (0 = disable)")
    p.add_argument("--eval_n_episodes", type=int, default=10,
                   help="Number of episodes for deterministic eval")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_interval", type=int, default=50)
    p.add_argument("--save_video_interval", type=int, default=10)
    p.add_argument("--log_interval", type=int, default=1)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="AlphaBrain_RLT")
    p.add_argument("--run_name", type=str, default=None)
    args = p.parse_args()
    # Backward-compat aliases: code may still reference args.G / args.num_envs
    args.G = args.G_per_task
    args.num_envs = args.num_envs_per_task
    return args
