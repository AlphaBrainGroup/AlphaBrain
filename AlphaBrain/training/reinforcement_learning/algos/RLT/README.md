# RLT — Reference-Faithful RL Token Encoder-Decoder

This module implements the encoder-decoder and Phase-1 reconstruction
training of the **RL Token** recipe (Physical Intelligence, 2026) as
close to the reference as this codebase permits. It is the sibling of
the production track `RLT_a/` and is meant to be *read* against
that module's `README.md`, which documents that track's pragmatic
deviations.

> Reference: "RL Token: Bootstrapping Online RL with Vision-Language-Action
> Models" (Physical Intelligence, 2026), Sec. IV-A and Algorithm 1, lines
> 1–3.

---

## What this module contains

| File | Role |
|:-----|:-----|
| `encoder_decoder.py` | `RLTokenEncoder` (Eq. 1) and `RLTokenDecoder` (Eq. 2) with an encoder-decoder cross-attention layout, wrapped by `RLTokenEncoderDecoder` |
| `vla_features.py`   | `get_vla_hidden_states(vla, ...)` helper that calls the frozen framework's public interface to return the full VLM last-layer sequence plus an attention mask — without modifying any framework file |
| `__init__.py`       | Re-exports the public surface |

Phase-1 pretraining lives one level up in
`trainers/train_rlt_pretrain.py` and is selected via
`--phase pretrain_rlt`.

---

## What matches the reference

### Encoder (Eq. 1)

`z_rl = g_φ([z_{1:M}, e_rl])_{M+1}`

- Input: the VLA's **full last-layer token embeddings** `z_{1:M}` for
  every input token (images + language [+ optional action placeholders])
  — obtained via `get_vla_hidden_states`, not via
  `get_action_queries`.
- `e_rl` is a learnable embedding appended to the sequence; the output
  at its position is the RL token.
- **No extra per-token projection.** `z_rl ∈ ℝ^{1 × H}` at the VLA
  hidden dim. The bottleneck comes from collapsing M tokens to 1 token.
- `key_padding_mask` plumbed through so the self-attention ignores
  right-padding positions emitted by the HF tokenizer.

### Decoder (Eq. 2)

```
L_ro = E_D[ Σ_{i=1}^{M}  ‖ h_φ(d_φ([z_rl, sg(z_{1:i-1})]))_i − sg(z_i) ‖² ]
```

- `nn.TransformerDecoderLayer` with `memory = z_rl` and a shifted-right,
  causally-masked target stream `[BOS, sg(z_1), ..., sg(z_{M-1})]`. This
  is the encoder-decoder construction the reference shows (distinct from
  the sibling track's self-attention-only decoder).
- `h_φ` is a final `Linear(H, H)`.
- The reconstruction loss is MSE against **stop-gradient** VLA
  embeddings; gradients flow back only through `z_rl` into the encoder.
- Padding positions are excluded from the loss via the attention mask.

### Joint VLA fine-tune (Algorithm 1, line 3)

```
ϕ, θ_vla = argmin L_ro(ϕ) + α L_vla(θ_vla)
```

- Passing `--alpha_vla > 0` together with `--demo_config <yaml>` unfreezes
  the VLA and adds its own imitation loss (`Qwenvl_OFT.forward` returns
  the framework's L1 action regression loss) to the objective.
- `α = 0` keeps the VLA frozen — the default.

### Demo-driven Phase-1 data

The reference trains Phase-1 on a small task-specific demonstration
dataset `D`. When `--demo_config <yaml>` is supplied, this trainer loads
the same `LeRobotMixtureDataset` the SFT pipeline uses and iterates
over demo samples batch-by-batch. Action labels are used for `L_vla`
when `α > 0`; otherwise only the images + language are used.

If `--demo_config` is omitted the trainer falls back to the sibling
track's random-rollout observation collector (`collect_observations_fast`),
which **is** a deviation — noted in the help text.

---

## What to be aware of

- **Base VLA.** Two backbones are wired in:
  - `Qwenvl_OFT` (Qwen2.5-VL-3B + MLP action head) — original target.
  - `PaliGemmaPi05` (SigLIP + Gemma 2B + flow-matching action expert) —
    routes through the Pi05 inference adapter
    (`pi05_inference.py`) which fuses prefix Gemma forward +
    diffusion in one call per rollout step.
  Both go through the same `RLTokenEncoderDecoder`; only the
  `get_vla_hidden_states` dispatch differs (see `vla_features.py` vs
  `vla_features_pi05.py`). The encoder-decoder construction is
  base-VLA-agnostic — it only sees `z_{1:M}` — but the meaning of
  "task-relevant information" inside `z_{1:M}` is whatever the pretrained
  backbone you pass in happens to encode.
- **Chunk length and hidden dim.** Taken from the loaded VLA. For Qwen,
  via `qwen_vl_interface.model.config.hidden_size`; for Pi05, via the
  framework's interior PaliGemma config. Both expose `vla.chunk_len`.
- **Max sequence length.** The decoder's positional embedding is
  allocated at `--max_len` (default 4096). For longer inputs (e.g. more
  cameras or longer instructions) pass a larger `--max_len`.
- **Action-token positions.** The reference describes `z_{1:M}` as the
  VLA's embeddings of its *input* tokens; the action placeholders are
  *prediction* targets, not inputs. The default (`--drop_action_tokens`)
  drops those positions from the encoder input. Pass
  `--keep_action_tokens` to include them (e.g. for ablations).

## How to run

### Phase-1 (encoder-decoder pretrain)

```bash
# Frozen VLA, demo-driven, reference-faithful:
bash scripts/run_rl_scripts/run_rlt_pretrain.sh 0

# Or directly:
python AlphaBrain/training/reinforcement_learning/trainers/train.py \
    --phase pretrain_rlt \
    --ckpt_path results/training/QwenOFT-5traj-libero_goal/final_model \
    --demo_config benchmarks/LIBERO/train/alphabrain_neurovla_libero.yaml \
    --output_dir results/rlt_training/pretrain \
    --suite libero_goal --all_tasks \
    --encoder_layers 2 --decoder_layers 2 --encoder_heads 8 \
    --pretrain_lr 1e-4 --pretrain_batch_size 8 --pretrain_epochs 50 \
    --alpha_vla 0.0
```

Enable the joint VLA fine-tune (Algorithm 1, line 3) with e.g.
`--alpha_vla 1.0 --lr_vla 5e-6`.

### Phase-2 (TD3 with frozen VLA + frozen encoder + trainable actor/critic)

Reuses the production Phase-2 trainer with `--encoder_mode rlt`:

```bash
# Qwen backbone:
bash scripts/run_rl_scripts/run_rlt_rl_task0_release.sh 0

# Pi05 backbone (1traj or 5traj):
VARIANT=1traj TASK_ID=0 bash scripts/run_rl_scripts/run_rlt_rl_task0_release_pi05.sh 0
VARIANT=5traj TASK_ID=3 bash scripts/run_rl_scripts/run_rlt_rl_task0_release_pi05.sh 0
```

The best encoder-decoder checkpoint lands at:

```
<output_dir>/checkpoints/pretrain_best/encoder.pt
```

Feed that path into a downstream RL phase as
`--encoder_path <...>/encoder.pt`. The state dict is compatible with
`RLTokenEncoderDecoder(hidden_dim=..., num_heads=..., encoder_layers=...,
decoder_layers=..., max_len=...)` — those four hyperparameters must
match at load time.

---

## Relationship to `RLT_a/`

| | `RLT_a` (production) | `RLT` (this module) |
|---|---|---|
| Encoder input | Action-query slice `(B, M=chunk_len, H)` | Full VLM tokens `(B, L, H)` |
| z_rl width | Projected to `D = 256` | Kept at VLA hidden dim `H` |
| Decoder | Self-attention, causal, `z_rl` prefix | Encoder-decoder cross-attention, causal target, `z_rl` memory |
| Phase-1 data | Random-rollout observations | Demonstrations (fallback: rollouts) |
| Joint VLA SFT | Not in Phase-1 | `α L_vla` term (Algorithm 1, line 3) |
| Phase 2 (actor/critic, TD3) | Shipped | Reuses the same Phase-2 trainer via `--encoder_mode rlt` |

Both live side-by-side; pick whichever matches your study's goal.
