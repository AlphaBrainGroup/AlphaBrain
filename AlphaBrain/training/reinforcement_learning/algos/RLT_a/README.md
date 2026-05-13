# RLT_a — Design Notes & Delta vs. the RL Token Paper

This directory contains our implementation of an off-policy, bottleneck-token
online RL recipe for VLA models, under the name **`RLT_a`**. The name
was deliberately chosen to be distinct from the **RL Token** paper
(Physical Intelligence, 2026), whose high-level idea inspired this module
but whose implementation we do **not** reproduce line-by-line.

This README explains what lives in this directory, how the pieces fit
together, and — most importantly — **where we deviate from the paper**, so
that readers comparing this code against the paper are not surprised.

---

## File layout

| File | Role |
|:-----|:-----|
| `action_token_encoder_decoder.py` | `ActionTokenEncoder` and `ActionTokenDecoder`: bottleneck encoder + reconstruction decoder used in Phase 1 pretraining |
| `action_token_actor_critic.py` | TD3 actor `μ_θ(x, ã)` and twin critics `Q_{ψ1,2}(x, a)` |
| `action_token_trainer.py` | Loss terms, TD-update step, and checkpoint plumbing |
| `action_token_rollout_fast.py` | Step-lock batched rollout over parallel envs; feeds the replay buffer |
| `__init__.py` | Re-exports the public surface |

The trainer entrypoints live one level up in `trainers/`; the scripts
that drive an end-to-end run live in `scripts/run_rl_scripts/`.

---

## High-level shape

```
obs ──► frozen VLA ──► action-query hidden states (M × H)
                             │
                             ▼
                        ActionTokenEncoder  ──►  z_rl ∈ ℝ^{1 × D}
                             │                        │
          Phase 1: ─────────►│                        │
          ActionTokenDecoder ◄───  reconstruct sg(action_queries)
                                                      │
          Phase 2: ─────────────────────────────────► │
                                      x = (z_rl, s_p)
                                             │
                                             ▼
                                   TD3 actor / twin critics
                                      (trained off-policy)
```

- **Phase 1** trains the encoder/decoder with a reconstruction objective.
- **Phase 2** freezes the VLA (and optionally the encoder) and trains the
  small actor/critic on transitions collected by the rollout workers.

---

## Delta vs. the RL Token paper

`RLT_a` keeps the paper's high-level shape — a frozen VLA feeds a
compact state into a small TD3 actor/critic — but differs in several concrete
choices. If you are comparing this code against the paper, please read this
section before assuming equivalence.

### Encoder (`ActionTokenEncoder`)

- **Input.** We feed the VLA's **action-query hidden states**
  `action_queries ∈ ℝ^{B × M × H}` (gathered at the action-token positions
  out of `last_hidden`), where `M = chunk_len = 8` and `H = 2048` for
  Qwen2.5-VL-3B. The paper's Fig. 2 instead feeds the **full image-token
  embeddings** `N × 2048` from the VLM backbone.
- **Structure.** We append a learnable `e_rl` CLS token to the sequence and
  run it through a small self-attention encoder (default 2 layers, 4 heads,
  `d_model = H`), then take the output at the `e_rl` position.
- **Extra bottleneck projection.** We apply a `Linear(H → bottleneck_dim)`
  after the encoder, giving `z_rl ∈ ℝ^{B × 1 × D}` with `D = 256` by default.
  **The paper keeps `z_rl` at the VLA hidden dim (`1 × 2048`)** — its
  bottleneck comes from collapsing `N` tokens to `1` token, not from reducing
  per-token width. Our extra projection is a pragmatic knob for the small
  actor/critic MLPs downstream, but it is a genuine deviation.

### Decoder (`ActionTokenDecoder`)

- Used only during Phase-1 pretraining, for the reconstruction loss that
  enforces the bottleneck to be information-preserving.
- We `Linear(D → H)` expand `z_rl` back to VLA hidden dim, prepend it as a
  prefix to a teacher-forced shifted VLA token sequence, add a learned
  positional embedding, and run a causal-masked self-attention stack
  (`TransformerEncoderLayer` with `src_mask = triu`, 2 layers by default).
- **This is a prefix + causal-self-attention setup, not the encoder-decoder
  cross-attention structure shown in the paper's Eq. 2.** Functionally
  close, architecturally different.
- `L_ro = MSE(reconstructed, sg(action_queries))` — MSE over all
  reconstructed positions, with a stop-gradient on the VLA tokens.

### Other deviations worth knowing

- **Pretrain data.** Paper uses task demonstrations `D`. We currently
  collect observations by rolling out **random actions** in the env
  (`collect_observations_fast`) for simplicity; this mismatches the paper's
  distribution.
- **Joint VLA fine-tune during pretrain.** Paper's Algorithm 1 (line 3)
  optimizes `ϕ, θ_vla = argmin L_ro(ϕ) + α L_vla(θ_vla)`. Our Phase-1 trains
  the encoder/decoder only; `--finetune_vla` exists but activates during the
  RL phase, not during encoder pretraining.
- **Base VLA.** Paper uses `π0.6` (SigLIP + Gemma 4B + 860M flow/diffusion
  action expert). We build on QwenOFT (Qwen2.5-VL-3B + MLP action head), so
  the VLA reference action is unimodal rather than a multimodal diffusion
  sample — the `ref-action pass-through` still helps, but for a different
  reason than in the paper.
- **Environment / chunking.** Paper: 14-D bimanual real-world, RL chunk
  `C = 10`, VLA `H = 50`, 50 Hz. We ship: 7-D LIBERO simulation, VLA chunk
  `= 8`, actor chunk `= 4`.
- **Human-in-the-loop, critical-phase switching, policy-handover learning**
  — all paper-only. The current release is a pure autonomous sim recipe.

### What matches the paper

- TD3 off-policy with twin-Q and target policy smoothing.
- Actor `μ_θ(x, ã)` + fixed small Gaussian std, BC regularizer
  `β ‖a − ã‖²`, 50% reference-action dropout.
- Chunk-subsampling at stride 2 when pushing transitions to the replay
  buffer.
- `x = (z_rl, s_p)` state construction, where `s_p` is proprioception.

---

## "Official" paper-accurate implementation

We are actively testing a second track that follows the paper more
literally — image-token input, `1 × 2048` RL token with no extra projection,
cross-attention decoder, demo-driven Phase-1, joint VLA fine-tune. It is
**not yet stable enough to release**; when it lands it will ship as a
sibling module rather than a replacement, so existing users of
`RLT_a` are not broken.

---

## Entry points

- **Train (Phase 1 + Phase 2)**: `scripts/run_rl_scripts/run_action_token_5traj_alltasks.sh`
- **Eval**: `scripts/run_rl_scripts/run_eval_action_token.sh`
- **Recipe YAML**: `configs/rl_recipes/QwenOFT_LIBERO_ActionToken.yaml`
- **Script-level README** (CLI flags, rollout math, gotchas): `scripts/run_rl_scripts/README.md`
