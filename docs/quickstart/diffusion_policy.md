# Diffusion Policy

Diffusion Policy predicts a normalized action chunk by denoising a trajectory conditioned on recent multi-camera images and robot state. AlphaBrain's integration exposes the same framework interface as ACT: training returns `action_loss`, while inference returns `normalized_actions`.

## Configuration

The default architecture is in [`configs/models/diffusion_policy.yaml`](https://github.com/AlphaBrainGroup/AlphaBrain/blob/main/configs/models/diffusion_policy.yaml). Set the action and state dimensions for your robot, and keep `image_keys` in the same order as the views in each example's `image` list. `horizon` is the supervised action sequence length; `n_action_steps` is the chunk returned at inference.

The ResNet-18 image encoders load ImageNet weights by default. Set `pretrained_backbone: false` for offline construction or tests that must not download weights.

## Select the Framework

Use `diffusion_policy` as the model default in a composed training recipe:

```yaml
defaults:
  model: diffusion_policy
  dataset: your_dataset
  trainer: default
```

Equivalently, a standalone or merged configuration must contain:

```yaml
framework:
  name: DiffusionPolicy
  action_model:
    action_dim: 8
    state_dim: 8
```

## Implementation Notes

The policy, conditional U-Net, observation encoder, ResNet-18 support code, and DDPM utilities are a trimmed MIT-licensed vendored subset of [`real-stanford/diffusion_policy`](https://github.com/real-stanford/diffusion_policy). AlphaBrain keeps internal normalizers as identity transforms so its data pipeline remains the single owner of normalization. Exponential-moving-average weights are saved under `ema_averaged.*`; loading older checkpoints without those keys safely reseeds inference weights from the trained action model.
