#!/usr/bin/env python
"""Compare vision encoder output for two image sources (MP4 vs env render)
at the same robot state. Tests whether the visual encoder produces near-
identical embeddings (in which case downstream divergence is not from
vision) or systematically different embeddings (vision-encoder OOD).

Loads the model directly (no server). Run on a free GPU.

Usage (vla env):
    CUDA_VISIBLE_DEVICES=5 python benchmarks/LIBERO/eval/probe_vision_embedding.py \
        --ckpt ./results/training/Pi05-goal-task0/checkpoints/steps_32500 \
        --lerobot_dataset_path /share/zhanghe/Datasets/libero_goal_no_noops_1.0.0_lerobot \
        --episode_index 20 \
        --env_frames_npz /tmp/envframes_ep20.npz \
        --frames "0,30,60,90,120"
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
import imageio.v3 as iio


def preprocess(img_uint8):
    """Mirror PaliGemmaPi05._process_single_img exactly."""
    img = torch.from_numpy(img_uint8.copy()).float() / 255.0
    if img.ndim == 3 and img.shape[-1] == 3:
        img = img.permute(2, 0, 1)
    img = TF.resize(img, [224, 224], antialias=True)
    img = TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return img


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lerobot_dataset_path", required=True)
    ap.add_argument("--episode_index", type=int, required=True)
    ap.add_argument("--env_frames_npz", required=True)
    ap.add_argument("--frames", default="0,30,60,90,120",
                    help="comma-separated frame indices to compare")
    args = ap.parse_args()

    print(f"loading {args.ckpt}...")
    from AlphaBrain.model.framework.PaliGemmaPi05 import PaliGemma_Pi05
    model = PaliGemma_Pi05.from_pretrained(args.ckpt)
    model = model.cuda().eval()
    print(f"model dtype: {next(model.parameters()).dtype}")

    # Load env-rendered frames (from earlier dump)
    env = np.load(args.env_frames_npz)
    env_p, env_w = env["primary"], env["wrist"]
    # Load MP4
    ds = Path(args.lerobot_dataset_path)
    chunk = args.episode_index // 1000
    mp4_p = list(iio.imiter(str(ds / f"videos/chunk-{chunk:03d}/observation.images.image/episode_{args.episode_index:06d}.mp4")))
    mp4_w = list(iio.imiter(str(ds / f"videos/chunk-{chunk:03d}/observation.images.wrist_image/episode_{args.episode_index:06d}.mp4")))

    frames = [int(f) for f in args.frames.split(",")]
    print(f"\nProbing vision encoder at frames {frames} for both primary + wrist:\n")

    paligemma = model.vlm_interface.model
    dtype = next(paligemma.parameters()).dtype
    print(f"vlm dtype: {dtype}\n")

    # Header
    print(f"{'frame':>5}  {'cam':>7}  {'pix L1':>8}  {'pix max':>7}  "
          f"{'tok L2':>10}  {'tok max':>9}  {'cosine':>9}  {'tok rel L1':>11}")
    print("-" * 95)

    for t in frames:
        for cam_name, env_arr, mp4_arr in [
            ("primary", env_p, mp4_p),
            ("wrist",   env_w, mp4_w),
        ]:
            ei = env_arr[t]
            mi = mp4_arr[t]
            # Pixel diff
            d_pix = np.abs(ei.astype(np.int32) - mi.astype(np.int32))
            pix_l1 = float(d_pix.mean())
            pix_max = int(d_pix.max())
            # Vision encoder forward
            ti_env = preprocess(ei).unsqueeze(0).cuda().to(dtype)
            ti_mp4 = preprocess(mi).unsqueeze(0).cuda().to(dtype)
            feat_env = paligemma.get_image_features(ti_env)  # [1, num_tokens, hidden]
            feat_mp4 = paligemma.get_image_features(ti_mp4)
            d = (feat_env - feat_mp4).float()
            l2 = float(d.norm())
            mx = float(d.abs().max())
            # cosine sim across all tokens flattened
            cos = float(torch.nn.functional.cosine_similarity(
                feat_env.flatten().float(), feat_mp4.flatten().float(), dim=0))
            # relative L1: |env - mp4| / (|mp4| + eps)
            rel = float((d.abs() / (feat_mp4.abs().float() + 1e-6)).mean())
            print(f"{t:>5}  {cam_name:>7}  {pix_l1:>8.2f}  {pix_max:>7d}  "
                  f"{l2:>10.2f}  {mx:>9.4f}  {cos:>9.5f}  {rel:>11.4f}")

    # Show shape for context
    print(f"\nfeat shape (per image): {feat_mp4.shape}, total elements: {feat_mp4.numel()}")
    print(f"\n=== verdict ===")
    print("If cosine ≥ 0.999 across all frames: vision encoder is robust → look downstream.")
    print("If cosine < 0.99 or rel L1 > 0.1: vision encoder amplifies the pixel diff → it's vision-encoder OOD.")


if __name__ == "__main__":
    main()
