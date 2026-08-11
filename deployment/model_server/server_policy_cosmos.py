# Copyright 2025 VLA-Engine. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
#
# server_policy_cosmos.py
#
# Cosmos-Policy specific inference server for LIBERO evaluation.
# Loads CosmosPolicy directly from the LIBERO fine-tuned checkpoint
# (Cosmos-Policy-LIBERO-Predict2-2B) without requiring a VLA-Engine
# checkpoint directory structure.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 python deployment/model_server/server_policy_cosmos.py \
#       --ckpt_dir data/pretrained_models/Cosmos-Policy-LIBERO-Predict2-2B \
#       --pretrained_dir data/pretrained_models/Cosmos-Predict2-2B-Video2World \
#       --port 10093

import argparse
import json
import logging
import os
import socket

import numpy as np

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer


class CosmosPolicyServer:
    """
    Wraps CosmosPolicy for the WebsocketPolicyServer interface.

    predict_action(**data) is called by the server with:
        batch_images: list of lists of np.ndarray  (shape: [[img1, img2], ...])
        instructions: list of str
        states: np.ndarray of shape (B, n, proprio_dim)  (optional)

    Returns:
        {"normalized_actions": np.ndarray of shape (B, chunk_size, action_dim)}
    but with actions already unnormalized using libero_dataset_statistics.json.
    The client (M1Inference) will call unnormalize_actions again, so we return
    the raw normalized_actions and let the client handle unnormalization.

    NOTE: The official cosmos-policy uses actions_min/actions_max for unnormalization.
    VLA-Engine's M1Inference.unnormalize_actions uses q01/q99 (or min/max).
    To bridge this, we expose a cosmos_unnormalize_actions() method and also
    return normalized_actions directly so the client can unnormalize.
    """

    def __init__(self, ckpt_dir: str, pretrained_dir: str, device: str = "cuda"):
        self.device = device
        self.ckpt_dir = ckpt_dir

        # Load dataset statistics (cosmos format: actions_min/actions_max)
        stats_path = os.path.join(ckpt_dir, "libero_dataset_statistics.json")
        assert os.path.exists(stats_path), f"Dataset stats not found: {stats_path}"
        with open(stats_path) as f:
            self.dataset_stats = json.load(f)
        for k, v in self.dataset_stats.items():
            self.dataset_stats[k] = np.array(v)
        logging.info(f"[CosmosPolicyServer] Loaded dataset stats from {stats_path}")

        # Load T5 embeddings
        t5_path = os.path.join(ckpt_dir, "libero_t5_embeddings.pkl")
        assert os.path.exists(t5_path), f"T5 embeddings not found: {t5_path}"

        # cuDNN settings handled in CosmosPolicy.__init__

        # Build CosmosPolicy config
        config = self._build_config(pretrained_dir, t5_path, ckpt_dir=ckpt_dir)

        # Instantiate CosmosPolicy
        from AlphaBrain.model.framework.CosmosPolicy import CosmosPolicy
        self.model = CosmosPolicy(config=config)

        # Pass dataset stats to model for proprio normalization
        self.model.set_dataset_stats(self.dataset_stats)

        # Move to device
        self.model = self.model.to(device).eval()
        logging.info(f"[CosmosPolicyServer] Model loaded on {device}")

    @staticmethod
    def _find_dit_checkpoint(ckpt_dir: str) -> str:
        """Find DIT checkpoint file in various formats (pretrained or VLA-trained)."""
        candidates = [
            os.path.join(ckpt_dir, "cosmos_dit.pt"),                           # VLA-trained (net. prefix)
            os.path.join(ckpt_dir, "Cosmos-Policy-LIBERO-Predict2-2B.pt"),     # Official pretrained
            os.path.join(ckpt_dir, "pytorch_model.pt"),                        # VLA generic save
        ]
        for path in candidates:
            if os.path.exists(path):
                logging.info(f"[CosmosPolicyServer] Using checkpoint: {path}")
                return path
        # Fallback: find any .pt file
        for f in os.listdir(ckpt_dir):
            if f.endswith(".pt") and "optimizer" not in f.lower():
                path = os.path.join(ckpt_dir, f)
                logging.info(f"[CosmosPolicyServer] Using checkpoint (fallback): {path}")
                return path
        raise FileNotFoundError(f"No DIT checkpoint found in {ckpt_dir}")

    def _build_config(self, pretrained_dir: str, t5_path: str, ckpt_dir: str | None = None):
        """Build a minimal config namespace for CosmosPolicy."""
        from types import SimpleNamespace

        def ns(**kwargs):
            return SimpleNamespace(**kwargs)

        config = ns(
            framework=ns(
                cosmos_policy=ns(
                    action_dim=7,
                    chunk_size=16,
                    proprio_dim=9,
                    state_t=9,
                    checkpoint=ns(
                        pretrained_dir=pretrained_dir,
                        load_path=self._find_dit_checkpoint(ckpt_dir),
                    ),
                    t5_embeddings_path=t5_path,
                    dit=ns(
                        max_img_h=240,
                        max_img_w=240,
                        max_frames=128,
                        patch_spatial=2,
                        patch_temporal=1,
                        model_channels=2048,
                        num_blocks=28,
                        num_heads=16,
                        crossattn_emb_channels=1024,
                        use_crossattn_projection=False,
                        pos_emb_cls="rope3d",
                        mlp_ratio=4.0,
                        use_wan_fp32_strategy=False,
                        pos_emb_learnable=True,
                        rope_enable_fps_modulation=False,
                        rope_h_extrapolation_ratio=3.0,
                        rope_w_extrapolation_ratio=3.0,
                    ),
                    sde=ns(
                        p_mean=1.3862943611198906,
                        p_std=1.2,
                        sigma_max=200.0,
                        sigma_min=0.01,
                    ),
                    loss=ns(
                        action_loss_multiplier=1.0,
                        world_model_loss_weight=1.0,
                        value_loss_weight=0.0,
                        sigma_data=1.0,  # RectifiedFlowScaling
                    ),
                    inference=ns(
                        action_num_steps=5,
                        sigma_min=4.0,
                        sigma_max=80.0,
                    ),
                    sigma_conditional=0.0,  # Matches original (EDMScaling: c_noise clamped for sigma=0)
                )
            )
        )
        return config

    def predict_action(self, batch_images=None, instructions=None, states=None, **kwargs):
        """
        Interface called by WebsocketPolicyServer.

        Args:
            batch_images: list of image lists, e.g. [[primary_img, wrist_img], ...]
                          Each image is np.ndarray (H, W, 3) uint8.
            instructions: list of str task descriptions
            states: np.ndarray (B, n, proprio_dim) — last n proprio states

        Returns:
            dict with "normalized_actions": np.ndarray (B, chunk_size, action_dim)
            NOTE: actions are already unnormalized to robot action space.
        """
        if batch_images is None or instructions is None:
            raise ValueError("batch_images and instructions are required")

        B = len(batch_images)

        # Upscale images from client (224x224) back to env resolution (256x256)
        # because CosmosPolicy._preprocess_image expects raw images and does its own
        # JPEG + resize + center_crop pipeline matching training augmentation.
        from PIL import Image as _PILImage
        def _upscale(img, target_size=256):
            if img is not None and img.shape[0] == 224:
                pil = _PILImage.fromarray(img)
                pil = pil.resize((target_size, target_size), _PILImage.BILINEAR)
                return np.array(pil)
            return img

        if batch_images is not None:
            # Cosmos eval client sends images with H flip only ([::-1]).
            # Pass through as-is. Only upscale if client sent 224x224.
            batch_images = [
                [np.ascontiguousarray(_upscale(img) if img is not None else img) for img in imgs]
                for imgs in batch_images
            ]

        # Extract primary and wrist images
        primary_images = []
        wrist_images = []
        for imgs in batch_images:
            if len(imgs) >= 2:
                primary_images.append(imgs[0])
                wrist_images.append(imgs[1])
            else:
                primary_images.append(imgs[0])
                wrist_images.append(None)

        # Extract current proprio from states (last timestep)
        # Client sends [eef_pos(3), axisangle(3), gripper_qpos(2)] = 8dim
        # Cosmos model needs [gripper_qpos(2), eef_pos(3), eef_quat(4)] = 9dim
        proprios = []
        if states is not None:
            from scipy.spatial.transform import Rotation
            states_arr = np.array(states)  # (B, n, proprio_dim)
            for i in range(B):
                s = states_arr[i, -1, :]  # last timestep
                if s.shape[0] == 8:
                    # Convert: [eef_pos(3), axisangle(3), gripper(2)] -> [gripper(2), eef_pos(3), quat(4)]
                    eef_pos = s[0:3]
                    axisangle = s[3:6]
                    gripper = s[6:8]
                    quat = Rotation.from_rotvec(axisangle).as_quat()  # [x,y,z,w]
                    proprios.append(np.concatenate([gripper, eef_pos, quat]))
                elif s.shape[0] == 9:
                    proprios.append(s)  # already cosmos format
                else:
                    proprios.append(s)
        else:
            proprios = [np.zeros(self.model.proprio_dim)] * B

        # Build examples for predict_action
        examples = []
        for i in range(B):
            examples.append({
                "image": primary_images[i],
                "wrist_image": wrist_images[i],
                "lang": instructions[i],
                "proprio": proprios[i],
            })

        # Run inference
        result = self.model.predict_action(examples=examples)
        normalized_actions = result["normalized_actions"]  # (B, chunk_size, action_dim)

        # Return raw normalized actions to client.
        # The cosmos eval client (eval_libero_cosmos.py) does its own
        # unnormalization, so do NOT unnormalize or binarize gripper here.
        return {"normalized_actions": normalized_actions}

    def _unnormalize_actions(self, normalized_actions: np.ndarray) -> np.ndarray:
        """
        Unnormalize actions from [-1,1] to robot action space.
        Uses cosmos format: actions_min/actions_max.
        Formula: action = 0.5 * (norm + 1) * (max - min) + min
        """
        actions_min = self.dataset_stats["actions_min"]
        actions_max = self.dataset_stats["actions_max"]
        orig_shape = normalized_actions.shape
        actions = normalized_actions.reshape(-1, actions_min.shape[0])
        actions = np.clip(actions, -1.0, 1.0)
        actions = 0.5 * (actions + 1.0) * (actions_max - actions_min) + actions_min
        return actions.reshape(orig_shape)


def main(args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)

    policy = CosmosPolicyServer(
        ckpt_dir=args.ckpt_dir,
        pretrained_dir=args.pretrained_dir,
        device="cuda",
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata={"env": "libero", "model": "CosmosPolicy"},
        api_key_sha256=args.api_key_sha256,
        controller_api_key_sha256=args.controller_api_key_sha256,
        deployment_id=args.deployment_id,
    )
    logging.info("Cosmos policy server running on port %d ...", args.port)
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser(description="Cosmos-Policy LIBERO inference server")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="data/pretrained_models/Cosmos-Policy-LIBERO-Predict2-2B",
        help="Path to LIBERO fine-tuned checkpoint directory",
    )
    parser.add_argument(
        "--pretrained_dir",
        type=str,
        default="data/pretrained_models/Cosmos-Predict2-2B-Video2World",
        help="Path to base Cosmos Predict2 model (for VAE weights)",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Address to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument(
        "--idle_timeout",
        type=int,
        default=1800,
        help="Idle timeout in seconds, -1 means never close",
    )
    parser.add_argument(
        "--api-key-sha256",
        default=None,
        help="Optional SHA-256 hex digest of the API key; omit to retain unauthenticated legacy mode",
    )
    parser.add_argument(
        "--controller-api-key-sha256",
        default=None,
        help="Optional SHA-256 digest for the non-user-facing UI controller credential",
    )
    parser.add_argument("--deployment-id", default=None, help="Deployment identifier reported by GET /healthz")
    return parser


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    main(args)
