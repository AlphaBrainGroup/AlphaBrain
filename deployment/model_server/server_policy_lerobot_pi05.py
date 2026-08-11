"""LeRobot Pi0.5 WebSocket deployment adapter.

This adapter deliberately loads the LeRobot policy format directly.  It does
not route the checkpoint through AlphaBrain's ``PaliGemmaPi05`` framework,
because the two formats have different configs, processors, and state-dict
hierarchies.
"""

from __future__ import annotations

import argparse
import logging
import socket
from pathlib import Path
from typing import Any

import numpy as np

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer


class LeRobotPI05ServerPolicy:
    """Expose a LeRobot ``PI05Policy`` through the AlphaBrain protocol.

    The protocol uses positional camera views.  This checkpoint contract is:
    ``[front, left_wrist, right_wrist]``, a 7-D current state, and one task
    instruction per batch item.  Returned actions have already passed through
    the checkpoint's LeRobot postprocessor (unnormalization and relative-to-
    absolute conversion).
    """

    CAMERA_KEYS = (
        "observation.images.front",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    )

    def __init__(self, checkpoint: str, device: str = "cuda") -> None:
        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        checkpoint_path = Path(checkpoint).expanduser().resolve()
        config = PreTrainedConfig.from_pretrained(
            checkpoint_path,
            local_files_only=True,
        )
        if config.type != "pi05":
            raise ValueError(f"Expected a LeRobot pi05 checkpoint, found {config.type!r}")
        config.device = device
        self.device = torch.device(device)
        self.policy = PI05Policy.from_pretrained(
            checkpoint_path,
            config=config,
            local_files_only=True,
            strict=True,
        ).eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            config,
            pretrained_path=str(checkpoint_path),
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self._torch = torch
        logging.info(
            "Loaded LeRobot Pi0.5 checkpoint format from %s on %s",
            checkpoint_path,
            device,
        )

    @staticmethod
    def _validate_image(image: Any, *, batch_index: int, camera_index: int) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(
                f"batch_images[{batch_index}][{camera_index}] must have HxWx3 shape"
            )
        if array.dtype != np.uint8:
            raise ValueError(
                f"batch_images[{batch_index}][{camera_index}] must use uint8 pixels"
            )
        return np.ascontiguousarray(array)

    def _build_batch(self, batch_images: Any, instructions: Any, states: Any) -> dict[str, Any]:
        torch = self._torch
        if not isinstance(batch_images, (list, tuple)) or not batch_images:
            raise ValueError("batch_images is required and must be a non-empty batch")
        batch_size = len(batch_images)
        if not isinstance(instructions, (list, tuple)) or len(instructions) != batch_size:
            raise ValueError("instructions must contain one task string per batch item")
        if any(not isinstance(item, str) or not item.strip() for item in instructions):
            raise ValueError("each Pi0.5 task instruction must be a non-empty string")

        camera_batches: list[list[np.ndarray]] = [[], [], []]
        for batch_index, views in enumerate(batch_images):
            if not isinstance(views, (list, tuple)) or len(views) != len(self.CAMERA_KEYS):
                raise ValueError(
                    "LeRobot Pi0.5 requires exactly three ordered camera views: "
                    "front, left_wrist, right_wrist"
                )
            for camera_index, image in enumerate(views):
                camera_batches[camera_index].append(
                    self._validate_image(
                        image,
                        batch_index=batch_index,
                        camera_index=camera_index,
                    )
                )

        if states is None:
            raise ValueError("states is required for this relative-action Pi0.5 checkpoint")
        state_array = np.asarray(states, dtype=np.float32)
        if batch_size == 1 and state_array.ndim == 1:
            state_array = state_array[None, :]
        if state_array.ndim == 3:
            state_array = state_array[:, -1, :]
        if state_array.ndim != 2 or state_array.shape != (batch_size, 7):
            raise ValueError(
                f"states must have shape [B, 7] or [B, T, 7]; received {state_array.shape}"
            )
        if not np.isfinite(state_array).all():
            raise ValueError("states contains non-finite values")

        batch: dict[str, Any] = {
            "observation.state": torch.from_numpy(np.ascontiguousarray(state_array)).to(self.device),
            "task": list(instructions),
            "robot_type": [""] * batch_size,
        }
        for key, images in zip(self.CAMERA_KEYS, camera_batches, strict=True):
            tensor = torch.from_numpy(np.stack(images, axis=0)).permute(0, 3, 1, 2).contiguous()
            batch[key] = tensor.to(device=self.device, dtype=torch.float32).div_(255.0)
        return batch

    def predict_action(
        self,
        batch_images: Any = None,
        instructions: Any = None,
        states: Any = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        batch = self.preprocessor(self._build_batch(batch_images, instructions, states))
        with self._torch.inference_mode():
            action_chunk = self.policy.predict_action_chunk(batch)
            actions = self.postprocessor(action_chunk)
        actions_np = actions.detach().cpu().float().numpy()
        return {
            # Keep the legacy key consumed by current AlphaBrain clients while
            # making the actual action-space semantics explicit.
            "normalized_actions": actions_np,
            "actions": actions_np,
            "action_space": "lerobot_postprocessed",
            "checkpoint_format": "lerobot",
        }


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    policy = LeRobotPI05ServerPolicy(args.ckpt_path, device=args.device)
    hostname = socket.gethostname()
    logging.info("Creating LeRobot Pi0.5 server (host: %s, ip: %s)", hostname, socket.gethostbyname(hostname))
    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata={
            "model": "Pi0.5",
            "checkpoint_family": "pi05",
            "checkpoint_format": "lerobot",
            "action_space": "lerobot_postprocessed",
            "camera_order": ["front", "left_wrist", "right_wrist"],
        },
        api_key_sha256=args.api_key_sha256,
        controller_api_key_sha256=args.controller_api_key_sha256,
        deployment_id=args.deployment_id,
    )
    server.serve_forever()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LeRobot Pi0.5 WebSocket policy server")
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--idle_timeout", type=int, default=1800)
    parser.add_argument("--api-key-sha256", default=None)
    parser.add_argument("--controller-api-key-sha256", default=None)
    parser.add_argument("--deployment-id", default=None)
    return parser


if __name__ == "__main__":
    main(build_argparser().parse_args())
