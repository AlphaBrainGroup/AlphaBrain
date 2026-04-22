from collections import deque
from typing import Optional, Sequence
import os
import logging
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

from benchmarks.LIBERO.eval.auto_eval_scripts.adaptive_ensemble import AdaptiveEnsembler
from typing import Dict
import numpy as np
from pathlib import Path

from typing import List
from typing import Tuple, Dict

from AlphaBrain.model.framework.config_utils import read_mode_config



class M1Inference:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "franka",
        horizon: int = 0,
        action_ensemble = True,
        action_ensemble_horizon: Optional[int] = None,  # None = auto from action_horizon
        image_size: List[int] = [224, 224],
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha = 0.1,
        host="0.0.0.0",
        port=10095,
    ) -> None:
        
        # build client to connect server policy
        self.client = WebsocketClientPolicy(host, port)
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.horizon = horizon #0
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)

        self.action_norm_stats = self.get_action_stats(self.unnorm_key, policy_ckpt_path=policy_ckpt_path)
        self.action_chunk_size = self.get_action_chunk_size(policy_ckpt_path=policy_ckpt_path)

        # Auto-detect action_ensemble_horizon from checkpoint if not specified
        if self.action_ensemble_horizon is None:
            self.action_ensemble_horizon = 3  # original default
            logging.info(f"[M1Inference] Using default action_ensemble_horizon=3")

        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0
        # Detect gripper semantics: if gripper dim is unmasked (mask=False),
        # the model uses inverted gripper convention (high=close, low=open)
        mask = self.action_norm_stats.get("mask", None)
        self.gripper_inverted = (mask is not None and len(mask) >= 7 and not mask[6])
        self.cached_raw_actions = None
        self.step_counter = 0

        # Detect Pi0/Pi0.5 flow matching models that do MEAN_STD unnorm internally.
        # These models return already-unnormalized actions despite the key name
        # "normalized_actions"; skip client-side unnormalization to avoid double-unnorm.
        self.skip_client_unnorm = False
        self.invert_gripper_after_unnorm = False  # Pi0/Pi0.5 without internal norm needs gripper invert
        try:
            from AlphaBrain.model.framework.config_utils import read_mode_config
            _mc, _ = read_mode_config(Path(policy_ckpt_path))
            _fw_name = _mc.get('framework', {}).get('name', '')
            _norm_cfg = _mc.get('framework', {}).get('normalization', {})
            if _fw_name in ('PaliGemmaPi0', 'Pi0OFT', 'LlamaPi0') and _norm_cfg.get('enabled', False):
                self.skip_client_unnorm = True
                logging.info(f"[M1Inference] Detected {_fw_name} with internal MEAN_STD norm → skipping client unnorm")
            elif _fw_name in ('PaliGemmaPi0', 'Pi0OFT', 'LlamaPi0') and not _norm_cfg.get('enabled', False):
                self.invert_gripper_after_unnorm = True
                logging.info(f"[M1Inference] Detected {_fw_name} without internal norm → will invert gripper after unnorm")
        except Exception as e:
            logging.warning(f"[M1Inference] Could not detect Pi0 norm mode: {e}")


    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.cached_raw_actions = None
        self.step_counter = 0


    def step(
        self,
        images,
        task_description: Optional[str] = None,
        **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        执行一步推理（带 Action Chunking）
        每 action_chunk_size 步调用一次模型，中间步使用缓存的动作序列。
        :param images: 输入图像列表
        :param task_description: 任务描述文本
        :return: (原始动作字典)
        """

        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        # 仅在缓存用完或首次调用时进行推理
        if self.cached_raw_actions is None or self.step_counter % len(self.cached_raw_actions) == 0:
            images = [self._resize_image(image) for image in images]
            vla_input = {
                "batch_images": [images],
                "instructions": [self.task_description],
                "unnorm_key": self.unnorm_key,
                "states": kwargs.get("states", None),
                "do_sample": False,
                "use_ddim": self.use_ddim,
                "num_ddim_steps": self.num_ddim_steps,
                "return_predicted_frame": kwargs.get("return_predicted_frame", False),
            }

            response = self.client.infer(vla_input)

            # Check for server-side inference errors
            if response.get("status") == "error" or "data" not in response:
                error_msg = response.get("error", {}).get("message", "Unknown server error")
                raise RuntimeError(f"Policy server inference failed: {error_msg}")

            # unnormalize the action
            normalized_actions = np.array(response["data"]["normalized_actions"])
            # Handle both (B, chunk, D) and (chunk, D) shapes
            if normalized_actions.ndim == 3:
                normalized_actions = normalized_actions[0]  # remove batch dim
            elif normalized_actions.ndim == 1:
                normalized_actions = normalized_actions[np.newaxis, :]  # (D,) -> (1, D)
            # [DEBUG gripper] log raw gripper dim before any transform
            try:
                import os as _os
                _dbg_path = _os.environ.get("GRIPPER_DEBUG_LOG", "")
                if _dbg_path:
                    with open(_dbg_path, "a") as _f:
                        _g = normalized_actions[:, 6]
                        _f.write(f"grip_raw: min={_g.min():+.4f} max={_g.max():+.4f} mean={_g.mean():+.4f} vals={_g.tolist()}\n")
            except Exception:
                pass
            if self.skip_client_unnorm:
                # Pi0 models already unnormalized actions internally (MEAN_STD);
                # use raw output directly without clip/unnorm.
                # Invert gripper (dim 6): Pi0 convention may be opposite to LIBERO
                normalized_actions[:, 6] = -normalized_actions[:, 6]
                self.cached_raw_actions = normalized_actions
            else:
                self.cached_raw_actions = self.unnormalize_actions(
                    normalized_actions=normalized_actions, action_norm_stats=self.action_norm_stats
                )
                if self.invert_gripper_after_unnorm:
                    self.cached_raw_actions[:, 6] = 1.0 - self.cached_raw_actions[:, 6]
            # Cache predicted frame if server sent one (WM backbones only)
            if "predicted_frame" in response.get("data", {}):
                self._last_predicted_frame = response["data"]["predicted_frame"]
            else:
                self._last_predicted_frame = None

        # 从缓存中取出当前 chunk 内的动作
        idx = self.step_counter % len(self.cached_raw_actions)
        raw_actions = self.cached_raw_actions[idx:idx+1]

        # Gripper from unnormalized actions (range [0,1] for mask=True models)
        gripper_val = raw_actions[0, 6]

        self.step_counter += 1

        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array([gripper_val]),  # range [0, 1]; 1 = open; 0 = close
        }

        result = {"raw_action": raw_action}
        if getattr(self, "_last_predicted_frame", None) is not None:
            result["predicted_frame"] = self._last_predicted_frame
            self._last_predicted_frame = None
        return result

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        # Auto-detect normalization mode from stats:
        # If training used min_max normalization, unnormalize with min/max.
        # If training used q99 normalization, unnormalize with q01/q99.
        norm_mode = action_norm_stats.get("norm_mode", "q99")
        if norm_mode == "min_max":
            ref_key_high, ref_key_low = "max", "min"
        else:
            ref_key_high, ref_key_low = "q99", "q01"
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats[ref_key_low], dtype=bool))
        action_high = np.array(action_norm_stats[ref_key_high])
        action_low = np.array(action_norm_stats[ref_key_low])
        normalized_actions = np.clip(normalized_actions, -1, 1)

        if mask[6]:  # PaliGemmaOFT: gripper is masked (unnormed), don't binarize here
            pass
        else:  # LlamaOFT: gripper is unmasked, binarize before unnorm (legacy behavior)
            normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1)

        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions

    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        """
        Duplicate stats accessor (retained for backward compatibility).
        """
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)  # read config and norm_stats

        unnorm_key = M1Inference._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["action"]

    @staticmethod
    def get_action_chunk_size(policy_ckpt_path) -> int:
        """
        从 config.yaml 中读取 action chunk size = future_action_window_size + 1
        """
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, _ = read_mode_config(policy_ckpt_path)
        return model_config['framework']['action_model']['future_action_window_size'] + 1



    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)
    
    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        """
        Duplicate helper (retained for backward compatibility).
        See primary _check_unnorm_key above.
        """
        if len(norm_stats) == 0:
            # Models with internal normalization (Pi0/LlamaPi0) may have no
            # dataset_statistics.json.  Return a sentinel key so downstream
            # code can proceed; the stats will be ignored when
            # skip_client_unnorm is True.
            _dummy_key = "__dummy__"
            norm_stats[_dummy_key] = {
                "action": {
                    "min": [0.0] * 7,
                    "max": [1.0] * 7,
                    "mean": [0.0] * 7,
                    "std": [1.0] * 7,
                    "q01": [0.0] * 7,
                    "q99": [1.0] * 7,
                    "mask": [True] * 7,
                }
            }
            logging.warning("[M1Inference] norm_stats empty, using dummy stats (model likely does internal normalization)")
            return _dummy_key
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key