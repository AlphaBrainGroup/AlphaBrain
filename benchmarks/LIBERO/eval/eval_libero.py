import sys, os
# Allow running from the libero conda env where these repos may not be installed as packages
# Override via VLA_EXTRA_SYSPATH env var (colon-separated), e.g. "/path/to/LIBERO:/path/to/AlphaBrain"
for _p in [p for p in os.environ.get("VLA_EXTRA_SYSPATH", "").split(":") if p]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dataclasses
import datetime as dt
import json
import logging
import math
import os
import pathlib
from pathlib import Path
import requests
import time

import imageio
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from benchmarks.LIBERO.model2libero_interface import M1Inference
from typing import Union


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

# ANSI palette for log messages. NO_COLOR env var disables (per no-color.org).
# Emitted unconditionally otherwise so `tail -f eval.log` in a terminal renders
# them as colors; text editors show raw escapes (acceptable for ephemeral logs).
if os.environ.get("NO_COLOR"):
    _C0 = _CB = _CD = _CG = _CR = _CY = _CC = _CV = _CH = ""
else:
    _C0 = "\033[0m";          _CB = "\033[1m";          _CD = "\033[2m"
    _CG = "\033[1;38;5;46m"   # bold green   (success)
    _CR = "\033[1;38;5;196m"  # bold red     (failure / very low SR)
    _CY = "\033[1;38;5;220m"  # bold yellow  (mid SR)
    _CC = "\033[1;38;5;45m"   # bold sky     (task / episode marker)
    _CV = "\033[38;5;228m"    # light yellow (value)
    _CH = "\033[1;38;5;214m"  # bold orange  (highlight)

def _sr_color(sr: float) -> str:
    """Pick a color tier by SR value: green ≥0.45, yellow 0.25–0.45, red <0.25."""
    if sr >= 0.45: return _CG
    if sr >= 0.25: return _CY
    return _CR

def _binarize_gripper_open(open_val: Union[np.ndarray, float]) -> np.ndarray:
    """Convert gripper value to LIBERO action space {-1, 1}.

    After unnormalization, gripper is in [0, 1] range:
      - 0.0 = fully closed, 1.0 = fully open (dataset convention)
      - LIBERO robosuite expects: -1 = open, +1 = close (INVERTED!)

    Threshold at 0.5:
      > 0.5 (open in dataset)  → -1.0 (open in robosuite)
      ≤ 0.5 (close in dataset) → +1.0 (close in robosuite)
    """
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * float(v > 0.5)  # 0→+1(close), 1→-1(open)
    return np.asarray([bin_val], dtype=np.float32)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093
    resize_size = [224,224]

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 10  # Number of rollouts per task


    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "experiments/libero/logs"  # Path to save videos
    predict_video: bool = False  # Output side-by-side predicted vs actual video

    seed: int = 7  # Random Seed (for reproducibility)

    pretrained_path: str = ""

    post_process_action: bool = True

    job_name: str = "test"

    num_views: int = 2  # Number of camera views to send (1=primary only, 2=primary+wrist)

    norm_mode: str = "q99"  # "q99" → q01/q99 percentile (VLA default); "min_max" → absolute min/max (ACT)



def eval_libero(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")

    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    # args.video_out_path = f"{date_base}+{args.job_name}"
    
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps #TODO: debug
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    # client = websocket_policy_client.WebsocketClientPolicy(args.host, args.port)

    model = M1Inference(
        policy_ckpt_path=args.pretrained_path, # to get unnormalization stats
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
    )


    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\n{_CC}Task:{_C0} {_CV}{task_description}{_C0}")

            # Reset environment
            model.reset(task_description=task_description)  # Reset the client connection
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            predicted_images = []
            full_actions = []
            # --- 新增：初始化 states 队列 ---
            n = 16  # 可根据需要调整 n
            state = np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
            )
            states = np.tile(state, (n, 1))  # shape (n, 8)
            # --- 新增结束 ---

            logging.info(f"{_CC}Starting episode {_CH}{task_episodes + 1}{_C0}{_CC}...{_C0}")
            step = 0
            
            # full_actions = np.load("./debug/action.npy")

            while t < max_steps + args.num_steps_wait:
                # try:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # IMPORTANT: rotate 180 degrees to match train preprocessing
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(
                    obs["robot0_eye_in_hand_image"][::-1, ::-1]
                )

                # Save preprocessed image for replay video
                replay_images.append(img)

                state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                )

                # --- 新增：更新 states 队列 ---
                states = np.vstack([states[1:], state])  # 保持最新的 n 个 state
                # --- 新增结束 ---

                observation = { # key 要和 和模型API对齐
                    "observation.primary": np.expand_dims(
                        img, axis=0
                    ),  # (H, W, C), dtype=unit8, range(0-255)
                    "observation.wrist_image": np.expand_dims(
                        wrist_img, axis=0
                    ),  # (H, W, C)
                    "observation.states": np.expand_dims(states, axis=0),
                    "instruction": [str(task_description)],
                }

                # align key with model API
                obs_input = {
                "images": [observation["observation.primary"][0]] + (
                    [observation["observation.wrist_image"][0]] if args.num_views >= 2 else []
                ),
                "states": observation["observation.states"].astype(np.float32),  # shape (n, 8)
                "task_description": observation["instruction"][0],
                "return_predicted_frame": args.predict_video,
                }

                
                start_time = time.time()
                
                response = model.step(**obs_input) 
                
                if args.predict_video and "predicted_frame" in response:
                    predicted_images.append(response["predicted_frame"][0])
                
                end_time = time.time()
                # print(f"time: {end_time - start_time}")
                
                # # 
                raw_action = response["raw_action"]
                
                world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
                rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
                open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
                gripper = _binarize_gripper_open(open_gripper)
                if step < 3 or step % 50 == 0:
                    logging.info(f"[GRIP DEBUG] step={step} open_gripper={open_gripper} → binarized={gripper}")

                if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                    logging.warning(f"Unexpected action sizes: "
                                    f"wv={world_vector_delta.shape}, rot={rotation_delta.shape}, grip={gripper.shape}. "
                                    f"Falling back to LIBERO_DUMMY_ACTION.")
                    raise ValueError(
                        f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                        f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
                    )
                else:
                    delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)

                full_actions.append(delta_action)
                
                # __import__("ipdb").set_trace()
                # see ../robosuite/controllers/controller_factory.py
                obs, reward, done, info = env.step(delta_action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1
                step += 1

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            if not args.predict_video:
                # Normal mode: output actual video only
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path)
                    / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

            if args.predict_video and len(predicted_images) > 0:
                import cv2
                sbs_frames = []
                caption = f"Task: {task_description}"

                # Sync: expand predicted frames to match actual frames
                # predicted_images has fewer frames (only when model is called)
                # replay_images has every timestep
                # Repeat each predicted frame to fill the gap until next prediction
                chunk_size = max(1, (len(replay_images) - 1) // max(1, len(predicted_images)))
                synced_preds = []
                for pi, pred in enumerate(predicted_images):
                    start = pi * chunk_size
                    end = min(start + chunk_size, len(replay_images) - 1)
                    for fi in range(start, end):
                        synced_preds.append((pred, fi))
                # Fill remaining frames with last prediction
                if synced_preds and synced_preds[-1][1] < len(replay_images) - 2:
                    last_p = predicted_images[-1]
                    for fi in range(synced_preds[-1][1] + 1, len(replay_images) - 1):
                        synced_preds.append((last_p, fi))

                target_size = 384  # larger render size
                label_h = 48  # height for label bar
                gap_w = 16  # gap between two videos
                caption_h = 44  # height for caption bar
                pad = 16  # padding around videos

                # Use PIL for clean font rendering
                from PIL import Image as PILImage, ImageDraw, ImageFont
                label_font = None
                caption_font = None
                # Try multiple font paths
                for fp in [
                    "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
                    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
                    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
                ]:
                    if os.path.exists(fp):
                        label_font = ImageFont.truetype(fp, 36)
                        caption_font = ImageFont.truetype(fp, 24)
                        break
                if label_font is None:
                    label_font = ImageFont.load_default(size=36)
                    caption_font = ImageFont.load_default(size=24)

                for pred_img, fi in synced_preds:
                    pred = np.asarray(pred_img)
                    actual = np.asarray(replay_images[fi + 1])

                    pred = cv2.resize(pred, (target_size, target_size))
                    actual = cv2.resize(actual, (target_size, target_size))

                    total_w = pad + target_size + gap_w + target_size + pad
                    total_h = pad + label_h + target_size + caption_h + pad

                    # White background
                    pil_frame = PILImage.new("RGB", (total_w, total_h), (255, 255, 255))
                    draw = ImageDraw.Draw(pil_frame)

                    # Place videos
                    x_pred = pad
                    x_actual = pad + target_size + gap_w
                    y_img = pad + label_h

                    pil_frame.paste(PILImage.fromarray(pred), (x_pred, y_img))
                    pil_frame.paste(PILImage.fromarray(actual), (x_actual, y_img))

                    # Labels centered above each video (black text)
                    pred_label = "Predicted"
                    actual_label = "Actual"
                    pred_bbox = draw.textbbox((0, 0), pred_label, font=label_font)
                    pred_tw = pred_bbox[2] - pred_bbox[0]
                    draw.text((x_pred + (target_size - pred_tw) // 2, pad + 4), pred_label, fill=(0, 0, 0), font=label_font)

                    actual_bbox = draw.textbbox((0, 0), actual_label, font=label_font)
                    actual_tw = actual_bbox[2] - actual_bbox[0]
                    draw.text((x_actual + (target_size - actual_tw) // 2, pad + 4), actual_label, fill=(0, 0, 0), font=label_font)

                    # Caption centered at bottom (black text)
                    cap_text = caption[:90]
                    cap_bbox = draw.textbbox((0, 0), cap_text, font=caption_font)
                    cap_tw = cap_bbox[2] - cap_bbox[0]
                    draw.text(((total_w - cap_tw) // 2, y_img + target_size + 6), cap_text, fill=(80, 80, 80), font=caption_font)

                    sbs_frames.append(np.array(pil_frame))

                if sbs_frames:
                    imageio.mimwrite(
                        pathlib.Path(args.video_out_path)
                        / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4",
                        sbs_frames, fps=10,
                    )
            
            full_actions = np.stack(full_actions)
            # np.save(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.npy", full_actions)
            
            # print(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4")
            # Log current results
            _succ_col = _CG if done else _CR
            _succ_sym = "✓" if done else "✗"
            logging.info(f"Success: {_succ_col}{_succ_sym} {done}{_C0}")
            logging.info(f"{_CD}# episodes completed so far: {total_episodes}{_C0}")
            _running_pct = total_successes / total_episodes
            _r_col = _sr_color(_running_pct)
            logging.info(
                f"{_CD}# successes: {_C0}{_r_col}{total_successes}{_C0} "
                f"({_r_col}{total_successes / total_episodes * 100:.1f}%{_C0})"
            )

        # Log final results
        _task_sr = float(task_successes) / float(task_episodes)
        _total_sr = float(total_successes) / float(total_episodes)
        logging.info(
            f"{_CB}Current task success rate:{_C0}  {_sr_color(_task_sr)}"
            f"{_task_sr:.2f}{_C0} {_CD}({_task_sr*100:.1f}%){_C0}"
        )
        logging.info(
            f"{_CB}Current total success rate:{_C0} {_sr_color(_total_sr)}"
            f"{_total_sr:.2f}{_C0} {_CD}({_total_sr*100:.1f}%){_C0}"
        )

        # Explicitly close the environment to avoid EGL cleanup errors during GC
        env.close()

    _final_sr = float(total_successes) / float(total_episodes)
    _final_col = _sr_color(_final_sr)
    logging.info(
        f"{_CB}{'━' * 60}{_C0}"
    )
    logging.info(
        f"{_CB}Total success rate: {_final_col}{_final_sr:.2f}{_C0} "
        f"{_CB}({_final_col}{_final_sr*100:.1f}%{_C0}{_CB}){_C0}"
    )
    logging.info(f"{_CD}Total episodes: {total_episodes}{_C0}")
    logging.info(
        f"{_CB}{'━' * 60}{_C0}"
    )


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(
        seed
    )  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    tyro.cli(eval_libero)