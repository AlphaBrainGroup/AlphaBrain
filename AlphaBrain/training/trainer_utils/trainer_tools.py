"""
metrics.py

Utility classes defining a Metrics container and multiple Trackers to enable model/stage-specific logging to various
endpoints (e.g., JSONL local logs, Weights & Biases).
"""

from typing import Tuple
import os
import re
import json
import sys
import numpy as np
import torch

from accelerate.logging import get_logger

# ── color helpers (auto-off when not a tty) ──────────────────────────────────
_USE_COLOR = sys.stdout.isatty()
def _c(t, *codes): return "\033[{}m{}\033[0m".format(";".join(map(str,codes)),t) if _USE_COLOR else str(t)
def _dim(t):         return _c(t, 2)
def _yellow(t):      return _c(t, 93)
def _cyan(t):        return _c(t, 96)
def _bold_green(t):  return _c(t, 1, 92)
def _bold_red(t):    return _c(t, 1, 91)
def _bold_yellow(t): return _c(t, 1, 93)
def _bold_cyan(t):   return _c(t, 1, 96)

logger = get_logger(__name__)


# === Define Tracker Interface ===
#

# utils/cli_parser.py


def normalize_dotlist_args(args):
    """
    Convert ['--x.y', 'val'] and ['--flag'] → ['x.y=val', 'flag=true']
    """
    normalized = []
    skip = False
    for i in range(len(args)):
        if skip:
            skip = False
            continue

        arg = args[i]
        if arg.startswith("--"):
            key = arg.lstrip("-")
            if "=" in key:
                normalized.append(key)
            elif i + 1 < len(args) and not args[i + 1].startswith("--"):
                normalized.append(f"{key}={args[i + 1]}")
                skip = True
            else:
                normalized.append(f"{key}=true")
        elif "=" in arg:
            # Bare dotlist format: key=value (Hydra/OmegaConf style)
            normalized.append(arg)
        else:
            pass  # skip orphaned values
    return normalized


def build_param_lr_groups(model, cfg):
    """
    build multiple param groups based on cfg.trainer.learning_rate.
    support specifying different learning rates for different modules, the rest use base.

    Args:
        vla: nn.Module model object
        cfg: config object, requires cfg.trainer.learning_rate dictionary

    Returns:
        List[Dict]: param_groups that can be used to build optimizer with torch.optim
    """

    lr_cfg = cfg.trainer.learning_rate
    base_lr = lr_cfg.get("base", 1e-4)  # default base learning rate

    freeze_modules = cfg.trainer.get("freeze_modules", "")
    if not isinstance(freeze_modules, str):
        freeze_modules = ""
    freeze_patterns = [p.strip() for p in freeze_modules.split(",") if p.strip()]

    used_params = set()
    frozen_params = set()
    param_groups = []

    for freeze_path in freeze_patterns:
        module = model
        try:
            for attr in freeze_path.split("."):
                module = getattr(module, attr)
            frozen_params.update(id(p) for p in module.parameters())
        except AttributeError:
            print(f"{_bold_yellow('[warn]')} freeze path not found: {_dim(str(freeze_path))}")
            continue

    for module_name, lr in lr_cfg.items():
        if module_name == "base":
            continue
        # try to find the module under vla by module_name (support nested paths)
        module = model
        try:
            for attr in module_name.split("."):
                module = getattr(module, attr)
            # filter out frozen parameters (config-based and requires_grad-based)
            params = [p for p in module.parameters() if id(p) not in frozen_params and p.requires_grad]
            if params:  # only add param group if there are trainable parameters
                param_groups.append({"params": params, "lr": lr, "name": module_name})
                used_params.update(id(p) for p in params)
        except AttributeError:
            ReferenceError(f"⚠️ module path `{module_name}` not found in vla")

    # assign base learning rate to the remaining unused parameters (exclude frozen ones)
    other_params = [p for p in model.parameters() if id(p) not in used_params and id(p) not in frozen_params and p.requires_grad]
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr, "name": "base"})

    return param_groups


import torch.distributed as dist


def only_main_process(func):
    """
    decorator: only run in main process (rank=0)
    """

    def wrapper(*args, **kwargs):
        if dist.is_initialized() and dist.get_rank() != 0:
            return None  # non-main process does not execute
        return func(*args, **kwargs)

    return wrapper


from torchvision.ops import box_iou
from PIL import Image


def resize_images(images, target_size=(224, 224)):
    """
    recursively resize all images in the nested list.

    :param images: nested list of images or single image.
    :param target_size: target size (width, height) after resizing.
    :return: resized images list, keeping the original nested structure.
    """
    if isinstance(images, np.ndarray):  # numpy array -> convert to PIL first
        return Image.fromarray(images).resize(target_size)
    if isinstance(images, Image.Image):  # if it is a single PIL image
        return images.resize(target_size)
    elif isinstance(images, list):  # if it is a list, recursively process each element
        return [resize_images(img, target_size) for img in images]
    else:
        raise ValueError("Unsupported image type or structure.")


class TrainerUtils:
    @staticmethod
    def freeze_backbones(model, freeze_modules=""):
        """
        directly freeze the specified submodules based on the relative module path list (patterns), no longer recursively find all submodule names:
          - patterns: read from config.trainer.freeze_modules, separated by commas to get the "relative path" list
            for example "qwen_vl_interface, action_model.net",
            it means to freeze model.qwen_vl_interface and model.action_model.net.

        Args:
            model: nn.Module model object
            freeze_modules: relative module path list (patterns)

        Returns:
            model: nn.Module model object
        return:
          - model:
        """
        frozen = []
        if freeze_modules:
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"🧊 freeze_modules: {freeze_modules}")
        if freeze_modules and type(freeze_modules) == str:
            # split and remove whitespace
            patterns = [p.strip() for p in freeze_modules.split(",") if p.strip()] if freeze_modules else []

            for path in patterns:
                # split the "relative path" by dots, for example "action_model.net" → ["action_model", "net"]
                attrs = path.split(".")
                module = model
                try:
                    for attr in attrs:
                        module = getattr(module, attr)
                    # if the module is successfully get, freeze it and its all submodule parameters
                    for param in module.parameters():
                        param.requires_grad = False
                    frozen.append(path)
                except AttributeError:
                    # if the attribute does not exist, skip and print warning
                    print(f"{_bold_yellow('[warn]')} module path not found, skipping freeze: {_yellow(path)}")
                    continue

        # accelerator.wait_for_everyone()  # synchronize when distributed training
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"{_cyan('[freeze]')} frozen: {_dim(str(frozen))}")
        return model

    @staticmethod
    def print_trainable_parameters(model):
        """
        print the total number of parameters and trainable parameters of the model
        :param model: PyTorch model instance
        """
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"\033[1;96m[model]\033[0m  "
            f"Total \033[93m{num_params / 10**6:.2f}M\033[0m  "
            f"Trainable \033[1;93m{num_trainable_params / 10**6:.2f}M\033[0m"
        )
        return num_params, num_trainable_params

    @staticmethod
    def load_pretrained_backbones(
        model,
        checkpoint_path=None,
        reload_modules=None,
        min_match_ratio: float = 0.5,
    ):
        """
        load checkpoint:
        - if reload_modules is set, load by path part
        - otherwise → load the entire model parameters (overwrite model)

        Args:
            min_match_ratio: warn loudly if matched / total model params falls below
                this fraction during full load. Default 0.5. Set to 0.0 to silence.

        return:
            replace, loaded_modules: list of module paths that successfully loaded parameters; if global load, then ["<full_model>"]
        """
        if not checkpoint_path:
            return []
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"{_cyan('[ckpt]')} loading {_dim(checkpoint_path)}")

        resolved_checkpoint_path = checkpoint_path
        if os.path.isdir(checkpoint_path):
            safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
            pt_path = os.path.join(checkpoint_path, "pytorch_model.pt")
            if os.path.exists(safetensors_path):
                resolved_checkpoint_path = safetensors_path
            elif os.path.exists(pt_path):
                resolved_checkpoint_path = pt_path
            else:
                raise RuntimeError(
                    f"{_bold_red('[error]')} checkpoint directory does not contain "
                    f"`model.safetensors` or `pytorch_model.pt`: {checkpoint_path}"
                )

        try:
            if _is_safetensors_path(resolved_checkpoint_path):
                from safetensors.torch import load_file

                sf_path = str(checkpoint_path)
                if os.path.isdir(sf_path):
                    sf_path = os.path.join(sf_path, "model.safetensors")
                checkpoint = load_file(sf_path)
            else:
                checkpoint = torch.load(resolved_checkpoint_path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"{_bold_red('[error]')} loading checkpoint failed: {e}")

        loaded_modules = []

        if reload_modules:  # partial load
            module_paths = [p.strip() for p in reload_modules.split(",") if p.strip()]
            for path in module_paths:
                reload_modules = path.split(".")
                module = model
                try:
                    for module_name in reload_modules:  # find the module to modify level by level
                        module = getattr(module, module_name)
                    prefix = path + "."
                    sub_state_dict = {k[len(prefix) :]: v for k, v in checkpoint.items() if k.startswith(prefix)}
                    if sub_state_dict:
                        module.load_state_dict(sub_state_dict, strict=True)
                        if not dist.is_initialized() or dist.get_rank() == 0:
                            print(f"{_bold_green('[ok]')} loaded module {_yellow(repr(path))}")
                        loaded_modules.append(path)
                    else:
                        print(f"{_bold_yellow('[warn]')} key not found in checkpoint: {_yellow(repr(path))}")
                except AttributeError:
                    print(f"{_bold_red('[error]')} module path not found: {_yellow(repr(path))}")
        else:  # full load
            try:
                # Filter out shape-mismatched keys (e.g. action_dim 32→7)
                model_state = model.state_dict()
                filtered_checkpoint = {}
                skipped_keys = []
                for k, v in checkpoint.items():
                    if k in model_state and model_state[k].shape != v.shape:
                        skipped_keys.append(f"{k}: ckpt {tuple(v.shape)} vs model {tuple(model_state[k].shape)}")
                    else:
                        filtered_checkpoint[k] = v
                is_main = not dist.is_initialized() or dist.get_rank() == 0
                if skipped_keys and is_main:
                    print(f"{_bold_yellow('[warn]')} skipped {len(skipped_keys)} shape-mismatched keys:")
                    for sk in skipped_keys:
                        print(f"  {_dim(sk)}")
                incompatible = model.load_state_dict(filtered_checkpoint, strict=False)
                missing_keys = list(getattr(incompatible, "missing_keys", []))
                unexpected_keys = list(getattr(incompatible, "unexpected_keys", []))
                total = len(model_state)
                matched = total - len(missing_keys)
                ratio = matched / total if total else 0.0

                if is_main:
                    if ratio >= 0.95:
                        tag, color = "[ok]", _bold_green
                    elif ratio >= min_match_ratio:
                        tag, color = "[partial]", _bold_yellow
                    else:
                        tag, color = "[low-coverage]", _bold_red
                    print(
                        f"{color(tag)} loaded {_bold_cyan('<full_model>')} "
                        f"matched={matched}/{total} ({ratio*100:.1f}%)  "
                        f"missing={len(missing_keys)}  unexpected={len(unexpected_keys)}  "
                        f"shape_skipped={len(skipped_keys)}"
                    )
                    if ratio < min_match_ratio:
                        print(
                            f"{_bold_red('[low-coverage]')} matched ratio {ratio*100:.1f}% < "
                            f"min_match_ratio {min_match_ratio*100:.1f}%. "
                            f"Most of the model is still randomly initialized — "
                            f"check checkpoint key naming. For openpi/lerobot π₀ checkpoints, "
                            f"use AlphaBrain.model.modules.action_model.pi0_flow_matching_head."
                            f"weight_bridge.load_pi0_weights instead."
                        )
                        sample_missing = missing_keys[:5]
                        sample_unexpected = unexpected_keys[:5]
                        if sample_missing:
                            print(f"  {_dim('sample missing:')} {sample_missing}")
                        if sample_unexpected:
                            print(f"  {_dim('sample unexpected:')} {sample_unexpected}")
                loaded_modules = ["<full_model>"]
            except Exception as e:
                raise RuntimeError(f"{_bold_red('[error]')} loading full model failed: {e}")
        return model

    @staticmethod
    def print_freeze_status(model):
        """
        print the freezing status of each parameter in the model
        :param model: PyTorch model instance
        """
        for name, param in model.named_parameters():
            status = "Frozen" if not param.requires_grad else "Trainable"
            print(f"{name:60s}  |  {status}")

    @staticmethod
    def setup_distributed_training(accelerator, *components):
        """
        use Accelerator to prepare distributed training components
        :param accelerator: Accelerate instance
        :param components: any number of components (such as model, optimizer, dataloader, etc.)
        :return: prepared distributed components (in the same order as input)
        """

        # use accelerator.prepare method to wrap components
        prepared_components = accelerator.prepare(*components)
        
        # For DDP with parameter reuse (e.g. shared attention in PaliGemmaOFT),
        # set static_graph to allow parameters being used multiple times
        for comp in (prepared_components if isinstance(prepared_components, tuple) else [prepared_components]):
            if hasattr(comp, "module") and hasattr(comp, "_set_static_graph"):
                comp._set_static_graph()
        
        return prepared_components

    @staticmethod
    def euclidean_distance(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        return np.linalg.norm(predicted - ground_truth)

    @staticmethod
    def _reset_dataloader(dataloader, epoch_counter):
        """safe reset dataloader iterator"""
        # 1. update epoch counter
        epoch_counter += 1

        # 2. set new epoch (distributed core)
        if hasattr(dataloader, "sampler") and callable(getattr(dataloader.sampler, "set_epoch", None)):
            dataloader.sampler.set_epoch(epoch_counter)

        # 3. create new iterator
        return iter(dataloader), epoch_counter

    @staticmethod
    def compute_grad_angle_with_stats(grads_a: list[torch.Tensor], grads_v: list[torch.Tensor]) -> Tuple[float, float]:
        """
        compute the cosine angle between two groups of gradient vectors (degrees), and calculate the average angle and variance.
        grads_a, grads_v: gradient Tensor list corresponding to the same parameter list interface_params
        return:
            mean_angle_deg: average angle (degrees)
            angle_variance: angle variance
        """
        angle_degs = []

        # compute the cosine angle between each gradient block grads_a[0].shape = 1280, 3, 14, 14
        # grads_1 = grads_a[0][0]  # [3, 14, 14]
        # grads_2 = grads_v[0][0]
        # grads_a = grads_1.view(-1, 3)  # reshape to [196, 3]
        # grads_v = grads_2.view(-1, 3)

        # lang linear
        # reshape to 14*14, 3
        # layer
        grads_action = grads_a[0]  # [2048, 11008]
        grads_action = grads_action[
            :32, :7
        ]  # only take the first 7 elements, avoid cosim failure in high-dimensional space
        grads_vl = grads_v[0]  # [2048, 11008]
        grads_vl = grads_vl[
            :32, :7
        ]  # only take the first 32 elements, 7 dimensions, avoid cosim failure in high-dimensional space
        for g_a, g_v in zip(grads_action, grads_vl):
            dot = torch.sum(g_a * g_v)
            norm_a_sq = torch.sum(g_a * g_a)
            norm_v_sq = torch.sum(g_v * g_v)

            # avoid division by zero
            norm_a = torch.sqrt(norm_a_sq + 1e-16)
            norm_v = torch.sqrt(norm_v_sq + 1e-16)

            cos_sim = (dot / (norm_a * norm_v)).clamp(-1.0, 1.0)
            angle_rad = torch.acos(cos_sim)
            angle_deg = angle_rad * (180.0 / torch.pi)

            angle_degs.append(angle_deg.item())

        # compute the average angle and variance
        angle_degs_tensor = torch.tensor(angle_degs)
        mean_angle_deg = torch.mean(angle_degs_tensor).item()
        angle_variance = torch.sqrt(torch.var(angle_degs_tensor)).item()
        # accelerator.wait_for_everyone()
        return mean_angle_deg, angle_variance

    @staticmethod
    def pcgrad_project(grads_a: list[torch.Tensor], grads_v: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        apply PCGrad projection to the second group of gradients grads_v, suppress negative transfer between grads_a and grads_v
        if the dot product of two groups of gradients < 0, then:
            grads_v <- grads_v - (dot / ||grads_a||^2) * grads_a
        return the new grads_v list
        """
        # first compute dot and ||grads_a||^2
        dot, norm_a_sq = 0.0, 0.0
        for g_a, g_v in zip(grads_a, grads_v):
            dot += torch.sum(g_a * g_v)
            norm_a_sq += torch.sum(g_a * g_a)

        if dot < 0:
            coeff = dot / (norm_a_sq + 1e-6)
            # projection
            grads_v = [g_v - coeff * g_a for g_a, g_v in zip(grads_a, grads_v)]

        return grads_v

    @staticmethod
    def eval_qwenpi(qwenpi, dataloader, num_batches=20):
        """
        evaluate QwenQFormerDiT model, compute IoU and action distance.

        Args:
            qwenpi: QwenQFormerDiT model instance.
            dataloader: data loader.
            num_batches: number of batches to evaluate.

        Returns:
            dict: contains IoU and action distance evaluation results.
        """
        iou_scores = []
        action_distances = []
        count = 0

        dataset_iter = iter(dataloader)
        while count < num_batches:
            try:
                batch_samples = next(dataset_iter)
                count += 1
            except StopIteration:
                break

            # extract data
            images = [example["image"] for example in batch_samples]
            instructions = [example["lang"] for example in batch_samples]
            actions = [example["action"] for example in batch_samples]
            solutions = [example["solution"] for example in batch_samples]

            # model prediction
            predicted_solutions, normalized_actions = qwenpi.predict_action_withCoT(
                images=images, instructions=instructions, use_ddim=False, num_ddim_steps=20
            )

            # extract and convert predicted results
            parsed_solutions = []
            for solution in predicted_solutions:
                parsed_solution = TrainerUtils.extract_json_from_string(solution)
                parsed_solutions.append(parsed_solution)

            # compute IoU
            for pred_dict, gt_dict in zip(parsed_solutions, solutions):
                pred_pick_bbox = torch.tensor(pred_dict["pick"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
                gt_pick_bbox = torch.tensor(gt_dict["pick"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
                pred_place_bbox = torch.tensor(pred_dict["place"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
                gt_place_bbox = torch.tensor(gt_dict["place"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)

                pick_iou = box_iou(pred_pick_bbox, gt_pick_bbox).item()
                place_iou = box_iou(pred_place_bbox, gt_place_bbox).item()

                iou_scores.append({"pick_iou": pick_iou, "place_iou": place_iou})

            # compute action distance
            actions = np.array(actions)  # convert to numpy array
            num_elements = np.prod(actions.shape)  # B*len*dim
            action_distance = TrainerUtils.euclidean_distance(normalized_actions, actions)
            average_action_distance = action_distance / num_elements
            action_distances.append(average_action_distance)

        # summarize results
        avg_action_distance = np.mean(action_distances)
        return {"iou_scores": iou_scores, "average_action_distance": avg_action_distance}

    @staticmethod
    def extract_json_from_string(input_string):
        """
        extract valid JSON part from string and convert to dictionary.

        Args:
            input_string (str): string containing extra characters.

        Returns:
            dict: dictionary extracted and parsed.
        """
        json_match = re.search(r"{.*}", input_string, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"JSON decode failed: {e}")
                return None
        else:
            print("No valid JSON part found")
            return None

    def _get_latest_checkpoint(self, checkpoint_dir):
        """Find the latest checkpoint in the directory based on step number."""
        if not os.path.exists(checkpoint_dir):
            self.accelerator.print(f"No checkpoint directory found at {checkpoint_dir}")
            return None, 0

        # # origin0309: 原始查找方式，仅支持文件格式
        # checkpoints = [
        #     f for f in os.listdir(checkpoint_dir)
        #     if re.match(r"steps_(\d+)_(?:pytorch_model\.pt|model\.safetensors)$", f)
        #     and os.path.isfile(os.path.join(checkpoint_dir, f))  # 确保是文件
        # ]

        # lpt0309: 同时支持文件格式和目录格式的checkpoint
        checkpoints = []
        for f in os.listdir(checkpoint_dir):
            full_path = os.path.join(checkpoint_dir, f)
            # lpt0309: 支持新的目录格式 steps_N/
            if os.path.isdir(full_path) and re.match(r"steps_(\d+)$", f):
                # 检查目录内是否有权重文件
                if (os.path.exists(os.path.join(full_path, "model.safetensors")) or
                    os.path.exists(os.path.join(full_path, "pytorch_model.pt"))):
                    checkpoints.append(f)
            # origin0309: 支持旧的文件格式 steps_N_model.safetensors
            elif os.path.isfile(full_path) and re.match(r"steps_(\d+)_(?:pytorch_model\.pt|model\.safetensors)$", f):
                checkpoints.append(f)

        if not checkpoints:
            self.accelerator.print(f"No checkpoints found in {checkpoint_dir}")
            return None, 0

        # 提取步数并排序
        try:
            checkpoints_with_steps = []
            for ckpt in checkpoints:
                # lpt0309: 支持两种格式的步数提取
                match = re.search(r"steps_(\d+)", ckpt)
                if match:
                    checkpoints_with_steps.append((ckpt, int(match.group(1))))
        except AttributeError as e:
            self.accelerator.print(f"Error parsing checkpoint filenames: {e}")
            return None, 0

        # 按步数排序，获取最新的 checkpoint
        checkpoints_with_steps.sort(key=lambda x: x[1])
        latest_checkpoint, completed_steps = checkpoints_with_steps[-1]

        latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        self.accelerator.print(f"Latest checkpoint found: {latest_checkpoint_path}")
        return latest_checkpoint_path, completed_steps


def is_main_process():
    rank = int(os.environ.get("RANK", 0))  # if RANK is not set, default to 0
    return rank == 0


def _is_safetensors_path(path):
    """Check if a path refers to a safetensors file or a directory containing one."""
    path = str(path)
    if path.endswith(".safetensors"):
        return True
    # Support checkpoint directories containing model.safetensors
    if os.path.isdir(path) and os.path.exists(os.path.join(path, "model.safetensors")):
        return True
    return False
