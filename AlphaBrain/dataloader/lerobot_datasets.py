# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Fangjing Wang/ SUST University] in [2025]. 
# Modification: [return raw data and suport multi-dataset mixture].
# Modified by [Jinhui YE/ HKUST University] in [2025]. 
# Modification: [suport topdowm processing, suport param from config].

from pathlib import Path
from typing import Sequence
from omegaconf import OmegaConf
import glob

from AlphaBrain.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, LeRobotMixtureDataset
from AlphaBrain.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from AlphaBrain.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from AlphaBrain.dataloader.gr00t_lerobot.embodiment_tags import ROBOT_TYPE_TO_EMBODIMENT_TAG, EmbodimentTag

def collate_fn(batch):
    return batch

def make_LeRobotSingleDataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,
    delete_pause_frame: bool = False,
    data_cfg: dict | None = None,
) -> LeRobotSingleDataset:
    """
    Make a LeRobotSingleDataset object.

    :param data_root_dir: The root directory of the dataset.
    :param data_name: The name of the dataset.
    :param robot_type: The robot type config to use.
    :param crop_obs_camera: Whether to crop the observation camera images.
    :return: A LeRobotSingleDataset object.
    """
    
    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    modality_config = data_config.modality_config()
    transforms = data_config.transform()
    data_root_dir = Path(data_root_dir)
    dataset_path = Path(data_name) if Path(data_name).is_absolute() else data_root_dir / data_name
    if robot_type not in ROBOT_TYPE_TO_EMBODIMENT_TAG:
        print(f"Warning: Robot type {robot_type} not found in ROBOT_TYPE_TO_EMBODIMENT_TAG, using {EmbodimentTag.NEW_EMBODIMENT} as default")
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    else:
        embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type]
    
    video_backend = data_cfg.get("video_backend", "decord") if data_cfg else "torchvision_av"
    dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=video_backend, # decord is more efficiency | torchvision_av for video.av1
        delete_pause_frame=delete_pause_frame,
        data_cfg=data_cfg,
    )
    # Keep mixture members unique even when many datasets end with the same leaf directory name, e.g. `lerobot`.
    dataset._dataset_name = str(data_name)
    return dataset


def _expand_mixture_entry(data_root_dir: Path, data_name: str, robot_type: str) -> list[tuple[str, str]]:
    pattern_chars = set("*?[]")
    if not any(ch in data_name for ch in pattern_chars):
        return [(data_name, robot_type)]

    matches = sorted(glob.glob(str(data_root_dir / data_name)))
    expanded = []
    for match in matches:
        match_path = Path(match)
        if match_path.is_dir():
            expanded.append((str(match_path.relative_to(data_root_dir)), robot_type))
    return expanded

def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    **kwargs: dict,
) -> LeRobotMixtureDataset:
    """
    Get a LeRobotMixtureDataset object.
    """
    data_root_dir = data_cfg.data_root_dir
    dataset_mix = data_cfg.dataset_mix
    delete_pause_frame = data_cfg.get("delete_pause_frame", False)
    mixture_spec = DATASET_NAMED_MIXTURES[dataset_mix]
    data_root_dir = Path(data_root_dir)
    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight, robot_type in mixture_spec:
        expanded_entries = _expand_mixture_entry(data_root_dir, d_name, robot_type)
        if not expanded_entries:
            print(f"Warning: No datasets matched `{d_name}` under `{data_root_dir}`")
        for expanded_name, expanded_robot_type in expanded_entries:
            dataset_key = (expanded_name, expanded_robot_type)
            if dataset_key in included_datasets:
                print(f"Skipping Duplicate Dataset: `{(expanded_name, d_weight, expanded_robot_type)}`")
                continue

            included_datasets.add(dataset_key)
            filtered_mixture_spec.append((expanded_name, d_weight, expanded_robot_type))

    dataset_mixture = []
    for d_name, d_weight, robot_type in filtered_mixture_spec:
        dataset_mixture.append((make_LeRobotSingleDataset(data_root_dir, d_name, robot_type, delete_pause_frame=delete_pause_frame, data_cfg=data_cfg), d_weight))

    return LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        data_cfg=data_cfg,
        **kwargs,
    )



if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./configs/train_recipes/QwenFM_OXE.yaml", help="Path to YAML config")
    args, extra_cli_args = parser.parse_known_args()

    # debugpy.listen(("0.0.0.0", 10092))
    # print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    # debugpy.wait_for_client()
    args.config_yaml = "./benchmarks/MultiRobot/train/alphabrain_cotrain_multiRobot.yaml"
    cfg = OmegaConf.load(args.config_yaml)
    # cfg.datasets.vla_data.dataset_mix = "robotwin"
    vla_dataset_cfg = cfg.datasets.vla_data
    # cfg.datasets.vla_data.include_state = True
    vla_dataset_cfg.task_id = 1
    for task_id in ["all"]:
        vla_dataset_cfg.task_id = task_id
        print(f"Testing Task ID: {task_id}")
        dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
        # dataset
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1, # For Debug
        collate_fn=collate_fn,
    )

    cfg.output_dir = "./results/debug"
    output_dir = Path(cfg.output_dir)
    dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")

    from tqdm import tqdm
    count = 0
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        # print(batch)
        # print(1)
        if count > 100:
            break
        count += 1
        pass
