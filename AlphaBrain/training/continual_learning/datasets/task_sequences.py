"""
continual_learning.py

Defines continual learning task sequences for sequential task training.
Each sequence specifies a base data_mix and task ordering.
Provides utilities to filter datasets by task_index for per-task training.
"""

import copy
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset

import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Continual Learning Task Sequences
# ============================================================================
# Each sequence defines:
#   - base_data_mix: the data_mix name in DATASET_NAMED_MIXTURES
#   - num_tasks: number of tasks (auto-detected from dataset if not specified)
#   - task_order: optional explicit ordering of task indices (default: 0..num_tasks-1)

CL_TASK_SEQUENCES = {
    # LIBERO suites — each suite contains 10 tasks with 50 demos each.
    # Default task_stream_mode is `by_task_index`: a single multi-task parquet
    # whose `task_index` column distinguishes the 10 tasks.
    "libero_spatial": {
        "base_data_mix": "libero_spatial",
        "num_tasks": 10,
    },
    "libero_object": {
        "base_data_mix": "libero_object",
        "num_tasks": 10,
    },
    "libero_goal": {
        "base_data_mix": "libero_goal",
        "num_tasks": 10,
    },
    "libero_long": {
        "base_data_mix": "libero_long",
        "num_tasks": 10,
    },

    # Robocasa365 lifelong benchmark — sub-sampled from the official 125-task
    # 4-phase spec (https://robocasa.ai/docs/build/html/benchmarking/lifelong_learning.html).
    # `by_dataset` mode: each lerobot sub-dataset in the mixture is one CL task,
    # in the order it appears in the mixture.
    "robocasa365_cl_atomic10": {
        "base_data_mix": "robocasa365_cl_atomic10",
        "num_tasks": 10,
        "task_stream_mode": "by_dataset",
    },
}


def get_task_sequence(sequence_name: str) -> dict:
    """Retrieve a CL task sequence by name.

    Returns a *copy* so callers can mutate freely without polluting the
    registry.  Fills in ``task_stream_mode`` default (``by_task_index``) when
    the entry doesn't specify one (LIBERO compatibility).
    """
    if sequence_name not in CL_TASK_SEQUENCES:
        raise ValueError(
            f"Unknown CL task sequence: {sequence_name}. "
            f"Available: {list(CL_TASK_SEQUENCES.keys())}"
        )
    cfg = dict(CL_TASK_SEQUENCES[sequence_name])
    cfg.setdefault("task_stream_mode", "by_task_index")
    return cfg


# ============================================================================
# Episode-to-Task Mapping
# ============================================================================
# Two partitioning strategies are supported, selected per-sequence via the
# ``task_stream_mode`` key:
#
#   - ``by_task_index``: LIBERO-style.  One multi-task parquet with a
#     ``task_index`` column that distinguishes tasks across all episodes.
#     :func:`build_episode_task_map` reads that column.
#
#   - ``by_dataset``: Robocasa-style.  Each sub-dataset of a MixtureDataset
#     represents one CL task; ordinal position in the mixture = task id.
#     :func:`build_dataset_task_map` ignores per-episode task_index (which is
#     local to each sub-dataset and not globally unique) and groups episodes
#     by their source sub-dataset.


def build_dataset_task_map(dataset) -> Dict[int, List[int]]:
    """Per-sub-dataset partition for ``task_stream_mode=by_dataset``.

    Args:
        dataset: A LeRobotMixtureDataset with a ``.datasets`` list of
            sub-datasets.  Each sub-dataset becomes one CL task whose id is
            its ordinal in the list.

    Returns:
        ``{task_idx: [episode_id, ...]}`` where episode_ids are the raw
        ``trajectory_ids`` from each sub-dataset.  Note that across
        sub-datasets these ids are **not globally unique** (each lerobot
        dataset counts from 0) — but :class:`TaskFilteredDataset`'s
        mixture path indexes by ``ds_idx`` before looking up steps, so
        collisions between sub-datasets are harmless.
    """
    if not hasattr(dataset, "datasets"):
        raise ValueError(
            "build_dataset_task_map requires a MixtureDataset with a `.datasets` "
            "attribute; got a single-dataset object.  Check that the configured "
            "`dataset_mix` expands to more than one source dataset."
        )

    task_to_episodes: Dict[int, List[int]] = {}
    for task_idx, sub_ds in enumerate(dataset.datasets):
        eps = sorted({traj_id for traj_id, _ in sub_ds.all_steps})
        task_to_episodes[task_idx] = eps
        logger.info(
            f"Task {task_idx} ← sub-dataset[{task_idx}] "
            f"(tag={getattr(sub_ds, 'tag', '?')}): {len(eps)} episodes, "
            f"{len(sub_ds.all_steps)} steps"
        )

    logger.info(
        f"Built dataset-task map (by_dataset mode): {len(task_to_episodes)} tasks, "
        f"{sum(len(v) for v in task_to_episodes.values())} total episodes"
    )
    return task_to_episodes


def build_episode_task_map(dataset) -> Dict[int, List[int]]:
    """Build mapping from task_index to list of episode_ids by reading episode data.

    Args:
        dataset: A LeRobotSingleDataset instance.

    Returns:
        Dict mapping task_index -> list of trajectory_ids (episode indices).
    """
    task_to_episodes: Dict[int, List[int]] = defaultdict(list)
    seen_episodes = set()

    for traj_id in dataset.trajectory_ids:
        if traj_id in seen_episodes:
            continue
        seen_episodes.add(traj_id)

        try:
            data = dataset.get_trajectory_data(traj_id)
            if "task_index" in data.columns:
                task_idx = int(data["task_index"].iloc[0])
            else:
                # Fallback: try annotation-based task index
                annotation_cols = [c for c in data.columns if "task" in c.lower()]
                if annotation_cols:
                    task_idx = int(data[annotation_cols[0]].iloc[0])
                else:
                    logger.warning(
                        f"No task_index column found for episode {traj_id}, assigning task 0"
                    )
                    task_idx = 0
            task_to_episodes[task_idx].append(traj_id)
        except Exception as e:
            logger.warning(f"Failed to read task_index for episode {traj_id}: {e}")
            continue

    # Clear dataset cache
    dataset.curr_traj_data = None
    dataset.curr_traj_id = None

    logger.info(
        f"Built episode-task map: {len(task_to_episodes)} tasks, "
        f"{sum(len(v) for v in task_to_episodes.values())} total episodes"
    )
    for task_idx in sorted(task_to_episodes.keys()):
        logger.info(f"  Task {task_idx}: {len(task_to_episodes[task_idx])} episodes")

    return dict(task_to_episodes)


# ============================================================================
# Task-Filtered Dataset Wrapper
# ============================================================================

class TaskFilteredDataset(Dataset):
    """Wraps a LeRobotMixtureDataset to only expose steps from specific task indices.

    This is a lightweight wrapper that filters the base dataset's step sampling
    without copying data or modifying the underlying dataset.
    """

    def __init__(
        self,
        base_dataset,
        task_indices: List[int],
        episode_task_map: Dict[int, List[int]],
        *,
        task_stream_mode: str = "by_task_index",
    ):
        """
        Args:
            base_dataset: A LeRobotMixtureDataset (or LeRobotSingleDataset).
            task_indices: List of task_index values to include.
            episode_task_map: Mapping from task_index -> list of episode_ids.
            task_stream_mode: How to interpret ``task_indices``.
                * ``by_task_index`` (LIBERO): filter every sub-dataset's steps
                  by whether the episode's traj_id is in
                  ``episode_task_map[task_index]`` (traj_ids are globally
                  unique in this mode).
                * ``by_dataset`` (Robocasa): each ``task_index`` IS a
                  sub-dataset ordinal; keep all steps from those
                  sub-datasets and drop all others.  This is essential
                  when sub-datasets share the same local traj_id
                  numbering (as Robocasa does).
        """
        self.base_dataset = base_dataset
        self.task_indices = task_indices
        self.task_stream_mode = task_stream_mode

        # Build set of valid episode ids for fast lookup (used by by_task_index mode).
        self.valid_episodes = set()
        for ti in task_indices:
            if ti in episode_task_map:
                self.valid_episodes.update(episode_task_map[ti])

        # For MixtureDataset: filter each sub-dataset's all_steps
        # For SingleDataset: filter directly
        if hasattr(base_dataset, 'datasets'):
            # MixtureDataset
            self._filtered_steps_per_dataset = []
            self._total_steps = 0
            if task_stream_mode == "by_dataset":
                # by_dataset: keep all steps from selected sub-datasets,
                # drop every other sub-dataset entirely.  This avoids the
                # cross-dataset traj_id collision that by_task_index filtering
                # would let through.
                keep_ds = set(task_indices)
                for ds_idx, ds in enumerate(base_dataset.datasets):
                    if ds_idx in keep_ds:
                        filtered = list(ds.all_steps)
                    else:
                        filtered = []
                    self._filtered_steps_per_dataset.append(filtered)
                    self._total_steps += len(filtered)
            else:
                # by_task_index: filter by traj_id membership in valid_episodes.
                for ds in base_dataset.datasets:
                    filtered = [
                        (traj_id, base_idx)
                        for traj_id, base_idx in ds.all_steps
                        if traj_id in self.valid_episodes
                    ]
                    self._filtered_steps_per_dataset.append(filtered)
                    self._total_steps += len(filtered)
        else:
            # SingleDataset — always by_task_index (no per-dataset notion).
            self._filtered_steps = [
                (traj_id, base_idx)
                for traj_id, base_idx in base_dataset.all_steps
                if traj_id in self.valid_episodes
            ]
            self._total_steps = len(self._filtered_steps)

        logger.info(
            f"TaskFilteredDataset (mode={task_stream_mode}): tasks={task_indices}, "
            f"episodes={len(self.valid_episodes)}, steps={self._total_steps}"
        )

    def __len__(self) -> int:
        return self._total_steps

    def __getitem__(self, index: int) -> dict:
        max_retries = 10
        for attempt in range(max_retries):
            try:
                if hasattr(self.base_dataset, 'datasets'):
                    return self._getitem_mixture(index)
                else:
                    return self._getitem_single(index)
            except Exception as e:
                import random
                logger.warning(
                    f"[TaskFilteredDataset] Error loading index {index} "
                    f"(attempt {attempt + 1}/{max_retries}): {e}"
                )
                index = random.randint(0, len(self) - 1)
        raise RuntimeError(
            f"[TaskFilteredDataset] Failed to load data after {max_retries} retries"
        )

    def _getitem_single(self, index: int) -> dict:
        """Get item from filtered single dataset."""
        traj_id, base_idx = self._filtered_steps[index % len(self._filtered_steps)]
        ds = self.base_dataset
        raw_data = ds.get_step_data(traj_id, base_idx)
        data = ds.transforms(raw_data)
        sample = ds._pack_sample(data)
        if hasattr(ds, 'tag'):
            sample["robot_tag"] = ds.tag
        return sample

    def _getitem_mixture(self, index: int) -> dict:
        """Get item from filtered mixture dataset with weighted sampling."""
        import random
        # Weighted random selection across sub-datasets
        dataset_weights = []
        for i, steps in enumerate(self._filtered_steps_per_dataset):
            if len(steps) > 0:
                dataset_weights.append((i, len(steps)))

        if not dataset_weights:
            raise ValueError("No valid steps in filtered dataset")

        total = sum(w for _, w in dataset_weights)
        r = random.random() * total
        cumulative = 0
        ds_idx = dataset_weights[0][0]
        for idx, w in dataset_weights:
            cumulative += w
            if r <= cumulative:
                ds_idx = idx
                break

        steps = self._filtered_steps_per_dataset[ds_idx]
        step_idx = random.randint(0, len(steps) - 1)
        traj_id, base_idx = steps[step_idx]

        ds = self.base_dataset.datasets[ds_idx]
        raw_data = ds.get_step_data(traj_id, base_idx)
        data = ds.transforms(raw_data)
        sample = ds._pack_sample(data)
        sample["robot_tag"] = ds.tag
        return sample

    @property
    def datasets(self):
        """Expose underlying datasets for compatibility."""
        if hasattr(self.base_dataset, 'datasets'):
            return self.base_dataset.datasets
        return [self.base_dataset]

    def save_dataset_statistics(self, path):
        """Delegate to base dataset."""
        if hasattr(self.base_dataset, 'save_dataset_statistics'):
            self.base_dataset.save_dataset_statistics(path)
