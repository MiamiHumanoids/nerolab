#!/usr/bin/env python3
"""Rebuild a local LeRobot dataset without one selected episode."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

META_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def delete_episode(root: Path, episode_index: int) -> None:
    dataset = LeRobotDataset(
        repo_id="adrian/nero_manual",
        root=root,
        download_videos=False,
        video_backend="pyav",
    )
    if episode_index < 0 or episode_index >= dataset.num_episodes:
        raise ValueError(f"Episode index {episode_index} is out of range.")

    import json

    info = json.loads((root / "meta" / "info.json").read_text())
    features = {
        key: {**value, "shape": tuple(value["shape"])}
        for key, value in info["features"].items()
        if key not in META_FEATURES
    }
    temporary_root = root.with_name(f"{root.name}.__without_episode_{episode_index}")
    if temporary_root.exists():
        shutil.rmtree(temporary_root)

    rebuilt = LeRobotDataset.create(
        repo_id="adrian/nero_manual",
        fps=int(info["fps"]),
        features=features,
        robot_type=info.get("robot_type", "nero"),
        root=temporary_root,
        use_videos=any(
            feature.get("dtype") in {"image", "video"}
            for feature in features.values()
        ),
        video_backend="pyav",
    )

    new_episode = 0
    for old_episode in range(dataset.num_episodes):
        if old_episode == episode_index:
            continue
        episode = dataset.meta.episodes[old_episode]
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
        for frame_index in range(start, end):
            sample = dataset[frame_index]
            frame = {
                key: as_numpy(sample[key])
                for key in features
                if key in sample
            }
            frame["task"] = sample["task"]
            rebuilt.add_frame(frame)
        rebuilt.save_episode()
        new_episode += 1

    if hasattr(rebuilt.meta, "_flush_metadata_buffer"):
        rebuilt.meta._flush_metadata_buffer()

    backup_root = root.with_name(f"{root.name}.__before_episode_delete")
    if backup_root.exists():
        shutil.rmtree(backup_root)
    root.rename(backup_root)
    try:
        temporary_root.rename(root)
    except Exception:
        backup_root.rename(root)
        raise
    shutil.rmtree(backup_root)
    print(f"Deleted episode {episode_index} from {root}")
    print(f"Remaining episodes: {new_episode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete one episode from a local NERO LeRobot dataset.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    args = parser.parse_args()
    delete_episode(args.dataset_root, args.episode_index)
