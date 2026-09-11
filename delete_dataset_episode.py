#!/usr/bin/env python3
"""Rebuild a local LeRobot dataset without one selected episode."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from dataset_episode_labels import load_episode_labels, save_episode_label
from lerobot.datasets import delete_episodes
from lerobot.datasets import lerobot_dataset as lerobot_dataset_module
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def remove_dataset_if_only_episode(root: Path, total_episodes: int) -> bool:
    if total_episodes != 1:
        return False
    shutil.rmtree(root)
    return True


def force_pyav_backend() -> None:
    lerobot_dataset_module.get_safe_default_video_backend = lambda: "pyav"


def delete_episode(root: Path, episode_index: int) -> None:
    force_pyav_backend()
    print(f"Loading dataset metadata: {root}", flush=True)
    episode_labels = load_episode_labels(root)
    repo_id = (
        "adrian/nero_replayed"
        if root.name.startswith("nero_replayed__")
        else "adrian/nero_manual"
    )
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        download_videos=False,
        video_backend="pyav",
    )
    if episode_index < 0 or episode_index >= dataset.num_episodes:
        raise ValueError(f"Episode index {episode_index} is out of range.")
    if remove_dataset_if_only_episode(root, dataset.num_episodes):
        print(f"Deleted episode {episode_index} and its now-empty dataset: {root}", flush=True)
        print("Remaining episodes: 0", flush=True)
        return

    temporary_root = root.with_name(f"{root.name}.__without_episode_{episode_index}")
    if temporary_root.exists():
        shutil.rmtree(temporary_root)
    print(
        f"Deleting episode {episode_index}; processing affected data and video chunks...",
        flush=True,
    )
    delete_episodes(
        dataset,
        [episode_index],
        output_dir=temporary_root,
        repo_id=repo_id,
    )

    new_episode = 0
    for old_episode in range(dataset.num_episodes):
        if old_episode == episode_index:
            continue
        label = episode_labels.get(old_episode)
        if label is not None:
            save_episode_label(
                temporary_root,
                new_episode,
                label["task"],
                label["variation"],
            )
        new_episode += 1

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
    print(f"Deleted episode {episode_index} from {root}", flush=True)
    print(f"Remaining episodes: {new_episode}", flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Delete one episode from a local NERO LeRobot dataset.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    args = parser.parse_args()
    delete_episode(args.dataset_root, args.episode_index)
