#!/usr/bin/env python3
"""Rebuild a local LeRobot dataset without one selected episode."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path

from dataset_episode_labels import load_episode_labels, save_episode_label
from lerobot.datasets import delete_episodes
from lerobot.datasets import dataset_tools as dataset_tools_module
from lerobot.datasets import lerobot_dataset as lerobot_dataset_module
from lerobot.datasets.lerobot_dataset import LeRobotDataset


STANDARD_COPY = shutil.copy


def link_or_copy(source: str | Path, destination: str | Path, *, follow_symlinks: bool = True) -> str:
    source_path = Path(source)
    destination_path = Path(destination)
    if destination_path.is_dir():
        destination_path /= source_path.name
    try:
        os.link(source_path, destination_path)
    except OSError:
        return STANDARD_COPY(
            source_path,
            destination_path,
            follow_symlinks=follow_symlinks,
        )
    return str(destination_path)


def has_one_data_file_per_episode(dataset: LeRobotDataset) -> bool:
    meta = getattr(dataset, "meta", None)
    if meta is None or not hasattr(meta, "get_data_file_path"):
        return False
    try:
        paths = [
            meta.get_data_file_path(episode_index)
            for episode_index in range(meta.total_episodes)
        ]
    except (AttributeError, KeyError, TypeError):
        return False
    return len(paths) == len(set(paths))


def copy_and_reindex_episode_files(dataset, destination_meta, episode_mapping):
    if dataset.meta.episodes is None:
        dataset.meta.episodes = dataset_tools_module.load_episodes(dataset.meta.root)

    if destination_meta.tasks is None and dataset.meta.tasks is not None:
        destination_meta.save_episode_tasks(list(dataset.meta.tasks.index))

    global_index = 0
    episode_data_metadata = {}
    for old_episode, new_episode in sorted(
        episode_mapping.items(), key=lambda item: item[1]
    ):
        source_episode = dataset.meta.episodes[old_episode]
        source_path = dataset.meta.get_data_file_path(old_episode)
        chunk_index = source_episode["data/chunk_index"]
        file_index = source_episode["data/file_index"]
        destination_path = destination_meta.root / dataset_tools_module.DEFAULT_DATA_PATH.format(
            chunk_index=chunk_index,
            file_index=file_index,
        )
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        episode_length = int(source_episode["length"])
        source_from_index = int(source_episode["dataset_from_index"])

        if old_episode == new_episode and source_from_index == global_index:
            link_or_copy(dataset.root / source_path, destination_path)
        else:
            frame_data = dataset_tools_module.pd.read_parquet(dataset.root / source_path)
            frame_data["episode_index"] = new_episode
            frame_data["index"] = range(global_index, global_index + len(frame_data))
            dataset_tools_module._write_parquet(
                frame_data,
                destination_path,
                destination_meta,
            )

        episode_data_metadata[new_episode] = {
            "data/chunk_index": chunk_index,
            "data/file_index": file_index,
            "dataset_from_index": global_index,
            "dataset_to_index": global_index + episode_length,
        }
        global_index += episode_length

    return episode_data_metadata


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
    original_copy = dataset_tools_module.shutil.copy
    original_data_copier = dataset_tools_module._copy_and_reindex_data
    fast_layout = has_one_data_file_per_episode(dataset)
    if fast_layout:
        print(
            "Fast delete: linking unchanged episode files and rewriting only shifted indices...",
            flush=True,
        )
        dataset_tools_module.shutil.copy = link_or_copy
        dataset_tools_module._copy_and_reindex_data = copy_and_reindex_episode_files
    try:
        delete_episodes(
            dataset,
            [episode_index],
            output_dir=temporary_root,
            repo_id=repo_id,
        )
    finally:
        dataset_tools_module.shutil.copy = original_copy
        dataset_tools_module._copy_and_reindex_data = original_data_copier

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
