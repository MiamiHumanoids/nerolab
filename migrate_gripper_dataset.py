#!/usr/bin/env python3
"""Migrate legacy NERO datasets to width-and-force state/action vectors."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets import LeRobotDataset, recompute_stats

JOINT_NAMES = [f"joint{i}.pos" for i in range(1, 8)]
FEATURE_NAMES = [*JOINT_NAMES, "gripper.width_m", "gripper.force"]
BACKUP_SUFFIX = ".gripper-v1.bak"


def backup(path: Path) -> None:
    destination = path.with_name(path.name + BACKUP_SUFFIX)
    if not destination.exists():
        shutil.copy2(path, destination)


def migrated_schema_metadata(
    metadata: dict[bytes, bytes] | None,
) -> dict[bytes, bytes] | None:
    if not metadata or b"huggingface" not in metadata:
        return metadata
    updated = dict(metadata)
    payload = json.loads(updated[b"huggingface"])
    features = payload["info"]["features"]
    features["observation.state"]["length"] = 9
    features["action"]["length"] = 9
    updated[b"huggingface"] = json.dumps(payload).encode()
    return updated


def fixed_float_list(values: np.ndarray) -> pa.FixedSizeListArray:
    width = values.shape[1]
    return pa.FixedSizeListArray.from_arrays(
        pa.array(values.reshape(-1), type=pa.float32()), width
    )


def migrate_data_file(path: Path, default_force: float) -> None:
    parquet = pq.ParquetFile(path)
    state_type = parquet.schema_arrow.field("observation.state").type
    action_type = parquet.schema_arrow.field("action").type
    if state_type.list_size == 9 and action_type.list_size == 9:
        return
    if state_type.list_size != 7 or action_type.list_size != 8:
        raise ValueError(
            f"Unsupported state/action dimensions in {path}: "
            f"{state_type.list_size}/{action_type.list_size}"
        )

    backup(path)
    temporary = path.with_name(path.name + ".gripper-migration.tmp")
    temporary.unlink(missing_ok=True)
    writer = None
    try:
        for batch in parquet.iter_batches(batch_size=8):
            state_index = batch.schema.get_field_index("observation.state")
            action_index = batch.schema.get_field_index("action")
            states = np.asarray(batch.column(state_index).to_pylist(), dtype=np.float32)
            actions = np.asarray(batch.column(action_index).to_pylist(), dtype=np.float32)
            forces = np.full((len(actions), 1), default_force, dtype=np.float32)
            migrated_states = np.concatenate((states, actions[:, 7:8], forces), axis=1)
            migrated_actions = np.concatenate((actions, forces), axis=1)
            batch = batch.set_column(
                state_index, "observation.state", fixed_float_list(migrated_states)
            )
            batch = batch.set_column(
                action_index, "action", fixed_float_list(migrated_actions)
            )
            if writer is None:
                schema = batch.schema.with_metadata(
                    migrated_schema_metadata(parquet.schema_arrow.metadata)
                )
                writer = pq.ParquetWriter(temporary, schema, compression="zstd")
            writer.write_batch(batch)
    finally:
        if writer is not None:
            writer.close()

    expected_rows = parquet.metadata.num_rows
    parquet.close()
    migrated = pq.ParquetFile(temporary)
    migrated_rows = migrated.metadata.num_rows
    migrated.close()
    if migrated_rows != expected_rows:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Frame count changed while migrating {path}")
    os.replace(temporary, path)


def update_info(root: Path) -> None:
    path = root / "meta" / "info.json"
    backup(path)
    info = json.loads(path.read_text())
    for key in ("observation.state", "action"):
        info["features"][key]["shape"] = [9]
        info["features"][key]["names"] = FEATURE_NAMES
    path.write_text(json.dumps(info, indent=4) + "\n")


def update_episode_stats(root: Path) -> None:
    stats = json.loads((root / "meta" / "stats.json").read_text())
    for path in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        backup(path)
        table = pq.read_table(path)
        for feature in ("observation.state", "action"):
            for statistic, values in stats[feature].items():
                name = f"stats/{feature}/{statistic}"
                index = table.schema.get_field_index(name)
                if index >= 0:
                    field_type = table.schema.field(index).type
                    table = table.set_column(
                        index, name, pa.array([values], type=field_type)
                    )
        temporary = path.with_name(path.name + ".gripper-migration.tmp")
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, path)


def migrate(root: Path, repo_id: str, default_force: float) -> None:
    info = json.loads((root / "meta" / "info.json").read_text())
    state_size = info["features"]["observation.state"]["shape"][0]
    action_size = info["features"]["action"]["shape"][0]
    if (state_size, action_size) == (9, 9):
        print("Dataset already uses the 9D gripper contract")
        return
    if (state_size, action_size) != (7, 8):
        raise ValueError(f"Unsupported dataset dimensions: {state_size}/{action_size}")

    for path in sorted((root / "data").rglob("*.parquet")):
        migrate_data_file(path, default_force)
    update_info(root)
    backup(root / "meta" / "stats.json")
    dataset = LeRobotDataset(repo_id=repo_id, root=root, video_backend="pyav")
    recompute_stats(dataset, skip_image_video=True)
    update_episode_stats(root)
    print(f"Migrated dataset to 9D state/action vectors: {root}")
    print(f"Legacy files were preserved with the {BACKUP_SUFFIX} suffix")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--repo-id", default="adrian/nero_replayed")
    parser.add_argument("--default-force", type=float, default=3.0)
    args = parser.parse_args()
    migrate(args.dataset_root, args.repo_id, args.default_force)


if __name__ == "__main__":
    main()