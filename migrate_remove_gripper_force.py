#!/usr/bin/env python3
"""Remove the gripper-force dimension from existing NERO datasets."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

FEATURE_KEYS = ("observation.state", "action")
FEATURE_NAMES = [
    "Joint_1",
    "Joint_2",
    "Joint_3",
    "Joint_4",
    "Joint_5",
    "Joint_6",
    "Joint_7",
    "Gripper",
]
BACKUP_SUFFIX = ".with-gripper-force.bak"


def backup(path: Path) -> None:
    destination = path.with_name(path.name + BACKUP_SUFFIX)
    if not destination.exists():
        shutil.copy2(path, destination)


def updated_schema_metadata(
    metadata: dict[bytes, bytes] | None,
) -> dict[bytes, bytes] | None:
    if not metadata or b"huggingface" not in metadata:
        return metadata
    updated = dict(metadata)
    payload = json.loads(updated[b"huggingface"])
    features = payload.get("info", {}).get("features", {})
    for key in FEATURE_KEYS:
        if key in features:
            features[key]["length"] = 8
    updated[b"huggingface"] = json.dumps(payload).encode()
    return updated


def fixed_list(values: list[list[float]], value_type: pa.DataType) -> pa.Array:
    flattened = [value for row in values for value in row]
    return pa.FixedSizeListArray.from_arrays(pa.array(flattened, type=value_type), 8)


def replace_parquet_columns(path: Path, columns: set[str]) -> bool:
    table = pq.read_table(path)
    changed = False
    for name in columns:
        index = table.schema.get_field_index(name)
        if index < 0:
            continue
        field_type = table.schema.field(index).type
        if not pa.types.is_fixed_size_list(field_type):
            continue
        if field_type.list_size == 8:
            continue
        if field_type.list_size != 9:
            raise ValueError(
                f"Unsupported {name} dimension in {path}: {field_type.list_size}"
            )
        values = [row[:8] for row in table.column(index).to_pylist()]
        table = table.set_column(
            index,
            name,
            fixed_list(values, field_type.value_type),
        )
        changed = True
    if not changed:
        return False

    backup(path)
    temporary = path.with_name(path.name + ".remove-force.tmp")
    temporary.unlink(missing_ok=True)
    table = table.replace_schema_metadata(updated_schema_metadata(table.schema.metadata))
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, path)
    return True


def update_info(root: Path) -> None:
    path = root / "meta" / "info.json"
    info = json.loads(path.read_text())
    for key in FEATURE_KEYS:
        feature = info["features"][key]
        dimension = int(feature["shape"][0])
        if dimension not in (8, 9):
            raise ValueError(f"Unsupported {key} dimension: {dimension}")
        feature["shape"] = [8]
        feature["names"] = FEATURE_NAMES
    backup(path)
    path.write_text(json.dumps(info, indent=4) + "\n")


def update_stats(root: Path) -> None:
    path = root / "meta" / "stats.json"
    if not path.exists():
        return
    stats = json.loads(path.read_text())
    changed = False
    for key in FEATURE_KEYS:
        for statistic, values in stats.get(key, {}).items():
            if statistic == "count" or not isinstance(values, list):
                continue
            if len(values) == 9:
                stats[key][statistic] = values[:8]
                changed = True
    if changed:
        backup(path)
        path.write_text(json.dumps(stats, indent=4) + "\n")


def migrate(root: Path) -> None:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Dataset metadata not found: {info_path}")

    for path in sorted((root / "data").rglob("*.parquet")):
        replace_parquet_columns(path, set(FEATURE_KEYS))

    episode_columns: set[str] = set()
    for feature in FEATURE_KEYS:
        for statistic in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99"):
            episode_columns.add(f"stats/{feature}/{statistic}")
    for path in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        replace_parquet_columns(path, episode_columns)

    update_info(root)
    update_stats(root)
    print(f"Removed gripper force from dataset: {root}")
    print(f"Original metadata and Parquet files use suffix {BACKUP_SUFFIX}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a 9D NERO dataset to 8D joints-plus-gripper-width data."
    )
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args()
    migrate(args.dataset_root)


if __name__ == "__main__":
    main()