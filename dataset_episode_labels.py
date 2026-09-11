"""Store and read human-readable labels for NERO dataset episodes."""

from __future__ import annotations

import json
import os
from pathlib import Path

LABELS_FILENAME = "episode_variations.json"
DATASET_METADATA_FILENAME = "nero_dataset.json"


def dataset_metadata_path(root: Path) -> Path:
    return root / "meta" / DATASET_METADATA_FILENAME


def load_dataset_display_name(root: Path) -> str:
    path = dataset_metadata_path(root)
    if not path.exists():
        return ""
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return ""
    return str(payload.get("display_name", "")).strip()


def save_dataset_display_name(root: Path, display_name: str) -> None:
    name = display_name.strip()
    if not name:
        return
    path = dataset_metadata_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps({"display_name": name}, indent=2) + "\n")
    os.replace(temporary, path)


def labels_path(root: Path) -> Path:
    return root / "meta" / LABELS_FILENAME


def load_episode_labels(root: Path) -> dict[int, dict[str, str]]:
    path = labels_path(root)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        int(index): {
            "task": str(value.get("task", "")).strip(),
            "variation": str(value.get("variation", "")).strip(),
        }
        for index, value in payload.items()
        if isinstance(value, dict)
    }


def save_episode_label(
    root: Path,
    episode_index: int,
    task: str,
    variation: str,
) -> None:
    labels = load_episode_labels(root)
    labels[episode_index] = {
        "task": task.strip(),
        "variation": variation.strip(),
    }
    path = labels_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(labels, indent=2) + "\n")
    os.replace(temporary, path)


def reindex_episode_labels(root: Path, deleted_episode: int) -> None:
    labels = load_episode_labels(root)
    updated = {
        index - (1 if index > deleted_episode else 0): value
        for index, value in labels.items()
        if index != deleted_episode
    }
    path = labels_path(root)
    if not updated:
        path.unlink(missing_ok=True)
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(updated, indent=2) + "\n")
    os.replace(temporary, path)


def format_episode_label(
    episode_index: int,
    task: str,
    variation: str,
) -> str:
    name = task.strip() or f"Episode {episode_index}"
    note = variation.strip()
    return f"{name} - {note}" if note else name


def read_episode_display_labels(root: Path, total_episodes: int) -> list[str]:
    tasks: dict[int, str] = {}
    try:
        import pyarrow.parquet as pq

        for path in sorted((root / "meta" / "episodes").rglob("*.parquet")):
            schema_names = pq.read_schema(path).names
            if "episode_index" not in schema_names or "tasks" not in schema_names:
                continue
            for row in pq.read_table(
                path, columns=["episode_index", "tasks"]
            ).to_pylist():
                episode_tasks = row.get("tasks") or []
                tasks[int(row["episode_index"])] = (
                    str(episode_tasks[0]) if episode_tasks else ""
                )
    except Exception:
        tasks = {}

    notes = load_episode_labels(root)
    return [
        format_episode_label(
            index,
            notes.get(index, {}).get("task") or tasks.get(index, ""),
            notes.get(index, {}).get("variation", ""),
        )
        for index in range(total_episodes)
    ]