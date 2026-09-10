import json

import pyarrow as pa
import pyarrow.parquet as pq

from dataset_episode_labels import (
    format_episode_label,
    load_episode_labels,
    read_episode_display_labels,
    reindex_episode_labels,
    save_episode_label,
)


def test_episode_labels_save_format_and_reindex(tmp_path):
    save_episode_label(tmp_path, 0, "Stack red on blue", "Centered")
    save_episode_label(tmp_path, 1, "Stack red on blue", "Red cube left")
    save_episode_label(tmp_path, 2, "Stack red on blue", "Red cube right")

    assert format_episode_label(1, **load_episode_labels(tmp_path)[1]) == (
        "Stack red on blue - Red cube left"
    )

    reindex_episode_labels(tmp_path, 1)

    assert load_episode_labels(tmp_path) == {
        0: {"task": "Stack red on blue", "variation": "Centered"},
        1: {"task": "Stack red on blue", "variation": "Red cube right"},
    }
    payload = json.loads(
        (tmp_path / "meta" / "episode_variations.json").read_text()
    )
    assert sorted(payload) == ["0", "1"]


def test_display_labels_combine_native_task_and_variation(tmp_path):
    episodes = tmp_path / "meta" / "episodes" / "chunk-000"
    episodes.mkdir(parents=True)
    pq.write_table(
        pa.table({
            "episode_index": [0, 1],
            "tasks": [
                ["Pick up red and stack it on blue"],
                ["Pick up red and stack it on blue"],
            ],
        }),
        episodes / "file-000.parquet",
    )
    save_episode_label(
        tmp_path,
        1,
        "Pick up red and stack it on blue",
        "Red cube 20 cm left",
    )

    assert read_episode_display_labels(tmp_path, 2) == [
        "Pick up red and stack it on blue",
        "Pick up red and stack it on blue - Red cube 20 cm left",
    ]