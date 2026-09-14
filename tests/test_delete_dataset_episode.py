from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from delete_dataset_episode import (
    copy_and_reindex_episode_files,
    delete_episode,
    force_pyav_backend,
    link_or_copy,
    lerobot_dataset_module,
    remove_dataset_if_only_episode,
)


def test_removing_only_episode_deletes_dataset_instead_of_leaving_shell(tmp_path):
    dataset_root = tmp_path / "nero_replayed__single__30fps"
    metadata = dataset_root / "meta"
    metadata.mkdir(parents=True)
    (metadata / "info.json").write_text("{}")

    assert remove_dataset_if_only_episode(dataset_root, 1) is True
    assert not dataset_root.exists()


def test_multi_episode_dataset_is_preserved_for_rebuild(tmp_path):
    dataset_root = tmp_path / "nero_replayed__multiple__30fps"
    dataset_root.mkdir()

    assert remove_dataset_if_only_episode(dataset_root, 2) is False
    assert dataset_root.exists()


def test_unchanged_episode_files_are_hard_linked(tmp_path):
    source = tmp_path / "source.parquet"
    destination = tmp_path / "destination.parquet"
    source.write_bytes(b"large immutable episode data")

    link_or_copy(source, destination)

    assert destination.read_bytes() == source.read_bytes()
    assert destination.stat().st_ino == source.stat().st_ino


def test_fast_data_copy_rewrites_only_shifted_episode(tmp_path):
    source_root = tmp_path / "source"
    destination_root = tmp_path / "destination"
    source_data = source_root / "data" / "chunk-000"
    source_data.mkdir(parents=True)
    (source_data / "file-000.parquet").write_bytes(b"unchanged")
    (source_data / "file-002.parquet").write_bytes(b"shifted")
    episodes = {
        0: {
            "length": 2,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": 0,
        },
        2: {
            "length": 2,
            "data/chunk_index": 0,
            "data/file_index": 2,
            "dataset_from_index": 4,
        },
    }
    source_meta = SimpleNamespace(
        episodes=episodes,
        tasks=None,
        get_data_file_path=lambda episode: Path(
            f"data/chunk-000/file-{episode:03d}.parquet"
        ),
    )
    dataset = SimpleNamespace(meta=source_meta, root=source_root)
    destination_meta = SimpleNamespace(root=destination_root, tasks=None)
    frame_data = __import__("pandas").DataFrame(
        {"episode_index": [2, 2], "index": [4, 5]}
    )

    with patch(
        "delete_dataset_episode.dataset_tools_module.pd.read_parquet",
        return_value=frame_data,
    ) as read_parquet, patch(
        "delete_dataset_episode.dataset_tools_module._write_parquet"
    ) as write_parquet:
        metadata = copy_and_reindex_episode_files(
            dataset,
            destination_meta,
            {0: 0, 2: 1},
        )

    assert read_parquet.call_count == 1
    assert (destination_root / "data/chunk-000/file-000.parquet").exists()
    rewritten = write_parquet.call_args.args[0]
    assert rewritten["episode_index"].tolist() == [1, 1]
    assert rewritten["index"].tolist() == [2, 3]
    assert metadata[1]["dataset_from_index"] == 2


def test_episode_delete_forces_pyav_for_internal_result_validation():
    with patch.object(
        lerobot_dataset_module,
        "get_safe_default_video_backend",
        return_value="torchcodec",
    ):
        force_pyav_backend()

        assert lerobot_dataset_module.get_safe_default_video_backend() == "pyav"


def test_multi_episode_delete_uses_native_editor_and_reindexes_labels(tmp_path):
    dataset_root = tmp_path / "nero_replayed__multiple__30fps"
    dataset_root.mkdir()
    dataset = SimpleNamespace(num_episodes=3)

    def build_native_output(source, indices, output_dir, repo_id):
        assert source is dataset
        assert indices == [1]
        assert repo_id == "adrian/nero_replayed"
        output_dir.mkdir()

    with patch(
        "delete_dataset_episode.LeRobotDataset",
        return_value=dataset,
    ), patch(
        "delete_dataset_episode.load_episode_labels",
        return_value={
            0: {"task": "stack", "variation": "standard"},
            2: {"task": "stack", "variation": "offset"},
        },
    ), patch(
        "delete_dataset_episode.delete_episodes",
        side_effect=build_native_output,
    ) as native_delete, patch(
        "delete_dataset_episode.save_episode_label"
    ) as save_label:
        delete_episode(dataset_root, 1)

    native_delete.assert_called_once()
    assert dataset_root.exists()
    temporary_root = dataset_root.with_name(
        f"{dataset_root.name}.__without_episode_1"
    )
    save_label.assert_any_call(temporary_root, 0, "stack", "standard")
    save_label.assert_any_call(temporary_root, 1, "stack", "offset")