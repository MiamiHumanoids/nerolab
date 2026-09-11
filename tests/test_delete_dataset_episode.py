from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from delete_dataset_episode import (
    delete_episode,
    force_pyav_backend,
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