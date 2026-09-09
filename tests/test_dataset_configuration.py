import tempfile
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from manual_record_dataset import FEATURES as MANUAL_FEATURES
from replay_record_task import FEATURES as REPLAY_FEATURES


def test_recorders_declare_camera_features_as_video():
    expected_video_keys = {
        "observation.images.wrist",
        "observation.images.overview",
    }

    for features in (MANUAL_FEATURES, REPLAY_FEATURES):
        assert features["observation.state"]["shape"] == (9,)
        assert features["action"]["shape"] == (9,)
        assert features["observation.state"]["names"][-2:] == [
            "gripper.width_m",
            "gripper.force",
        ]
        assert features["action"]["names"][-2:] == [
            "gripper.width_m",
            "gripper.force",
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = LeRobotDataset.create(
                repo_id="local/nero-schema-test",
                fps=30,
                features=features,
                robot_type="nero",
                root=Path(temp_dir) / "dataset",
                use_videos=True,
                video_backend="pyav",
            )

            assert set(dataset.meta.video_keys) == expected_video_keys
            assert dataset.meta.fps == 30
            assert dataset._video_backend == "pyav"
            assert dataset.writer.episode_buffer["episode_index"] == 0

            resumed = LeRobotDataset.resume(
                repo_id="local/nero-schema-test",
                root=Path(temp_dir) / "dataset",
                video_backend="pyav",
            )
            assert resumed.writer.episode_buffer["episode_index"] == 0
            assert resumed._video_backend == "pyav"