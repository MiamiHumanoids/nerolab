import tempfile
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.lerobot_dataset_viz import get_feature_names

from manual_record_dataset import FEATURES as MANUAL_FEATURES
from lerobot_dataset_viz_pyav import NERO_SCALAR_NAMES, install_named_scalar_logging
from replay_record_task import (
    FEATURES as REPLAY_FEATURES,
    GRIPPER_REPLAY_FORCE as RECORDER_GRIPPER_REPLAY_FORCE,
)
from task_trajectory import GRIPPER_REPLAY_FORCE


def test_recorders_declare_camera_features_as_video():
    assert RECORDER_GRIPPER_REPLAY_FORCE == GRIPPER_REPLAY_FORCE
    expected_video_keys = {
        "observation.images.wrist",
        "observation.images.overview",
    }

    for features in (MANUAL_FEATURES, REPLAY_FEATURES):
        assert features["observation.state"]["shape"] == (8,)
        assert features["action"]["shape"] == (8,)
        expected_names = [
            "Joint_1",
            "Joint_2",
            "Joint_3",
            "Joint_4",
            "Joint_5",
            "Joint_6",
            "Joint_7",
            "Gripper",
        ]
        assert features["observation.state"]["names"] == expected_names
        assert features["action"]["names"] == expected_names
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
            assert get_feature_names(dataset, "action") == expected_names

            resumed = LeRobotDataset.resume(
                repo_id="local/nero-schema-test",
                root=Path(temp_dir) / "dataset",
                video_backend="pyav",
            )
            assert resumed.writer.episode_buffer["episode_index"] == 0
            assert resumed._video_backend == "pyav"


def test_rerun_wrapper_splits_action_batch_into_named_series():
    calls = []

    class ArrowValues:
        def to_pylist(self):
            return list(range(8))

    class ScalarBatch:
        def as_arrow_array(self):
            return ArrowValues()

    class Scalars:
        def __init__(self, value):
            self.value = value
            self.scalars = ScalarBatch()

    class SeriesLines:
        def __init__(self, names):
            self.names = names

    class FakeRerun:
        @staticmethod
        def log(path, entity, *args, **kwargs):
            calls.append((path, entity, kwargs))

    rerun = FakeRerun()
    rerun.Scalars = Scalars
    rerun.SeriesLines = SeriesLines
    install_named_scalar_logging(rerun)
    rerun.log("action", Scalars(range(8)))

    scalar_calls = [call for call in calls if isinstance(call[1], Scalars)]
    assert [call[0] for call in scalar_calls] == [
        f"action/{name}" for name in NERO_SCALAR_NAMES
    ]
    assert [call[1].value for call in scalar_calls] == list(range(8))