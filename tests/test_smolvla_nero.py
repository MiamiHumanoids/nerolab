import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from run_smolvla_nero import build_policy, decode_policy_action


def test_build_policy_uses_requested_policy_type(tmp_path):
    metadata = MagicMock()
    metadata.features = {
        "observation.state": {"shape": [8]},
        "action": {"shape": [8]},
    }
    config = MagicMock()

    with patch("run_smolvla_nero.LeRobotDatasetMetadata", return_value=metadata), patch(
        "run_smolvla_nero.make_policy_config", return_value=config
    ) as make_config, patch("run_smolvla_nero.make_policy"), patch(
        "run_smolvla_nero.make_pre_post_processors", return_value=(MagicMock(), MagicMock())
    ):
        build_policy(tmp_path / "checkpoint", tmp_path / "dataset", "cuda", "groot")

    make_config.assert_called_once_with(
        "groot",
        pretrained_path=tmp_path / "checkpoint",
        device="cuda",
    )


def test_decode_policy_action_uses_configured_force_for_closed_gripper():
    joints, width, force = decode_policy_action(
        np.asarray([0.1] * 7 + [0.0], dtype=np.float32),
        3.0,
    )

    assert joints.tolist() == pytest.approx([0.1] * 7)
    assert width == 0.0
    assert force == 3.0


def test_decode_policy_action_rejects_ninth_value_and_invalid_force():
    with pytest.raises(ValueError, match="expected 8"):
        decode_policy_action(np.zeros(9, dtype=np.float32), 3.0)
    with pytest.raises(ValueError, match="between 0 and 30"):
        decode_policy_action(np.zeros(8, dtype=np.float32), 31.0)