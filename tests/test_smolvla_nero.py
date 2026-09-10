import numpy as np
import pytest

from run_smolvla_nero import decode_policy_action


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