from unittest.mock import patch

from lerobot_replay_pyav import install_enabled_disconnect
from lerobot_robot_nero import Nero


def test_lerobot_replay_disconnect_keeps_arm_enabled():
    robot = object()
    with patch.object(Nero, "disconnect", autospec=True) as disconnect:
        install_enabled_disconnect()
        Nero.disconnect(robot)

    disconnect.assert_called_once_with(robot, disable_arm=False)