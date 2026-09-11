from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import can.interfaces
import pytest

from lerobot_robot_nero import Nero
from lerobot_robot_nero.config import NeroConfig


def test_windows_can_backend_detects_configured_device_channel():
    cfg = NeroConfig(id="test-arm", can_interface="gs_usb", can_channel="0")
    robot = Nero(cfg)

    with patch(
        "lerobot_robot_nero.robot.can.detect_available_configs",
        return_value=[{"interface": "gs_usb", "channel": 0}],
    ):
        assert robot.check_can_interface() is True


def test_windows_can_backend_reports_missing_device():
    cfg = NeroConfig(id="test-arm", can_interface="gs_usb", can_channel="0")
    robot = Nero(cfg)

    with patch(
        "lerobot_robot_nero.robot.can.detect_available_configs", return_value=[]
    ):
        assert robot.check_can_interface() is False


def test_windows_arm_uses_gs_usb_backend():
    cfg = NeroConfig(id="test-arm", can_interface="gs_usb", can_channel="0")
    robot = Nero(cfg)
    arm = MagicMock()
    arm.enable.return_value = True

    with (
        patch.object(robot, "check_can_interface", return_value=True),
        patch("lerobot_robot_nero.robot.create_agx_arm_config", return_value={}) as create_config,
        patch("lerobot_robot_nero.robot.AgxArmFactory.create_arm", return_value=arm),
    ):
        robot.connect()

    assert create_config.call_args.kwargs["interface"] == "gs_usb"
    assert can.interfaces.BACKENDS["gs_usb"] == (
        "lerobot_robot_nero.windows_gs_usb",
        "WindowsGsUsbBus",
    )


def test_connect_can_skip_motor_enable_for_emergency_stopped_gui():
    cfg = NeroConfig(
        id="test-arm",
        can_interface="gs_usb",
        can_channel="0",
        enable_on_connect=False,
    )
    robot = Nero(cfg)
    arm = MagicMock()

    with (
        patch.object(robot, "check_can_interface", return_value=True),
        patch("lerobot_robot_nero.robot.create_agx_arm_config", return_value={}),
        patch("lerobot_robot_nero.robot.AgxArmFactory.create_arm", return_value=arm),
    ):
        robot.connect()

    arm.enable.assert_not_called()


def test_enable_can_feedback_preserves_cached_mode():
    cfg = NeroConfig(id="test-arm", can_channel="can0")
    robot = Nero(cfg)

    class Reporting:
        INVALID = 0
        ENABLE = 1

    class Enums:
        pass

    Enums.CanActiveMsgReporting = Reporting

    class Mode:
        move_mode = 2
        enable_can_push = Reporting.INVALID

    Mode.Enums = Enums

    class DummyArm:
        def __init__(self):
            self._msg_mode = Mode()
            self.sent = []

        def _set_mode(self):
            self.sent.append(
                (self._msg_mode.move_mode, self._msg_mode.enable_can_push)
            )

    robot._arm = DummyArm()

    with patch("lerobot_robot_nero.robot.time.sleep"):
        robot._enable_can_feedback()

    assert robot._arm.sent == [(255, Reporting.ENABLE)]
    assert robot._arm._msg_mode.move_mode == 2
    assert robot._arm._msg_mode.enable_can_push == Reporting.ENABLE


def test_joint_key_mapping_and_action_conversion():
    cfg = NeroConfig(id="test-arm", can_channel="can0")
    robot = Nero(cfg)

    action = {
        "joint1.pos": 0.1,
        "joint2.pos": 0.2,
        "joint3.pos": 0.3,
        "joint4.pos": 0.4,
        "joint5.pos": 0.5,
        "joint6.pos": 0.6,
        "joint7.pos": 0.7,
    }

    target = robot._build_target(action)
    assert len(target) == 7
    assert target[0] == 0.1
    assert target[-1] == 0.7


def test_lerobot_dataset_action_names_drive_joints_and_gripper():
    cfg = NeroConfig(id="test-arm", can_channel="can0")
    robot = Nero(cfg)
    arm = MagicMock()
    arm.is_connected.return_value = True
    effector = MagicMock()
    robot._arm = arm
    robot._gripper_effector = effector
    action = {
        **{f"Joint_{index}": index / 10 for index in range(1, 8)},
        "Gripper": 0.04,
    }

    result = robot.send_action(action)

    arm.move_j.assert_called_once_with([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    effector.move_gripper_m.assert_called_once_with(value=0.04, force=3.0)
    assert result == {"action": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.04, 3.0]}


def test_joint_normalization_handles_list_and_dict():
    cfg = NeroConfig(id="test-arm", can_channel="can0")
    robot = Nero(cfg)

    values = robot._normalize_joint_angles([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert len(values) == 7
    assert values[-1] == 0.6

    payload = {"joint1": 1.0, "joint2": 2.0, "joint3": 3.0, "joint4": 4.0, "joint5": 5.0, "joint6": 6.0, "joint7": 7.0}
    values = robot._normalize_joint_angles(payload)
    assert values[0] == 1.0
    assert values[6] == 7.0


def test_teach_mode_helpers_expose_sdk_methods():
    cfg = NeroConfig(id="test-arm", can_channel="can0")
    robot = Nero(cfg)

    class DummyArm:
        def __init__(self):
            self.mode = "idle"
            self.events = []
            self.targets = []

        def is_connected(self):
            return True

        def set_leader_mode(self):
            self.mode = "leader"

        def electronic_emergency_stop(self):
            self.events.append("emergency_stop")

        def disable(self):
            self.events.append("disable")

        def set_follower_mode(self):
            self.mode = "follower"
            self.events.append("follower")

        def _send_msg(self, message):
            self.events.append(("teach_exit", message.grag_teach_ctrl))

        def set_normal_mode(self):
            self.mode = "normal"

        def reset(self):
            self.events.append("reset")

        def enable(self):
            self.events.append("enable")
            return True

        def set_motion_mode(self, mode):
            self.events.append("motion_mode")

        def move_js(self, target):
            self.events.append("move_js")
            self.targets.append(target)

        def get_leader_joint_angles(self):
            return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    robot._arm = DummyArm()

    robot.set_teach_mode(True)
    assert robot._arm.mode == "leader"
    assert "disable" not in robot._arm.events
    assert robot.get_teach_joint_angles() == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    hold_target = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    with patch.object(robot, "_enable_can_feedback") as enable_can_feedback:
        robot.set_teach_mode(False, hold_target=hold_target)
    enable_can_feedback.assert_called_once_with()
    assert robot._arm.mode == "follower"
    assert robot._arm.events == [
        ("teach_exit", 0x02),
        "follower",
        "move_js",
        "motion_mode",
        "move_js",
        "enable",
        "move_js",
    ]
    assert robot._arm.targets == [hold_target, hold_target, hold_target]
    assert robot._teach_mode_enabled is False

    with patch.object(robot, "_enable_can_feedback") as enable_can_feedback:
        robot.set_teach_mode(False)
    enable_can_feedback.assert_called_once_with()
    assert robot._arm.events[-5:] == [
        ("teach_exit", 0x02),
        "follower",
        "reset",
        "motion_mode",
        "enable",
    ]


def test_get_joint_angles_wrapper_returns_underlying_arm_values():
    cfg = NeroConfig(id="test-arm", can_channel="can0")
    robot = Nero(cfg)

    class DummyArm:
        def is_connected(self):
            return True

        def get_joint_angles(self):
            return [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    robot._arm = DummyArm()

    assert robot.get_joint_angles() == [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def test_live_arm_feedback_rejects_transport_only_connection():
    cfg = NeroConfig(id="test-arm", can_channel="can0")
    robot = Nero(cfg)

    class StaleArm:
        def is_connected(self):
            return True

        def get_arm_status(self):
            return None

        def get_joint_angles(self):
            return None

    robot._arm = StaleArm()

    assert robot.is_connected
    assert not robot.has_live_arm_feedback()


def test_restore_live_arm_feedback_retries_can_reporting():
    cfg = NeroConfig(id="test-arm", can_channel="can0")
    robot = Nero(cfg)

    with (
        patch.object(
            robot,
            "_wait_for_feedback_and_can_ids",
            side_effect=[(False, set()), (False, set()), (True, set())],
        ) as wait_for_feedback,
        patch.object(robot, "_enable_can_feedback") as enable_feedback,
    ):
        assert robot.restore_live_arm_feedback(
            attempts=3, timeout_per_attempt=0.1
        )

    assert wait_for_feedback.call_count == 3
    assert enable_feedback.call_count == 2


def test_restore_live_arm_feedback_resets_error_control_traffic():
    cfg = NeroConfig(id="test-arm", can_channel="can0")
    robot = Nero(cfg)

    with (
        patch.object(
            robot,
            "_wait_for_feedback_and_can_ids",
            return_value=(False, {0x151, 0x155, 0x159}),
        ),
        patch.object(robot, "_recover_error_control_traffic") as recover,
        patch.object(robot, "has_live_arm_feedback", return_value=True),
    ):
        assert robot.restore_live_arm_feedback()

    recover.assert_called_once_with()


def test_error_control_recovery_exits_teach_and_selects_follower_mode():
    cfg = NeroConfig(id="test-arm", can_channel="can0")
    robot = Nero(cfg)
    arm = MagicMock()
    robot._arm = arm

    with (
        patch("lerobot_robot_nero.robot.time.sleep"),
        patch.object(robot, "configure") as configure,
        patch.object(robot, "_enable_can_feedback") as enable_feedback,
    ):
        robot._recover_error_control_traffic()

    motion_ctrl = arm._send_msg.call_args.args[0]
    assert motion_ctrl.grag_teach_ctrl == 0x02
    arm.set_follower_mode.assert_called_once_with()
    arm.reset.assert_called_once_with()
    configure.assert_called_once_with()
    enable_feedback.assert_called_once_with()


def test_recover_stuck_teach_state_requires_can_control_confirmation():
    cfg = NeroConfig(id="test-arm", can_channel="can0")
    robot = Nero(cfg)
    statuses = iter(
        [
            SimpleNamespace(
                msg=SimpleNamespace(
                    ctrl_mode="LINKAGE_TEACHING_INPUT_MODE(0x6)",
                    teach_status="TERMINATE_EXECUTION(0x6)",
                )
            ),
            SimpleNamespace(
                msg=SimpleNamespace(
                    ctrl_mode="CAN_CTRL(0x1)",
                    teach_status="DISABLED(0x0)",
                )
            ),
        ]
    )

    with (
        patch.object(robot, "get_arm_status", side_effect=lambda: next(statuses)),
        patch.object(robot, "_recover_error_control_traffic") as recover,
    ):
        assert robot.recover_stuck_teach_state()

    recover.assert_called_once_with()


def test_task_handoff_disconnect_can_preserve_enabled_motors():
    cfg = NeroConfig(id="test-arm", can_channel="can0")
    robot = Nero(cfg)

    class DummyArm:
        def __init__(self):
            self.disable_calls = 0
            self.disconnect_calls = 0

        def is_connected(self):
            return True

        def disable(self):
            self.disable_calls += 1

        def disconnect(self):
            self.disconnect_calls += 1

    dummy = DummyArm()
    robot._arm = dummy

    robot.disconnect(disable_arm=False)

    assert dummy.disable_calls == 0
    assert dummy.disconnect_calls == 1
    assert robot._arm is None


def test_emergency_disconnect_stops_then_releases_transport():
    cfg = NeroConfig(id="test-arm", can_channel="can0")
    robot = Nero(cfg)
    events = []

    class DummyArm:
        def is_connected(self):
            return True

        def electronic_emergency_stop(self):
            events.append("emergency_stop")

        def disconnect(self):
            events.append("disconnect")

    robot._arm = DummyArm()

    robot.emergency_disconnect()

    assert events == ["emergency_stop", "disconnect"]
    assert robot._arm is None


def test_engage_brakes_waits_for_emergency_stop_and_rest():
    cfg = NeroConfig(id="test-arm", can_channel="can0")
    robot = Nero(cfg)

    class DummyArm:
        def __init__(self):
            self.events = []

        def is_connected(self):
            return True

        def electronic_emergency_stop(self):
            self.events.append("emergency_stop")

        def get_arm_status(self):
            class Message:
                arm_status = "EMERGENCY_STOP"

            class Status:
                msg = Message()

            return Status()

        def get_joint_angles(self):
            return [0.0] * 7

    dummy = DummyArm()
    robot._arm = dummy

    robot.engage_brakes(settle_time=0.0, sample_interval=0.0)

    assert dummy.events == ["emergency_stop"]


def test_robot_observation_and_action_keys_match_lerobot_schema():
    cfg = NeroConfig(id="test-arm", can_channel="can0")
    robot = Nero(cfg)

    obs_features = robot.observation_features
    action_features = robot.action_features

    assert "observation.state" in obs_features
    assert "observation.images.wrist" in obs_features
    assert "action" in action_features

    assert obs_features["observation.state"] == (9,)
    assert obs_features["observation.images.wrist"] == (480, 640, 3)
    assert action_features["action"] == (9,)


def test_robot_can_advertise_overview_camera_feature():
    cfg = NeroConfig(id="test-arm", can_channel="can0", has_overview_camera=True)
    robot = Nero(cfg)

    assert robot.observation_features["observation.images.overview"] == (480, 640, 3)


def test_observation_restarts_wrist_before_overview_camera():
    cfg = NeroConfig(
        id="test-arm",
        can_channel="can0",
        has_camera=True,
        has_overview_camera=True,
    )
    robot = Nero(cfg)
    events = []

    class DummyArm:
        def is_connected(self):
            return True

        def get_joint_angles(self):
            return [0.0] * 7

    class WristCamera:
        is_connected = False
        height = 480
        width = 640

        def connect(self):
            events.append("wrist_connect")
            self.is_connected = True

        def capture_frame(self):
            return None, None

    class OverviewCamera:
        is_connected = True

        def disconnect(self):
            events.append("overview_disconnect")
            self.is_connected = False

        def connect(self):
            events.append("overview_connect")
            self.is_connected = True

        def capture_frame(self):
            return None

    robot._arm = DummyArm()
    robot._camera = WristCamera()
    robot._overview_camera = OverviewCamera()

    robot.get_observation()

    assert events == [
        "overview_disconnect",
        "wrist_connect",
        "overview_connect",
    ]


def test_send_action_accepts_standard_lerobot_action_format():
    cfg = NeroConfig(id="test-arm", can_channel="can0")
    robot = Nero(cfg)

    class DummyArm:
        def __init__(self):
            self.target = None

        def is_connected(self):
            return True

        def move_j(self, target):
            self.target = list(target)

    class DummyEffector:
        def __init__(self):
            self.command = None

        def move_gripper_m(self, value, force):
            self.command = (value, force)

    dummy = DummyArm()
    robot._arm = dummy
    effector = DummyEffector()
    robot._gripper_effector = effector

    result = robot.send_action(
        {"action": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.08, 3.0]}
    )

    assert result == {
        "action": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.08, 3.0]
    }
    assert dummy.target == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    assert effector.command == (0.08, 3.0)


def test_gripper_feedback_reports_width_and_force():
    cfg = NeroConfig(id="test-arm", can_channel="can0")
    robot = Nero(cfg)

    class Message:
        mode = "width"
        value = 0.042
        force = 2.5

    class Status:
        msg = Message()

    effector = MagicMock()
    effector.get_gripper_status.return_value = Status()
    robot._arm = MagicMock()
    robot._gripper_effector = effector

    assert robot.get_gripper_feedback() == (0.042, 2.5)


def test_gripper_feedback_rejects_angle_mode():
    cfg = NeroConfig(id="test-arm", can_channel="can0")
    robot = Nero(cfg)

    class Message:
        mode = "angle"
        value = 30.0
        force = 2.5

    class Status:
        msg = Message()

    effector = MagicMock()
    effector.get_gripper_status.return_value = Status()
    robot._arm = MagicMock()
    robot._gripper_effector = effector

    with pytest.raises(RuntimeError, match="width-mode feedback is required"):
        robot.get_gripper_feedback()


def test_gripper_helper_methods_call_sdk_effectors():
    cfg = NeroConfig(id="test-arm", can_channel="can0")
    robot = Nero(cfg)

    class DummyEffector:
        def __init__(self):
            self.calls = []

        def move_gripper_m(self, value=0.0, force=1.0):
            self.calls.append((value, force))

    class DummyArm:
        class OPTIONS:
            class EFFECTOR:
                AGX_GRIPPER = "agx_gripper"

        def __init__(self):
            self.effectors = {}

        def is_connected(self):
            return True

        def init_effector(self, effector_type):
            self.effectors[effector_type] = DummyEffector()
            return self.effectors[effector_type]

    dummy_arm = DummyArm()
    robot._arm = dummy_arm

    robot.close_gripper()
    robot.open_gripper()

    gripper = dummy_arm.effectors["agx_gripper"]
    assert gripper.calls[0] == (0.0, 1.0)
    assert gripper.calls[1] == (0.055, 1.0)
