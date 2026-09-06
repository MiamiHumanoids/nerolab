from lerobot_robot_nero import Nero
from lerobot_robot_nero.config import NeroConfig


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

        def is_connected(self):
            return True

        def set_leader_mode(self):
            self.mode = "leader"

        def set_follower_mode(self):
            self.mode = "follower"
            self.events.append("follower")

        def set_normal_mode(self):
            self.mode = "normal"

        def reset(self):
            self.events.append("reset")

        def enable(self):
            self.events.append("enable")
            return True

        def get_leader_joint_angles(self):
            return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    robot._arm = DummyArm()

    robot.set_teach_mode(True)
    assert robot._arm.mode == "leader"
    assert robot.get_teach_joint_angles() == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    robot.set_teach_mode(False)
    assert robot._arm.mode == "follower"
    assert robot._arm.events == ["follower", "reset", "enable"]
    assert robot._teach_mode_enabled is False


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

    assert obs_features["observation.state"] == (7,)
    assert obs_features["observation.images.wrist"] == (480, 640, 3)
    assert action_features["action"] == (7,)


def test_robot_can_advertise_overview_camera_feature():
    cfg = NeroConfig(id="test-arm", can_channel="can0", has_overview_camera=True)
    robot = Nero(cfg)

    assert robot.observation_features["observation.images.overview"] == (480, 640, 3)


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

    dummy = DummyArm()
    robot._arm = dummy

    result = robot.send_action({"action": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]})

    assert result == {"action": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]}
    assert dummy.target == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def test_gripper_helper_methods_call_sdk_effectors():
    cfg = NeroConfig(id="test-arm", can_channel="can0")
    robot = Nero(cfg)

    class DummyEffector:
        def __init__(self):
            self.calls = []

        def move_gripper_m(self, value=0.0, force=1.0):
            self.calls.append((value, force))

    class DummyArm:
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
