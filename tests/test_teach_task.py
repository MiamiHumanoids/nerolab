from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from teach_task import (
    capture_initial_table_setup,
    enter_gravity_compensation,
    follower_hold_target,
    play_recording_start_beep,
    show_recording_stopped_countdown,
    wait_for_gravity_compensation,
)


def test_initial_table_setup_capture_writes_image_and_metadata(tmp_path):
    frame = __import__("numpy").zeros((480, 640, 3), dtype="uint8")
    camera = SimpleNamespace(
        is_connected=True,
        device_index=2,
        capture_frame=Mock(return_value=frame),
    )
    robot = SimpleNamespace(_overview_camera=camera)
    output = tmp_path / "stack-red.json"

    with patch("teach_task.cv2.imwrite", return_value=True) as write_image:
        metadata = capture_initial_table_setup(robot, output)

    assert metadata is not None
    assert metadata["path"] == "setup_images/stack-red__initial-table.jpg"
    assert metadata["camera"] == "overview_webcam"
    assert metadata["device_index"] == 2
    assert metadata["width"] == 640
    assert metadata["height"] == 480
    write_image.assert_called_once()


def test_initial_table_setup_capture_is_nonfatal_without_webcam(tmp_path):
    robot = SimpleNamespace(_overview_camera=None)

    assert capture_initial_table_setup(robot, tmp_path / "task.json") is None


class FakeTeachArm:
    def __init__(
        self,
        ctrl_mode: str,
        enabled: list[bool],
        teach_status: str = "DISABLED(0x0)",
        leader_timestamps: list[float] | None = None,
    ):
        self.ctrl_mode = ctrl_mode
        self.enabled = enabled
        self.teach_status = teach_status
        self.leader_timestamps = iter(leader_timestamps or [])

    def get_arm_status(self):
        return SimpleNamespace(
            msg=SimpleNamespace(
                ctrl_mode=self.ctrl_mode,
                teach_status=self.teach_status,
            )
        )

    def get_joints_enable_status_list(self):
        return self.enabled

    def get_leader_joint_angles(self):
        timestamp = next(self.leader_timestamps, None)
        if timestamp is None:
            return None
        return SimpleNamespace(timestamp=timestamp, msg=[0.0] * 7)


def test_recording_start_beep_uses_windows_tone():
    beep = Mock()
    with (
        patch("teach_task.os.name", "nt"),
        patch.dict("sys.modules", {"winsound": SimpleNamespace(Beep=beep)}),
    ):
            play_recording_start_beep()

    beep.assert_called_once_with(1000, 250)


def test_gravity_compensation_accepts_enabled_teach_mode():
    arm = FakeTeachArm(
        "LINKAGE_TEACHING_INPUT_MODE(0x6)",
        [True] * 7,
        leader_timestamps=[1.0],
    )

    wait_for_gravity_compensation(arm, timeout=0.01)


def test_gravity_compensation_accepts_drag_recording_status():
    arm = FakeTeachArm(
        "CAN_CTRL(0x1)",
        [True] * 7,
        teach_status="START_RECORDING(0x1)",
        leader_timestamps=[1.0],
    )

    wait_for_gravity_compensation(arm, timeout=0.01)


def test_gravity_compensation_accepts_fresh_leader_feedback():
    arm = FakeTeachArm(
        "CAN_CTRL(0x1)",
        [True] * 7,
        leader_timestamps=[12.5],
    )

    wait_for_gravity_compensation(
        arm, previous_leader_timestamp=12.0, timeout=0.01
    )


def test_gravity_compensation_rejects_plain_can_control():
    arm = FakeTeachArm("CAN_CTRL(0x1)", [True] * 7)

    with pytest.raises(RuntimeError, match="teach_status=DISABLED"):
        wait_for_gravity_compensation(arm, timeout=0.001)


def test_gravity_compensation_rejects_disabled_motors():
    arm = FakeTeachArm("LINKAGE_TEACHING_INPUT_MODE(0x6)", [False] * 7)

    with pytest.raises(RuntimeError, match="refusing to record"):
        wait_for_gravity_compensation(arm, timeout=0.0)


def test_gravity_compensation_rejects_terminated_linkage_without_leader_feedback():
    arm = FakeTeachArm(
        "LINKAGE_TEACHING_INPUT_MODE(0x6)",
        [True] * 7,
        teach_status="TERMINATE_EXECUTION(0x6)",
    )

    with pytest.raises(RuntimeError, match="Fresh leader-joint feedback"):
        wait_for_gravity_compensation(arm, timeout=0.001)


def test_recording_stopped_countdown_warns_and_counts_to_zero():
    with (
        patch("teach_task.time.monotonic", side_effect=[0, 0, 1, 2, 3, 4, 5]),
        patch("teach_task.cv2.putText") as put_text,
        patch("teach_task.cv2.imshow") as show,
        patch("teach_task.cv2.waitKey") as wait_key,
    ):
        show_recording_stopped_countdown(seconds=5)

    messages = [call.args[1] for call in put_text.call_args_list]
    assert "Recording stopped - release robot and step away" in messages
    assert "Returning to Safe Bicep in 5 seconds" in messages
    assert "Returning to Safe Bicep in 0 seconds" in messages
    assert show.call_count == 6
    assert wait_key.call_count == 6


def test_follower_hold_target_uses_fresh_leader_pose():
    target = follower_hold_target(
        [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        [0.1] * 7,
        [1.0] * 7,
    )

    assert target == pytest.approx([1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8])


def test_gravity_compensation_retries_unacknowledged_mode_commands():
    events = []
    normalized_anchor = [0.1, -1.7, 0.02, 2.1, -0.03, 0.08, 1.6]
    arm = SimpleNamespace(
        _send_msg=lambda message: events.append("drag_teach"),
    )
    robot = SimpleNamespace(
        _arm=arm,
        get_arm_status=lambda: SimpleNamespace(
            msg=SimpleNamespace(
                ctrl_mode="STANDBY(0x0)",
                teach_status="DISABLED(0x0)",
            )
        ),
        get_joint_angles=lambda: normalized_anchor,
        set_teach_mode=lambda enabled: events.append(
            "leader" if enabled else "follower_reset"
        ),
    )
    with (
        patch("teach_task.leader_feedback_timestamp", return_value=0.0),
        patch("teach_task.time.sleep"),
        patch(
            "teach_task.wait_for_gravity_compensation",
            side_effect=[RuntimeError("not ready"), None],
        ) as wait_for_mode,
    ):
        result = enter_gravity_compensation(robot)

    assert wait_for_mode.call_count == 2
    assert result == normalized_anchor
    assert events == [
        "follower_reset",
        "leader",
        "drag_teach",
        "follower_reset",
        "leader",
        "drag_teach",
    ]


def test_gravity_compensation_skips_reset_when_controller_is_ready():
    events = []
    anchor = [0.0, -1.68, 0.023, 2.08, -0.026, 0.076, 1.5]
    arm = SimpleNamespace(
        get_joints_enable_status_list=lambda: [True] * 7,
        get_leader_joint_angles=lambda: None,
        _send_msg=lambda message: events.append("drag_teach"),
    )
    robot = SimpleNamespace(
        _arm=arm,
        get_arm_status=lambda: SimpleNamespace(
            msg=SimpleNamespace(
                ctrl_mode="CAN_CTRL(0x1)",
                teach_status="DISABLED(0x0)",
            )
        ),
        get_joint_angles=lambda: anchor,
        set_teach_mode=lambda enabled: events.append(
            "leader" if enabled else "follower_reset"
        ),
    )

    with (
        patch("teach_task.time.sleep"),
        patch("teach_task.wait_for_gravity_compensation"),
    ):
        result = enter_gravity_compensation(robot)

    assert result == anchor
    assert events == ["leader", "drag_teach"]