import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from nero_lab import (
    NeroLab,
    is_lerobot_replay_start_pose,
    lerobot_replay_command,
    newest_dataset_paths,
    next_task_output_path,
    read_dataset_info,
    requires_enabled_safe_bicep_completion,
    task_layout_image_path,
)
from dataset_episode_labels import save_dataset_display_name
from task_trajectory import SAFE_BICEP_BRAKED_JOINTS, SAFE_BICEP_JOINTS


def test_repeated_task_variations_get_unique_recording_paths(tmp_path):
    first = next_task_output_path(tmp_path, "stack-red__standard-layout")
    first.write_text("{}")
    second = next_task_output_path(tmp_path, "stack-red__standard-layout")
    second.write_text("{}")

    assert first.name == "stack-red__standard-layout.json"
    assert second.name == "stack-red__standard-layout__2.json"
    assert next_task_output_path(
        tmp_path, "stack-red__standard-layout"
    ).name == "stack-red__standard-layout__3.json"


def test_task_layout_image_path_resolves_saved_setup_image(tmp_path):
    task_file = tmp_path / "stack-red.json"
    image = tmp_path / "setup_images" / "stack-red__initial-table.jpg"
    image.parent.mkdir()
    image.write_bytes(b"image")

    assert task_layout_image_path(
        task_file,
        {"initial_table_setup": {"path": "setup_images/stack-red__initial-table.jpg"}},
    ) == image


def test_task_layout_image_path_rejects_path_outside_setup_directory(tmp_path):
    task_file = tmp_path / "stack-red.json"
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"image")

    assert task_layout_image_path(
        task_file,
        {"initial_table_setup": {"path": "outside.jpg"}},
    ) is None


def test_open_task_layout_image_uses_windows_default_viewer(tmp_path):
    image = tmp_path / "setup.jpg"
    image.write_bytes(b"image")
    app = SimpleNamespace(
        _selected_task_layout_image=MagicMock(return_value=image),
        log_message=MagicMock(),
    )

    with patch("nero_lab.os.name", "nt"), patch(
        "nero_lab.os.startfile", create=True
    ) as startfile:
        NeroLab.open_task_layout_image(app)

    startfile.assert_called_once_with(image)
    app.log_message.assert_called_once_with(f"Opened task layout image: {image}")


def test_connect_is_deferred_while_task_process_owns_can():
    app = SimpleNamespace(
        process=SimpleNamespace(poll=lambda: None),
        log_message=MagicMock(),
    )

    NeroLab.connect_robot(app)

    app.log_message.assert_called_once_with(
        "Arm connection deferred while the active task process owns CAN."
    )


def test_all_robot_motion_workflows_restore_enabled_safe_bicep():
    for label in (
        "Teach task",
        "Replay task",
        "Replay trained task",
        "LeRobot replay",
    ):
        assert requires_enabled_safe_bicep_completion(label)

    assert not requires_enabled_safe_bicep_completion("Replay")
    assert not requires_enabled_safe_bicep_completion("Rerun")
    assert not requires_enabled_safe_bicep_completion("Episode deletion")


def test_datasets_are_sorted_newest_first(tmp_path):
    older = tmp_path / "older"
    newer = tmp_path / "newer"
    for path in (older, newer):
        (path / "meta").mkdir(parents=True)
        (path / "meta" / "info.json").write_text("{}")
    os.utime(older / "meta" / "info.json", (10, 10))
    os.utime(newer / "meta" / "info.json", (20, 20))

    assert newest_dataset_paths([older, newer]) == [newer, older]


def test_dataset_display_name_cleans_legacy_storage_folder(tmp_path):
    root = tmp_path / "nero_replayed__bin-cubes__30fps"
    root.mkdir()

    assert read_dataset_info(root).task == "Bin Cubes"


def test_dataset_display_name_preserves_exact_gui_name(tmp_path):
    root = tmp_path / "nero_replayed__bin-cubes__30fps"
    save_dataset_display_name(root, "Bin Cubes")

    assert read_dataset_info(root).task == "Bin Cubes"


def test_lerobot_replay_command_targets_selected_episode(tmp_path):
    dataset_root = tmp_path / "nero_replayed__pick-up__30fps"

    assert lerobot_replay_command("lerobot_replay_pyav.py", dataset_root, 12) == [
        os.sys.executable,
        "lerobot_replay_pyav.py",
        "--robot.type=nero",
        "--robot.reset_on_connect=false",
        "--dataset.repo_id=adrian/nero_replayed",
        f"--dataset.root={dataset_root}",
        "--dataset.episode=12",
    ]


def test_lerobot_replay_requires_safe_bicep_start_pose():
    assert is_lerobot_replay_start_pose(SAFE_BICEP_JOINTS)
    assert is_lerobot_replay_start_pose(SAFE_BICEP_BRAKED_JOINTS)
    assert not is_lerobot_replay_start_pose(
        [0.002, 0.253, 0.076, 1.154, -0.057, 0.044, 1.679]
    )


def test_lerobot_replay_completion_returns_to_safe_bicep_enabled():
    robot = SimpleNamespace(
        is_connected=True,
        get_joint_angles=MagicMock(return_value=SAFE_BICEP_JOINTS),
        engage_brakes=MagicMock(),
    )
    app = SimpleNamespace(
        robot=robot,
        safe_bicep_position_reached=False,
        log_message=MagicMock(),
        connect_robot=MagicMock(),
        reenable_arm=MagicMock(),
        safe_bicep_reset=MagicMock(),
        is_safe_bicep_position=lambda values: values == SAFE_BICEP_JOINTS,
        set_arm_status_display=MagicMock(),
        set_joint_slider_values=MagicMock(),
    )

    NeroLab._restore_safe_bicep_after_lerobot_replay(app)

    app.connect_robot.assert_called_once_with()
    app.reenable_arm.assert_not_called()
    app.safe_bicep_reset.assert_not_called()
    robot.engage_brakes.assert_not_called()
    assert app.safe_bicep_position_reached
    app.set_arm_status_display.assert_called_with(
        "Arm status: Safe Bicep | enabled", color="#008000"
    )


def test_lerobot_replay_completion_does_not_brake_before_safe_pose():
    unsafe_joints = [0.0] * 7
    robot = SimpleNamespace(
        is_connected=True,
        get_joint_angles=MagicMock(return_value=unsafe_joints),
        engage_brakes=MagicMock(),
    )
    app = SimpleNamespace(
        robot=robot,
        safe_bicep_position_reached=True,
        log_message=MagicMock(),
        connect_robot=MagicMock(),
        reenable_arm=MagicMock(),
        safe_bicep_reset=MagicMock(),
        is_safe_bicep_position=MagicMock(return_value=False),
        set_arm_status_display=MagicMock(),
        set_joint_slider_values=MagicMock(),
    )

    NeroLab._restore_safe_bicep_after_lerobot_replay(app)

    robot.engage_brakes.assert_not_called()
    assert not app.safe_bicep_position_reached
    assert "FAILED" in app.log_message.call_args.args[0]


def test_teach_completion_recovers_unsafe_pose_and_stays_enabled():
    unsafe_joints = [0.0] * 7
    robot = SimpleNamespace(
        is_connected=True,
        get_joint_angles=MagicMock(
            side_effect=[unsafe_joints, SAFE_BICEP_JOINTS]
        ),
    )
    app = SimpleNamespace(
        robot=robot,
        safe_bicep_position_reached=False,
        log_message=MagicMock(),
        connect_robot=MagicMock(),
        reenable_arm=MagicMock(),
        safe_bicep_reset=MagicMock(),
        is_safe_bicep_position=lambda values: values == SAFE_BICEP_JOINTS,
        set_arm_status_display=MagicMock(),
        set_joint_slider_values=MagicMock(),
    )

    NeroLab._restore_safe_bicep_after_teach(app)

    app.reenable_arm.assert_called_once_with()
    app.safe_bicep_reset.assert_called_once_with()
    assert app.safe_bicep_position_reached
    app.set_arm_status_display.assert_called_with(
        "Arm status: Safe Bicep | enabled", color="#008000"
    )


def test_reenable_latches_fresh_measured_joints_before_enable():
    measured = [0.01, -1.76, 0.02, 2.19, -0.03, 0.08, 1.68]
    status = SimpleNamespace(
        msg=SimpleNamespace(arm_status="NORMAL", ctrl_mode="CAN_CTRL")
    )
    arm = SimpleNamespace(set_speed_percent=MagicMock(return_value=True))
    robot = SimpleNamespace(
        _arm=arm,
        has_live_arm_feedback=MagicMock(return_value=True),
        get_joint_angles=MagicMock(return_value=measured),
        set_teach_mode=MagicMock(),
        get_arm_status=MagicMock(return_value=status),
    )
    app = SimpleNamespace(
        robot=robot,
        require_robot=MagicMock(return_value=robot),
        safe_bicep_position_reached=True,
        log_arm_debug=MagicMock(),
        log_message=MagicMock(),
        arm_debug_text=MagicMock(),
        speed_var=SimpleNamespace(set=MagicMock()),
        set_arm_status_display=MagicMock(),
    )

    NeroLab.reenable_arm(app)

    robot.set_teach_mode.assert_called_once_with(False, hold_target=measured)


def test_task_handoff_resets_brake_settled_safe_bicep_to_powered_target():
    status = SimpleNamespace(msg=SimpleNamespace(arm_status="NORMAL"))
    robot = SimpleNamespace(
        is_connected=True,
        get_arm_status=MagicMock(return_value=status),
        get_joint_angles=MagicMock(
            side_effect=[SAFE_BICEP_BRAKED_JOINTS, SAFE_BICEP_JOINTS]
        ),
    )
    app = SimpleNamespace(
        robot=robot,
        connect_robot=MagicMock(),
        log_message=MagicMock(),
        safe_bicep_reset=MagicMock(),
        task_handoff_anchor=None,
        disconnect_robot=MagicMock(),
    )

    NeroLab._prepare_task_process(app, emergency_brake=False)

    app.safe_bicep_reset.assert_called_once_with()


def test_normal_close_preserves_enabled_safe_bicep():
    robot = SimpleNamespace(
        is_connected=True,
        get_joint_angles=MagicMock(return_value=SAFE_BICEP_JOINTS),
        disconnect=MagicMock(),
        emergency_disconnect=MagicMock(),
    )
    app = SimpleNamespace(
        robot=robot,
        safe_bicep_reset=MagicMock(),
        stop_windows_can=MagicMock(),
        destroy=MagicMock(),
    )

    NeroLab.close_application(app)

    robot.disconnect.assert_called_once_with(disable_arm=False)
    robot.emergency_disconnect.assert_not_called()
    assert app.robot is None
    app.destroy.assert_called_once_with()