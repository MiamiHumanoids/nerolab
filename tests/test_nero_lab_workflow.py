import os
from datetime import datetime, timezone
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
    sync_tracked_tasks,
    task_layout_image_path,
    timestamp_activity_entry,
)
from dataset_episode_labels import save_dataset_display_name
from task_trajectory import SAFE_BICEP_BRAKED_JOINTS, SAFE_BICEP_JOINTS


def test_sync_tracked_tasks_copies_updates_and_preserves_local_files(tmp_path):
    source = tmp_path / "tracked"
    destination = tmp_path / "runtime"
    (source / "setup_images").mkdir(parents=True)
    destination.mkdir()
    (source / "task.json").write_text('{"task": "tracked"}')
    (source / "setup_images" / "layout.jpg").write_bytes(b"image")
    (destination / "task.json").write_text('{"task": "stale"}')
    (destination / "local-only.json").write_text('{"task": "local"}')

    copied = sync_tracked_tasks(source, destination)

    assert copied == 2
    assert (destination / "task.json").read_text() == '{"task": "tracked"}'
    assert (destination / "setup_images" / "layout.jpg").read_bytes() == b"image"
    assert (destination / "local-only.json").exists()
    assert sync_tracked_tasks(source, destination) == 0


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


def test_activity_trace_timestamp_is_hidden_from_visible_log():
    app = SimpleNamespace(
        activity_trace=[],
        log=MagicMock(),
        status_var=MagicMock(),
    )

    with patch(
        "nero_lab.timestamp_activity_entry",
        return_value="[2026-09-11T10:15:30.123-04:00] Arm connected\n",
    ):
        NeroLab.log_message(app, "Arm connected")

    assert app.activity_trace == [
        "[2026-09-11T10:15:30.123-04:00] Arm connected\n"
    ]
    app.log.insert.assert_called_once_with("end", "Arm connected\n")


def test_timestamp_activity_entry_timestamps_each_line():
    timestamp = datetime(2026, 9, 11, 14, 15, 30, 123000, tzinfo=timezone.utc)

    assert timestamp_activity_entry("first\nsecond", timestamp) == (
        "[2026-09-11T14:15:30.123+00:00] first\n"
        "[2026-09-11T14:15:30.123+00:00] second\n"
    )


def test_gui_spacebar_requests_active_teach_task_stop(tmp_path):
    stop_signal = tmp_path / "teach.stop"
    app = SimpleNamespace(
        active_process_label="Teach task",
        process=SimpleNamespace(poll=lambda: None),
        teach_stop_signal=stop_signal,
        log_message=MagicMock(),
    )

    result = NeroLab._handle_spacebar(app)

    assert result == "break"
    assert stop_signal.exists()
    app.log_message.assert_called_once_with(
        "GUI Spacebar requested Teach task save and Safe Bicep return."
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


def test_teach_completion_retries_partial_safe_bicep_return():
    partial = [0.0, -0.9, 0.023, 2.08, -0.026, 0.076, 1.5]
    robot = SimpleNamespace(
        is_connected=True,
        get_joint_angles=MagicMock(
            side_effect=[[0.6, 0.2, 0.8, 1.3, -0.3, -0.06, 1.44], partial, SAFE_BICEP_JOINTS]
        ),
    )
    app = SimpleNamespace(
        robot=robot,
        safe_bicep_position_reached=False,
        log_message=MagicMock(),
        connect_robot=MagicMock(),
        reenable_arm=MagicMock(),
        safe_bicep_reset=MagicMock(),
        set_arm_status_display=MagicMock(),
        set_joint_slider_values=MagicMock(),
    )

    NeroLab._restore_safe_bicep_after_teach(app)

    assert app.reenable_arm.call_count == 2
    assert app.safe_bicep_reset.call_count == 2
    assert app.safe_bicep_position_reached
    assert any(
        "partial progress" in call.args[0]
        for call in app.log_message.call_args_list
    )


def test_lerobot_replay_completion_retries_partial_safe_bicep_return():
    replay_pose = [0.99, 0.32, 0.31, 1.06, -0.04, -0.33, 1.56]
    partial = [0.98, 0.30, 0.29, 1.08, -0.04, -0.32, 1.56]
    robot = SimpleNamespace(
        is_connected=True,
        get_joint_angles=MagicMock(
            side_effect=[replay_pose, partial, SAFE_BICEP_JOINTS]
        ),
    )
    app = SimpleNamespace(
        robot=robot,
        safe_bicep_position_reached=False,
        log_message=MagicMock(),
        connect_robot=MagicMock(),
        reenable_arm=MagicMock(),
        safe_bicep_reset=MagicMock(),
        set_arm_status_display=MagicMock(),
        set_joint_slider_values=MagicMock(),
    )

    NeroLab._restore_safe_bicep_after_lerobot_replay(app)

    assert app.reenable_arm.call_count == 2
    assert app.safe_bicep_reset.call_count == 2
    assert app.safe_bicep_position_reached
    assert any(
        "LeRobot replay Safe Bicep attempt 1/3 made partial progress"
        in call.args[0]
        for call in app.log_message.call_args_list
    )


def test_reset_path_uses_four_second_profile():
    target = [0.0, 0.0, 0.0, 2.08, 0.0, 0.0, 0.0]
    arm = SimpleNamespace(
        move_js=MagicMock(),
    )
    robot = SimpleNamespace(
        _arm=arm,
        get_joint_angles=MagicMock(return_value=[0.0] * 7),
    )
    app = SimpleNamespace(
        log_message=MagicMock(),
        wait_for_joint_target=MagicMock(),
    )

    with patch("nero_lab.time.sleep"):
        NeroLab.move_joint_path(app, robot, target, "Test reset")

    waypoints = [call.args[0] for call in arm.move_js.call_args_list]
    largest_step = max(
        abs(current[3] - previous[3])
        for previous, current in zip(waypoints, waypoints[1:])
    )
    assert largest_step <= 0.016
    assert len(waypoints) == 201
    assert "duration=4.00s" in app.log_message.call_args_list[0].args[0]
    app.wait_for_joint_target.assert_called_once_with(
        robot,
        target,
        "Test reset",
        timeout=12.0,
        motion_command=arm.move_js,
    )


def test_reset_path_duration_can_be_overridden():
    target = [0.0, 0.0, 0.0, 2.08, 0.0, 0.0, 0.0]
    arm = SimpleNamespace(move_js=MagicMock())
    robot = SimpleNamespace(
        _arm=arm,
        get_joint_angles=MagicMock(return_value=[0.0] * 7),
    )
    app = SimpleNamespace(
        arm_motion_cancel_requested=False,
        log_message=MagicMock(),
        wait_for_joint_target=MagicMock(),
    )

    with patch("nero_lab.time.sleep"):
        NeroLab.move_joint_path(
            app,
            robot,
            target,
            "Upright Reset",
            duration=2.0,
        )

    assert arm.move_js.call_count == 101
    assert "duration=2.00s" in app.log_message.call_args_list[0].args[0]


def test_reset_path_limits_waypoints_to_live_encoder_progress():
    arm = SimpleNamespace(move_js=MagicMock())
    robot = SimpleNamespace(
        _arm=arm,
        get_joint_angles=MagicMock(return_value=[0.0] * 7),
    )
    app = SimpleNamespace(
        arm_motion_cancel_requested=False,
        log_message=MagicMock(),
        wait_for_joint_target=MagicMock(),
    )

    with patch("nero_lab.time.sleep"):
        NeroLab.move_joint_path(
            app,
            robot,
            [1.0] * 7,
            "Feedback-following reset",
            duration=0.02,
        )

    assert arm.move_js.call_args.args[0] == [0.05] * 7


def test_reset_completion_continues_encoder_following_stream():
    current = [0.0] * 7

    class Arm:
        def __init__(self):
            self.targets = []

        def move_js(self, target):
            self.targets.append(target)
            current[:] = target

    arm = Arm()
    robot = SimpleNamespace(
        _arm=arm,
        get_joint_angles=lambda: current.copy(),
    )
    app = SimpleNamespace(
        arm_motion_cancel_requested=False,
        log_message=MagicMock(),
        log_arm_debug=MagicMock(),
    )

    with patch("nero_lab.time.sleep"):
        NeroLab.wait_for_joint_target(
            app,
            robot,
            [0.1] * 7,
            "Completing reset",
            timeout=0.1,
            motion_command=arm.move_js,
        )

    assert arm.targets == [[0.05] * 7, [0.1] * 7]


def test_upright_reset_runs_arm_motion_off_tk_thread():
    robot = SimpleNamespace()
    worker = MagicMock()
    app = SimpleNamespace(
        arm_motion_in_progress=False,
        arm_motion_cancel_requested=True,
        safe_bicep_position_reached=True,
        require_robot=MagicMock(return_value=robot),
        joint_motion_block_reason=MagicMock(return_value=None),
        cancel_slider_motion=MagicMock(),
        set_arm_status_display=MagicMock(),
        _run_upright_reset=MagicMock(),
    )

    with patch("nero_lab.threading.Thread", return_value=worker) as thread:
        NeroLab.upright_reset(app)

    assert app.arm_motion_in_progress
    assert not app.arm_motion_cancel_requested
    app._run_upright_reset.assert_not_called()
    thread.assert_called_once_with(
        target=app._run_upright_reset,
        args=(robot,),
        name="nero-upright-reset",
        daemon=True,
    )
    worker.start.assert_called_once_with()


def test_upright_reset_worker_uses_full_controller_speed():
    arm = SimpleNamespace(set_speed_percent=MagicMock(return_value=None))
    gripper = SimpleNamespace(move_gripper_m=MagicMock())
    robot = SimpleNamespace(
        _arm=arm,
        _get_gripper_effector=MagicMock(return_value=gripper),
    )
    app = SimpleNamespace(
        log_arm_debug=MagicMock(),
        log_message=MagicMock(),
        prepare_reset_motion=MagicMock(),
        move_joint_path=MagicMock(),
        after=MagicMock(),
        _finish_upright_reset=MagicMock(),
    )

    NeroLab._run_upright_reset(app, robot)

    assert arm.set_speed_percent.call_args_list[0].args == (100,)
    app.move_joint_path.assert_called_once_with(
        robot,
        [0.0] * 7,
        "Upright Reset",
        duration=2.0,
    )


def test_safe_bicep_reset_uses_full_controller_speed():
    arm = SimpleNamespace(set_speed_percent=MagicMock(return_value=None))
    gripper = SimpleNamespace(move_gripper_m=MagicMock())
    robot = SimpleNamespace(
        _arm=arm,
        _get_gripper_effector=MagicMock(return_value=gripper),
    )
    app = SimpleNamespace(
        require_robot=MagicMock(return_value=robot),
        joint_motion_block_reason=MagicMock(return_value=None),
        cancel_slider_motion=MagicMock(),
        log_arm_debug=MagicMock(),
        log_message=MagicMock(),
        prepare_reset_motion=MagicMock(),
        move_joint_path=MagicMock(),
        set_joint_slider_values=MagicMock(),
        safe_bicep_position_reached=False,
        gripper_var=SimpleNamespace(set=MagicMock()),
    )

    NeroLab.safe_bicep_reset(app)

    assert arm.set_speed_percent.call_args_list[0].args == (100,)
    app.move_joint_path.assert_called_once()


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


def test_clear_emergency_stop_holds_pose_resets_and_enables():
    measured = [0.01, -1.7, 0.02, 2.1, -0.03, 0.08, 1.5]
    status = SimpleNamespace(
        msg=SimpleNamespace(arm_status="NORMAL", ctrl_mode="CAN_CTRL")
    )
    arm = SimpleNamespace(
        _send_msg=MagicMock(),
        set_follower_mode=MagicMock(),
        move_js=MagicMock(),
        reset=MagicMock(),
        set_motion_mode=MagicMock(),
        enable=MagicMock(return_value=True),
        get_joints_enable_status_list=MagicMock(return_value=[True] * 7),
        set_speed_percent=MagicMock(),
        OPTIONS=SimpleNamespace(MOTION_MODE=SimpleNamespace(J="joint")),
    )
    robot = SimpleNamespace(
        _arm=arm,
        get_joint_angles=MagicMock(return_value=measured),
        configure=MagicMock(),
        _enable_can_feedback=MagicMock(),
        get_arm_status=MagicMock(return_value=status),
    )
    app = SimpleNamespace(
        log_arm_debug=MagicMock(),
        log_message=MagicMock(),
        after=MagicMock(),
        _finish_clear_emergency_stop=MagicMock(),
    )

    with patch("nero_lab.time.sleep"):
        NeroLab._run_clear_emergency_stop(app, robot)

    arm.set_follower_mode.assert_called_once_with()
    arm.reset.assert_called_once_with()
    arm.move_js.assert_any_call(measured)
    arm.enable.assert_called()
    robot.configure.assert_called_once_with()
    robot._enable_can_feedback.assert_called_once_with()
    app.after.assert_called_once_with(0, app._finish_clear_emergency_stop, None)


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