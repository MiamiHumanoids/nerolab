#!/usr/bin/env python3
"""Replay a taught NERO trajectory without recording a dataset."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from lerobot_robot_nero import Nero, NeroConfig
from task_trajectory import (
    command_recorded_gripper,
    prepare_replay_samples,
    safe_bicep_shutdown,
    smooth_move_with_recovery,
    stream_recorded_trajectory,
)

REPLAY_SPEED_PERCENT = 25
TRAJECTORY_SPEED_PERCENT = 100


def arm_status_text(robot: Nero) -> str:
    status = robot.get_arm_status()
    message = getattr(status, "msg", status)
    return (
        f"arm_status={getattr(message, 'arm_status', '?')} "
        f"ctrl_mode={getattr(message, 'ctrl_mode', '?')} "
        f"motion_status={getattr(message, 'motion_status', '?')} "
        f"err_status={getattr(message, 'err_status', '?')}"
    )


def main(task_file: Path) -> None:
    recording = json.loads(task_file.read_text())
    samples = prepare_replay_samples(recording)

    robot = Nero(NeroConfig(
        id="nero_task_replay",
        can_channel="can0",
        bitrate=1_000_000,
        firmware_version="v121",
        speed_percent=REPLAY_SPEED_PERCENT,
        has_gripper=True,
        has_camera=True,
        has_overview_camera=True,
        overview_camera_index=0,
        reset_on_connect=False,
    ))
    robot.connect(calibrate=False)
    effector = robot._get_gripper_effector()
    if effector is None:
        raise RuntimeError("NERO gripper effector is unavailable")
    if hasattr(effector, "set_gripper_teaching_pendant_param"):
        effector.set_gripper_teaching_pendant_param(max_range_config=0.1, timeout=5.0)
    if hasattr(robot._arm, "get_joints_enable_status_list"):
        enabled = robot._arm.get_joints_enable_status_list()
        if not all(enabled):
            raise RuntimeError(f"Replay aborted: arm joints are disabled: {enabled}; {arm_status_text(robot)}")
    robot._arm.set_motion_mode(robot._arm.OPTIONS.MOTION_MODE.J)
    robot._arm.set_speed_percent(REPLAY_SPEED_PERCENT)
    time.sleep(0.2)
    print(f"Replay arm ready: {arm_status_text(robot)}")

    print(f"Replaying task without recording: {task_file}")
    print("Interpolating recorded joint targets at 50 Hz on their original timeline.")
    gripper_modes = sorted({str(sample.get("gripper_mode", "width")) for sample in samples})
    gripper_values = [float(sample.get("gripper", 0.1)) for sample in samples]
    print(
        f"Recorded gripper: modes={gripper_modes} min={min(gripper_values):.6f} "
        f"max={max(gripper_values):.6f}."
    )
    previous_gripper: tuple[str, float] | None = None
    try:
        smooth_move_with_recovery(
            robot,
            [float(value) for value in samples[0]["joints"]],
            "Task replay",
        )
        robot._arm.set_speed_percent(TRAJECTORY_SPEED_PERCENT)
        print(f"Recorded trajectory speed set to {TRAJECTORY_SPEED_PERCENT}%.")
        def apply_sample(sample: dict[str, object], index: int) -> None:
            nonlocal previous_gripper
            previous_gripper = command_recorded_gripper(effector, sample, previous_gripper)
            if index == 0 or index % 25 == 0:
                print(f"Replay sample {index + 1}/{len(samples)} | {arm_status_text(robot)}")
        stream_recorded_trajectory(robot, samples, apply_sample)
        final_target = [float(value) for value in samples[-1]["joints"]]
        final_joints = [float(value) for value in robot.get_joint_angles()]
        final_error = max(abs(value - target) for value, target in zip(final_joints, final_target))
        print(
            f"Replay final tracking error={final_error:.6f} rad "
            f"target={final_target} current={final_joints}."
        )
    finally:
        safe_bicep_shutdown(robot, "Replay shutdown")
    print("Task replay complete; no dataset was recorded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay a taught NERO task without recording data.")
    parser.add_argument("--task-file", type=Path, required=True)
    args = parser.parse_args()
    main(args.task_file)
