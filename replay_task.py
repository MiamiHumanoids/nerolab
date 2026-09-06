#!/usr/bin/env python3
"""Replay a taught NERO trajectory without recording a dataset."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from lerobot_robot_nero import Nero, NeroConfig
REPLAY_SPEED_PERCENT = 25
JOINT_TOLERANCE = 0.01
MOTION_TIMEOUT = 5.0


def to_bgr(value: object, label: str) -> np.ndarray:
    if value is None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(image, f"No {label} camera", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        return image
    image = np.asarray(value)
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0 if image.max() <= 1.0 else image, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.shape[-1] == 3 else image


def arm_status_text(robot: Nero) -> str:
    status = robot.get_arm_status()
    message = getattr(status, "msg", status)
    return (
        f"arm_status={getattr(message, 'arm_status', '?')} "
        f"ctrl_mode={getattr(message, 'ctrl_mode', '?')} "
        f"motion_status={getattr(message, 'motion_status', '?')} "
        f"err_status={getattr(message, 'err_status', '?')}"
    )


def wait_for_target(robot: Nero, target: list[float], timeout: float = MOTION_TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if joints_are_close(robot, target, tolerance=JOINT_TOLERANCE):
            return
        time.sleep(0.02)
    current = robot.get_joint_angles()
    raise RuntimeError(
        f"Replay did not reach recorded target {target}; current joints={current}; "
        f"{arm_status_text(robot)}"
    )


def joints_are_close(robot: Nero, target: list[float], tolerance: float = 0.01) -> bool:
    try:
        current = robot.get_joint_angles()
    except Exception:
        return False
    return all(abs(float(value) - goal) <= tolerance for value, goal in zip(current, target))


def move_to_recorded_target(robot: Nero, target: list[float]) -> None:
    status = robot.get_arm_status()
    message = getattr(status, "msg", status)
    ctrl_mode = getattr(message, "ctrl_mode", None)
    linkage_mode = "LINKAGE" in str(ctrl_mode)
    try:
        linkage_mode = linkage_mode or int(ctrl_mode) == 6
    except (TypeError, ValueError):
        pass

    robot._arm.move_j(target)
    if linkage_mode:
        time.sleep(0.25)
        print("Control transitioned from linkage to CAN; resending the first recorded target.")
        robot._arm.move_j(target)
    wait_for_target(robot, target)


def main(task_file: Path) -> None:
    recording = json.loads(task_file.read_text())
    samples = recording.get("samples", [])
    if len(samples) < 2:
        raise ValueError("Taught task contains fewer than two samples.")

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
    robot.set_teach_mode(False)
    if hasattr(robot._arm, "get_joints_enable_status_list"):
        enabled = robot._arm.get_joints_enable_status_list()
        if not all(enabled):
            raise RuntimeError(f"Replay aborted: arm joints are disabled: {enabled}; {arm_status_text(robot)}")
    robot._arm.set_motion_mode(robot._arm.OPTIONS.MOTION_MODE.J)
    robot._arm.set_speed_percent(REPLAY_SPEED_PERCENT)
    time.sleep(0.2)
    print(f"Replay arm ready: {arm_status_text(robot)}")

    cv2.namedWindow("NERO task replay", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("NERO task replay", 1280, 480)
    print(f"Replaying task without recording: {task_file}")
    print("Sending every recorded joint target and waiting for joint feedback.")
    stop_requested = False
    previous_gripper = None
    try:
        for index, sample in enumerate(samples):
            target = [float(value) for value in sample["joints"]]
            if len(target) != 7:
                raise ValueError(f"Recorded sample {index + 1} has {len(target)} joints; expected 7")
            gripper = float(np.clip(float(sample.get("gripper", 0.1)), 0.0, 0.1))
            if previous_gripper is None or abs(gripper - previous_gripper) > 0.002:
                effector.move_gripper_m(value=gripper, force=30.0)
                previous_gripper = gripper
            move_to_recorded_target(robot, target)
            if index == 0 or index % 25 == 0:
                print(f"Replay sample {index + 1}/{len(samples)} | {arm_status_text(robot)}")
            previous_time = float(samples[index - 1]["time"]) if index else 0.0
            deadline = time.monotonic() + max(1.0 / 15.0, float(sample["time"]) - previous_time)
            while time.monotonic() < deadline:
                observation = robot.get_observation()
                wrist = to_bgr(observation.get("observation.images.wrist"), "wrist")
                overview = to_bgr(observation.get("observation.images.overview"), "overview")
                combined = np.hstack([wrist, overview])
                cv2.putText(combined, "Joint replay - q: stop", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow("NERO task replay", combined)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_requested = True
                    break
                time.sleep(0.005)
            if stop_requested:
                break
    finally:
        cv2.destroyAllWindows()
        try:
            robot._arm.set_follower_mode()
            robot._arm.reset()
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and not robot._arm.enable():
                time.sleep(0.1)
            robot._arm.set_motion_mode(robot._arm.OPTIONS.MOTION_MODE.J)
        except Exception:
            pass
        robot.disconnect()
    print("Task replay complete; no dataset was recorded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay a taught NERO task without recording data.")
    parser.add_argument("--task-file", type=Path, required=True)
    args = parser.parse_args()
    main(args.task_file)
