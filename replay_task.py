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
from task_trajectory import prepare_replay_samples, smooth_move_to_target

REPLAY_SPEED_PERCENT = 25


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

    cv2.namedWindow("NERO task replay", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("NERO task replay", 1280, 480)
    print(f"Replaying task without recording: {task_file}")
    print("Streaming recorded joint targets on their original timeline.")
    stop_requested = False
    previous_gripper = None
    try:
        smooth_move_to_target(robot, [float(value) for value in samples[0]["joints"]], "Task replay")
        replay_started = time.monotonic()
        for index, sample in enumerate(samples):
            target = [float(value) for value in sample["joints"]]
            gripper = float(np.clip(float(sample.get("gripper", 0.1)), 0.0, 0.1))
            if previous_gripper is None or abs(gripper - previous_gripper) > 0.002:
                effector.move_gripper_m(value=gripper, force=30.0)
                previous_gripper = gripper
            robot._arm.move_js(target)
            if index == 0 or index % 25 == 0:
                print(f"Replay sample {index + 1}/{len(samples)} | {arm_status_text(robot)}")
            next_time = float(samples[index + 1]["time"]) if index + 1 < len(samples) else float(sample["time"]) + 1.0 / 15.0
            deadline = replay_started + next_time
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
