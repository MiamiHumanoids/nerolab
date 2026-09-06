#!/usr/bin/env python3
"""Record a manually taught NERO joint trajectory for later replay."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from lerobot_robot_nero import Nero, NeroConfig
from pyAgxArm.protocols.can_protocol.msgs.nero.default import ArmMsgMotionCtrl

FPS = 15
JOINT_TOLERANCE = 0.01
MOTION_TIMEOUT = 5.0
DEFAULT_TASK_DIR = Path.home() / "Nero" / "tasks"
JOINT_LIMITS = [
    (-2.695261, 2.695261),
    (-1.73533, 1.73533),
    (-2.747621, 2.747621),
    (-1.002291, 2.136755),
    (-2.747621, 2.747621),
    (-0.723039, 0.949932),
    (-1.560797, 1.560797),
]


def read_gripper_width(effector, fallback: float = 0.1) -> float:
    try:
        status = effector.get_gripper_status()
        value = getattr(getattr(status, "msg", status), "value", None)
        if value is not None:
            return float(np.clip(float(value), 0.0, 0.1))
    except Exception:
        pass
    return fallback


def wait_for_target(robot: Nero, target: list[float], timeout: float = MOTION_TIMEOUT) -> None:
    time.sleep(0.05)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = robot.get_joint_angles()
        if all(abs(float(value) - goal) <= JOINT_TOLERANCE for value, goal in zip(current, target)):
            return
        time.sleep(0.02)
    raise RuntimeError(
        f"Replay did not reach recorded target {target}; current joints={robot.get_joint_angles()}"
    )


def main(task: str, output: Path) -> None:
    robot = Nero(NeroConfig(
        id="nero_teach",
        can_channel="can0",
        bitrate=1_000_000,
        firmware_version="v121",
        speed_percent=25,
        has_gripper=True,
        has_camera=True,
        has_overview_camera=True,
        overview_camera_index=0,
    ))
    robot.connect(calibrate=False)
    effector = robot._get_gripper_effector()
    if effector is None:
        raise RuntimeError("NERO gripper effector is unavailable")
    if hasattr(effector, "disable_gripper"):
        effector.disable_gripper()
    if hasattr(effector, "set_gripper_teaching_pendant_param"):
        configured = effector.set_gripper_teaching_pendant_param(
            max_range_config=0.1,
            timeout=5.0,
        )
        print(f"Gripper teaching range {'configured' if configured else 'not acknowledged'}.")
    sequence: list[dict[str, object]] = []
    interval = 1.0 / FPS
    next_sample = time.monotonic()
    last_gripper = 0.1
    cv2.namedWindow("NERO teach task", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("NERO teach task", 900, 180)
    print("Teach mode active. Move the robot manually; samples are recorded at 15 FPS.")
    print("Press q in the teach window to save the task.")

    try:
        robot.set_teach_mode(True)
        robot._arm._send_msg(ArmMsgMotionCtrl(grag_teach_ctrl=1))
        while True:
            now = time.monotonic()
            if now >= next_sample:
                raw_state = [float(value) for value in robot.get_teach_joint_angles()]
                state = [
                    max(lower, min(upper, value))
                    for value, (lower, upper) in zip(raw_state, JOINT_LIMITS)
                ]
                if state != raw_state:
                    print(f"Clamped taught joint sample: {raw_state} -> {state}")
                last_gripper = read_gripper_width(effector, last_gripper)
                sequence.append({
                    "time": time.monotonic(),
                    "joints": state,
                    "gripper": last_gripper,
                })
                next_sample += interval
                if next_sample < now:
                    next_sample = now + interval

            canvas = np.zeros((180, 900, 3), dtype=np.uint8)
            cv2.putText(canvas, "TEACH MODE - move the robot manually", (24, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"Samples: {len(sequence)}    Gripper: {last_gripper:.3f} m    Press q to save", (24, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 220, 255), 1, cv2.LINE_AA)
            cv2.imshow("NERO teach task", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            time.sleep(0.005)
    finally:
        try:
            robot._arm._send_msg(ArmMsgMotionCtrl(grag_teach_ctrl=2))
        except Exception:
            pass
        finally:
            robot.set_teach_mode(False)
            cv2.destroyAllWindows()

    if len(sequence) < 2:
        raise RuntimeError("Teach task was too short; record at least two samples.")
    start_time = float(sequence[0]["time"])
    for sample in sequence:
        sample["time"] = float(sample["time"]) - start_time
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"task": task, "fps": FPS, "samples": sequence}, indent=2))
    print(f"Saved taught task: {output} ({len(sequence)} samples)")
    replay_trigger = output.with_suffix(".replay")
    replay_trigger.unlink(missing_ok=True)
    print("Waiting for Replay task from NERO Lab...")
    while not replay_trigger.exists():
        time.sleep(0.1)
    replay_trigger.unlink(missing_ok=True)
    print("Sending every recorded joint target and waiting for joint feedback.")
    robot._arm.set_motion_mode(robot._arm.OPTIONS.MOTION_MODE.J)
    robot._arm.set_speed_percent(25)
    cv2.namedWindow("NERO task replay", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("NERO task replay", 1280, 480)
    previous_gripper = None
    for index, sample in enumerate(sequence):
        target = [float(value) for value in sample["joints"]]
        if len(target) != 7:
            raise ValueError(f"Recorded sample {index + 1} has {len(target)} joints; expected 7")
        gripper = float(np.clip(float(sample.get("gripper", 0.1)), 0.0, 0.1))
        if previous_gripper is None or abs(gripper - previous_gripper) > 0.002:
            effector.move_gripper_m(value=gripper, force=30.0)
            previous_gripper = gripper
        robot._arm.move_j(target)
        wait_for_target(robot, target)
        deadline = time.monotonic() + max(
            1.0 / FPS,
            float(sample["time"]) - (float(sequence[index - 1]["time"]) if index else 0.0),
        )
        while time.monotonic() < deadline:
            observation = robot.get_observation()
            frames = []
            for key, label in (("observation.images.wrist", "wrist"), ("observation.images.overview", "overview")):
                frame = observation.get(key)
                if frame is None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, f"No {label} camera", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    frame = np.asarray(frame)
                    if frame.ndim == 3 and frame.shape[0] in (1, 3):
                        frame = np.transpose(frame, (1, 2, 0))
                    frame = np.clip(frame * 255.0 if frame.dtype != np.uint8 and frame.max() <= 1.0 else frame, 0, 255).astype(np.uint8)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frames.append(frame)
            combined = np.hstack(frames)
            cv2.putText(combined, "Joint replay - q: stop", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("NERO task replay", combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                replay_trigger = None
                break
            time.sleep(0.005)
        if replay_trigger is None:
            break
    cv2.destroyAllWindows()
    try:
        robot.set_teach_mode(False)
    except Exception:
        pass
    robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record a manually taught NERO trajectory.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.task, args.output)
