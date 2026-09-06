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
from task_trajectory import SAFE_BICEP_JOINTS, convert_leader_samples, smooth_move_to_target

FPS = 15
DEFAULT_TASK_DIR = Path.home() / "Nero" / "tasks"


def read_gripper_state(
    effector, fallback: tuple[str, float] = ("width", 0.1)
) -> tuple[tuple[str, float], float]:
    try:
        status = effector.get_gripper_status()
        message = getattr(status, "msg", status)
        value = getattr(message, "value", None)
        if value is not None:
            mode = str(getattr(message, "mode", "width"))
            if mode == "width":
                value = float(np.clip(float(value), 0.0, 0.1))
            return (mode, float(value)), float(getattr(status, "timestamp", 0.0))
    except Exception:
        pass
    return fallback, 0.0


def enable_can_feedback_push(arm) -> None:
    mode = arm._msg_mode
    previous_push = mode.enable_can_push
    previous_move_mode = mode.move_mode
    try:
        mode.enable_can_push = mode.Enums.CanActiveMsgReporting.ENABLE
        mode.move_mode = 255
        arm._set_mode()
    finally:
        mode.enable_can_push = previous_push
        mode.move_mode = previous_move_mode


def wait_for_fresh_gripper_feedback(
    effector, previous_timestamp: float, timeout: float = 2.0
) -> tuple[str, float]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state, timestamp = read_gripper_state(effector)
        if timestamp > previous_timestamp:
            status = effector.get_gripper_status()
            print(
                f"Physical gripper feedback active: timestamp={timestamp:.6f} "
                f"hz={float(getattr(status, 'hz', 0.0)):.1f}",
                flush=True,
            )
            return state
        time.sleep(0.01)
    raise RuntimeError(
        "No fresh physical gripper feedback after entering Teach mode; "
        "CAN 0x2A8 did not resume."
    )


def return_to_safe_bicep_and_disconnect(robot: Nero) -> None:
    try:
        robot._arm.set_speed_percent(25)
        smooth_move_to_target(robot, SAFE_BICEP_JOINTS, "Teach shutdown")
        print("Teach shutdown: Safe Bicep reached.", flush=True)
    finally:
        try:
            robot.engage_brakes()
            print("Teach shutdown: emergency-stop resting pose settled.", flush=True)
        finally:
            robot.disconnect(disable_arm=False)


def main(task: str, output: Path, follower_anchor: list[float]) -> None:
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
    last_gripper = ("width", 0.1)
    last_reported_gripper: tuple[str, float] | None = None
    _, gripper_timestamp = read_gripper_state(effector, last_gripper)
    cv2.namedWindow("NERO teach task", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("NERO teach task", 900, 180)
    print("Teach mode active. Move the robot manually; samples are recorded at 15 FPS.")
    print("Press q in the teach window to save the task.")

    try:
        robot.set_teach_mode(True)
        enable_can_feedback_push(robot._arm)
        robot._arm._send_msg(ArmMsgMotionCtrl(grag_teach_ctrl=1))
        last_gripper = wait_for_fresh_gripper_feedback(
            effector, gripper_timestamp
        )
        while True:
            now = time.monotonic()
            if now >= next_sample:
                state = [float(value) for value in robot.get_teach_joint_angles()]
                last_gripper, _ = read_gripper_state(effector, last_gripper)
                threshold = 0.5 if last_gripper[0] == "angle" else 0.0005
                if (
                    last_reported_gripper is None
                    or last_gripper[0] != last_reported_gripper[0]
                    or abs(last_gripper[1] - last_reported_gripper[1]) > threshold
                ):
                    print(
                        f"Teach gripper feedback: mode={last_gripper[0]} "
                        f"value={last_gripper[1]:.6f}",
                        flush=True,
                    )
                    last_reported_gripper = last_gripper
                sequence.append({
                    "time": time.monotonic(),
                    "joints": state,
                    "gripper": last_gripper[1],
                    "gripper_mode": last_gripper[0],
                })
                next_sample += interval
                if next_sample < now:
                    next_sample = now + interval

            canvas = np.zeros((180, 900, 3), dtype=np.uint8)
            cv2.putText(canvas, "TEACH MODE - move the robot manually", (24, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"Samples: {len(sequence)}    Gripper: {last_gripper[1]:.3f} {last_gripper[0]}", (24, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, "Backdrive arm and gripper    q: save", (24, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 220, 255), 1, cv2.LINE_AA)
            cv2.imshow("NERO teach task", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
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
        return_to_safe_bicep_and_disconnect(robot)
        raise RuntimeError("Teach task was too short; record at least two samples.")
    start_time = float(sequence[0]["time"])
    for sample in sequence:
        sample["time"] = float(sample["time"]) - start_time
    gripper_values = [float(sample["gripper"]) for sample in sequence]
    print(
        f"Recorded gripper range: min={min(gripper_values):.6f} "
        f"max={max(gripper_values):.6f}",
        flush=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    recording = {
        "task": task,
        "fps": FPS,
        "joint_space": "leader",
        "follower_anchor": follower_anchor,
        "samples": sequence,
    }
    output.write_text(json.dumps(recording, indent=2))
    try:
        sequence = convert_leader_samples(sequence, follower_anchor)
    except ValueError as exc:
        recording["replay_ready"] = False
        recording["replay_error"] = str(exc)
        output.write_text(json.dumps(recording, indent=2))
        print(f"Saved taught task: {output} ({len(sequence)} samples)", flush=True)
        print(f"Replay unavailable: {exc}", flush=True)
        return_to_safe_bicep_and_disconnect(robot)
        return
    output.write_text(json.dumps({
        "task": task,
        "fps": FPS,
        "joint_space": "follower",
        "follower_anchor": follower_anchor,
        "replay_ready": True,
        "samples": sequence,
    }, indent=2))
    print(f"Saved taught task: {output} ({len(sequence)} samples)", flush=True)
    return_to_safe_bicep_and_disconnect(robot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record a manually taught NERO trajectory.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--follower-anchor", type=float, nargs=7, required=True)
    args = parser.parse_args()
    main(args.task, args.output, args.follower_anchor)
