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
from task_trajectory import convert_leader_samples

FPS = 15
DEFAULT_TASK_DIR = Path.home() / "Nero" / "tasks"


def read_gripper_state(
    effector, fallback: tuple[str, float] = ("width", 0.1)
) -> tuple[str, float]:
    try:
        status = effector.get_gripper_status()
        message = getattr(status, "msg", status)
        value = getattr(message, "value", None)
        if value is not None:
            mode = str(getattr(message, "mode", "width"))
            if mode == "width":
                return mode, float(np.clip(float(value), 0.0, 0.1))
            return mode, float(value)
    except Exception:
        pass
    return fallback


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
                state = [float(value) for value in robot.get_teach_joint_angles()]
                last_gripper = read_gripper_state(effector, last_gripper)
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
            cv2.putText(canvas, f"Samples: {len(sequence)}    Gripper: {last_gripper[1]:.3f} {last_gripper[0]}    Press q to save", (24, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 220, 255), 1, cv2.LINE_AA)
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
        robot.disconnect()
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
    robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record a manually taught NERO trajectory.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--follower-anchor", type=float, nargs=7, required=True)
    args = parser.parse_args()
    main(args.task, args.output, args.follower_anchor)
