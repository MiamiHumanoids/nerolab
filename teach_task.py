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
from task_trajectory import append_safe_bicep_return, convert_leader_samples, safe_bicep_shutdown

FPS = 50
DEFAULT_TASK_DIR = Path.home() / "Nero" / "tasks"


def decode_gripper_status(status, control_state: bool = False):
    if status is None:
        return None
    message = getattr(status, "msg", status)
    value = getattr(message, "value", None)
    if value is None:
        return None
    mode = getattr(message, "mode", None)
    if mode is None and control_state:
        status_code = int(getattr(message, "status_code", 0))
        mode = "angle" if status_code in {4, 5, 6, 7} else "width"
    mode = str(mode or "width")
    value = float(value)
    if mode == "width":
        value = float(np.clip(value, 0.0, 0.1))
    return (mode, value), float(getattr(status, "timestamp", 0.0)), float(
        getattr(status, "hz", 0.0)
    )


def read_gripper_channels(effector):
    channels = {}
    for source, getter_name, control_state in (
        ("physical-0x2A8", "get_gripper_status", False),
        ("leader-0x159", "get_gripper_ctrl_states", True),
    ):
        getter = getattr(effector, getter_name, None)
        if getter is None:
            continue
        try:
            decoded = decode_gripper_status(getter(), control_state)
            if decoded is not None:
                channels[source] = decoded
        except Exception:
            continue
    return channels


def read_gripper_state(effector, fallback: tuple[str, float] = ("width", 0.1)):
    channels = read_gripper_channels(effector)
    physical = channels.get("physical-0x2A8")
    if physical is not None:
        return physical[0], physical[1]
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
        reset_on_connect=False,
    ))
    robot.connect(calibrate=False)
    effector = robot._get_gripper_effector()
    if effector is None:
        raise RuntimeError("NERO gripper effector is unavailable")
    if hasattr(effector, "disable_gripper"):
        effector.disable_gripper()
    sequence: list[dict[str, object]] = []
    interval = 1.0 / FPS
    next_sample = time.monotonic()
    last_gripper = ("width", 0.1)
    last_reported_gripper: tuple[str, float] | None = None
    active_gripper_source = "physical-0x2A8"
    initial_channels = read_gripper_channels(effector)
    channel_baselines = {
        source: timestamp
        for source, (_, timestamp, _) in initial_channels.items()
    }
    channel_states = {
        source: state for source, (state, _, _) in initial_channels.items()
    }
    _, gripper_timestamp = read_gripper_state(effector, last_gripper)
    cv2.namedWindow("NERO teach task", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("NERO teach task", 900, 180)
    print(f"Teach mode active. Move the robot manually; samples are recorded at {FPS} FPS.")
    print("Press q in the teach window to save the task.")

    shutdown_hold_target: list[float] | None = None
    try:
        robot.set_teach_mode(True)
        enable_can_feedback_push(robot._arm)
        robot._arm._send_msg(ArmMsgMotionCtrl(grag_teach_ctrl=1))
        if hasattr(effector, "set_gripper_teaching_pendant_param"):
            configured = effector.set_gripper_teaching_pendant_param(
                teaching_range_per=100,
                max_range_config=0.1,
                teaching_friction=1,
                timeout=5.0,
            )
            print(
                f"Leader gripper teaching parameters "
                f"{'configured' if configured else 'not acknowledged'}.",
                flush=True,
            )
        last_gripper = wait_for_fresh_gripper_feedback(
            effector, gripper_timestamp
        )
        while True:
            now = time.monotonic()
            if now >= next_sample:
                joint_state = [float(value) for value in robot.get_teach_joint_angles()]
                channels = read_gripper_channels(effector)
                for source, (gripper_state, timestamp, hz) in channels.items():
                    baseline = channel_baselines.get(source, 0.0)
                    previous_state = channel_states.get(source)
                    source_changed = previous_state is not None and (
                        gripper_state[0] != previous_state[0]
                        or abs(gripper_state[1] - previous_state[1]) > 0.0001
                    )
                    if timestamp > baseline and source_changed:
                        active_gripper_source = source
                        last_gripper = gripper_state
                        print(
                            f"Teach gripper source={source} timestamp={timestamp:.6f} "
                            f"hz={hz:.1f}",
                            flush=True,
                        )
                    channel_baselines[source] = max(baseline, timestamp)
                    channel_states[source] = gripper_state
                threshold = 0.5 if last_gripper[0] == "angle" else 0.0005
                if (
                    last_reported_gripper is None
                    or last_gripper[0] != last_reported_gripper[0]
                    or abs(last_gripper[1] - last_reported_gripper[1]) > threshold
                ):
                    print(
                        f"Teach gripper feedback: mode={last_gripper[0]} "
                        f"value={last_gripper[1]:.6f} source={active_gripper_source}",
                        flush=True,
                    )
                    last_reported_gripper = last_gripper
                sequence.append({
                    "time": time.monotonic(),
                    "joints": joint_state,
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
                if sequence:
                    converted = convert_leader_samples(sequence, follower_anchor)
                    shutdown_hold_target = [
                        float(value) for value in converted[-1]["joints"]
                    ]
                break
            time.sleep(0.005)
    finally:
        try:
            robot._arm._send_msg(ArmMsgMotionCtrl(grag_teach_ctrl=2))
        except Exception:
            pass
        finally:
            robot.set_teach_mode(False, hold_target=shutdown_hold_target)
            cv2.destroyAllWindows()

    if len(sequence) < 2:
        safe_bicep_shutdown(robot, "Teach shutdown")
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
        safe_bicep_shutdown(robot, "Teach shutdown")
        return
    sequence = append_safe_bicep_return(sequence)
    output.write_text(json.dumps({
        "task": task,
        "fps": FPS,
        "joint_space": "follower",
        "follower_anchor": follower_anchor,
        "safe_bicep_return": True,
        "replay_ready": True,
        "samples": sequence,
    }, indent=2))
    print(f"Saved taught task: {output} ({len(sequence)} samples)", flush=True)
    safe_bicep_shutdown(robot, "Teach shutdown")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record a manually taught NERO trajectory.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--follower-anchor", type=float, nargs=7, required=True)
    args = parser.parse_args()
    main(args.task, args.output, args.follower_anchor)
