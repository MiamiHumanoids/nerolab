#!/usr/bin/env python3
"""Record a manually taught NERO joint trajectory for later replay."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np

from lerobot_robot_nero import Nero, NeroConfig
from pyAgxArm.protocols.can_protocol.msgs.nero.default import ArmMsgMotionCtrl
from task_trajectory import append_safe_bicep_return, convert_leader_samples, safe_bicep_shutdown

FPS = 50
GRIPPER_CONFIG_TIMEOUT_S = 0.25
DEFAULT_TASK_DIR = Path.home() / "Nero" / "tasks"
TEACH_SHUTDOWN_COUNTDOWN_S = 3


def capture_initial_table_setup(robot, output: Path) -> dict[str, object] | None:
    camera = getattr(robot, "_overview_camera", None)
    if camera is None:
        print("Initial table setup image unavailable: no overview webcam configured.", flush=True)
        return None
    try:
        if not camera.is_connected:
            camera.connect()
        frame = None
        for _ in range(5):
            frame = camera.capture_frame()
            if frame is not None:
                break
            time.sleep(0.05)
        if frame is None:
            raise RuntimeError("overview webcam returned no frame")
        rgb = np.asarray(frame)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise RuntimeError(f"unexpected overview image shape {rgb.shape}")
        image_dir = output.parent / "setup_images"
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{output.stem}__initial-table.jpg"
        if not cv2.imwrite(str(image_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"could not write {image_path}")
        metadata = {
            "path": image_path.relative_to(output.parent).as_posix(),
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "camera": "overview_webcam",
            "device_index": getattr(camera, "device_index", None),
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
        }
        print(f"Captured initial table setup: {image_path}", flush=True)
        return metadata
    except Exception as exc:
        print(f"Initial table setup image unavailable: {exc}", flush=True)
        return None


def play_recording_start_beep() -> None:
    try:
        if os.name == "nt":
            import winsound

            winsound.Beep(1000, 250)
        else:
            print("\a", end="", flush=True)
    except (ImportError, OSError, RuntimeError):
        print("\a", end="", flush=True)


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


def leader_feedback_timestamp(arm) -> float:
    getter = getattr(arm, "get_leader_joint_angles", None)
    if getter is None:
        return 0.0
    feedback = getter()
    return float(getattr(feedback, "timestamp", 0.0)) if feedback is not None else 0.0


def follower_hold_target(
    leader_joints: list[float],
    leader_start: list[float],
    follower_anchor: list[float],
) -> list[float]:
    if not (len(leader_joints) == len(leader_start) == len(follower_anchor) == 7):
        raise ValueError("Teach shutdown hold pose requires seven joint values")
    return [
        float(value) + float(anchor) - float(start)
        for value, start, anchor in zip(
            leader_joints, leader_start, follower_anchor
        )
    ]


def wait_for_gravity_compensation(
    arm, previous_leader_timestamp: float = 0.0, timeout: float = 3.0
) -> None:
    deadline = time.monotonic() + timeout
    last_mode = "unknown"
    last_teach_status = "unknown"
    last_leader_timestamp = previous_leader_timestamp
    last_enabled = None
    while time.monotonic() < deadline:
        status = arm.get_arm_status()
        message = getattr(status, "msg", status)
        last_mode = str(getattr(message, "ctrl_mode", "unknown"))
        last_teach_status = str(getattr(message, "teach_status", "unknown"))
        enabled_getter = getattr(arm, "get_joints_enable_status_list", None)
        last_enabled = enabled_getter() if enabled_getter is not None else None
        last_leader_timestamp = leader_feedback_timestamp(arm)
        leader_ready = last_leader_timestamp > previous_leader_timestamp
        motors_ready = last_enabled is None or all(last_enabled)
        if leader_ready and motors_ready:
            print(
                f"Gravity compensation active: ctrl_mode={last_mode} "
                f"teach_status={last_teach_status} "
                f"leader_timestamp={last_leader_timestamp:.6f} "
                f"enabled={last_enabled}",
                flush=True,
            )
            return
        time.sleep(0.05)
    raise RuntimeError(
        "Teach mode did not enter gravity compensation; refusing to record. "
        f"ctrl_mode={last_mode} teach_status={last_teach_status} "
        f"leader_timestamp={last_leader_timestamp:.6f} "
        f"enabled={last_enabled}. Fresh leader-joint feedback is required."
    )


def enter_gravity_compensation(robot, attempts: int = 3) -> list[float]:
    last_error: RuntimeError | None = None
    normalized_anchor: list[float] | None = None
    for attempt in range(1, attempts + 1):
        print(
            f"Preparing controller for Teach mode ({attempt}/{attempts})...",
            flush=True,
        )
        status = robot.get_arm_status()
        message = getattr(status, "msg", status)
        ctrl_mode = str(getattr(message, "ctrl_mode", ""))
        teach_status = str(getattr(message, "teach_status", ""))
        enabled_getter = getattr(robot._arm, "get_joints_enable_status_list", None)
        enabled = enabled_getter() if enabled_getter is not None else None
        controller_ready = (
            attempt == 1
            and "CAN_CTRL" in ctrl_mode
            and "DISABLED" in teach_status
            and (enabled is None or all(enabled))
        )
        if controller_ready:
            print("Controller already ready; skipping follower/reset normalization.", flush=True)
        else:
            robot.set_teach_mode(False)
        normalized_anchor = [float(value) for value in robot.get_joint_angles()]
        leader_timestamp = leader_feedback_timestamp(robot._arm)
        robot.set_teach_mode(True)
        time.sleep(0.2)
        robot._arm._send_msg(ArmMsgMotionCtrl(grag_teach_ctrl=1))
        try:
            wait_for_gravity_compensation(
                robot._arm, leader_timestamp, timeout=3.0
            )
            return normalized_anchor
        except RuntimeError as exc:
            last_error = exc
            if attempt < attempts:
                print(
                    f"Teach mode entry not acknowledged; retrying "
                    f"({attempt + 1}/{attempts}).",
                    flush=True,
                )
    if last_error is not None:
        raise last_error
    raise RuntimeError("Teach mode entry was not attempted")


def show_recording_stopped_countdown(seconds: int = 5) -> None:
    deadline = time.monotonic() + seconds
    while True:
        remaining = max(0, math.ceil(deadline - time.monotonic()))
        canvas = np.zeros((180, 900, 3), dtype=np.uint8)
        cv2.putText(
            canvas,
            "Recording stopped - release robot and step away",
            (24, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Returning to Safe Bicep in {remaining} seconds",
            (24, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (80, 210, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("NERO teach task", canvas)
        cv2.waitKey(50)
        if remaining == 0:
            return


def main(
    task: str,
    output: Path,
    follower_anchor: list[float],
    variation: str = "",
) -> None:
    robot = Nero(NeroConfig(
        id="nero_teach",
        bitrate=1_000_000,
        firmware_version="v121",
        speed_percent=25,
        has_gripper=True,
        has_camera=True,
        has_overview_camera=True,
        reset_on_connect=False,
    ))
    robot.connect(calibrate=False)
    initial_table_setup = capture_initial_table_setup(robot, output)
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
    print("Entering gravity-compensated Teach mode...", flush=True)

    shutdown_hold_target: list[float] | None = None
    recording_anchor = follower_anchor.copy()
    teach_mode_active = False
    try:
        recording_anchor = enter_gravity_compensation(robot)
        anchor_shift = [
            actual - requested
            for actual, requested in zip(recording_anchor, follower_anchor)
        ]
        print(
            "Teach conversion anchor captured after controller normalization: "
            f"anchor={recording_anchor} shift={anchor_shift}",
            flush=True,
        )
        teach_mode_active = True
        print(f"Teach mode active. Move the robot manually; samples are recorded at {FPS} FPS.")
        print("Press Spacebar in the teach window to save the task.")
        if hasattr(effector, "set_gripper_teaching_pendant_param"):
            configured = effector.set_gripper_teaching_pendant_param(
                teaching_range_per=100,
                max_range_config=0.1,
                teaching_friction=1,
                timeout=GRIPPER_CONFIG_TIMEOUT_S,
            )
            print(
                f"Leader gripper teaching parameters "
                f"{'configured' if configured else 'not acknowledged'}.",
                flush=True,
            )
        last_gripper = wait_for_fresh_gripper_feedback(
            effector, gripper_timestamp
        )
        play_recording_start_beep()
        next_sample = time.monotonic()
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
            cv2.putText(canvas, "Backdrive arm and gripper    Spacebar: save", (24, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 220, 255), 1, cv2.LINE_AA)
            cv2.imshow("NERO teach task", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                print(
                    "Recording stopped; release robot and step away. "
                    f"Returning to Safe Bicep in {TEACH_SHUTDOWN_COUNTDOWN_S} seconds.",
                    flush=True,
                )
                show_recording_stopped_countdown(TEACH_SHUTDOWN_COUNTDOWN_S)
                if sequence:
                    shutdown_hold_target = follower_hold_target(
                        robot.get_teach_joint_angles(),
                        [float(value) for value in sequence[0]["joints"]],
                        recording_anchor,
                    )
                break
            time.sleep(0.005)
    finally:
        try:
            robot.set_teach_mode(False, hold_target=shutdown_hold_target)
        except Exception as exc:
            if teach_mode_active:
                raise
            print(f"Teach entry cleanup command was rejected: {exc}", flush=True)
        finally:
            cv2.destroyAllWindows()
            if not teach_mode_active or len(sequence) < 2:
                try:
                    robot.emergency_disconnect()
                except Exception as exc:
                    print(f"Teach entry transport cleanup failed: {exc}", flush=True)

    if len(sequence) < 2:
        safe_bicep_shutdown(
            robot,
            "Teach shutdown",
            engage_brakes=False,
            brake_on_failure=False,
        )
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
        "variation": variation,
        "fps": FPS,
        "joint_space": "leader",
        "follower_anchor": recording_anchor,
        "initial_table_setup": initial_table_setup,
        "samples": sequence,
    }
    output.write_text(json.dumps(recording, indent=2))
    try:
        sequence = convert_leader_samples(sequence, recording_anchor)
    except ValueError as exc:
        recording["replay_ready"] = False
        recording["replay_error"] = str(exc)
        output.write_text(json.dumps(recording, indent=2))
        print(f"Saved taught task: {output} ({len(sequence)} samples)", flush=True)
        print(f"Replay unavailable: {exc}", flush=True)
        safe_bicep_shutdown(
            robot,
            "Teach shutdown",
            engage_brakes=False,
            brake_on_failure=False,
        )
        return
    sequence = append_safe_bicep_return(sequence)
    output.write_text(json.dumps({
        "task": task,
        "variation": variation,
        "fps": FPS,
        "joint_space": "follower",
        "follower_anchor": recording_anchor,
        "safe_bicep_return": True,
        "replay_ready": True,
        "initial_table_setup": initial_table_setup,
        "samples": sequence,
    }, indent=2))
    print(f"Saved taught task: {output} ({len(sequence)} samples)", flush=True)
    try:
        safe_bicep_shutdown(
            robot,
            "Teach shutdown",
            engage_brakes=False,
            brake_on_failure=False,
        )
    except RuntimeError as exc:
        print(
            f"Teach shutdown handed Safe Bicep recovery to Nero Lab: {exc}",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record a manually taught NERO trajectory.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--variation", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--follower-anchor", type=float, nargs=7, required=True)
    args = parser.parse_args()
    try:
        main(args.task, args.output, args.follower_anchor, args.variation)
    except RuntimeError as exc:
        print(f"Teach task unavailable: {exc}", flush=True)
        if "Teach mode did not enter gravity compensation" in str(exc):
            print(
                "The controller rejected Teach mode. Power-cycle the arm controller "
                "before trying Teach Task again.",
                flush=True,
            )
        raise SystemExit(2) from None
