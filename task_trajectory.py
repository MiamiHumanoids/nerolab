"""Shared conversion and playback helpers for taught NERO joint trajectories."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

SAFE_BICEP_JOINTS = [0.0, -1.68, 0.023, 2.08, -0.026, 0.076, 1.5]
SAFE_BICEP_BRAKED_JOINTS = [0.0, -1.7655, 0.023, 2.1964, -0.026, 0.0765, 1.6895]
CONTROL_PRIME_POSE = [-0.4, 0.0, 0.4, -1.57, 0.0, -3.14]
COMMAND_JOINT_LIMITS = [
    (-2.705261, 2.705261),
    (-1.74533, 1.74533),
    (-2.757621, 2.757621),
    (-1.012291, 2.146755),
    (-2.757621, 2.757621),
    (-0.733039, 0.959932),
    (-1.570797, 1.570797),
]
STREAM_INTERVAL_S = 0.02
STREAM_SPEED_RAD_S = 0.4
TARGET_TOLERANCE = 0.01
TARGET_TIMEOUT_S = 5.0


def is_safe_bicep_pose(
    values: list[float],
    target_tolerance: float = 0.1,
    braked_tolerance: float = 0.06,
) -> bool:
    if len(values) != len(SAFE_BICEP_JOINTS):
        return False
    poses = (
        (SAFE_BICEP_JOINTS, target_tolerance),
        (SAFE_BICEP_BRAKED_JOINTS, braked_tolerance),
    )
    return any(
        all(abs(float(value) - target) <= tolerance for value, target in zip(values, pose))
        for pose, tolerance in poses
    )


def interpolated_joint_trajectory(
    samples: list[dict[str, Any]], interval: float = STREAM_INTERVAL_S
) -> list[tuple[float, list[float], int | None]]:
    points = [(float(samples[0]["time"]), [float(value) for value in samples[0]["joints"]], 0)]
    for sample_index in range(len(samples) - 1):
        current = samples[sample_index]
        following = samples[sample_index + 1]
        start_time = float(current["time"])
        end_time = float(following["time"])
        duration = end_time - start_time
        if duration <= 0.0:
            raise ValueError(f"Recorded sample {sample_index + 2} has a non-increasing timestamp.")
        start = [float(value) for value in current["joints"]]
        target = [float(value) for value in following["joints"]]
        step_count = max(1, math.ceil(duration / interval))
        for step_index in range(1, step_count + 1):
            fraction = step_index / step_count
            points.append((
                start_time + duration * fraction,
                [value + (goal - value) * fraction for value, goal in zip(start, target)],
                sample_index + 1 if step_index == step_count else None,
            ))
    return points


def stream_recorded_trajectory(
    robot: Any,
    samples: list[dict[str, Any]],
    sample_callback: Callable[[dict[str, Any], int], None] | None = None,
) -> None:
    points = interpolated_joint_trajectory(samples)
    started = time.monotonic() - points[0][0]
    for sample_time, target, sample_index in points:
        remaining = started + sample_time - time.monotonic()
        if remaining > 0.0:
            time.sleep(remaining)
        robot._arm.move_js(target)
        if sample_index is not None and sample_callback is not None:
            sample_callback(samples[sample_index], sample_index)


def command_recorded_gripper(
    effector: Any,
    sample: dict[str, Any],
    previous: tuple[str, float] | None,
) -> tuple[str, float]:
    mode = str(sample.get("gripper_mode", "width"))
    value = float(sample.get("gripper", 0.1))
    threshold = 0.5 if mode == "angle" else 0.0005
    if previous is not None and mode == previous[0] and abs(value - previous[1]) <= threshold:
        return previous
    if mode == "angle":
        move = getattr(effector, "move_gripper_deg", None)
        if move is None:
            raise RuntimeError("Recorded angle-mode gripper motion requires move_gripper_deg")
    else:
        move = effector.move_gripper_m
    move(value=value, force=1.0)
    print(f"Gripper replay sample: mode={mode} value={value:.6f}")
    return mode, value


def convert_leader_samples(
    samples: list[dict[str, Any]], follower_anchor: list[float]
) -> list[dict[str, Any]]:
    if not samples:
        return []
    first_joints = samples[0].get("joints", [])
    if (
        isinstance(first_joints, (list, tuple))
        and len(first_joints) == 2
        and first_joints[0] in {"width", "angle"}
    ):
        raise ValueError(
            "This task was recorded by build 37 with corrupted joint samples; "
            "the arm trajectory is not recoverable. Re-record the task with build 38."
        )
    leader_start = [float(value) for value in samples[0]["joints"]]
    offsets = [anchor - start for anchor, start in zip(follower_anchor, leader_start)]
    converted: list[dict[str, Any]] = []
    for sample in samples:
        leader_joints = [float(value) for value in sample["joints"]]
        follower_joints = [value + offset for value, offset in zip(leader_joints, offsets)]
        converted.append({**sample, "leader_joints": leader_joints, "joints": follower_joints})
    _validate_targets(converted)
    return converted


def prepare_replay_samples(recording: dict[str, Any]) -> list[dict[str, Any]]:
    samples = list(recording.get("samples", []))
    if len(samples) < 2:
        raise ValueError("Taught task contains fewer than two samples.")
    joint_space = recording.get("joint_space")
    if joint_space == "follower":
        _validate_targets(samples)
        return samples

    follower_anchor = [
        float(value) for value in recording.get("follower_anchor", SAFE_BICEP_JOINTS)
    ]
    if len(follower_anchor) != 7:
        raise ValueError("Taught task follower anchor must contain seven joints.")
    print("Leader-space task detected; converting it with its follower anchor.")
    return convert_leader_samples(samples, follower_anchor)


def smooth_move_to_target(robot: Any, target: list[float], label: str) -> None:
    start = [float(value) for value in robot.get_joint_angles()]
    largest_delta = max(abs(goal - value) for value, goal in zip(start, target))
    duration = max(0.75, largest_delta / STREAM_SPEED_RAD_S)
    step_count = max(1, int(duration / STREAM_INTERVAL_S) + 1)
    move_js = getattr(robot._arm, "move_js", None)
    if move_js is None:
        raise RuntimeError("Installed pyAgxArm does not provide move_js for taught-task replay")
    print(
        f"{label}: smooth move to first sample in {duration:.2f}s ({step_count} steps) "
        f"start={start} target={target}."
    )
    started = time.monotonic()
    for step_index in range(1, step_count + 1):
        progress = step_index / step_count
        fraction = progress * progress * (3.0 - 2.0 * progress)
        waypoint = [
            value + (goal - value) * fraction
            for value, goal in zip(start, target)
        ]
        move_js(waypoint)
        remaining = started + step_index * duration / step_count - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
    deadline = time.monotonic() + TARGET_TIMEOUT_S
    while time.monotonic() < deadline:
        current = [float(value) for value in robot.get_joint_angles()]
        if all(abs(value - goal) <= TARGET_TOLERANCE for value, goal in zip(current, target)):
            return
        time.sleep(STREAM_INTERVAL_S)
    raise RuntimeError(
        f"{label} did not reach target; current joints={robot.get_joint_angles()}"
    )


def prepare_safe_bicep_motion(robot: Any, label: str) -> None:
    current = [float(value) for value in robot.get_joint_angles()]
    status = robot.get_arm_status()
    message = getattr(status, "msg", status)
    arm_status = str(getattr(message, "arm_status", ""))
    outside_limits = any(
        value < lower or value > upper
        for value, (lower, upper) in zip(current, COMMAND_JOINT_LIMITS)
    )
    if outside_limits or "NO_SOLUTION" in arm_status or "SINGULARITY" in arm_status:
        reason = "current joints outside command limits" if outside_limits else arm_status
        print(f"{label}: running P-to-J recovery because {reason}.", flush=True)
        robot._arm.set_motion_mode(robot._arm.OPTIONS.MOTION_MODE.P)
        time.sleep(0.2)
        recovered = False
        for attempt in range(1, 5):
            start = [float(value) for value in robot.get_joint_angles()]
            robot._arm.move_p(CONTROL_PRIME_POSE)
            moved = False
            saw_in_progress = False
            started_at = time.monotonic()
            deadline = started_at + 10.0
            while time.monotonic() < deadline:
                current = [float(value) for value in robot.get_joint_angles()]
                moved = moved or any(
                    abs(value - initial) > 0.002
                    for value, initial in zip(current, start)
                )
                status = robot.get_arm_status()
                message = getattr(status, "msg", status)
                arm_status = str(getattr(message, "arm_status", ""))
                motion_status = str(getattr(message, "motion_status", ""))
                saw_in_progress = saw_in_progress or "FAILED" in motion_status
                if (
                    moved
                    and saw_in_progress
                    and "NORMAL" in arm_status
                    and "SUCCESSFULLY" in motion_status
                ):
                    recovered = True
                    break
                if not moved and time.monotonic() - started_at >= 1.5:
                    break
                time.sleep(0.05)
            if recovered:
                print(f"{label}: P recovery completed on attempt {attempt}.", flush=True)
                break
        if not recovered:
            raise RuntimeError(f"{label} P-to-J recovery did not complete")
    robot._arm.set_motion_mode(robot._arm.OPTIONS.MOTION_MODE.J)
    time.sleep(0.2)


def smooth_move_with_recovery(robot: Any, target: list[float], label: str) -> None:
    prepare_safe_bicep_motion(robot, label)
    smooth_move_to_target(robot, target, label)


def safe_bicep_shutdown(robot: Any, label: str) -> None:
    try:
        robot._arm.set_speed_percent(25)
        smooth_move_with_recovery(robot, SAFE_BICEP_JOINTS, label)
        print(f"{label}: Safe Bicep reached.", flush=True)
    finally:
        robot.engage_brakes()
        print(f"{label}: emergency-stop resting pose settled.", flush=True)
        robot.disconnect(disable_arm=False)


def _validate_targets(samples: list[dict[str, Any]]) -> None:
    for index, sample in enumerate(samples, 1):
        target = [float(value) for value in sample["joints"]]
        if len(target) != 7:
            raise ValueError(f"Recorded sample {index} has {len(target)} joints; expected 7")
        if not all(math.isfinite(value) for value in target):
            raise ValueError(f"Recorded sample {index} contains a non-finite joint value: {target}")
