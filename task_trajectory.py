"""Shared conversion and playback helpers for taught NERO joint trajectories."""

from __future__ import annotations

import time
from typing import Any

SAFE_BICEP_JOINTS = [0.0, -1.68, 0.023, 2.08, -0.026, 0.076, 1.5]
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
LIMIT_MARGIN = 0.005
TARGET_TOLERANCE = 0.01
TARGET_TIMEOUT_S = 5.0


def joints_within_limits(joints: list[float]) -> bool:
    return len(joints) == 7 and all(
        lower <= float(value) <= upper
        for value, (lower, upper) in zip(joints, COMMAND_JOINT_LIMITS)
    )


def convert_leader_samples(
    samples: list[dict[str, Any]], follower_anchor: list[float]
) -> list[dict[str, Any]]:
    if not samples:
        return []
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

    targets = [[float(value) for value in sample["joints"]] for sample in samples]
    if joint_space is None and all(joints_within_limits(target) for target in targets):
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
    print(f"{label}: smooth move to first sample in {duration:.2f}s ({step_count} steps).")
    started = time.monotonic()
    for step_index in range(1, step_count + 1):
        progress = step_index / step_count
        fraction = progress * progress * (3.0 - 2.0 * progress)
        waypoint = [
            min(
                max(value + (goal - value) * fraction, lower + LIMIT_MARGIN),
                upper - LIMIT_MARGIN,
            )
            for value, goal, (lower, upper) in zip(start, target, COMMAND_JOINT_LIMITS)
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
        f"{label} did not reach the first recorded target; current joints={robot.get_joint_angles()}"
    )


def _validate_targets(samples: list[dict[str, Any]]) -> None:
    for index, sample in enumerate(samples, 1):
        target = [float(value) for value in sample["joints"]]
        if len(target) != 7:
            raise ValueError(f"Recorded sample {index} has {len(target)} joints; expected 7")
        if not joints_within_limits(target):
            raise ValueError(f"Recorded sample {index} cannot be converted inside command limits: {target}")
