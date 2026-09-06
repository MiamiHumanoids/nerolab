#!/usr/bin/env python3
"""Replay a taught NERO trajectory while recording a LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lerobot_robot_nero import Nero, NeroConfig

REPLAY_SPEED_PERCENT = 25
JOINT_TOLERANCE = 0.01
MOTION_TIMEOUT = 5.0

REPO_ID = "adrian/nero_replayed"
FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (7,), "names": None},
    "action": {"dtype": "float32", "shape": (8,), "names": None},
    "observation.images.wrist": {"dtype": "image", "shape": (3, 480, 640), "names": ["channel", "height", "width"]},
    "observation.images.overview": {"dtype": "image", "shape": (3, 480, 640), "names": ["channel", "height", "width"]},
    "observation.depth": {"dtype": "float32", "shape": (480, 640), "names": None},
}


def rgb_image(value: object, label: str) -> np.ndarray:
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


def main(task_file: Path, dataset_root: Path) -> None:
    recording = json.loads(task_file.read_text())
    samples = recording.get("samples", [])
    if len(samples) < 2:
        raise ValueError("Taught task contains fewer than two samples.")
    task = str(recording.get("task", task_file.stem))

    if dataset_root.exists():
        if (dataset_root / "meta" / "info.json").exists():
            dataset = LeRobotDataset(repo_id=REPO_ID, root=dataset_root, download_videos=False)
            episode_index = dataset.meta.total_episodes
        else:
            shutil.rmtree(dataset_root)
            dataset = None
    else:
        dataset = None
    if dataset is None:
        dataset = LeRobotDataset.create(repo_id=REPO_ID, fps=15, features=FEATURES, robot_type="nero", root=dataset_root, use_videos=True)
        episode_index = 0
    dataset.episode_buffer = dataset.create_episode_buffer(episode_index=episode_index)

    robot = Nero(NeroConfig(
        id="nero_replay_record",
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
        enabled_joints = robot._arm.get_joints_enable_status_list()
        if not all(enabled_joints):
            raise RuntimeError(f"Replay aborted: arm joints are disabled: {enabled_joints}")
    robot._arm.set_motion_mode(robot._arm.OPTIONS.MOTION_MODE.J)
    robot._arm.set_speed_percent(REPLAY_SPEED_PERCENT)
    cv2.namedWindow("NERO replay recording", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("NERO replay recording", 1280, 480)
    print(f"Replaying taught task and recording dataset: {dataset_root}")
    print("Sending every recorded joint target and waiting for joint feedback.")

    stop_requested = False
    try:
        start = time.monotonic()
        previous_gripper = None
        for index, sample in enumerate(samples):
            target = [float(value) for value in sample["joints"]]
            if len(target) != 7:
                raise ValueError(f"Recorded sample {index + 1} has {len(target)} joints; expected 7")
            gripper = float(np.clip(float(sample.get("gripper", 0.1)), 0.0, 0.1))
            if previous_gripper is None or abs(gripper - previous_gripper) > 0.002:
                effector.move_gripper_m(value=gripper, force=30.0)
                previous_gripper = gripper
            robot._arm.move_j(target)
            wait_for_target(robot, target)
            duration = float(sample["time"]) - (float(samples[index - 1]["time"]) if index else 0.0)
            deadline = time.monotonic() + max(1.0 / 15.0, duration)
            while time.monotonic() < deadline:
                obs = robot.get_observation()
                state = np.asarray(obs["observation.state"], dtype=np.float32)
                wrist = rgb_image(obs.get("observation.images.wrist"), "wrist")
                overview = rgb_image(obs.get("observation.images.overview"), "overview")
                depth = np.asarray(obs.get("observation.images.wrist_depth", np.zeros((480, 640), dtype=np.float32)), dtype=np.float32)
                if depth.ndim == 3:
                    depth = depth[..., 0]
                dataset.add_frame({
                    "observation.state": state,
                    "action": np.concatenate([state, np.asarray([gripper], dtype=np.float32)]),
                    "observation.images.wrist": wrist,
                    "observation.images.overview": overview,
                    "observation.depth": depth,
                    "task": task,
                })
                combined = np.hstack([wrist, overview])
                cv2.putText(combined, f"Replay recording {index + 1}/{len(samples)}  q: stop", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow("NERO replay recording", combined)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_requested = True
                    break
                time.sleep(0.005)
            if stop_requested:
                break
        print("Replay complete; saving episode.")
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

    dataset.save_episode()
    print(f"Saved replay dataset: {dataset_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay a taught NERO task and record cameras and joint data.")
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    main(args.task_file, args.dataset_root)
