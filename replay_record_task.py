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
from task_trajectory import (
    amplify_gripper_samples,
    command_recorded_gripper,
    is_amplified_gripper_opening,
    prepare_gripper_for_replay,
    prepare_replay_samples,
    resample_replay_samples,
    safe_bicep_shutdown,
    smooth_move_with_recovery,
    wait_for_gripper_release_pose,
)

REPLAY_SPEED_PERCENT = 25
DATASET_FPS = 30

REPO_ID = "adrian/nero_replayed"
JOINT_NAMES = [f"joint{i}.pos" for i in range(1, 8)]
STATE_NAMES = [*JOINT_NAMES, "gripper.width_m", "gripper.force"]
ACTION_NAMES = [*JOINT_NAMES, "gripper.width_m", "gripper.force"]

FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (9,), "names": STATE_NAMES},
    "action": {"dtype": "float32", "shape": (9,), "names": ACTION_NAMES},
    "observation.images.wrist": {"dtype": "video", "shape": (3, 480, 640), "names": ["channel", "height", "width"]},
    "observation.images.overview": {"dtype": "video", "shape": (3, 480, 640), "names": ["channel", "height", "width"]},
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
    return image


def main(task_file: Path, dataset_root: Path, amplified_gripper: bool = False) -> None:
    recording = json.loads(task_file.read_text())
    samples = prepare_replay_samples(recording)
    if amplified_gripper:
        samples = amplify_gripper_samples(samples)
    samples = resample_replay_samples(samples, DATASET_FPS)
    task = str(recording.get("task", task_file.stem))

    if dataset_root.exists():
        if (dataset_root / "meta" / "info.json").exists():
            dataset = LeRobotDataset.resume(
                repo_id=REPO_ID,
                root=dataset_root,
                video_backend="pyav",
            )
        else:
            shutil.rmtree(dataset_root)
            dataset = None
    else:
        dataset = None
    if dataset is None:
        dataset = LeRobotDataset.create(
            repo_id=REPO_ID,
            fps=DATASET_FPS,
            features=FEATURES,
            robot_type="nero",
            root=dataset_root,
            use_videos=True,
            video_backend="pyav",
        )

    robot = Nero(NeroConfig(
        id="nero_replay_record",
        bitrate=1_000_000,
        firmware_version="v121",
        speed_percent=REPLAY_SPEED_PERCENT,
        has_gripper=True,
        has_camera=True,
        has_overview_camera=True,
        reset_on_connect=False,
    ))
    robot.connect(calibrate=False)
    effector = robot._get_gripper_effector()
    if effector is None:
        raise RuntimeError("NERO gripper effector is unavailable")
    prepare_gripper_for_replay(effector)
    if hasattr(robot._arm, "get_joints_enable_status_list"):
        enabled_joints = robot._arm.get_joints_enable_status_list()
        if not all(enabled_joints):
            raise RuntimeError(f"Replay aborted: arm joints are disabled: {enabled_joints}")
    robot._arm.set_motion_mode(robot._arm.OPTIONS.MOTION_MODE.J)
    robot._arm.set_speed_percent(REPLAY_SPEED_PERCENT)
    cv2.namedWindow("NERO replay recording", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("NERO replay recording", 1280, 480)
    print(f"Replaying taught task and recording dataset: {dataset_root}")
    print("Streaming recorded joint targets on their original timeline.")

    stop_requested = False
    try:
        previous_gripper: tuple[str, float, float] | None = None
        previous_grasping = False
        smooth_move_with_recovery(
            robot,
            [float(value) for value in samples[0]["joints"]],
            "Replay recording",
        )
        replay_started = time.monotonic()
        for index, sample in enumerate(samples):
            target = [float(value) for value in sample["joints"]]
            mode = str(sample.get("gripper_mode", "width"))
            if mode != "width":
                raise RuntimeError(
                    "ML dataset recording requires gripper width-mode samples; "
                    f"got {mode!r} at frame {index}"
                )
            gripper = float(np.clip(float(sample.get("gripper", 0.1)), 0.0, 0.1))
            gripper_force = float(sample.get("gripper_force", GRIPPER_REPLAY_FORCE))
            robot._arm.move_js(target)
            if amplified_gripper and is_amplified_gripper_opening(sample, previous_grasping):
                pause_started = time.monotonic()
                wait_for_gripper_release_pose(robot, target, "Replay recording")
                replay_started += time.monotonic() - pause_started
            previous_gripper = command_recorded_gripper(effector, sample, previous_gripper)
            previous_grasping = bool(sample.get("gripper_grasping", False))
            obs = robot.get_observation()
            state = np.asarray(obs["observation.state"], dtype=np.float32)
            wrist = rgb_image(obs.get("observation.images.wrist"), "wrist")
            overview = rgb_image(obs.get("observation.images.overview"), "overview")
            depth = np.asarray(obs.get("observation.images.wrist_depth", np.zeros((480, 640), dtype=np.float32)), dtype=np.float32)
            if depth.ndim == 3:
                depth = depth[..., 0]
            dataset.add_frame({
                "observation.state": state,
                "action": np.concatenate([
                    np.asarray(target, dtype=np.float32),
                    np.asarray([gripper, gripper_force], dtype=np.float32),
                ]),
                "observation.images.wrist": wrist,
                "observation.images.overview": overview,
                "observation.depth": depth,
                "task": task,
            })
            combined = np.hstack([
                cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(overview, cv2.COLOR_RGB2BGR),
            ])
            cv2.putText(combined, f"Replay recording {index + 1}/{len(samples)}  q: stop", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("NERO replay recording", combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_requested = True
                break
            next_time = float(samples[index + 1]["time"]) if index + 1 < len(samples) else float(sample["time"]) + 1.0 / DATASET_FPS
            deadline = replay_started + next_time
            while time.monotonic() < deadline:
                time.sleep(0.005)
        print("Replay complete; saving episode.")
    finally:
        cv2.destroyAllWindows()
        safe_bicep_shutdown(robot, "Replay recording shutdown")

    dataset.save_episode()
    print(f"Saved replay dataset: {dataset_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay a taught NERO task and record cameras and joint data.")
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--amplified-gripper", action="store_true")
    args = parser.parse_args()
    main(
        args.task_file,
        args.dataset_root,
        amplified_gripper=args.amplified_gripper,
    )
