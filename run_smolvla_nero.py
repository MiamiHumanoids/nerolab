#!/usr/bin/env python3
"""Run a LeRobot SmolVLA checkpoint on NERO with a guarded 8D adapter."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_policy_config, make_pre_post_processors
from lerobot_robot_nero import Nero, NeroConfig

JOINT_COUNT = 7
GRIPPER_MIN_M = 0.0
GRIPPER_MAX_M = 0.1
GRIPPER_FORCE_MAX = 30.0
POLICY_ACTION_SIZE = 8


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def normalize_camera_image(image) -> torch.Tensor:
    image_array = to_numpy(image)
    if image_array.ndim != 3:
        raise ValueError(f"Expected a 3D camera image, got shape {image_array.shape}.")
    if image_array.shape[-1] == 3:
        image_array = np.transpose(image_array, (2, 0, 1))
    elif image_array.shape[0] != 3:
        raise ValueError(f"Expected RGB camera image, got shape {image_array.shape}.")
    image_tensor = torch.from_numpy(image_array)
    if image_tensor.max() > 1.0:
        image_tensor = image_tensor / 255.0
    return image_tensor


def decode_policy_action(action, max_gripper_force: float) -> tuple[np.ndarray, float, float]:
    values = to_numpy(action).reshape(-1)
    if values.size != POLICY_ACTION_SIZE:
        raise ValueError(
            f"SmolVLA returned {values.size} values; expected {POLICY_ACTION_SIZE}."
        )
    if not 0.0 <= max_gripper_force <= GRIPPER_FORCE_MAX:
        raise ValueError("Max gripper force must be between 0 and 30 N.")
    joints = values[:JOINT_COUNT]
    gripper_width = float(np.clip(values[JOINT_COUNT], GRIPPER_MIN_M, GRIPPER_MAX_M))
    return joints, gripper_width, max_gripper_force


def build_policy(checkpoint: Path, dataset_root: Path, device: str):
    metadata = LeRobotDatasetMetadata(
        repo_id="adrian/nero_manual",
        root=dataset_root,
    )
    if metadata.features.get("observation.state", {}).get("shape", [0])[0] != 8:
        raise ValueError(
            "Selected dataset must contain an 8D state: 7 joints and gripper width."
        )
    if metadata.features.get("action", {}).get("shape", [0])[0] != POLICY_ACTION_SIZE:
        raise ValueError(
            "Selected dataset must contain an 8D action: 7 joints and gripper width."
        )
    config = make_policy_config(
        "smolvla",
        pretrained_path=checkpoint,
        device=device,
    )
    policy = make_policy(config, ds_meta=metadata)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint),
        dataset_stats=metadata.stats,
    )
    return metadata, policy, preprocessor, postprocessor


def run(checkpoint: Path, dataset_root: Path, task: str, device: str, confirm: bool, dry_run: bool, fast_motion: bool, max_gripper_force: float) -> None:
    if not confirm and not dry_run:
        raise SystemExit("Hardware motion is blocked. Re-run with --confirm, or use --dry-run.")

    device = auto_device() if device == "auto" else device
    metadata, policy, preprocessor, postprocessor = build_policy(
        checkpoint, dataset_root, device
    )
    robot = Nero(NeroConfig(
        id="smolvla_nero",
        bitrate=1_000_000,
        firmware_version="v121",
        speed_percent=100,
        has_gripper=True,
        has_camera=True,
        has_overview_camera=True,
    ))
    robot.connect(calibrate=False)
    robot.set_teach_mode(False)
    effector = robot._get_gripper_effector()
    if effector is None:
        raise RuntimeError("NERO gripper effector is unavailable")

    print(f"Task: {task}")
    print("Press q in the policy window or Ctrl-C to stop.")
    try:
        while True:
            observation = robot.get_observation()
            observation["observation.state"] = torch.as_tensor(
                observation["observation.state"][:8], dtype=torch.float32
            )
            if "observation.images.wrist" not in observation:
                raise RuntimeError("NERO camera did not provide observation.images.wrist")
            observation["observation.images.wrist"] = normalize_camera_image(
                observation["observation.images.wrist"]
            )
            if "observation.images.overview" in metadata.features:
                observation["observation.images.overview"] = normalize_camera_image(
                    observation["observation.images.overview"]
                )
            if "observation.depth" in metadata.features:
                observation["observation.depth"] = torch.as_tensor(
                    observation["observation.images.wrist_depth"], dtype=torch.float32
                )
            observation["task"] = task
            batch = preprocessor(observation)
            with torch.inference_mode():
                action = policy.select_action(batch)
            action = to_numpy(postprocessor(action)).reshape(-1)
            joints, gripper_width, gripper_force = decode_policy_action(
                action, max_gripper_force
            )
            print(
                f"joints={np.round(joints, 3).tolist()} "
                f"gripper_width_m={gripper_width:.4f} "
                f"max_gripper_force={max_gripper_force:.2f}N"
            )
            if not dry_run:
                if fast_motion:
                    robot._arm.move_js(joints.tolist())
                else:
                    robot._arm.set_motion_mode(robot._arm.OPTIONS.MOTION_MODE.J)
                    robot._arm.move_j(joints.tolist())
                effector.move_gripper_m(value=gripper_width, force=gripper_force)
            if not dry_run:
                key = 255
                try:
                    import cv2
                    key = cv2.waitKey(1) & 0xFF
                except Exception:
                    pass
                if key == ord("q"):
                    break
            time.sleep(0.1)
    finally:
        try:
            if hasattr(effector, "disable_gripper"):
                effector.disable_gripper()
        except Exception:
            pass
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SmolVLA on NERO through an 8D joint/gripper adapter.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Local SmolVLA checkpoint directory.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Local 8D NERO dataset used for normalization.")
    parser.add_argument("--task", required=True, help="Language instruction, for example: pick up the banana")
    parser.add_argument("--device", default="auto", help="Torch device or auto (cuda/ROCm, xpu, mps, or cpu).")
    parser.add_argument("--confirm", action="store_true", help="Allow commands to be sent to the physical robot.")
    parser.add_argument("--dry-run", action="store_true", help="Run inference without sending robot commands.")
    parser.add_argument("--fast-motion", action="store_true", help="Use instantaneous move_js joint commands; may cause mechanical shock.")
    parser.add_argument("--max-gripper-force", type=float, default=3.0, help="Maximum gripper force in newtons (0-30).")
    args = parser.parse_args()
    if not 0.0 <= args.max_gripper_force <= GRIPPER_FORCE_MAX:
        parser.error("--max-gripper-force must be between 0 and 30 N")
    run(args.checkpoint, args.dataset_root, args.task, args.device, args.confirm, args.dry_run, args.fast_motion, args.max_gripper_force)
