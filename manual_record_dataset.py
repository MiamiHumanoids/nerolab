import argparse
import cv2
import numpy as np
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot_robot_nero import Nero, NeroConfig
from lerobot_robot_nero.console import read_key_nonblocking


DATASET_BASE = Path.home() / "Nero" / "datasets"
REPO_ID = "adrian/nero_manual"
DATASET_FPS = 30
JOINT_NAMES = [f"Joint_{i}" for i in range(1, 8)]
STATE_NAMES = [*JOINT_NAMES, "Gripper"]
ACTION_NAMES = [*JOINT_NAMES, "Gripper"]

FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (8,), "names": STATE_NAMES},
    "action": {"dtype": "float32", "shape": (8,), "names": ACTION_NAMES},
    "observation.images.wrist": {
        "dtype": "video",
        "shape": (3, 480, 640),
        "names": ["channel", "height", "width"],
    },
    "observation.images.overview": {
        "dtype": "video",
        "shape": (3, 480, 640),
        "names": ["channel", "height", "width"],
    },
    "observation.depth": {
        "dtype": "float32",
        "shape": (480, 640),
        "names": None,
    },
}
GRIPPER_CLOSED_WIDTH_M = 0.0
GRIPPER_OPEN_WIDTH_M = 0.1
GRIPPER_FORCE_N = 30.0


def dataset_slug(task: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", task.lower()).strip("-")
    return slug[:50] or "task"


def move_gripper(robot: Nero, effector, width_m: float) -> None:
    try:
        robot.set_teach_mode(False)
        if hasattr(robot._arm, "set_motion_mode") and hasattr(robot._arm.OPTIONS, "MOTION_MODE"):
            robot._arm.set_motion_mode(robot._arm.OPTIONS.MOTION_MODE.P)
        effector.move_gripper_m(value=width_m, force=GRIPPER_FORCE_N)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            status = effector.get_gripper_status()
            position = getattr(status.msg, "value", None) if status is not None else None
            if position is not None and abs(float(position) - width_m) < 0.002:
                break
            time.sleep(0.05)
    finally:
        robot.set_teach_mode(True)


def main(task: str, dataset_root: Path | None = None) -> None:
    root = dataset_root or DATASET_BASE / f"nero_manual__{dataset_slug(task)}"
    info_path = root / "meta" / "info.json"
    tasks_path = root / "meta" / "tasks.parquet"
    if info_path.exists() and tasks_path.exists():
        info_text = info_path.read_text()
        suffixes = []
        if "observation.depth" not in info_text:
            suffixes.append("depth")
        if "observation.images.overview" not in info_text:
            suffixes.append("overview")
        if suffixes:
            root = root.with_name(f"{root.name}__{'_'.join(suffixes)}")
        info_path = root / "meta" / "info.json"
        tasks_path = root / "meta" / "tasks.parquet"
    if info_path.exists() and tasks_path.exists():
        dataset = LeRobotDataset.resume(
            repo_id=REPO_ID,
            root=root,
            video_backend="pyav",
        )
        episode_index = dataset.meta.total_episodes
    else:
        if root.exists():
            print(f"Removing incomplete local dataset: {root}")
            shutil.rmtree(root)
        dataset = LeRobotDataset.create(
            repo_id=REPO_ID,
            fps=DATASET_FPS,
            features=FEATURES,
            robot_type="nero",
            root=root,
            use_videos=True,
            video_backend="pyav",
        )
        episode_index = 0

    cfg = NeroConfig(
        id="manual_record",
        bitrate=1_000_000,
        firmware_version="v121",
        speed_percent=50,
        has_gripper=True,
        has_camera=True,
        has_overview_camera=True,
    )

    robot = Nero(cfg)
    robot.connect(calibrate=False)
    effector = robot._get_gripper_effector()
    if effector is None:
        raise RuntimeError("Nero gripper effector is unavailable")
    if hasattr(effector, "disable_gripper"):
        effector.disable_gripper()
    if hasattr(effector, "set_gripper_teaching_pendant_param"):
        range_set = effector.set_gripper_teaching_pendant_param(
            max_range_config=GRIPPER_OPEN_WIDTH_M, timeout=5.0
        )
        print(f"Gripper range configuration {'succeeded' if range_set else 'not acknowledged'}.")
    robot.set_teach_mode(True)

    print(f"Task: {task}")
    print(f"Dataset: {root} | episode: {episode_index}")
    print("Teach mode enabled. Move the robot manually. Press '1' to close (0 mm), '0' to open (100 mm), 'r' to release, 'q' to stop and save the dataset.")
    cv2.namedWindow("Nero manual record", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Nero manual record", 1600, 480)
    gripper_width_m = GRIPPER_OPEN_WIDTH_M
    last_key = -1

    try:
        while True:
            obs = robot.get_observation()
            state = np.asarray(obs["observation.state"][:8], dtype=np.float32)
            if np.allclose(state, 0.0):
                print("Warning: joint state is still all zeros; check the arm is not in a neutral/no-feedback state.")

            image = obs.get("observation.images.wrist")
            overview = obs.get("observation.images.overview")
            depth = obs.get("observation.images.wrist_depth")

            if image is None:
                print("Warning: no wrist image captured; camera may not be connected or streaming.")
                image = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                image = np.asarray(image, dtype=np.float32)
                if image.max() <= 1.0:
                    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
                if image.ndim == 3 and image.shape[0] in (1, 3):
                    image = np.transpose(image, (1, 2, 0))

            if overview is None:
                print("Warning: no overview image captured; webcam may not be connected.")
                overview = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                overview = np.asarray(overview, dtype=np.float32)
                if overview.max() <= 1.0:
                    overview = np.clip(overview * 255.0, 0, 255).astype(np.uint8)
                else:
                    overview = overview.astype(np.uint8)
                if overview.ndim == 3 and overview.shape[0] in (1, 3):
                    overview = np.transpose(overview, (1, 2, 0))

            if depth is not None:
                depth = np.asarray(depth, dtype=np.float32)
                if depth.ndim == 3 and depth.shape[0] in (1, 3):
                    depth = np.transpose(depth, (1, 2, 0))
                depth = depth[:, :, 0] if depth.ndim == 3 and depth.shape[2] == 1 else depth
                if depth.size == 0:
                    depth = np.zeros((480, 640), dtype=np.float32)
                depth_min = np.nanmin(depth)
                depth_max = np.nanmax(depth)
                if np.isclose(depth_min, depth_max):
                    depth_vis = np.zeros((480, 640, 3), dtype=np.uint8)
                else:
                    depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-6)
                    # close = red, far = blue/teal like RealSense viewer
                    depth_norm = np.clip(1.0 - depth_norm, 0.0, 1.0)
                    depth8 = (depth_norm * 255.0).astype(np.uint8)
                    depth_vis = cv2.applyColorMap(depth8, cv2.COLORMAP_JET)
                    deep_mask = depth > 0
                    depth_vis[~deep_mask] = 0
            else:
                depth_min = 0.0
                depth_max = 0.0
                depth_vis = np.zeros((480, 640, 3), dtype=np.uint8)

            depth_text = f"Depth range: {depth_min:.2f} .. {depth_max:.2f}"
            cv2.putText(depth_vis, depth_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            if depth is not None:
                depth_array = np.asarray(depth, dtype=np.float32)
                if depth_array.ndim == 3:
                    depth_array = depth_array[..., 0]
                depth_array = np.nan_to_num(depth_array, nan=0.0, posinf=0.0, neginf=0.0)
                depth_profile = np.clip((1.0 - ((depth_array - depth_min) / (depth_max - depth_min + 1e-6))) * 255.0, 0, 255).astype(np.uint8)
                side_view = np.zeros((480, 200, 3), dtype=np.uint8)
                h, w = side_view.shape[:2]
                side_view[:] = (10, 14, 18)

                # Perspective floor/grid, similar to a RealSense side-view overlay.
                for i in range(-8, 9):
                    y = int(h * 0.72 + i * 24)
                    cv2.line(side_view, (0, y), (w, y), (80, 90, 100), 1)
                    x0 = int(w * 0.12 + i * 12)
                    x1 = int(w * 0.88 + i * 12)
                    cv2.line(side_view, (x0, h), (x1, 0), (120, 130, 140), 1)

                cx, cy = w // 2, h // 2
                cv2.line(side_view, (0, cy), (w, cy), (0, 255, 0), 2)
                cv2.line(side_view, (cx, 0), (cx, h), (0, 0, 255), 2)

                # Add a sparse depth silhouette using the nearest points to simulate the object surface.
                valid = depth_array > 0
                if np.any(valid):
                    ys, xs = np.where(valid)
                    if len(xs) > 0:
                        sample_x = xs[::max(1, len(xs) // 200)]
                        sample_y = ys[::max(1, len(ys) // 120)]
                        for x, y in zip(sample_x, sample_y):
                            if x >= depth_array.shape[1] or y >= depth_array.shape[0]:
                                continue
                            d = float(depth_array[y, x])
                            if d <= 0.0:
                                continue
                            norm = 1.0 - np.clip((d - depth_min) / (depth_max - depth_min + 1e-6), 0.0, 1.0)
                            px = int((x / depth_array.shape[1]) * (w - 1))
                            py = int((1.0 - norm) * (h * 0.7) + 20)
                            if 0 <= px < w and 0 <= py < h:
                                cv2.circle(side_view, (px, py), 2, (0, 165, 255), -1)

                # Add a subtle center point like the reference view.
                cv2.drawMarker(side_view, (cx, cy), (0, 0, 255), cv2.MARKER_STAR, 18, 2)
            else:
                side_view = np.zeros((480, 200, 3), dtype=np.uint8)
                side_view[:] = (10, 14, 18)
                cv2.line(side_view, (0, 240), (200, 240), (0, 255, 0), 2)
                cv2.line(side_view, (100, 0), (100, 480), (0, 0, 255), 2)

            combined = np.hstack([
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(overview, cv2.COLOR_RGB2BGR),
                depth_vis,
                side_view,
            ])
            cv2.rectangle(combined, (0, 0), (combined.shape[1], 58), (10, 14, 18), -1)
            cv2.putText(
                combined,
                "1: close 0 mm    0: open 100 mm    r: release gripper",
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                combined,
                "q: save and quit",
                (12, 49),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (180, 220, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow("Nero manual record", combined)
            key = cv2.waitKey(round(1000 / DATASET_FPS))
            if key == -1:
                key = read_key_nonblocking()
            else:
                key &= 0xFF

            if key == -1:
                last_key = -1
            elif key != last_key:
                last_key = key
                if key == ord("1"):
                    try:
                        move_gripper(robot, effector, GRIPPER_CLOSED_WIDTH_M)
                        gripper_width_m = GRIPPER_CLOSED_WIDTH_M
                        print("Gripper closed to 0 mm")
                    except Exception as exc:
                        print(f"Could not close gripper: {exc}")
                elif key == ord("0"):
                    try:
                        move_gripper(robot, effector, GRIPPER_OPEN_WIDTH_M)
                        gripper_width_m = GRIPPER_OPEN_WIDTH_M
                        print("Gripper opened to 100 mm")
                    except Exception as exc:
                        print(f"Could not open gripper: {exc}")
                elif key == ord("r"):
                    try:
                        effector.disable_gripper()
                        print("Gripper released")
                    except Exception as exc:
                        print(f"Could not release gripper: {exc}")
                elif key == ord("q"):
                    print("Exiting teach mode and saving dataset...")
                    robot.set_teach_mode(False)
                    break

            frame = {
                "observation.state": state,
                "action": np.concatenate(
                    [
                        state[:7],
                        np.asarray([gripper_width_m], dtype=np.float32),
                    ]
                ),
                "observation.images.wrist": image,
                "observation.images.overview": overview,
                "observation.depth": (
                    np.asarray(depth, dtype=np.float32)
                    if depth is not None
                    else np.zeros((480, 640), dtype=np.float32)
                ),
                "task": task,
            }
            dataset.add_frame(frame)
    finally:
        cv2.destroyAllWindows()
        try:
            robot.set_teach_mode(False)
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass

    dataset.save_episode()
    print(f"Saved dataset to: {root}")
    print(f"Meta total episodes: {dataset.meta.total_episodes}")
    print(f"Meta total frames: {dataset.meta.total_frames}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record a manual NERO demonstration.")
    parser.add_argument(
        "--task",
        default="pick up the cactus and place it on the green notebook",
        help="Language instruction stored with every frame.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Existing dataset directory to append to; defaults to a task-named directory.",
    )
    args = parser.parse_args()
    main(task=args.task, dataset_root=args.dataset_root)
