import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


DEFAULT_BASE_DIR = Path.home() / "Nero" / "datasets"
DEFAULT_DATASET_PREFIX = "nero_manual__"


def find_latest_dataset(base_dir: Path = DEFAULT_BASE_DIR) -> Path:
    """Return the most recently modified NERO dataset directory."""
    candidates = []
    for path in base_dir.glob(f"{DEFAULT_DATASET_PREFIX}*"):
        if path.is_dir():
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"No dataset found in {base_dir}. Record one first or pass --dataset-root /path/to/dataset."
        )

    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_dataset(dataset_root: Path | str | None, repo_id: str | None = None) -> LeRobotDataset:
    root = Path(dataset_root) if dataset_root is not None else find_latest_dataset()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    dataset_id = repo_id or root.name or "adrian/nero_manual"
    ds = LeRobotDataset(repo_id=dataset_id, root=str(root), download_videos=False)
    print(f"Loaded dataset from: {root}")
    print(f"Episodes: {ds.num_episodes}, frames: {len(ds)}")
    return ds


def latest_episode_bounds(ds: LeRobotDataset) -> tuple[int, int]:
    if ds.meta.total_episodes == 0:
        raise ValueError("Dataset contains no saved episodes.")
    episode = ds.meta.episodes[-1]
    return int(episode["dataset_from_index"]), int(episode["dataset_to_index"])


def normalize_image(frame: np.ndarray | object) -> np.ndarray:
    image = np.asarray(frame)
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.dtype != np.uint8 and image.max() <= 1.0:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    return image


def replay_dataset(dataset_root: str | None = None, repo_id: str | None = None, fps: int = 15, max_frames: int | None = None, start_frame: int | None = None, episode_index: int | None = None) -> None:
    ds = load_dataset(dataset_root=dataset_root, repo_id=repo_id)
    if episode_index is None:
        episode_start, episode_end = latest_episode_bounds(ds)
        episode_index = ds.meta.total_episodes - 1
    else:
        if episode_index < 0 or episode_index >= ds.meta.total_episodes:
            raise ValueError(f"Episode index {episode_index} is out of range.")
        episode = ds.meta.episodes[episode_index]
        episode_start = int(episode["dataset_from_index"])
        episode_end = int(episode["dataset_to_index"])
    first_frame = episode_start if start_frame is None else episode_start + start_frame
    end_frame = episode_end if max_frames is None else min(episode_end, first_frame + max_frames)

    print(f"Replaying episode {episode_index}: frames {first_frame}..{end_frame - 1}")
    print("Press 'q' to stop replaying.")
    cv2.namedWindow("Nero dataset replay", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Nero dataset replay", 1480, 480)

    try:
        for idx in range(first_frame, end_frame):
            sample = ds[idx]
            state = sample["observation.state"]
            if hasattr(state, "cpu"):
                state = state.cpu().numpy()
            state = np.asarray(state, dtype=np.float32)
            image = sample["observation.images.wrist"]
            if hasattr(image, "cpu"):
                image = image.cpu().numpy()
            image = normalize_image(image)
            overview = sample.get("observation.images.overview")
            if overview is None:
                overview = np.zeros_like(image)
                cv2.putText(overview, "No overview camera data", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                if hasattr(overview, "cpu"):
                    overview = overview.cpu().numpy()
                overview = normalize_image(overview)

            depth = np.zeros((480, 640), dtype=np.float32)
            if "observation.depth" in sample or "observation.images.wrist_depth" in sample:
                depth = sample.get("observation.depth", sample.get("observation.images.wrist_depth"))
                if hasattr(depth, "cpu"):
                    depth = depth.cpu().numpy()
                depth = np.asarray(depth, dtype=np.float32)
                if depth.ndim == 3 and depth.shape[0] in (1, 3):
                    depth = np.transpose(depth, (1, 2, 0))
                if depth.ndim == 3 and depth.shape[2] == 1:
                    depth = depth[:, :, 0]
            else:
                depth = np.zeros((480, 640), dtype=np.float32)
            depth_min = np.nanmin(depth)
            depth_max = np.nanmax(depth)
            if np.isclose(depth_min, depth_max):
                depth_vis = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(depth_vis, "No depth data", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-6)
                depth_norm = np.clip(1.0 - depth_norm, 0.0, 1.0)
                depth8 = (depth_norm * 255.0).astype(np.uint8)
                depth_vis = cv2.applyColorMap(depth8, cv2.COLORMAP_JET)
                deep_mask = depth > 0
                depth_vis[~deep_mask] = 0
            cv2.putText(depth_vis, f"Depth range: {depth_min:.2f} .. {depth_max:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            depth_array = np.asarray(depth, dtype=np.float32)
            if depth_array.ndim == 3:
                depth_array = depth_array[..., 0]
            depth_array = np.nan_to_num(depth_array, nan=0.0, posinf=0.0, neginf=0.0)
            depth_3d = np.clip((1.0 - ((depth_array - depth_min) / (depth_max - depth_min + 1e-6))) * 255.0, 0, 255).astype(np.uint8)
            side_view = np.zeros((480, 200, 3), dtype=np.uint8)
            side_view[:] = (10, 14, 18)
            for i in range(-8, 9):
                y = int(480 * 0.72 + i * 24)
                cv2.line(side_view, (0, y), (200, y), (80, 90, 100), 1)
                x0 = int(200 * 0.12 + i * 12)
                x1 = int(200 * 0.88 + i * 12)
                cv2.line(side_view, (x0, 480), (x1, 0), (120, 130, 140), 1)
            cx, cy = 100, 240
            cv2.line(side_view, (0, cy), (200, cy), (0, 255, 0), 2)
            cv2.line(side_view, (cx, 0), (cx, 480), (0, 0, 255), 2)
            valid = depth_array > 0
            if np.any(valid):
                ys, xs = np.where(valid)
                for x, y in zip(xs[::max(1, len(xs) // 200)], ys[::max(1, len(ys) // 120)]):
                    d = float(depth_array[y, x])
                    norm = 1.0 - np.clip((d - depth_min) / (depth_max - depth_min + 1e-6), 0.0, 1.0)
                    px = int((x / depth_array.shape[1]) * 199)
                    py = int((1.0 - norm) * 336 + 20)
                    if 0 <= px < 200 and 0 <= py < 480:
                        cv2.circle(side_view, (px, py), 2, (0, 165, 255), -1)
            cv2.drawMarker(side_view, (cx, cy), (0, 0, 255), cv2.MARKER_STAR, 18, 2)

            print(f"frame {idx}: state={state.tolist()}")
            combined = np.hstack([image, overview, depth_vis, side_view])
            cv2.rectangle(combined, (0, 0), (combined.shape[1], 58), (10, 14, 18), -1)
            cv2.putText(combined, "1: close 0 mm    0: open 100 mm    r: release gripper", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(combined, f"Task: {sample.get('task', '')}", (12, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 220, 255), 1, cv2.LINE_AA)
            cv2.imshow("Nero dataset replay", combined)
            key = cv2.waitKey(max(1, int(1000 / max(1, fps)))) & 0xFF
            if key == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay the latest NERO LeRobot dataset and print joint angles.")
    parser.add_argument("--dataset-root", type=str, default=None, help="Path to a saved dataset root. Defaults to the newest dataset in ~/Nero/datasets.")
    parser.add_argument("--repo-id", type=str, default=None, help="Optional repo id override.")
    parser.add_argument("--fps", type=int, default=15, help="Replay speed in frames per second.")
    parser.add_argument("--start-frame", type=int, default=None, help="Offset within the latest episode; defaults to its first frame.")
    parser.add_argument("--episode-index", type=int, default=None, help="Episode to replay; defaults to the latest episode.")
    parser.add_argument("--max-frames", type=int, default=None, help="Replay this many frames, default is the whole dataset.")
    args = parser.parse_args()

    replay_dataset(
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        fps=args.fps,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
        episode_index=args.episode_index,
    )
