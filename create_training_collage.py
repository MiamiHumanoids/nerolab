"""Create text-free vertical or landscape collages from dataset overview videos."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import av
import cv2
import numpy as np
import pyarrow.parquet as pq


FPS = 30
PAGE_SECONDS = 6


class ClipReader:
    def __init__(self, path: Path) -> None:
        self.container = av.open(str(path))
        self.stream = self.container.streams.video[0]
        self.duration = float(self.stream.duration * self.stream.time_base)
        self.frames = iter(self.container.decode(self.stream))
        self.current: np.ndarray | None = None
        self.current_time = 0.0

    def frame_at(self, timestamp: float) -> np.ndarray:
        while self.current is None or self.current_time < timestamp:
            try:
                frame = next(self.frames)
            except StopIteration:
                break
            self.current = frame.to_ndarray(format="bgr24")
            self.current_time = float(frame.pts * frame.time_base)
        if self.current is None:
            raise RuntimeError("Video contains no decodable frames")
        return self.current

    def close(self) -> None:
        self.container.close()


class DepthReader:
    def __init__(self, path: Path) -> None:
        table = pq.read_table(path, columns=["observation.depth"])
        depth_column = table.column("observation.depth").combine_chunks()
        depth_frames = depth_column.values.values.to_numpy(zero_copy_only=False).reshape(
            len(depth_column),
            480,
            640,
        )
        source_indices = np.linspace(
            0,
            len(depth_column) - 1,
            PAGE_SECONDS * FPS,
            dtype=np.int32,
        )
        self.frames = [self._colorize(depth_frames[index]) for index in source_indices]
        self.duration = len(self.frames) / FPS

    @staticmethod
    def _colorize(values: np.ndarray) -> np.ndarray:
        depth = np.asarray(values, dtype=np.float32)
        valid = np.isfinite(depth) & (depth > 0) & (depth < 65_000)
        clipped = np.clip(depth, 800, 10_000)
        normalized = ((10_000 - clipped) / 9_200 * 255).astype(np.uint8)
        colorized = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        colorized[~valid] = 0
        return colorized

    def frame_at(self, timestamp: float) -> np.ndarray:
        index = min(round(timestamp * FPS), len(self.frames) - 1)
        return self.frames[index]

    def close(self) -> None:
        self.frames.clear()


def render_page_frame(
    readers: list[ClipReader],
    elapsed: float,
    layout: str,
) -> np.ndarray:
    if layout == "landscape":
        width, height = 1920, 1080
        columns, rows = 3, 2
        tile_width, tile_height = 640, 480
        top = (height - rows * tile_height) // 2
    else:
        width, height = 1080, 1920
        columns, rows = 2, 5
        tile_width, tile_height = 540, 384
        top = 0
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    for local_index, reader in enumerate(readers):
        row, column = divmod(local_index, columns)
        x = column * tile_width
        y = top + row * tile_height

        source_time = min(
            reader.duration - 1 / FPS,
            elapsed / PAGE_SECONDS * reader.duration,
        )
        frame = reader.frame_at(max(0.0, source_time))
        if layout != "landscape":
            source_height, source_width = frame.shape[:2]
            crop_height = round(source_width * tile_height / tile_width)
            crop_top = max(0, (source_height - crop_height) // 2)
            frame = frame[crop_top : crop_top + crop_height]
        frame = cv2.resize(frame, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        canvas[y : y + tile_height, x : x + tile_width] = frame
    return canvas


def create_collage(
    dataset_root: Path,
    output: Path,
    layout: str = "vertical",
    mixed_views: bool = False,
) -> None:
    video_root = dataset_root / "videos" / "observation.images.overview"
    videos = sorted(video_root.rglob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No overview videos found under {video_root}")
    if mixed_views and layout != "landscape":
        raise ValueError("Mixed views are only supported by the landscape layout")

    sources: list[tuple[str, Path]] = []
    for index, overview_path in enumerate(videos):
        if not mixed_views or index % 3 == 0:
            sources.append(("rgb", overview_path))
        elif index % 3 == 1:
            wrist_path = (
                dataset_root
                / "videos"
                / "observation.images.wrist"
                / overview_path.relative_to(video_root)
            )
            sources.append(("rgb", wrist_path))
        else:
            depth_path = (
                dataset_root
                / "data"
                / overview_path.relative_to(video_root).with_suffix(".parquet")
            )
            sources.append(("depth", depth_path))

    output.parent.mkdir(parents=True, exist_ok=True)
    clips_per_page = 6 if layout == "landscape" else 10
    width, height = (1920, 1080) if layout == "landscape" else (1080, 1920)
    page_count = math.ceil(len(sources) / clips_per_page)
    container = av.open(str(output), mode="w")
    stream = container.add_stream("libx264", rate=FPS)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "20", "preset": "medium", "movflags": "+faststart"}

    try:
        for page_index in range(page_count):
            page_sources = sources[
                page_index * clips_per_page : (page_index + 1) * clips_per_page
            ]
            readers = [
                DepthReader(path) if source_type == "depth" else ClipReader(path)
                for source_type, path in page_sources
            ]
            try:
                for frame_index in range(PAGE_SECONDS * FPS):
                    elapsed = frame_index / FPS
                    image = render_page_frame(
                        readers,
                        elapsed,
                        layout,
                    )
                    frame = av.VideoFrame.from_ndarray(image, format="bgr24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
            finally:
                for reader in readers:
                    reader.close()
            print(f"Rendered set {page_index + 1}/{page_count}", flush=True)

        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layout", choices=("vertical", "landscape"), default="vertical")
    parser.add_argument("--mixed-views", action="store_true")
    args = parser.parse_args()
    create_collage(
        args.dataset_root.resolve(),
        args.output.resolve(),
        args.layout,
        args.mixed_views,
    )


if __name__ == "__main__":
    main()