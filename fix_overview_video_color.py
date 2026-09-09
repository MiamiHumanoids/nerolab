#!/usr/bin/env python3
"""Correct red/blue channel reversal in legacy NERO overview videos."""

from __future__ import annotations

import argparse
import os
import shutil
from fractions import Fraction
from pathlib import Path

import av


def correct_video(source: Path) -> None:
    temporary = source.with_name(f"{source.stem}.color-corrected.tmp.mp4")
    backup = source.with_name(f"{source.name}.color-swapped.bak")
    temporary.unlink(missing_ok=True)
    if not backup.exists():
        shutil.copy2(source, backup)

    with av.open(str(source)) as input_container:
        input_stream = input_container.streams.video[0]
        expected_frames = input_stream.frames
        rate = input_stream.average_rate or 30
        with av.open(str(temporary), mode="w") as output_container:
            output_stream = output_container.add_stream("libsvtav1", rate=rate)
            output_stream.width = input_stream.width
            output_stream.height = input_stream.height
            output_stream.pix_fmt = "yuv420p"
            output_stream.options = {"crf": "30", "preset": "12"}
            time_base = Fraction(rate.denominator, rate.numerator)
            for index, frame in enumerate(input_container.decode(input_stream)):
                rgb = frame.to_ndarray(format="rgb24")
                corrected = av.VideoFrame.from_ndarray(
                    rgb[:, :, ::-1].copy(), format="rgb24"
                )
                corrected.pts = index
                corrected.time_base = time_base
                for packet in output_stream.encode(corrected):
                    output_container.mux(packet)
            for packet in output_stream.encode():
                output_container.mux(packet)

    with av.open(str(temporary)) as validation_container:
        validated_frames = sum(1 for _ in validation_container.decode(video=0))
    if validated_frames != expected_frames:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Expected {expected_frames} frames, encoded {validated_frames}"
        )
    os.replace(temporary, source)
    print(f"Corrected {validated_frames} overview frames: {source}")
    print(f"Original backup: {backup}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args()
    videos = sorted(
        (args.dataset_root / "videos" / "observation.images.overview").rglob(
            "*.mp4"
        )
    )
    if not videos:
        raise FileNotFoundError("No overview videos found in the dataset")
    for video in videos:
        correct_video(video)


if __name__ == "__main__":
    main()