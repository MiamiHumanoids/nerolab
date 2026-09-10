#!/usr/bin/env python3
"""Launch LeRobot dataset visualization with the PyAV video decoder."""

from lerobot.utils import import_utils

NERO_SCALAR_NAMES = [
    "Joint_1",
    "Joint_2",
    "Joint_3",
    "Joint_4",
    "Joint_5",
    "Joint_6",
    "Joint_7",
    "Gripper",
]


def install_named_scalar_logging(rerun) -> None:
    original_log = rerun.log
    styled_paths: set[str] = set()

    def named_log(entity_path, entity, *args, **kwargs):
        path = str(entity_path)
        scalars = getattr(entity, "scalars", None)
        if path in {"action", "state"} and scalars is not None:
            values = scalars.as_arrow_array().to_pylist()
            if len(values) == len(NERO_SCALAR_NAMES):
                for name, value in zip(NERO_SCALAR_NAMES, values):
                    series_path = f"{path}/{name}"
                    if series_path not in styled_paths:
                        original_log(
                            series_path,
                            rerun.SeriesLines(names=name),
                            static=True,
                        )
                        styled_paths.add(series_path)
                    original_log(series_path, rerun.Scalars(value), *args, **kwargs)
                return None
        return original_log(entity_path, entity, *args, **kwargs)

    rerun.log = named_log


def main() -> None:
    import_utils.get_safe_default_video_backend = lambda: "pyav"

    import rerun as rr

    install_named_scalar_logging(rr)

    from lerobot.scripts.lerobot_dataset_viz import main as dataset_viz_main

    dataset_viz_main()


if __name__ == "__main__":
    main()