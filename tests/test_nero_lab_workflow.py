import os

from nero_lab import newest_dataset_paths, next_task_output_path


def test_repeated_task_variations_get_unique_recording_paths(tmp_path):
    first = next_task_output_path(tmp_path, "stack-red__standard-layout")
    first.write_text("{}")
    second = next_task_output_path(tmp_path, "stack-red__standard-layout")
    second.write_text("{}")

    assert first.name == "stack-red__standard-layout.json"
    assert second.name == "stack-red__standard-layout__2.json"
    assert next_task_output_path(
        tmp_path, "stack-red__standard-layout"
    ).name == "stack-red__standard-layout__3.json"


def test_datasets_are_sorted_newest_first(tmp_path):
    older = tmp_path / "older"
    newer = tmp_path / "newer"
    for path in (older, newer):
        (path / "meta").mkdir(parents=True)
        (path / "meta" / "info.json").write_text("{}")
    os.utime(older / "meta" / "info.json", (10, 10))
    os.utime(newer / "meta" / "info.json", (20, 20))

    assert newest_dataset_paths([older, newer]) == [newer, older]