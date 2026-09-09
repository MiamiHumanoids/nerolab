from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest

from lerobot_robot_nero.azure_storage import AzureNeroStorage


class Download:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def chunks(self):
        yield self.data


class FakeContainer:
    container_name = "nero"

    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}

    def list_blobs(self, name_starts_with: str):
        return [
            SimpleNamespace(name=name)
            for name in sorted(self.blobs)
            if name.startswith(name_starts_with)
        ]

    def download_blob(self, name: str) -> Download:
        return Download(self.blobs[name])

    def upload_blob(self, name: str, data, overwrite: bool) -> None:
        assert overwrite is True
        self.blobs[name] = data.read()

    def delete_blobs(self, *names: str) -> None:
        for name in names:
            self.blobs.pop(name, None)


def test_task_upload_and_download_round_trip():
    with TemporaryDirectory() as directory:
        local_root = Path(directory)
        container = FakeContainer()
        storage = AzureNeroStorage(container, local_root)
        task = local_root / "tasks" / "pick.json"
        task.parent.mkdir(parents=True)
        task.write_text('{"task": "pick"}')

        assert storage.upload_path(task, "tasks") == 1
        task.unlink()
        assert storage.sync_down("tasks") == 1

        assert task.read_text() == '{"task": "pick"}'


def test_dataset_upload_removes_stale_blobs():
    with TemporaryDirectory() as directory:
        local_root = Path(directory)
        container = FakeContainer()
        container.blobs["datasets/run/obsolete.txt"] = b"old"
        storage = AzureNeroStorage(container, local_root)
        dataset = local_root / "datasets" / "run"
        dataset.mkdir(parents=True)
        (dataset / "meta.json").write_bytes(b"current")

        assert storage.upload_path(dataset, "datasets") == 1

        assert container.blobs == {"datasets/run/meta.json": b"current"}


def test_dataset_upload_excludes_local_migration_artifacts():
    with TemporaryDirectory() as directory:
        local_root = Path(directory)
        container = FakeContainer()
        storage = AzureNeroStorage(container, local_root)
        dataset = local_root / "datasets" / "run"
        dataset.mkdir(parents=True)
        (dataset / "data.parquet").write_bytes(b"current")
        (dataset / "data.parquet.gripper-v1.bak").write_bytes(b"backup")
        (dataset / "data.parquet.gripper-migration.tmp").write_bytes(b"temporary")

        assert storage.upload_path(dataset, "datasets") == 1
        assert container.blobs == {"datasets/run/data.parquet": b"current"}


def test_delete_path_removes_only_selected_prefix():
    with TemporaryDirectory() as directory:
        local_root = Path(directory)
        container = FakeContainer()
        container.blobs = {
            "datasets/run/meta.json": b"one",
            "datasets/run/data/frame.json": b"two",
            "datasets/run-other/meta.json": b"keep",
        }
        storage = AzureNeroStorage(container, local_root)

        assert storage.delete_path(local_root / "datasets" / "run", "datasets") == 2
        assert set(container.blobs) == {"datasets/run-other/meta.json"}


def test_download_rejects_blob_path_outside_local_cache():
    with TemporaryDirectory() as directory:
        local_root = Path(directory)
        container = FakeContainer()
        container.blobs["tasks/../outside.txt"] = b"unsafe"
        storage = AzureNeroStorage(container, local_root)

        with pytest.raises(ValueError, match="Unsafe Azure blob path"):
            storage.sync_down("tasks")
