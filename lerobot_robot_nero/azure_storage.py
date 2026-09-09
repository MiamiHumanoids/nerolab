from __future__ import annotations

import os
from pathlib import Path

from azure.core.exceptions import ResourceExistsError
from azure.core.exceptions import HttpResponseError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContainerClient


class AzureNeroStorage:
    def __init__(self, container: ContainerClient, local_root: Path) -> None:
        self.container = container
        self.local_root = local_root.resolve()

    @classmethod
    def from_environment(cls, local_root: Path | None = None) -> AzureNeroStorage | None:
        account = os.getenv("NERO_AZURE_STORAGE_ACCOUNT", "").strip()
        if not account:
            return None
        container_name = os.getenv("NERO_AZURE_STORAGE_CONTAINER", "nero").strip()
        service = BlobServiceClient(
            account_url=f"https://{account}.blob.core.windows.net",
            credential=DefaultAzureCredential(),
        )
        container = service.get_container_client(container_name)
        try:
            container.create_container()
        except ResourceExistsError:
            pass
        return cls(container, local_root or Path.home() / "Nero")

    def sync_down(self, category: str) -> int:
        local_category = self._category_root(category)
        local_category.mkdir(parents=True, exist_ok=True)
        downloaded = 0
        prefix = f"{category}/"
        for blob in self.container.list_blobs(name_starts_with=prefix):
            relative = Path(blob.name.removeprefix(prefix))
            destination = self._safe_destination(local_category, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as output:
                for chunk in self.container.download_blob(blob.name).chunks():
                    output.write(chunk)
            downloaded += 1
        return downloaded

    def upload_path(self, path: Path, category: str) -> int:
        category_root = self._category_root(category)
        source = path.resolve()
        if not source.is_relative_to(category_root):
            raise ValueError(f"{source} is outside {category_root}")
        files = (
            [source]
            if source.is_file()
            else [
                item
                for item in source.rglob("*")
                if item.is_file() and not item.name.endswith((".bak", ".tmp"))
            ]
        )
        if source.is_dir():
            relative_root = source.relative_to(category_root).as_posix()
            blob_prefix = f"{category}/{relative_root}/"
            local_names = {
                f"{category}/{file_path.relative_to(category_root).as_posix()}"
                for file_path in files
            }
            stale_names = [
                blob.name
                for blob in self.container.list_blobs(name_starts_with=blob_prefix)
                if blob.name not in local_names
            ]
            for start in range(0, len(stale_names), 256):
                self.container.delete_blobs(*stale_names[start:start + 256])
        uploaded = 0
        for file_path in files:
            relative = file_path.relative_to(category_root).as_posix()
            with file_path.open("rb") as data:
                self.container.upload_blob(
                    name=f"{category}/{relative}", data=data, overwrite=True
                )
            uploaded += 1
        return uploaded

    def delete_path(self, path: Path, category: str) -> int:
        category_root = self._category_root(category)
        target = path.resolve()
        if not target.is_relative_to(category_root):
            raise ValueError(f"{target} is outside {category_root}")
        relative = target.relative_to(category_root).as_posix()
        blob_prefix = f"{category}/{relative}"
        names = [
            blob.name
            for blob in self.container.list_blobs(name_starts_with=blob_prefix)
            if blob.name == blob_prefix or blob.name.startswith(f"{blob_prefix}/")
        ]
        for start in range(0, len(names), 256):
            self.container.delete_blobs(*names[start:start + 256])
        return len(names)

    def _category_root(self, category: str) -> Path:
        if category not in {"tasks", "datasets"}:
            raise ValueError(f"Unsupported Azure storage category: {category}")
        return (self.local_root / category).resolve()

    @staticmethod
    def describe_error(exc: Exception) -> str:
        if isinstance(exc, HttpResponseError) and exc.status_code == 403:
            return (
                "Azure denied Blob access (HTTP 403). Grant the signed-in user "
                "the Storage Blob Data Contributor role on the Storage account."
            )
        return str(exc)

    @staticmethod
    def _safe_destination(root: Path, relative: Path) -> Path:
        destination = (root / relative).resolve()
        if not destination.is_relative_to(root):
            raise ValueError(f"Unsafe Azure blob path: {relative}")
        return destination