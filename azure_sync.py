import argparse
from pathlib import Path

from lerobot_robot_nero.azure_storage import AzureNeroStorage


def upload_existing(storage: AzureNeroStorage) -> int:
    uploaded = 0
    tasks_root = storage.local_root / "tasks"
    datasets_root = storage.local_root / "datasets"
    if tasks_root.exists():
        for task_file in tasks_root.glob("*.json"):
            uploaded += storage.upload_path(task_file, "tasks")
    if datasets_root.exists():
        for dataset_root in datasets_root.iterdir():
            if dataset_root.is_dir():
                uploaded += storage.upload_path(dataset_root, "datasets")
    return uploaded


def main() -> None:
    parser = argparse.ArgumentParser(description="Synchronize NERO data with Azure Blob Storage.")
    parser.add_argument("direction", choices=("upload", "download"))
    args = parser.parse_args()

    storage = AzureNeroStorage.from_environment(Path.home() / "Nero")
    if storage is None:
        raise SystemExit("Set NERO_AZURE_STORAGE_ACCOUNT before running Azure sync.")

    if args.direction == "upload":
        count = upload_existing(storage)
        print(f"Uploaded {count} file(s) to Azure.")
        return

    count = storage.sync_down("tasks") + storage.sync_down("datasets")
    print(f"Downloaded {count} file(s) from Azure.")


if __name__ == "__main__":
    main()
