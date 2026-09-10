import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from migrate_remove_gripper_force import BACKUP_SUFFIX, FEATURE_NAMES, migrate


def vector(values):
    return pa.FixedSizeListArray.from_arrays(pa.array(values, type=pa.float32()), 9)


def test_migration_removes_force_from_data_and_stats(tmp_path: Path):
    root = tmp_path / "dataset"
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    episode_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    data_path.parent.mkdir(parents=True)
    episode_path.parent.mkdir(parents=True)

    metadata = {
        b"huggingface": json.dumps({
            "info": {
                "features": {
                    "observation.state": {"length": 9},
                    "action": {"length": 9},
                }
            }
        }).encode()
    }
    data = pa.table({
        "observation.state": vector(list(range(9))),
        "action": vector(list(range(10, 19))),
    }).replace_schema_metadata(metadata)
    pq.write_table(data, data_path)
    pq.write_table(
        pa.table({
            "stats/action/mean": vector(list(range(9))),
            "stats/observation.state/max": vector(list(range(9))),
        }),
        episode_path,
    )
    info = {
        "features": {
            "observation.state": {"shape": [9], "names": [f"old_{i}" for i in range(9)]},
            "action": {"shape": [9], "names": [f"old_{i}" for i in range(9)]},
        }
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    stats = {
        "observation.state": {"mean": list(range(9)), "count": [1]},
        "action": {"mean": list(range(9)), "count": [1]},
    }
    (root / "meta" / "stats.json").write_text(json.dumps(stats))

    migrate(root)

    migrated_data = pq.read_table(data_path)
    assert migrated_data.schema.field("action").type.list_size == 8
    assert migrated_data["action"].to_pylist() == [list(range(10, 18))]
    embedded = json.loads(migrated_data.schema.metadata[b"huggingface"])
    assert embedded["info"]["features"]["action"]["length"] == 8
    migrated_episode = pq.read_table(episode_path)
    assert migrated_episode.schema.field("stats/action/mean").type.list_size == 8
    migrated_info = json.loads((root / "meta" / "info.json").read_text())
    assert migrated_info["features"]["action"] == {
        "shape": [8],
        "names": FEATURE_NAMES,
    }
    migrated_stats = json.loads((root / "meta" / "stats.json").read_text())
    assert migrated_stats["action"]["mean"] == list(range(8))
    assert migrated_stats["action"]["count"] == [1]
    for path in (data_path, episode_path, root / "meta" / "info.json", root / "meta" / "stats.json"):
        assert path.with_name(path.name + BACKUP_SUFFIX).exists()