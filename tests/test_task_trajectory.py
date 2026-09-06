import json
import unittest
from pathlib import Path

from task_trajectory import (
    SAFE_BICEP_JOINTS,
    joints_within_limits,
    prepare_replay_samples,
)


class TaskTrajectoryTest(unittest.TestCase):
    def test_legacy_cactus_recording_converts_to_safe_follower_targets(self):
        task_file = Path(__file__).parents[1] / "tasks" / "pick-up-the-cactus.json"
        recording = json.loads(task_file.read_text())

        samples = prepare_replay_samples(recording)

        self.assertEqual(len(samples), 257)
        self.assertEqual(samples[0]["joints"], SAFE_BICEP_JOINTS)
        self.assertEqual(samples[0]["leader_joints"], recording["samples"][0]["joints"])
        self.assertTrue(all(joints_within_limits(sample["joints"]) for sample in samples))
        self.assertTrue(all(
            float(current["time"]) <= float(following["time"])
            for current, following in zip(samples, samples[1:])
        ))

    def test_follower_recording_is_preserved(self):
        samples = [
            {"time": 0.0, "joints": SAFE_BICEP_JOINTS.copy(), "gripper": 0.1},
            {"time": 0.1, "joints": SAFE_BICEP_JOINTS.copy(), "gripper": 0.05},
        ]

        prepared = prepare_replay_samples({"joint_space": "follower", "samples": samples})

        self.assertIs(prepared[0], samples[0])
        self.assertNotIn("leader_joints", prepared[0])


if __name__ == "__main__":
    unittest.main()