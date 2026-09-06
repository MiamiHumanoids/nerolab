import json
import unittest
from pathlib import Path
from unittest.mock import patch

from task_trajectory import (
    SAFE_BICEP_JOINTS,
    command_recorded_gripper,
    interpolated_joint_trajectory,
    prepare_replay_samples,
    safe_bicep_shutdown,
)


class TaskTrajectoryTest(unittest.TestCase):
    def test_legacy_cactus_recording_preserves_taught_joint_deltas(self):
        task_file = Path(__file__).parents[1] / "tasks" / "pick-up-the-cactus.json"
        recording = json.loads(task_file.read_text())

        samples = prepare_replay_samples(recording)

        self.assertEqual(len(samples), 257)
        self.assertEqual(samples[0]["joints"], SAFE_BICEP_JOINTS)
        self.assertEqual(samples[0]["leader_joints"], recording["samples"][0]["joints"])
        for leader_start, leader_end, follower_start, follower_end in zip(
            recording["samples"][0]["joints"],
            recording["samples"][-1]["joints"],
            samples[0]["joints"],
            samples[-1]["joints"],
        ):
            self.assertAlmostEqual(follower_end - follower_start, leader_end - leader_start)
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

    def test_leader_recording_uses_its_saved_follower_anchor(self):
        anchor = SAFE_BICEP_JOINTS.copy()
        anchor[0] = 0.1
        leader_start = [0.0] * 7
        samples = [
            {"time": 0.0, "joints": leader_start, "gripper": 0.1},
            {"time": 0.1, "joints": [0.05] + [0.0] * 6, "gripper": 0.1},
        ]

        prepared = prepare_replay_samples({
            "joint_space": "leader",
            "follower_anchor": anchor,
            "replay_ready": False,
            "replay_error": "Recorded sample 2 cannot be converted inside command limits",
            "samples": samples,
        })

        self.assertEqual(prepared[0]["joints"], anchor)
        self.assertAlmostEqual(prepared[1]["joints"][0], 0.15)

    def test_recorded_pose_is_not_rejected_or_clamped(self):
        anchor = SAFE_BICEP_JOINTS.copy()
        samples = [
            {"time": 0.0, "joints": [0.0] * 7, "gripper": 0.1},
            {"time": 0.1, "joints": [0.0] * 5 + [0.973745, 0.0], "gripper": 0.1},
        ]

        prepared = prepare_replay_samples({
            "joint_space": "leader",
            "follower_anchor": anchor,
            "samples": samples,
        })

        self.assertEqual(prepared[1]["joints"][5], anchor[5] + 0.973745)

    def test_interpolation_hits_recorded_samples_with_20ms_max_spacing(self):
        samples = [
            {"time": 0.0, "joints": [0.0] * 7},
            {"time": 0.07, "joints": [0.7] * 7},
            {"time": 0.14, "joints": [0.0] * 7},
        ]

        points = interpolated_joint_trajectory(samples)
        original_points = [point for point in points if point[2] is not None]

        self.assertEqual([point[2] for point in original_points], [0, 1, 2])
        self.assertEqual([point[1] for point in original_points], [sample["joints"] for sample in samples])
        self.assertTrue(all(
            following[0] - current[0] <= 0.0200001
            for current, following in zip(points, points[1:])
        ))

    def test_gripper_replay_preserves_recorded_control_mode(self):
        class Effector:
            def __init__(self):
                self.calls = []

            def move_gripper_m(self, value, force):
                self.calls.append(("width", value, force))

            def move_gripper_deg(self, value, force):
                self.calls.append(("angle", value, force))

        effector = Effector()

        previous = command_recorded_gripper(
            effector, {"gripper_mode": "angle", "gripper": 17.5}, None
        )
        command_recorded_gripper(
            effector, {"gripper_mode": "width", "gripper": 0.04}, previous
        )

        self.assertEqual(effector.calls, [
            ("angle", 17.5, 1.0),
            ("width", 0.04, 1.0),
        ])

    def test_safe_shutdown_moves_brakes_then_disconnects(self):
        events = []

        class Arm:
            def set_speed_percent(self, speed):
                events.append(("speed", speed))

        class Robot:
            _arm = Arm()

            def engage_brakes(self):
                events.append(("brakes",))

            def disconnect(self, disable_arm=True):
                events.append(("disconnect", disable_arm))

        with patch(
            "task_trajectory.smooth_move_to_target",
            side_effect=lambda robot, target, label: events.append(
                ("move", target.copy(), label)
            ),
        ):
            safe_bicep_shutdown(Robot(), "Replay shutdown")

        self.assertEqual(events, [
            ("speed", 25),
            ("move", SAFE_BICEP_JOINTS, "Replay shutdown"),
            ("brakes",),
            ("disconnect", False),
        ])


if __name__ == "__main__":
    unittest.main()