import argparse
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from task_trajectory import (
    SAFE_BICEP_BRAKED_JOINTS,
    SAFE_BICEP_JOINTS,
    append_safe_bicep_return,
    command_recorded_gripper,
    format_cli_float,
    interpolated_joint_trajectory,
    is_safe_bicep_pose,
    prepare_replay_samples,
    safe_bicep_shutdown,
    smooth_move_with_recovery,
)


class TaskTrajectoryTest(unittest.TestCase):
    def test_negative_near_zero_anchor_is_accepted_by_argparse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--follower-anchor", type=float, nargs=7, required=True)
        anchor = [-3.490658503988659e-05, -1.68, 0.023, 2.08, -0.026, 0.076, 1.5]

        arguments = parser.parse_args([
            "--follower-anchor",
            *[format_cli_float(value) for value in anchor],
        ])

        for parsed, expected in zip(arguments.follower_anchor, anchor):
            self.assertAlmostEqual(parsed, expected, places=16)

    def test_safe_bicep_recognizes_commanded_and_brake_settled_poses(self):
        self.assertTrue(is_safe_bicep_pose(SAFE_BICEP_JOINTS))
        self.assertTrue(is_safe_bicep_pose(SAFE_BICEP_BRAKED_JOINTS))
        self.assertTrue(is_safe_bicep_pose([
            0.00014, -1.765453, 0.022881, 2.19641, -0.025621, 0.07669, 1.689479
        ]))
        self.assertFalse(is_safe_bicep_pose([0.0] * 7))

    def test_legacy_cactus_recording_preserves_taught_joint_deltas(self):
        task_file = Path(__file__).parents[1] / "tasks" / "pick-up-the-cactus.json"
        if not task_file.exists():
            self.skipTest("legacy cactus task fixture is not present")
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

    def test_post_processing_appends_exact_safe_bicep_return(self):
        final_pose = SAFE_BICEP_JOINTS.copy()
        final_pose[1] -= 0.2
        samples = [
            {"time": 0.0, "joints": [0.0] * 7, "gripper": 0.04},
            {"time": 1.0, "joints": final_pose, "gripper": 0.03},
        ]

        processed = append_safe_bicep_return(samples)

        self.assertEqual(processed[:-1], samples)
        self.assertEqual(processed[-1]["joints"], SAFE_BICEP_JOINTS)
        self.assertEqual(processed[-1]["gripper"], 0.1)
        self.assertEqual(processed[-1]["gripper_mode"], "width")
        self.assertAlmostEqual(processed[-1]["time"], 1.75)
        self.assertEqual(samples[-1]["joints"], final_pose)

    def test_post_processing_normalizes_existing_safe_bicep_endpoint(self):
        almost_safe = SAFE_BICEP_JOINTS.copy()
        almost_safe[0] += 1e-10
        samples = [{"time": 1.0, "joints": almost_safe, "gripper": 0.03}]

        processed = append_safe_bicep_return(samples)

        self.assertEqual(len(processed), 1)
        self.assertEqual(processed[-1]["joints"], SAFE_BICEP_JOINTS)
        self.assertEqual(processed[-1]["gripper"], 0.1)
        self.assertEqual(processed[-1]["gripper_mode"], "width")

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

    def test_build_37_corrupted_recording_has_clear_error(self):
        samples = [
            {"time": 0.0, "joints": ["width", 0.099], "gripper": 0.099},
            {"time": 0.1, "joints": ["width", 0.050], "gripper": 0.050},
        ]

        with self.assertRaisesRegex(ValueError, "Re-record the task with build 38"):
            prepare_replay_samples({"joint_space": "leader", "samples": samples})

    def test_interpolation_hits_recorded_samples_with_10ms_max_spacing(self):
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
            following[0] - current[0] <= 0.0100001
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
            ("angle", 17.5, 30.0),
            ("width", 0.04, 30.0),
        ])

    def test_amplified_gripper_is_binary_with_delayed_opening(self):
        class Effector:
            def __init__(self):
                self.calls = []

            def move_gripper_m(self, value, force):
                self.calls.append((value, force))

        effector = Effector()
        previous = None
        for value in (0.1, 0.08, 0.09, 0.097, 0.099):
            previous = command_recorded_gripper(
                effector,
                {"gripper_mode": "width", "gripper": value},
                previous,
                amplified=True,
            )

        self.assertEqual(effector.calls, [
            (0.1, 30.0),
            (0.0, 30.0),
            (0.1, 30.0),
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
            "task_trajectory.prepare_safe_bicep_motion",
            side_effect=lambda robot, label: events.append(("prepare", label)),
        ), patch(
            "task_trajectory.smooth_move_to_target",
            side_effect=lambda robot, target, label: events.append(
                ("move", target.copy(), label)
            ),
        ):
            safe_bicep_shutdown(Robot(), "Replay shutdown")

        self.assertEqual(events, [
            ("speed", 25),
            ("prepare", "Replay shutdown"),
            ("move", SAFE_BICEP_JOINTS, "Replay shutdown"),
            ("brakes",),
            ("disconnect", False),
        ])

    def test_replay_approach_prepares_before_joint_motion(self):
        events = []
        target = [0.1] * 7

        with patch(
            "task_trajectory.prepare_safe_bicep_motion",
            side_effect=lambda robot, label: events.append(("prepare", label)),
        ), patch(
            "task_trajectory.smooth_move_to_target",
            side_effect=lambda robot, goal, label: events.append(
                ("move", goal.copy(), label)
            ),
        ):
            smooth_move_with_recovery(object(), target, "Task replay")

        self.assertEqual(events, [
            ("prepare", "Task replay"),
            ("move", target, "Task replay"),
        ])


if __name__ == "__main__":
    unittest.main()