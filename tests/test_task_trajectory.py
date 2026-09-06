import argparse
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from task_trajectory import (
    SAFE_BICEP_BRAKED_JOINTS,
    SAFE_BICEP_JOINTS,
    amplify_gripper_samples,
    append_safe_bicep_return,
    command_recorded_gripper,
    format_cli_float,
    interpolated_joint_trajectory,
    is_amplified_gripper_opening,
    is_safe_bicep_pose,
    prepare_gripper_for_replay,
    prepare_replay_samples,
    safe_bicep_shutdown,
    safe_bicep_recovery_pose,
    smooth_move_with_recovery,
    wait_for_gripper_release_pose,
)


class TaskTrajectoryTest(unittest.TestCase):
    def test_safe_bicep_recovery_pose_uses_sdk_forward_kinematics(self):
        expected_pose = [-0.2, 0.0, 0.35, -1.4, 0.0, -3.0]

        class Arm:
            def fk(self, joints):
                self.joints = joints
                return expected_pose

        arm = Arm()

        self.assertEqual(safe_bicep_recovery_pose(arm), expected_pose)
        self.assertEqual(arm.joints, SAFE_BICEP_JOINTS)

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
            ("angle", 17.5, 3.0),
            ("width", 0.04, 3.0),
        ])

    def test_gripper_replay_latches_single_closed_command(self):
        class Effector:
            def __init__(self):
                self.calls = []

            def move_gripper_m(self, value, force):
                self.calls.append((value, force))

        effector = Effector()
        sample = {
            "gripper_mode": "width",
            "gripper": 0.02,
            "gripper_force": 3.0,
        }
        previous = command_recorded_gripper(effector, sample, None)
        command_recorded_gripper(effector, sample, previous)

        self.assertEqual(effector.calls, [(0.02, 3.0)])

    def test_gripper_replay_resets_control_before_configuring_range(self):
        events = []

        class Effector:
            def disable_gripper(self):
                events.append(("disable",))

            def set_gripper_teaching_pendant_param(self, **kwargs):
                events.append(("configure", kwargs))
                return True

        prepare_gripper_for_replay(Effector())

        self.assertEqual(events, [
            ("disable",),
            ("configure", {"max_range_config": 0.1, "timeout": 5.0}),
        ])

    def test_amplified_gripper_backdates_confirmed_opening(self):
        samples = [
            {"time": 0.0, "gripper_mode": "width", "gripper": 0.1},
            {"time": 0.1, "gripper_mode": "width", "gripper": 0.08},
            {"time": 0.2, "gripper_mode": "width", "gripper": 0.0993},
            {"time": 0.5, "gripper_mode": "width", "gripper": 0.0995},
            {"time": 0.8, "gripper_mode": "width", "gripper": 0.0995},
            {"time": 0.96, "gripper_mode": "width", "gripper": 0.0995},
        ]

        amplified = amplify_gripper_samples(samples)

        self.assertEqual(
            [sample["gripper"] for sample in amplified],
            [0.1, 0.08, 0.0993, 0.0995, 0.0995, 0.0995],
        )
        self.assertEqual(
            [sample["gripper_grasping"] for sample in amplified],
            [False, True, False, False, False, False],
        )
        self.assertEqual(
            [sample["gripper_force"] for sample in amplified],
            [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
        )
        self.assertEqual(
            [sample["gripper"] for sample in samples],
            [0.1, 0.08, 0.0993, 0.0995, 0.0995, 0.0995],
        )

    def test_amplified_gripper_preserves_terminal_open_command(self):
        samples = [
            {"time": 0.0, "gripper_mode": "width", "gripper": 0.04},
            {"time": 1.0, "gripper_mode": "width", "gripper": 0.1},
        ]

        amplified = amplify_gripper_samples(samples)

        self.assertEqual([sample["gripper"] for sample in amplified], [0.04, 0.1])
        self.assertEqual(
            [sample["gripper_force"] for sample in amplified], [3.0, 3.0]
        )

    def test_amplified_gripper_recognizes_calibrated_open_plateau(self):
        samples = [
            {"time": index * 0.1, "gripper_mode": "width", "gripper": value}
            for index, value in enumerate(
                [0.0993] * 10 + [0.06] * 10 + [0.0993] * 6 + [0.1]
            )
        ]

        amplified = amplify_gripper_samples(samples)

        self.assertEqual(
            [sample["gripper"] for sample in amplified],
            [sample["gripper"] for sample in samples],
        )
        self.assertTrue(amplified[10]["gripper_grasping"])
        self.assertFalse(amplified[20]["gripper_grasping"])
        self.assertEqual(amplified[10]["gripper_force"], 3.0)
        self.assertEqual(amplified[20]["gripper_force"], 3.0)

    def test_amplified_gripper_preserves_lower_small_object_release_plateau(self):
        values = [0.0993] * 10 + [0.02] * 10 + [0.0988] * 6 + [0.0993] * 10
        samples = [
            {"time": index * 0.1, "gripper_mode": "width", "gripper": value}
            for index, value in enumerate(values)
        ]

        amplified = amplify_gripper_samples(samples)

        self.assertEqual(
            [sample["gripper"] for sample in amplified],
            [sample["gripper"] for sample in samples],
        )
        self.assertTrue(amplified[19]["gripper_grasping"])
        self.assertFalse(amplified[20]["gripper_grasping"])
        self.assertEqual(amplified[19]["gripper_force"], 3.0)
        self.assertEqual(amplified[20]["gripper_force"], 3.0)

    def test_amplified_opening_waits_for_recorded_release_pose(self):
        target = [0.0, -1.7594, 0.0, 0.0, 0.0, 0.0, 0.0]

        class Arm:
            def __init__(self):
                self.targets = []

            def move_js(self, value):
                self.targets.append(value)

        class Robot:
            def __init__(self):
                self._arm = Arm()
                self.positions = [
                    [0.02, -1.72, 0.0, 0.0, 0.0, 0.01, 0.0],
                    [0.009, -1.7453, 0.0, 0.0, 0.0, 0.004, 0.0],
                ]

            def get_joint_angles(self):
                return self.positions.pop(0)

        robot = Robot()
        with patch("task_trajectory.time.sleep"):
            wait_for_gripper_release_pose(robot, target, "Test")

        self.assertEqual(robot._arm.targets, [target])
        self.assertTrue(is_amplified_gripper_opening(
            {"gripper_grasping": False}, True
        ))

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