import sys
import time

from pyAgxArm import AgxArmFactory, ArmModel, create_agx_arm_config
from lerobot_robot_nero.config import default_can_channel, default_can_interface
from lerobot_robot_nero.console import read_char

GRIPPER_CLOSED_WIDTH_M = 0.0
GRIPPER_OPEN_WIDTH_M = 0.1
# I think the max is 40 but it doesn't seem to work with 40... so using 30
GRIPPER_FORCE_N = 30.0


def create_demo_config():
    return create_agx_arm_config(
        robot=ArmModel.NERO,
        firmeware_version="v121",
        interface=default_can_interface(),
        channel=default_can_channel(),
        bitrate=1_000_000,
    )


def print_gripper_state(label: str, robot, effector) -> None:
    try:
        status = robot.get_arm_status()
        print(f"{label} arm_status={getattr(status.msg, 'arm_status', None) if status is not None else None}")
    except Exception as exc:
        print(f"{label} arm_status_error={exc}")
    try:
        gs = effector.get_gripper_status()
        if gs is None:
            print(f"{label} gripper_status=None")
        else:
            print(f"{label} gripper_status={gs.msg}")
    except Exception as exc:
        print(f"{label} gripper_status_error={exc}")


def print_gripper_position(label: str, effector, commanded_position_m=None) -> None:
    try:
        gs = effector.get_gripper_status()
        if gs is None:
            print(f"{label} commanded_position_m={commanded_position_m} feedback_position_m=None")
        else:
            print(
                f"{label} commanded_position_m={commanded_position_m} "
                f"feedback_position_m={getattr(gs.msg, 'value', None)}"
            )
    except Exception as exc:
        print(f"{label} gripper_position_error={exc}")


def calibrate_gripper(robot, effector) -> None:
    if not hasattr(effector, "calibrate_gripper"):
        print("This gripper does not expose calibrate_gripper().")
        return

    try:
        if hasattr(effector, "disable_gripper"):
            effector.disable_gripper()
        if hasattr(effector, "set_gripper_teaching_pendant_param"):
            range_set = effector.set_gripper_teaching_pendant_param(
                max_range_config=GRIPPER_OPEN_WIDTH_M, timeout=5.0
            )
            print(f"Gripper range configuration {'succeeded' if range_set else 'not acknowledged'}.")
        print("Gripper released. Close the jaws fully, then press Enter.")
        input()
        calibrated = effector.calibrate_gripper(timeout=5.0)
        print(f"Calibration {'succeeded' if calibrated else 'failed'}.")
        if hasattr(robot, "set_motion_mode") and hasattr(robot.OPTIONS, "MOTION_MODE"):
            robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.P)
        effector.move_gripper_m(
            value=GRIPPER_CLOSED_WIDTH_M, force=GRIPPER_FORCE_N
        )
        print("Closed position command sent. Press 'o' to test 100 mm open, then 'g' to close.")
    except Exception as exc:
        print(f"Gripper calibration failed: {exc}")


def run_gripper_trials(robot, effector) -> None:
    print("\n=== gripper control trials ===")
    tests = []

    if hasattr(effector, "move_gripper_m"):
        tests.extend([
            ("m_0.000", lambda: effector.move_gripper_m(value=0.000, force=GRIPPER_FORCE_N)),
            ("m_0.005", lambda: effector.move_gripper_m(value=0.005, force=GRIPPER_FORCE_N)),
            ("m_0.015", lambda: effector.move_gripper_m(value=0.015, force=GRIPPER_FORCE_N)),
            ("m_0.100", lambda: effector.move_gripper_m(value=GRIPPER_OPEN_WIDTH_M, force=GRIPPER_FORCE_N)),
        ])
    if hasattr(effector, "disable_gripper"):
        tests.append(("disable", lambda: effector.disable_gripper()))
    if hasattr(effector, "reset_gripper"):
        tests.append(("reset", lambda: effector.reset_gripper()))

    for name, action in tests:
        print(f"\n--- trial: {name} ---")
        try:
            print_gripper_state(f"before_{name}", robot, effector)
            action()
            time.sleep(0.75)
            print_gripper_state(f"after_{name}", robot, effector)
        except Exception as exc:
            print(f"trial {name} failed: {exc}")


def main() -> None:
    if not sys.stdin.isatty():
        print("This script requires a real terminal. Use Run in Terminal / terminal command, not VS Code debug console.")
        return

    cfg = create_demo_config()
    robot = AgxArmFactory.create_arm(cfg)
    robot.connect()

    robot.set_auto_set_motion_mode_enabled(False)
    robot.set_joint_limits_enabled(False)
    robot.enable()
    robot.set_speed_percent(100)

    effector = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
    if hasattr(effector, "set_gripper_teaching_pendant_param"):
        try:
            if hasattr(effector, "disable_gripper"):
                effector.disable_gripper()
            effector.set_gripper_teaching_pendant_param(
                max_range_config=GRIPPER_OPEN_WIDTH_M, timeout=5.0
            )
        except Exception as exc:
            print(f"Could not set gripper range: {exc}")
    print("Gripper test active.")
    print(f"Using width control: 0-{GRIPPER_OPEN_WIDTH_M:.3f} m, force {GRIPPER_FORCE_N:.1f} N.")
    print("Press '0' or 'o' to open, '1' or 'g' to close, 'r' to release, 'c' to calibrate, 't' for trials, 'q' to quit.")

    def dump_state(label: str) -> None:
        try:
            status = robot.get_arm_status()
            print(f"{label} arm_status={getattr(status.msg, 'arm_status', None) if status is not None else None}")
        except Exception as exc:
            print(f"{label} arm_status_error={exc}")
        try:
            gs = effector.get_gripper_status()
            if gs is None:
                print(f"{label} gripper_status=None")
            else:
                print(f"{label} gripper_status={gs.msg}")
        except Exception as exc:
            print(f"{label} gripper_status_error={exc}")

    try:
        commanded_position_m = None
        while True:
            ch = read_char().lower()
            if ch in ("g", "1"):
                commanded_position_m = GRIPPER_CLOSED_WIDTH_M
                print("Sending close gripper command")
                dump_state("before_close")
                try:
                    arm_before = robot.get_arm_status()
                    print(f"close arm_status_before={getattr(arm_before.msg, 'arm_status', None) if arm_before is not None else None}")
                except Exception as exc:
                    print(f"close arm_status_before_error={exc}")
                if hasattr(robot, "set_motion_mode") and hasattr(robot.OPTIONS, "MOTION_MODE"):
                    robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.P)
                effector.move_gripper_m(value=GRIPPER_CLOSED_WIDTH_M, force=GRIPPER_FORCE_N)
                try:
                    arm_after = robot.get_arm_status()
                    print(f"close arm_status_after={getattr(arm_after.msg, 'arm_status', None) if arm_after is not None else None}")
                except Exception as exc:
                    print(f"close arm_status_after_error={exc}")
                dump_state("after_close")
                print("close sent")
            elif ch in ("o", "0"):
                commanded_position_m = GRIPPER_OPEN_WIDTH_M
                print("Sending open gripper command")
                dump_state("before_open")
                try:
                    arm_before = robot.get_arm_status()
                    print(f"open arm_status_before={getattr(arm_before.msg, 'arm_status', None) if arm_before is not None else None}")
                except Exception as exc:
                    print(f"open arm_status_before_error={exc}")
                if hasattr(robot, "set_motion_mode") and hasattr(robot.OPTIONS, "MOTION_MODE"):
                    robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.P)
                effector.move_gripper_m(value=GRIPPER_OPEN_WIDTH_M, force=GRIPPER_FORCE_N)
                try:
                    arm_after = robot.get_arm_status()
                    print(f"open arm_status_after={getattr(arm_after.msg, 'arm_status', None) if arm_after is not None else None}")
                except Exception as exc:
                    print(f"open arm_status_after_error={exc}")
                dump_state("after_open")
                print("open sent")
            elif ch == "r":
                commanded_position_m = None
                if hasattr(effector, "disable_gripper"):
                    effector.disable_gripper()
                    print("Gripper released; it can be adjusted by hand.")
                else:
                    print("This gripper does not expose disable_gripper().")
            elif ch == "c":
                commanded_position_m = GRIPPER_CLOSED_WIDTH_M
                calibrate_gripper(robot, effector)
            elif ch == "t":
                print("Running gripper alternative trials...")
                run_gripper_trials(robot, effector)
            elif ch == "q":
                print("Quit")
                break
            else:
                print(f"Ignored key: {ch!r}")
            print_gripper_position(
                f"after_key_{ch!r}", effector, commanded_position_m
            )
    finally:
        try:
            if hasattr(effector, "disable_gripper"):
                effector.disable_gripper()
        except Exception:
            pass
        try:
            robot.set_follower_mode()
        except Exception:
            pass
        try:
            if hasattr(robot, "reset"):
                robot.reset()
                print("Robot reset")
        except Exception as exc:
            print(f"Robot reset failed: {exc}")
        try:
            robot.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
