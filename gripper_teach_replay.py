# Not using this file

import os
import select
import sys
import termios
import time
import tty

from pyAgxArm import AgxArmFactory, ArmModel, create_agx_arm_config


def create_demo_config():
    return create_agx_arm_config(
        robot=ArmModel.NERO,
        firmeware_version="v121",
        interface="socketcan",
        channel="can0",
        bitrate=1_000_000,
    )


def read_char():
    if not sys.stdin.isatty():
        raise RuntimeError("No real TTY available. Run this script in a system terminal, not the Python debug console.")

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            r, _, _ = select.select([fd], [], [], 0.1)
            if not r:
                continue
            ch = sys.stdin.buffer.read(1)
            if not ch:
                continue
            return ch.decode("utf-8", errors="ignore")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def get_gripper_state_label(robot, effector):
    try:
        arm = robot.get_arm_status()
        arm_s = getattr(arm.msg, "arm_status", None) if arm is not None else None
    except Exception:
        arm_s = "ERR"

    try:
        gs = effector.get_gripper_status()
        if gs is None:
            return f"arm={arm_s}, gripper=None"
        return f"arm={arm_s}, gripper={gs.msg}"
    except Exception as exc:
        return f"arm={arm_s}, gripper_error={exc}"


def wait_for_motion_done(robot, timeout=5.0, poll=0.1):
    start = time.monotonic()
    while True:
        status = robot.get_arm_status()
        if status is not None and getattr(status.msg, "motion_status", None) == 0:
            return True
        if time.monotonic() - start > timeout:
            return False
        time.sleep(poll)


def teach_joint_recording(robot, effector):
    print("\nTeach mode active. Use pyAgxArm native leader mode.")
    print("  s = save current leader joint state")
    print("  g = record gripper close")
    print("  o = record gripper open")
    print("  r = replay recorded sequence")
    print("  q = quit teaching")

    if hasattr(robot, "set_leader_mode"):
        robot.set_leader_mode()
        print("Robot switched to leader mode.")
    else:
        print("This robot does not expose set_leader_mode().")
        return []

    sequence = []

    while True:
        ch = read_char().lower()
        if ch == "s":
            if hasattr(robot, "get_leader_joint_angles"):
                angles = robot.get_leader_joint_angles()
                if angles is None:
                    print("No leader joint angles available.")
                    continue
                angles_list = list(angles.msg)
                print(f"Saved leader joint angles: {angles_list}")
                sequence.append({"type": "joint", "angles": angles_list})
            else:
                print("get_leader_joint_angles not available.")
        elif ch == "g":
            print("Recorded gripper close command.")
            sequence.append({"type": "gripper", "cmd": "close", "value": 0.0, "force": 1.0})
        elif ch == "o":
            print("Recorded gripper open command.")
            sequence.append({"type": "gripper", "cmd": "open", "value": 0.040, "force": 1.0})
        elif ch == "r":
            if not sequence:
                print("Nothing recorded yet.")
                continue
            print("Replaying recorded sequence...")
            replay_sequence(robot, effector, sequence)
        elif ch == "q":
            print("Leaving teach mode.")
            return sequence
        else:
            print(f"Ignored key: {ch!r}")


def replay_sequence(robot, effector, sequence):
    if hasattr(robot, "set_follower_mode"):
        robot.set_follower_mode()
        print("Robot switched to follower mode.")
    else:
        print("This robot does not expose set_follower_mode().")
        return

    for idx, step in enumerate(sequence, 1):
        name = step["type"]
        print(f"\nReplay step {idx}: {name}")
        if name == "joint":
            angles = list(step["angles"])
            print(f"move_j({angles})")
            robot.move_j(angles)
            wait_for_motion_done(robot, timeout=8.0, poll=0.1)
            print(f"after: {get_gripper_state_label(robot, effector)}")
        elif name == "gripper":
            cmd = step["cmd"]
            value = step["value"]
            force = step["force"]
            if cmd == "close":
                print(f"gripper close -> move_gripper_m(value={value}, force={force})")
                effector.move_gripper_m(value=value, force=force)
            elif cmd == "open":
                print(f"gripper open -> move_gripper_m(value={value}, force={force})")
                effector.move_gripper_m(value=value, force=force)
            else:
                print(f"Unknown gripper cmd: {cmd}")
            time.sleep(1.0)
            print(f"after: {get_gripper_state_label(robot, effector)}")
        else:
            print(f"Unknown recorded step: {step}")

    print("Replay complete.")


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

    print("pyAgxArm teach/replay test")
    print("Use leader mode to teach joint poses; use follower mode to replay them.")
    print("Press 't' to enter teach mode, 'r' to replay, 'q' to quit.")

    last_sequence = []

    while True:
        ch = read_char().lower()
        if ch == "t":
            last_sequence = teach_joint_recording(robot, effector)
            if not last_sequence:
                print("No sequence recorded.")
        elif ch == "r":
            if not last_sequence:
                print("No recorded sequence available. Press 't' to teach one first.")
            else:
                print("Replaying last recorded sequence...")
                replay_sequence(robot, effector, last_sequence)
        elif ch == "q":
            print("Quit")
            break
        else:
            print(f"Ignored key: {ch!r}")

    try:
        if hasattr(robot, "set_normal_mode"):
            robot.set_normal_mode()
    except Exception:
        pass

    try:
        robot.disconnect()
    except Exception:
        pass


if __name__ == "__main__":
    main()
