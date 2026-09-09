#!/usr/bin/env python3

import argparse
import faulthandler
import logging

from lerobot_robot_nero import NeroConfig, NeroRobot
from lerobot_robot_nero.config import default_can_channel, default_can_interface


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Connect to an Agilex NERO arm with an optional wrist camera"
    )
    parser.add_argument("--channel", default=default_can_channel())
    parser.add_argument("--bitrate", type=int, default=1_000_000)
    parser.add_argument("--interface", default=default_can_interface())
    parser.add_argument("--firmware-version", default="v121")
    parser.add_argument("--speed-percent", type=int, default=50)
    parser.add_argument("--no-gripper", action="store_true", help="Disable gripper metadata for the robot configuration")
    parser.add_argument("--no-camera", action="store_true", help="Disable camera metadata for the robot configuration")
    parser.add_argument("--check-can", action="store_true", help="Only check whether the configured CAN backend is available")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = NeroConfig(
        id="nero",
        can_interface=args.interface,
        can_channel=args.channel,
        bitrate=args.bitrate,
        firmware_version=args.firmware_version,
        speed_percent=args.speed_percent,
        has_gripper=not args.no_gripper,
        has_camera=not args.no_camera,
    )
    robot = NeroRobot(cfg)

    if args.check_can:
        print(f"can={robot.check_can_interface()} interface={args.interface} channel={args.channel}")
        return

    faulthandler.dump_traceback_later(15.0, repeat=True)
    try:
        robot.connect()
    finally:
        faulthandler.cancel_dump_traceback_later()
    print(f"is_connected={robot.is_connected}")
    print(robot.get_arm_status())
    robot.disconnect()


if __name__ == "__main__":
    main()
