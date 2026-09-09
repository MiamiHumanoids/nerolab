import argparse
import os
import signal

import can

from lerobot_robot_nero.config import default_can_channel, default_can_interface


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print raw NERO CAN frames until Ctrl+C is pressed."
    )
    parser.add_argument("--interface", default=default_can_interface())
    parser.add_argument("--channel", default=default_can_channel())
    parser.add_argument("--bitrate", type=int, default=1_000_000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if os.name == "nt" and args.interface == "gs_usb":
        from lerobot_robot_nero.windows_gs_usb import register_windows_gs_usb

        register_windows_gs_usb()
    running = True

    def stop(_signum, _frame) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    print(
        f"Listening on interface={args.interface} channel={args.channel} "
        f"bitrate={args.bitrate}. Press Ctrl+C to stop."
    )
    count = 0
    with can.Bus(
        interface=args.interface,
        channel=args.channel,
        bitrate=args.bitrate,
        receive_own_messages=False,
        local_loopback=False,
    ) as bus:
        while running:
            message = bus.recv(timeout=0.25)
            if message is None:
                continue
            count += 1
            frame_id = f"{message.arbitration_id:08X}" if message.is_extended_id else f"{message.arbitration_id:03X}"
            data = " ".join(f"{value:02X}" for value in message.data)
            print(
                f"{count:8d}  {message.timestamp:14.6f}  "
                f"{frame_id}  {message.dlc:3d}  {data}",
                flush=True,
            )


if __name__ == "__main__":
    main()