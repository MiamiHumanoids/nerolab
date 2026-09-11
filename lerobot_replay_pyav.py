#!/usr/bin/env python3
"""Launch LeRobot replay with the reliable PyAV video decoder."""

from lerobot.utils import import_utils


def install_enabled_disconnect() -> None:
    from lerobot_robot_nero import Nero

    original_disconnect = Nero.disconnect

    def disconnect_without_brakes(self, disable_arm: bool = True) -> None:
        original_disconnect(self, disable_arm=False)

    Nero.disconnect = disconnect_without_brakes


def main() -> None:
    import_utils.get_safe_default_video_backend = lambda: "pyav"
    install_enabled_disconnect()

    from lerobot.scripts.lerobot_replay import main as replay_main

    replay_main()


if __name__ == "__main__":
    main()