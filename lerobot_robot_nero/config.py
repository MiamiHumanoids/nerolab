from dataclasses import dataclass, field
import platform

from lerobot.robots import RobotConfig


def default_can_interface() -> str:
    return "gs_usb" if platform.system() == "Windows" else "socketcan"


def default_can_channel() -> str:
    return "0" if platform.system() == "Windows" else "can0"


def default_overview_camera_index() -> int:
    return 2 if platform.system() == "Windows" else 0


@RobotConfig.register_subclass("nero")
@dataclass
class NeroConfig(RobotConfig):
    """Configuration object for the Agilex NERO arm with gripper and wrist camera."""

    can_interface: str = field(default_factory=default_can_interface)
    can_channel: str = field(default_factory=default_can_channel)
    bitrate: int = 1_000_000
    firmware_version: str = "v121"
    enable_check_can: bool = True
    auto_connect: bool = True
    timeout: float = 1.0
    speed_percent: int = 50
    has_gripper: bool = True
    gripper_type: str = "agilex_piper_gripper"
    has_camera: bool = True
    camera_type: str = "intel_d405"
    camera_serial: str | None = None
    camera_device: str = "realsense"
    has_overview_camera: bool = False
    overview_camera_index: int = field(default_factory=default_overview_camera_index)
    reset_on_connect: bool = True
    enable_on_connect: bool = True
