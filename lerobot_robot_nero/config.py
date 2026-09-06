from dataclasses import dataclass

from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("nero")
@dataclass
class NeroConfig(RobotConfig):
    """Configuration object for the Agilex NERO arm with gripper and wrist camera."""

    can_interface: str = "socketcan"
    can_channel: str = "can0"
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
    overview_camera_index: int = 0
    reset_on_connect: bool = True
