import logging
import os
import shutil
import subprocess
import time
from typing import Any

from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots import Robot

from pyAgxArm import AgxArmFactory, ArmModel, create_agx_arm_config

from .camera import IntelRealSenseD405, OpenCVWebcam
from .config import NeroConfig

logger = logging.getLogger(__name__)


class Nero(Robot):
    config_class = NeroConfig
    name = "nero"
    JOINT_KEYS = tuple(f"joint{i}.pos" for i in range(1, 8))

    def __init__(self, config: NeroConfig):
        super().__init__(config)
        self.config = config
        self._arm = None
        self._gripper_effector = None
        self._teach_mode_enabled = False
        self._camera = IntelRealSenseD405(
            device_index=0,
            serial=config.camera_serial,
            width=640,
            height=480,
        ) if getattr(config, "has_camera", False) else None
        self._overview_camera = OpenCVWebcam(
            device_index=config.overview_camera_index,
            width=640,
            height=480,
        ) if getattr(config, "has_overview_camera", False) else None

    @property
    def observation_features(self) -> dict[str, tuple[int, ...] | tuple[int, int, int]]:
        features = {
            "observation.state": (len(self.JOINT_KEYS),),
            "observation.images.wrist": (self._camera.height, self._camera.width, 3)
            if self._camera is not None
            else (480, 640, 3),
        }
        if self._overview_camera is not None:
            features["observation.images.overview"] = (480, 640, 3)
        return features

    @property
    def action_features(self) -> dict[str, tuple[int, ...]]:
        return {"action": (len(self.JOINT_KEYS),)}

    @property
    def is_connected(self) -> bool:
        return bool(self._arm is not None and getattr(self._arm, "is_connected", lambda: False)())

    def check_can_interface(self) -> bool:
        channel = self.config.can_channel
        if os.path.exists(f"/sys/class/net/{channel}"):
            return True
        if shutil.which("ip") is not None:
            result = subprocess.run(["ip", "link", "show", channel], capture_output=True, text=True)
            if result.returncode == 0:
                return True
        return False

    def _normalize_joint_angles(self, raw_angles: Any) -> list[float]:
        if raw_angles is None:
            return [0.0] * len(self.JOINT_KEYS)
        if hasattr(raw_angles, "msg"):
            raw_angles = raw_angles.msg
        if hasattr(raw_angles, "tolist"):
            raw_angles = raw_angles.tolist()
        if isinstance(raw_angles, dict):
            values = []
            for key in self.JOINT_KEYS:
                joint_name = key.removesuffix(".pos")
                values.append(float(raw_angles.get(joint_name, 0.0)))
            return values
        if isinstance(raw_angles, (list, tuple)):
            values = [float(v) for v in raw_angles]
            if len(values) != len(self.JOINT_KEYS):
                raise ValueError(f"Expected {len(self.JOINT_KEYS)} joint values, got {len(values)}")
            return values
        raise TypeError(f"Unsupported joint angles payload: {type(raw_angles)!r}")

    def _build_target(self, action: RobotAction) -> list[float]:
        if "action" in action:
            values = list(action["action"])
            if len(values) != len(self.JOINT_KEYS):
                raise ValueError(f"Expected {len(self.JOINT_KEYS)} action values, got {len(values)}")
            return [float(v) for v in values]

        values: list[float] = []
        for key in self.JOINT_KEYS:
            if key not in action:
                raise KeyError(f"Missing required joint target '{key}' in action payload")
            values.append(float(action[key]))
        return values

    def connect(self, calibrate: bool = True) -> None:
        if self._arm is not None and self.is_connected:
            return

        if not self.check_can_interface():
            raise RuntimeError(
                f"SocketCAN interface '{self.config.can_channel}' is not available. "
                "Run: sudo ip link set can0 up type can bitrate 1000000"
            )

        cfg = create_agx_arm_config(
            robot=ArmModel.NERO,
            firmeware_version=self.config.firmware_version,
            comm="can",
            interface=self.config.can_interface,
            channel=self.config.can_channel,
            bitrate=self.config.bitrate,
            enable_check_can=self.config.enable_check_can,
            auto_connect=self.config.auto_connect,
            timeout=self.config.timeout,
        )

        self._arm = AgxArmFactory.create_arm(cfg)
        self._arm.connect()

        self._arm.set_auto_set_motion_mode_enabled(False)
        self._arm.set_joint_limits_enabled(False)
        self._arm.set_speed_percent(self.config.speed_percent)
        if self.config.reset_on_connect and hasattr(self._arm, "reset"):
            self._arm.reset()
        if hasattr(self._arm, "enable"):
            deadline = time.monotonic() + 5.0
            enabled = False
            while time.monotonic() < deadline:
                enabled = bool(self._arm.enable())
                if enabled:
                    break
                time.sleep(0.1)
            if not enabled:
                raise RuntimeError("NERO arm joints did not re-enable after reset")
        self.configure()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        if self._arm is None:
            return
        if hasattr(self._arm, "OPTIONS") and hasattr(self._arm.OPTIONS, "MOTION_MODE"):
            self._arm.set_motion_mode(self._arm.OPTIONS.MOTION_MODE.J)
        elif hasattr(self._arm, "set_motion_mode"):
            self._arm.set_motion_mode("J")

    def set_teach_mode(self, enabled: bool = True) -> None:
        if self._arm is None:
            raise RuntimeError("Nero is not connected")
        if enabled:
            if hasattr(self._arm, "set_leader_mode"):
                self._arm.set_leader_mode()
            elif hasattr(self._arm, "set_follower_mode"):
                self._arm.set_follower_mode()
            else:
                raise AttributeError("Nero robot does not expose a teaching mode API")
            if hasattr(self._arm, "disable"):
                self._arm.disable()
            self._teach_mode_enabled = True
        else:
            if hasattr(self._arm, "set_follower_mode"):
                self._arm.set_follower_mode()
            elif hasattr(self._arm, "set_normal_mode"):
                self._arm.set_normal_mode()
            else:
                raise AttributeError("Nero robot does not expose a non-teach mode API")
                if hasattr(self._arm, "reset"):
                    self._arm.reset()
                if hasattr(self._arm, "enable"):
                    if not self._arm.enable():
                        raise RuntimeError("NERO arm joints did not re-enable after leaving Teach mode")
                self._teach_mode_enabled = False

    def get_joint_angles(self) -> list[float]:
        if self._arm is None:
            raise RuntimeError("Nero is not connected")

        raw_angles = None
        getter_names = (
            ["get_leader_joint_angles", "get_joint_angles"]
            if self._teach_mode_enabled
            else ["get_joint_angles", "get_leader_joint_angles"]
        )
        for getter_name in getter_names:
            getter = getattr(self._arm, getter_name, None)
            if getter is None:
                continue
            try:
                raw_angles = getter()
                if raw_angles is not None:
                    break
            except Exception:
                continue

        if raw_angles is None:
            raise AttributeError("Nero robot does not expose usable joint angles")
        return self._normalize_joint_angles(raw_angles)

    def get_teach_joint_angles(self) -> list[float]:
        if self._arm is None:
            raise RuntimeError("Nero is not connected")
        if hasattr(self._arm, "get_leader_joint_angles"):
            data = self._arm.get_leader_joint_angles()
            if data is None:
                return [0.0] * len(self.JOINT_KEYS)
            if hasattr(data, "msg"):
                data = data.msg
            return self._normalize_joint_angles(data)
        if hasattr(self._arm, "get_joint_angles"):
            return self._normalize_joint_angles(self._arm.get_joint_angles())
        raise AttributeError("Nero robot does not expose leader joint angles")

    def get_observation(self) -> RobotObservation:
        if not self.is_connected:
            raise RuntimeError("Nero is not connected")

        raw_angles = None
        getter_names = (
            ["get_leader_joint_angles", "get_joint_angles"]
            if self._teach_mode_enabled
            else ["get_joint_angles", "get_leader_joint_angles"]
        )
        for getter_name in getter_names:
            getter = getattr(self._arm, getter_name, None)
            if getter is None:
                continue
            try:
                raw_angles = getter()
                if raw_angles is not None:
                    break
            except Exception:
                continue

        if raw_angles is None:
            raise RuntimeError("Nero does not expose usable joint-angle feedback")

        values = self._normalize_joint_angles(raw_angles)
        obs: RobotObservation = {"observation.state": [float(value) for value in values]}
        if self._camera is not None:
            if not self._camera.is_connected:
                self._camera.connect()
            color, depth = self._camera.capture_frame()
            if color is not None:
                obs["observation.images.wrist"] = color
            if depth is not None:
                obs["observation.images.wrist_depth"] = depth
        if self._overview_camera is not None:
            if not self._overview_camera.is_connected:
                self._overview_camera.connect()
            overview = self._overview_camera.capture_frame()
            if overview is not None:
                obs["observation.images.overview"] = overview
        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        if not self.is_connected:
            raise RuntimeError("Nero is not connected")

        target = self._build_target(action)
        self._arm.move_j(target)
        return {"action": [float(value) for value in target]}

    def disconnect(self) -> None:
        if self._overview_camera is not None:
            self._overview_camera.disconnect()
        if self._arm is not None:
            try:
                if self.is_connected and hasattr(self._arm, "disable"):
                    self._arm.disable()
            except Exception:
                logger.debug("Ignoring disconnect failure", exc_info=True)
            finally:
                self._arm = None

    def get_arm_status(self) -> Any:
        if not self.is_connected:
            raise RuntimeError("Nero is not connected")
        return self._arm.get_arm_status()

    def _get_gripper_effector(self):
        if self._arm is None:
            raise RuntimeError("Nero is not connected")
        if self._gripper_effector is not None:
            return self._gripper_effector
        if hasattr(self._arm, "init_effector"):
            effector = self._arm.init_effector(self._arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
            self._gripper_effector = effector
            return effector
        return None

    def configure_gripper(self, max_range_config: float = 0.07) -> None:
        if self._arm is None:
            raise RuntimeError("Nero is not connected")

        effector = self._get_gripper_effector()
        if effector is None:
            raise AttributeError("Nero robot does not expose a gripper effector")

        if hasattr(effector, "disable_gripper"):
            try:
                effector.disable_gripper()
            except Exception:
                pass

        if hasattr(effector, "set_gripper_teaching_pendant_param"):
            try:
                effector.set_gripper_teaching_pendant_param(max_range_config=max_range_config)
            except Exception:
                pass

        if hasattr(effector, "calibrate_gripper"):
            try:
                effector.calibrate_gripper(timeout=1.0)
            except Exception:
                pass

    def _prepare_gripper_for_move(self) -> Any:
        if self._arm is None:
            raise RuntimeError("Nero is not connected")
        effector = self._get_gripper_effector()
        if effector is None:
            raise AttributeError("Nero robot does not expose a gripper effector")

        was_teaching = bool(self._teach_mode_enabled)
        if was_teaching:
            self.set_teach_mode(False)

        if hasattr(self._arm, "set_motion_mode") and hasattr(self._arm.OPTIONS, "MOTION_MODE"):
            try:
                self._arm.set_motion_mode(self._arm.OPTIONS.MOTION_MODE.P)
            except Exception:
                pass

        try:
            self.configure_gripper(max_range_config=0.07)
        except Exception:
            pass

        return effector, was_teaching

    def open_gripper(self, width_m: float = 0.055, force: float = 1.0) -> None:
        effector, was_teaching = self._prepare_gripper_for_move()
        try:
            if hasattr(effector, "move_gripper_m"):
                effector.move_gripper_m(value=float(width_m), force=float(force))
                return
            if hasattr(effector, "move_gripper"):
                effector.move_gripper(width=float(width_m), force=float(force))
                return
            raise AttributeError("Nero gripper effector does not support direct movement")
        finally:
            if was_teaching:
                self.set_teach_mode(True)

    def close_gripper(self, width_m: float = 0.0, force: float = 1.0) -> None:
        effector, was_teaching = self._prepare_gripper_for_move()
        try:
            if hasattr(effector, "move_gripper_deg"):
                effector.move_gripper_deg(value=float(width_m), force=float(force))
                return
            if hasattr(effector, "move_gripper_m"):
                effector.move_gripper_m(value=float(width_m), force=float(force))
                return
            if hasattr(effector, "move_gripper"):
                effector.move_gripper(width=float(width_m), force=float(force))
                return
            raise AttributeError("Nero gripper effector does not support direct movement")
        finally:
            if was_teaching:
                self.set_teach_mode(True)
