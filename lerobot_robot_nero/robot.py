import logging
import os
import shutil
import subprocess
import time
from typing import Any

import can
import can.interfaces
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
    GRIPPER_WIDTH_MIN_M = 0.0
    GRIPPER_WIDTH_MAX_M = 0.1
    GRIPPER_FORCE_MIN = 0.0
    GRIPPER_FORCE_MAX = 30.0

    def __init__(self, config: NeroConfig):
        super().__init__(config)
        self.config = config
        self._arm = None
        self._gripper_effector = None
        self._last_gripper_feedback = (self.GRIPPER_WIDTH_MAX_M, 0.0)
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
        state_size = len(self.JOINT_KEYS) + (2 if self.config.has_gripper else 0)
        features = {
            "observation.state": (state_size,),
            "observation.images.wrist": (self._camera.height, self._camera.width, 3)
            if self._camera is not None
            else (480, 640, 3),
        }
        if self._overview_camera is not None:
            features["observation.images.overview"] = (480, 640, 3)
        return features

    @property
    def action_features(self) -> dict[str, tuple[int, ...]]:
        action_size = len(self.JOINT_KEYS) + (2 if self.config.has_gripper else 0)
        return {"action": (action_size,)}

    @property
    def is_connected(self) -> bool:
        return bool(self._arm is not None and getattr(self._arm, "is_connected", lambda: False)())

    def _prepare_can_backend(self) -> None:
        if os.name == "nt" and self.config.can_interface == "gs_usb":
            from .windows_gs_usb import register_windows_gs_usb

            register_windows_gs_usb()

    def check_can_interface(self) -> bool:
        self._prepare_can_backend()
        if self.config.can_interface in {"gs_usb", "agx_cando"}:
            configs = can.detect_available_configs(interfaces=self.config.can_interface)
            return any(
                str(config.get("channel")) == str(self.config.can_channel)
                for config in configs
            )
        if self.config.can_interface != "socketcan":
            return True

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
            expected = len(self.JOINT_KEYS) + (2 if self.config.has_gripper else 0)
            if len(values) != expected:
                raise ValueError(f"Expected {expected} action values, got {len(values)}")
            return [float(v) for v in values[: len(self.JOINT_KEYS)]]

        values: list[float] = []
        for key in self.JOINT_KEYS:
            if key not in action:
                raise KeyError(f"Missing required joint target '{key}' in action payload")
            values.append(float(action[key]))
        return values

    def get_gripper_feedback(self) -> tuple[float, float]:
        if not self.config.has_gripper:
            raise RuntimeError("Nero gripper feedback requested without a gripper")
        effector = self._get_gripper_effector()
        status = effector.get_gripper_status() if effector is not None else None
        if status is None:
            return self._last_gripper_feedback
        message = getattr(status, "msg", status)
        mode = str(getattr(message, "mode", "width"))
        if mode != "width":
            raise RuntimeError(
                f"Nero gripper feedback is {mode!r}; width-mode feedback is required"
            )
        width = min(
            max(float(message.value), self.GRIPPER_WIDTH_MIN_M),
            self.GRIPPER_WIDTH_MAX_M,
        )
        force = min(
            max(float(getattr(message, "force", 0.0)), self.GRIPPER_FORCE_MIN),
            self.GRIPPER_FORCE_MAX,
        )
        self._last_gripper_feedback = (width, force)
        return self._last_gripper_feedback

    def _enable_can_feedback(self) -> None:
        if self._arm is None:
            return
        msg_mode = getattr(self._arm, "_msg_mode", None)
        set_mode = getattr(self._arm, "_set_mode", None)
        enums = getattr(msg_mode, "Enums", None)
        reporting = getattr(enums, "CanActiveMsgReporting", None)
        if msg_mode is None or set_mode is None or reporting is None:
            return

        previous_move_mode = msg_mode.move_mode
        try:
            msg_mode.move_mode = 255
            msg_mode.enable_can_push = reporting.ENABLE
            set_mode()
        finally:
            msg_mode.move_mode = previous_move_mode
        time.sleep(0.25)

    def connect(self, calibrate: bool = True) -> None:
        if self._arm is not None and self.is_connected:
            return

        self._prepare_can_backend()
        logger.info("Checking CAN interface %s channel %s", self.config.can_interface, self.config.can_channel)
        if self.config.can_interface != "gs_usb" and not self.check_can_interface():
            if self.config.can_interface in {"gs_usb", "agx_cando"}:
                raise RuntimeError(
                    f"Agilex CAN device channel '{self.config.can_channel}' was not detected. "
                    "Close other CAN programs, reconnect the official Agilex USB-CAN "
                    "module, and confirm that it appears in Windows Device Manager."
                )
            raise RuntimeError(
                f"SocketCAN interface '{self.config.can_channel}' is not available. "
                f"On Linux run: sudo ip link set {self.config.can_channel} up type can "
                f"bitrate {self.config.bitrate}"
            )

        logger.info("Creating NERO SDK configuration")
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

        logger.info("Creating NERO arm driver")
        self._arm = AgxArmFactory.create_arm(cfg)
        logger.info("Opening NERO CAN connection")
        self._arm.connect()
        logger.info("NERO CAN connection opened")

        logger.info("Configuring NERO arm")
        self._arm.set_auto_set_motion_mode_enabled(False)
        self._arm.set_joint_limits_enabled(False)
        self._arm.set_speed_percent(self.config.speed_percent)
        if self.config.reset_on_connect and hasattr(self._arm, "reset"):
            logger.info("Resetting NERO arm")
            if hasattr(self._arm, "set_follower_mode"):
                self._arm.set_follower_mode()
                time.sleep(0.5)
            self._arm.reset()
            time.sleep(1.0)
        logger.info("Enabling NERO CAN feedback")
        self._enable_can_feedback()
        if hasattr(self._arm, "enable"):
            logger.info("Enabling NERO arm joints")
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
        logger.info("NERO arm connection complete")

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

    def set_teach_mode(
        self, enabled: bool = True, hold_target: list[float] | None = None
    ) -> None:
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
                time.sleep(0.5)
            elif hasattr(self._arm, "set_normal_mode"):
                self._arm.set_normal_mode()
            else:
                raise AttributeError("Nero robot does not expose a non-teach mode API")
            if hasattr(self._arm, "reset"):
                self._arm.reset()
                time.sleep(1.0)
            self.configure()
            if hold_target is not None:
                move_js = getattr(self._arm, "move_js", None)
                if move_js is None:
                    raise RuntimeError(
                        "NERO arm cannot preload the taught pose before releasing brakes"
                    )
                move_js([float(value) for value in hold_target])
            if hasattr(self._arm, "enable"):
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    if self._arm.enable():
                        break
                    time.sleep(0.1)
                else:
                    raise RuntimeError("NERO arm joints did not re-enable after leaving Teach mode")
            if hold_target is not None:
                self._arm.move_js([float(value) for value in hold_target])
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
        if self.config.has_gripper:
            values.extend(self.get_gripper_feedback())
        obs: RobotObservation = {"observation.state": [float(value) for value in values]}
        if self._camera is not None:
            if not self._camera.is_connected:
                overview_was_connected = bool(
                    self._overview_camera is not None
                    and self._overview_camera.is_connected
                )
                if overview_was_connected:
                    self._overview_camera.disconnect()
                self._camera.connect()
                if overview_was_connected:
                    self._overview_camera.connect()
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
        result = [float(value) for value in target]
        if self.config.has_gripper:
            values = list(action["action"])
            width = min(
                max(float(values[-2]), self.GRIPPER_WIDTH_MIN_M),
                self.GRIPPER_WIDTH_MAX_M,
            )
            force = min(
                max(float(values[-1]), self.GRIPPER_FORCE_MIN),
                self.GRIPPER_FORCE_MAX,
            )
            effector = self._get_gripper_effector()
            if effector is None:
                raise RuntimeError("Nero gripper effector is unavailable")
            effector.move_gripper_m(value=width, force=force)
            result.extend((width, force))
        return {"action": result}

    def engage_brakes(
        self,
        timeout: float = 8.0,
        settle_time: float = 0.75,
        sample_interval: float = 0.05,
        motion_tolerance: float = 0.001,
    ) -> None:
        if self._arm is None or not self.is_connected:
            raise RuntimeError("Nero is not connected")

        arm = self._arm
        arm.electronic_emergency_stop()
        deadline = time.monotonic() + timeout
        stable_since: float | None = None
        stable_joints: list[float] | None = None
        emergency_stop_seen = False
        while time.monotonic() < deadline:
            status = arm.get_arm_status()
            message = getattr(status, "msg", status)
            arm_status = str(getattr(message, "arm_status", message))
            emergency_stop_seen = emergency_stop_seen or any(
                value in arm_status
                for value in ("EMERGENCY_STOP", "EMERGENCY STOP", "BRAKE_NOT_RELEASED")
            )
            current_joints = self.get_joint_angles()
            now = time.monotonic()
            if stable_joints is None or any(
                abs(current - stable) > motion_tolerance
                for current, stable in zip(current_joints, stable_joints)
            ):
                stable_since = now
                stable_joints = current_joints
            if (
                emergency_stop_seen
                and stable_since is not None
                and now - stable_since >= settle_time
            ):
                return
            time.sleep(sample_interval)
        raise RuntimeError(
            "Nero emergency-stop resting pose did not settle before disconnect; "
            f"status={arm.get_arm_status()} joints={self.get_joint_angles()}"
        )

    def disconnect(self, disable_arm: bool = True) -> None:
        if self._overview_camera is not None:
            self._overview_camera.disconnect()
        if self._arm is not None:
            if disable_arm and self.is_connected:
                self.engage_brakes()
            try:
                if hasattr(self._arm, "disconnect"):
                    self._arm.disconnect()
            except Exception:
                logger.debug("Ignoring SDK disconnect failure", exc_info=True)
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
