import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class IntelRealSenseD405:
    """Minimal Intel RealSense D405 wrapper for gripper-mounted wrist camera capture."""

    def __init__(
        self,
        device_index: int = 0,
        serial: str | None = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        self.device_index = device_index
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self._stream_width = width
        self._stream_height = height
        self._pipeline: Any = None
        self._config: Any = None
        self._device: Any = None
        self._connected = False
        self._consecutive_capture_failures = 0
        self._awaiting_first_frame = False
        self._last_color: Any = None
        self._last_depth: Any = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        try:
            import pyrealsense2 as rs  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime dependency may be absent
            logger.warning("pyrealsense2 not available; D405 acquisition is disabled: %s", exc)
            self._connected = False
            return

        try:
            self._pipeline = rs.pipeline()
            self._config = rs.config()

            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) == 0:
                raise RuntimeError("No RealSense devices found")
            if self.serial:
                matching_devices = [
                    device
                    for device in devices
                    if device.get_info(rs.camera_info.serial_number) == self.serial
                ]
                if not matching_devices:
                    raise RuntimeError(f"RealSense device {self.serial} was not found")
                device = matching_devices[0]
            else:
                if self.device_index >= len(devices):
                    raise RuntimeError(f"Requested device index {self.device_index}, but only {len(devices)} device(s) found")
                device = devices[self.device_index]
            serial = device.get_info(rs.camera_info.serial_number)
            self._config.enable_device(serial)

            usb_type = device.get_info(rs.camera_info.usb_type_descriptor)
            if usb_type.startswith("2"):
                self._stream_width = 480
                self._stream_height = 270
                logger.warning(
                    "D405 is connected over USB %s; using 480x270 at %s FPS "
                    "to keep both cameras stable",
                    usb_type,
                    self.fps,
                )
            else:
                self._stream_width = self.width
                self._stream_height = self.height

            self._config.enable_stream(
                rs.stream.color,
                self._stream_width,
                self._stream_height,
                rs.format.rgb8,
                self.fps,
            )
            self._config.enable_stream(
                rs.stream.depth,
                self._stream_width,
                self._stream_height,
                rs.format.z16,
                self.fps,
            )
            self._pipeline.start(self._config)
            self._connected = True
            self._consecutive_capture_failures = 0
            self._awaiting_first_frame = True
            logger.info("Intel RealSense D405 connected")
        except Exception as exc:  # pragma: no cover - hardware may not be attached
            logger.warning("Could not initialize Intel RealSense D405: %s", exc)
            self._connected = False
            self._pipeline = None
            self._config = None

    def disconnect(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                logger.debug("Ignoring D405 stop failure", exc_info=True)
        self._pipeline = None
        self._config = None
        self._device = None
        self._connected = False

    def can_acquire(self) -> bool:
        return self.is_connected

    def capture_frame(self) -> tuple[Any | None, Any | None]:
        if not self.is_connected or self._pipeline is None:
            return self._last_color, self._last_depth

        timeout_ms = 2000 if self._awaiting_first_frame else 500
        try:
            success, frames = self._pipeline.try_wait_for_frames(timeout_ms=timeout_ms)
            if not success:
                raise RuntimeError(f"Frame did not arrive within {timeout_ms} ms")
            self._store_frames(frames)
        except Exception as exc:  # pragma: no cover - hardware runtime path
            logger.warning("Failed to capture D405 frame: %s", exc)
            self._consecutive_capture_failures += 1
            if self._consecutive_capture_failures >= 2:
                logger.warning("Restarting stalled D405 pipeline")
                self.disconnect()
        return self._last_color, self._last_depth

    def _store_frames(self, frames: Any) -> None:
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return

        color = self._frame_to_array(color_frame)
        depth = self._frame_to_array(depth_frame)
        if color.shape[:2] != (self.height, self.width):
            import cv2

            color = cv2.resize(color, (self.width, self.height))
            depth = cv2.resize(
                depth,
                (self.width, self.height),
                interpolation=cv2.INTER_NEAREST,
            )
        self._last_color = color
        self._last_depth = depth
        self._consecutive_capture_failures = 0
        self._awaiting_first_frame = False

    @staticmethod
    def _frame_to_array(frame: Any) -> Any:
        import numpy as np

        return np.asanyarray(frame.get_data())


class OpenCVWebcam:
    """OpenCV webcam wrapper for a fixed overview camera."""

    def __init__(
        self,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self._stream_width = 320 if os.name == "nt" else width
        self._stream_height = 240 if os.name == "nt" else height
        self._capture: Any = None
        self._last_frame: Any = None

    @property
    def is_connected(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def connect(self) -> None:
        try:
            import cv2
        except Exception as exc:  # pragma: no cover - runtime dependency may be absent
            logger.warning("OpenCV not available; overview webcam acquisition is disabled: %s", exc)
            return

        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        capture = cv2.VideoCapture(self.device_index, backend)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._stream_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._stream_height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            capture.release()
            logger.warning("Could not open overview webcam at device index %s", self.device_index)
            return
        self._capture = capture
        logger.info("Overview webcam connected at device index %s", self.device_index)

    def disconnect(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None

    def capture_frame(self) -> Any | None:
        if not self.is_connected:
            return self._last_frame
        import cv2

        success, frame = self._capture.read()
        if not success or frame is None:
            return self._last_frame
        converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if converted.shape[:2] != (self.height, self.width):
            converted = cv2.resize(converted, (self.width, self.height))
        self._last_frame = converted
        return self._last_frame
