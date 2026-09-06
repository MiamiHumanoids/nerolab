import logging
from typing import Any

logger = logging.getLogger(__name__)


class IntelRealSenseD405:
    """Minimal Intel RealSense D405 wrapper for gripper-mounted wrist camera capture."""

    def __init__(self, device_index: int = 0, serial: str | None = None, width: int = 640, height: int = 480):
        self.device_index = device_index
        self.serial = serial
        self.width = width
        self.height = height
        self._pipeline: Any = None
        self._config: Any = None
        self._device: Any = None
        self._connected = False

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

            if self.serial:
                self._config.enable_device(self.serial)
            else:
                ctx = rs.context()
                devices = ctx.query_devices()
                if len(devices) == 0:
                    raise RuntimeError("No RealSense devices found")
                if self.device_index >= len(devices):
                    raise RuntimeError(f"Requested device index {self.device_index}, but only {len(devices)} device(s) found")
                serial = devices[self.device_index].get_info(rs.camera_info.serial_number)
                self._config.enable_device(serial)

            self._config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 30)
            self._config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 30)
            self._pipeline.start(self._config)
            self._connected = True
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
            return None, None

        try:
            import pyrealsense2 as rs  # type: ignore

            frames = self._pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                return None, None

            color = self._frame_to_array(color_frame)
            depth = self._frame_to_array(depth_frame)
            return color, depth
        except Exception as exc:  # pragma: no cover - hardware runtime path
            logger.warning("Failed to capture D405 frame: %s", exc)
            return None, None

    @staticmethod
    def _frame_to_array(frame: Any) -> Any:
        import numpy as np

        return np.asanyarray(frame.get_data())


class OpenCVWebcam:
    """OpenCV webcam wrapper for a fixed overview camera."""

    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480):
        self.device_index = device_index
        self.width = width
        self.height = height
        self._capture: Any = None

    @property
    def is_connected(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def connect(self) -> None:
        try:
            import cv2
        except Exception as exc:  # pragma: no cover - runtime dependency may be absent
            logger.warning("OpenCV not available; overview webcam acquisition is disabled: %s", exc)
            return

        capture = cv2.VideoCapture(self.device_index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
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
            return None
        import cv2

        success, frame = self._capture.read()
        if not success or frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
