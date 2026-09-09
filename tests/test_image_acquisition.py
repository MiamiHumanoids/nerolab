import unittest
from unittest.mock import MagicMock, patch

from lerobot_robot_nero import NeroConfig
from lerobot_robot_nero.camera import IntelRealSenseD405, OpenCVWebcam
from lerobot_robot_nero.config import default_overview_camera_index


class TestImageAcquisition(unittest.TestCase):
    def test_config_tracks_camera_metadata(self):
        cfg = NeroConfig(id="demo", can_channel="can0")
        self.assertTrue(cfg.has_camera)
        self.assertEqual(cfg.camera_type, "intel_d405")

    def test_d405_starts_in_disabled_state_without_device(self):
        camera = IntelRealSenseD405(device_index=0, serial=None)
        self.assertFalse(camera.is_connected)
        self.assertFalse(camera.can_acquire())
        self.assertEqual((camera._stream_width, camera._stream_height), (640, 480))

    def test_d405_capture_reuses_last_frame_on_timeout(self):
        camera = IntelRealSenseD405()
        camera._connected = True
        camera._pipeline = MagicMock()
        camera._pipeline.try_wait_for_frames.return_value = (False, None)
        camera._last_color = "color"
        camera._last_depth = "depth"

        self.assertEqual(camera.capture_frame(), ("color", "depth"))
        self.assertTrue(camera.is_connected)

    def test_overview_camera_starts_disconnected(self):
        camera = OpenCVWebcam(device_index=99)
        self.assertFalse(camera.is_connected)

    @patch("lerobot_robot_nero.config.platform.system", return_value="Windows")
    def test_windows_overview_camera_defaults_to_ugreen_index(self, _system):
        self.assertEqual(default_overview_camera_index(), 2)

    def test_windows_overview_camera_uses_directshow(self):
        capture = MagicMock()
        capture.isOpened.return_value = True
        cv2 = MagicMock(
            CAP_DSHOW=700,
            CAP_ANY=0,
            CAP_PROP_FOURCC=6,
            CAP_PROP_FRAME_WIDTH=3,
            CAP_PROP_FRAME_HEIGHT=4,
            CAP_PROP_FPS=5,
            CAP_PROP_BUFFERSIZE=38,
        )
        cv2.VideoWriter_fourcc.return_value = 1196444237
        cv2.VideoCapture.return_value = capture
        frame = MagicMock()
        frame.shape = (240, 320, 3)
        converted = MagicMock()
        converted.shape = (240, 320, 3)
        capture.read.return_value = (True, frame)
        cv2.cvtColor.return_value = converted
        cv2.resize.return_value = "rgb-frame"
        camera = OpenCVWebcam(device_index=2)

        with (
            patch.dict("sys.modules", {"cv2": cv2}),
            patch("lerobot_robot_nero.camera.os.name", "nt"),
        ):
            camera.connect()
            captured = camera.capture_frame()

        cv2.VideoCapture.assert_called_once_with(2, cv2.CAP_DSHOW)
        cv2.VideoWriter_fourcc.assert_called_once_with(*"MJPG")
        capture.set.assert_any_call(cv2.CAP_PROP_FOURCC, 1196444237)
        capture.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 320)
        capture.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        capture.set.assert_any_call(cv2.CAP_PROP_FPS, 30)
        capture.set.assert_any_call(cv2.CAP_PROP_BUFFERSIZE, 1)
        cv2.resize.assert_called_with(converted, (640, 480))
        self.assertTrue(camera.is_connected)
        self.assertEqual(captured, "rgb-frame")

if __name__ == "__main__":
    unittest.main()
