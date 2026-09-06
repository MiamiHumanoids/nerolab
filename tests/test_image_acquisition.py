import unittest

from lerobot_robot_nero import NeroConfig
from lerobot_robot_nero.camera import IntelRealSenseD405, OpenCVWebcam


class TestImageAcquisition(unittest.TestCase):
    def test_config_tracks_camera_metadata(self):
        cfg = NeroConfig(id="demo", can_channel="can0")
        self.assertTrue(cfg.has_camera)
        self.assertEqual(cfg.camera_type, "intel_d405")

    def test_d405_starts_in_disabled_state_without_device(self):
        camera = IntelRealSenseD405(device_index=0, serial=None)
        self.assertFalse(camera.is_connected)
        self.assertFalse(camera.can_acquire())

    def test_overview_camera_starts_disconnected(self):
        camera = OpenCVWebcam(device_index=99)
        self.assertFalse(camera.is_connected)


if __name__ == "__main__":
    unittest.main()
