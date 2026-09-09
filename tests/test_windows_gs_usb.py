from unittest.mock import MagicMock, patch

import usb.core
from can.interfaces.gs_usb import GsUsbBus
from gs_usb.gs_usb import GsUsb
from lerobot_robot_nero.windows_gs_usb import (
    WindowsGsUsbBus,
    _device_capability_with_host_format,
    _send_classic_frame,
    _start_with_host_format,
    describe_windows_can_error,
)


def test_start_configures_little_endian_host_format():
    device = MagicMock()
    original_start = GsUsb._nero_original_start
    GsUsb._nero_original_start = MagicMock(return_value=True)
    try:
        assert _start_with_host_format(device, flags=0x10) is True
    finally:
        mocked_start = GsUsb._nero_original_start
        GsUsb._nero_original_start = original_start

    device.gs_usb.ctrl_transfer.assert_called_once_with(
        0x41, 0, 1, 0, b"\xef\xbe\x00\x00"
    )
    mocked_start.assert_called_once_with(device, 0x10)


def test_capability_query_configures_host_format_first():
    device = MagicMock()
    device.capability = None
    original_capability = GsUsb._nero_original_device_capability
    GsUsb._nero_original_device_capability = MagicMock()
    try:
        _device_capability_with_host_format(device)
    finally:
        queried_capability = GsUsb._nero_original_device_capability
        GsUsb._nero_original_device_capability = original_capability

    device.gs_usb.ctrl_transfer.assert_called_once_with(
        0x41, 0, 1, 0, b"\xef\xbe\x00\x00"
    )
    queried_capability.assert_called_once_with(device)


def test_classic_transmit_does_not_append_hardware_timestamp():
    device = MagicMock()
    frame = MagicMock()
    frame.pack.return_value = b"frame"

    assert _send_classic_frame(device, frame) is True

    frame.pack.assert_called_once_with(False)
    device.gs_usb.write.assert_called_once_with(0x02, b"frame")


def test_windows_shutdown_stops_and_disposes_without_rescan():
    bus = object.__new__(WindowsGsUsbBus)
    bus._is_shutdown = False
    bus.gs_usb = MagicMock()
    bus.stop_all_periodic_tasks = MagicMock()

    with (
        patch("lerobot_robot_nero.windows_gs_usb.usb.util.dispose_resources") as dispose,
        patch("lerobot_robot_nero.windows_gs_usb.GsUsb.scan") as scan,
    ):
        bus.shutdown()

    bus.gs_usb.stop.assert_called_once_with()
    dispose.assert_called_once_with(bus.gs_usb.gs_usb)
    scan.assert_not_called()
    assert bus._is_shutdown is True


def test_windows_open_retries_access_denied_only():
    bus = object.__new__(WindowsGsUsbBus)
    access_denied = usb.core.USBError("Access denied", errno=13)

    with (
        patch.object(GsUsbBus, "__init__", side_effect=[access_denied, None]) as init,
        patch.object(WindowsGsUsbBus, "_dispose_usb_device") as dispose,
        patch("lerobot_robot_nero.windows_gs_usb.time.sleep") as sleep,
    ):
        WindowsGsUsbBus.__init__(bus, channel=0, bitrate=1_000_000)

    assert init.call_count == 2
    dispose.assert_called_once_with()
    sleep.assert_called_once_with(0.25)


def test_missing_adapter_error_is_actionable():
    message = describe_windows_can_error("Cannot find device 0. Devices found: 0")

    assert "adapter not detected" in message
    assert "Windows Device Manager" in message


def test_access_denied_error_is_actionable():
    message = describe_windows_can_error("[Errno 13] Access denied")

    assert "adapter is in use" in message