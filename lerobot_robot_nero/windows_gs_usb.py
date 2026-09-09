import logging
import time

import libusb_package
import usb.core
import usb.backend.libusb1
import usb.util
import can.interfaces
from struct import pack


logger = logging.getLogger(__name__)


if not hasattr(usb.backend.libusb1, "_nero_original_get_backend"):
    usb.backend.libusb1._nero_original_get_backend = usb.backend.libusb1.get_backend


def _get_bundled_backend(*args, **kwargs):
    kwargs.setdefault("find_library", libusb_package.find_library)
    return usb.backend.libusb1._nero_original_get_backend(*args, **kwargs)


usb.backend.libusb1.get_backend = _get_bundled_backend

from can.interfaces.gs_usb import GsUsbBus
from gs_usb.constants import GS_CAN_MODE_HW_TIMESTAMP, GS_CAN_MODE_NORMAL
from gs_usb.gs_usb import GsUsb


if not hasattr(GsUsb, "_nero_original_start"):
    GsUsb._nero_original_start = GsUsb.start
if not hasattr(GsUsb, "_nero_original_device_capability"):
    GsUsb._nero_original_device_capability = GsUsb.device_capability.fget


def _set_host_format(device) -> None:
    device.gs_usb.ctrl_transfer(0x41, 0, 1, 0, pack("<I", 0x0000BEEF))


def _device_capability_with_host_format(device):
    if device.capability is None:
        _set_host_format(device)
    return GsUsb._nero_original_device_capability(device)


GsUsb.device_capability = property(_device_capability_with_host_format)


def _start_with_host_format(
    device, flags=GS_CAN_MODE_NORMAL | GS_CAN_MODE_HW_TIMESTAMP
):
    _set_host_format(device)
    return GsUsb._nero_original_start(device, flags)


def _send_classic_frame(device, frame):
    # GS-USB hardware timestamps extend RX frames only; classic TX stays 20 bytes.
    device.gs_usb.write(0x02, frame.pack(False))
    return True


GsUsb.start = _start_with_host_format
GsUsb.send = _send_classic_frame


class WindowsGsUsbBus(GsUsbBus):
    def __init__(self, *args, **kwargs):
        for attempt in range(6):
            try:
                super().__init__(*args, **kwargs)
                return
            except usb.core.USBError as exc:
                self._dispose_usb_device()
                self._is_shutdown = True
                if getattr(exc, "errno", None) != 13 or attempt == 5:
                    raise
                time.sleep(0.25)
            except Exception:
                self._is_shutdown = True
                raise

    def _dispose_usb_device(self) -> None:
        gs_usb = getattr(self, "gs_usb", None)
        usb_device = getattr(gs_usb, "gs_usb", None)
        if usb_device is None:
            return
        try:
            usb.util.dispose_resources(usb_device)
        except Exception:
            logger.debug("Ignoring GS-USB resource disposal failure", exc_info=True)

    def shutdown(self) -> None:
        if getattr(self, "_is_shutdown", False):
            return
        try:
            gs_usb = getattr(self, "gs_usb", None)
            if gs_usb is not None:
                try:
                    gs_usb.stop()
                except usb.core.USBError:
                    logger.debug("Ignoring GS-USB stop failure", exc_info=True)
                self._dispose_usb_device()
        finally:
            can.BusABC.shutdown(self)

    @staticmethod
    def _detect_available_configs():
        return [
            {"interface": "gs_usb", "channel": index}
            for index, _ in enumerate(GsUsb.scan())
        ]


def register_windows_gs_usb() -> None:
    can.interfaces.BACKENDS["gs_usb"] = (
        "lerobot_robot_nero.windows_gs_usb",
        "WindowsGsUsbBus",
    )


def describe_windows_can_error(error: str) -> str:
    if "Cannot find device" in error or "Devices found: 0" in error:
        return (
            "GS-USB adapter not detected. Reconnect the Agilex CAN USB module, "
            "confirm it appears in Windows Device Manager, then click Refresh status."
        )
    if "access denied" in error.lower() or "errno 13" in error.lower():
        return (
            "GS-USB adapter is in use. Close other NERO Lab, CAN viewer, and Python "
            "processes, then try again."
        )
    if "pipe error" in error.lower() or "errno 32" in error.lower():
        return "GS-USB adapter needs a USB reset. Unplug it, reconnect it, then try again."
    return error