from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass


_INTERFACE_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]+$")


@dataclass(frozen=True)
class SocketCanStatus:
    detected: bool
    is_can: bool
    is_up: bool
    bitrate: int | None = None
    error: str | None = None


def _validated_interface(interface: str) -> str:
    if not _INTERFACE_PATTERN.fullmatch(interface):
        raise ValueError(f"Invalid SocketCAN interface name: {interface!r}")
    return interface


def _is_root() -> bool:
    get_effective_user_id = getattr(os, "geteuid", None)
    return get_effective_user_id is not None and get_effective_user_id() == 0


def get_socketcan_status(interface: str = "can0") -> SocketCanStatus:
    interface = _validated_interface(interface)
    ip_command = shutil.which("ip")
    if ip_command is None:
        return SocketCanStatus(False, False, False, error="iproute2 is not installed")

    result = subprocess.run(
        [ip_command, "-details", "-json", "link", "show", "dev", interface],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        error = result.stderr.strip() or f"interface {interface} was not found"
        return SocketCanStatus(False, False, False, error=error)

    try:
        link = json.loads(result.stdout)[0]
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        return SocketCanStatus(False, False, False, error=f"Could not parse ip link output: {exc}")

    link_info = link.get("linkinfo", {})
    is_can = link_info.get("info_kind") == "can"
    bitrate = link_info.get("info_data", {}).get("bitrate")
    return SocketCanStatus(
        detected=True,
        is_can=is_can,
        is_up="UP" in link.get("flags", []),
        bitrate=int(bitrate) if bitrate is not None else None,
    )


def _run_privileged_ip(commands: list[list[str]]) -> None:
    ip_command = shutil.which("ip")
    if ip_command is None:
        raise RuntimeError("iproute2 is not installed")

    command_text = " && ".join(
        " ".join(shlex.quote(part) for part in [ip_command, *command])
        for command in commands
    )
    shell_command = ["/bin/sh", "-c", command_text]
    if not _is_root():
        pkexec = shutil.which("pkexec")
        if pkexec is None:
            raise RuntimeError(
                "Polkit pkexec is not installed; run the SocketCAN ip commands with sudo"
            )
        shell_command.insert(0, pkexec)

    result = subprocess.run(shell_command, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "authorization was cancelled"
        raise RuntimeError(detail)


def start_socketcan(interface: str = "can0", bitrate: int = 1_000_000) -> None:
    interface = _validated_interface(interface)
    _run_privileged_ip([
        ["link", "set", interface, "down"],
        [
            "link",
            "set",
            interface,
            "type",
            "can",
            "bitrate",
            str(int(bitrate)),
            "restart-ms",
            "100",
        ],
        ["link", "set", interface, "up"],
    ])


def stop_socketcan(interface: str = "can0") -> None:
    interface = _validated_interface(interface)
    _run_privileged_ip([["link", "set", interface, "down"]])