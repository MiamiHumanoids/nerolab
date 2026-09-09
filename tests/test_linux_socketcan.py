import json
from unittest.mock import MagicMock, patch

from lerobot_robot_nero.linux_socketcan import (
    get_socketcan_status,
    start_socketcan,
    stop_socketcan,
)


@patch("lerobot_robot_nero.linux_socketcan.shutil.which", return_value="/usr/sbin/ip")
@patch("lerobot_robot_nero.linux_socketcan.subprocess.run")
def test_get_socketcan_status_reads_kernel_link_state(run, _which):
    run.return_value = MagicMock(
        returncode=0,
        stdout=json.dumps([{
            "flags": ["NOARP", "UP"],
            "linkinfo": {
                "info_kind": "can",
                "info_data": {"bitrate": 1_000_000},
            },
        }]),
        stderr="",
    )

    status = get_socketcan_status("can0")

    assert status.detected
    assert status.is_can
    assert status.is_up
    assert status.bitrate == 1_000_000


@patch("lerobot_robot_nero.linux_socketcan._is_root", return_value=True)
@patch("lerobot_robot_nero.linux_socketcan.shutil.which", return_value="/usr/sbin/ip")
@patch("lerobot_robot_nero.linux_socketcan.subprocess.run")
def test_start_socketcan_configures_one_megabit_and_restart(run, _which, _is_root):
    run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    start_socketcan("can0")

    command_text = run.call_args.args[0][-1]
    assert "link set can0 down" in command_text
    assert "type can bitrate 1000000 restart-ms 100" in command_text
    assert "link set can0 up" in command_text


@patch("lerobot_robot_nero.linux_socketcan._is_root", return_value=True)
@patch("lerobot_robot_nero.linux_socketcan.shutil.which", return_value="/usr/sbin/ip")
@patch("lerobot_robot_nero.linux_socketcan.subprocess.run")
def test_stop_socketcan_brings_interface_down(run, _which, _is_root):
    run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    stop_socketcan("can0")

    assert run.call_args.args[0][-1] == "/usr/sbin/ip link set can0 down"


@patch("lerobot_robot_nero.linux_socketcan._is_root", return_value=False)
@patch("lerobot_robot_nero.linux_socketcan.shutil.which")
@patch("lerobot_robot_nero.linux_socketcan.subprocess.run")
def test_non_root_socketcan_uses_polkit(run, which, _is_root):
    which.side_effect = lambda command: {
        "ip": "/usr/sbin/ip",
        "pkexec": "/usr/bin/pkexec",
    }.get(command)
    run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    stop_socketcan("can0")

    assert run.call_args.args[0][:3] == ["/usr/bin/pkexec", "/bin/sh", "-c"]