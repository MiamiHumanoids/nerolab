# lerobot-nero

LeRobot hardware plugin and local workbench for the Agilex NERO 7-DOF arm,
Agilex gripper, and Intel RealSense D405 wrist camera. It supports:

- Windows 11 with the `gs_usb` backend and channel `0`.
- Ubuntu 22.04 with SocketCAN and interface `can0`.

The correct defaults are selected automatically. Explicit `--interface` and
`--channel` arguments can override them when using another adapter index or
Linux network interface.

## Requirements

- Python 3.10 or newer (Python 3.11 is recommended).
- Git, because `pyAgxArm` is installed from GitHub.
- The official CAN module supplied with the arm. The upstream SDK does not
	support arbitrary CAN adapters.
- The Intel RealSense runtime/driver if the D405 camera is used.

Keep the arm clear and be ready to stop it before running a command that
connects to hardware. Connecting can reset and enable the arm.

## Windows 11 setup

Open PowerShell in the repository root:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

The execution-policy change applies only to the current PowerShell process and
is discarded when that terminal closes. To avoid activation entirely, invoke
the virtual environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
```

The editable install also installs the Windows-only `gs_usb`, PyUSB, and
bundled libusb runtime used by the official candleLight-compatible CAN module.

Connect the official Agilex CAN module and the arm, then verify the selected
configuration:

```powershell
python -c "from lerobot_robot_nero import NeroConfig; c=NeroConfig(id='demo'); print(c.type, c.can_interface, c.can_channel)"
lerobot-nero --check-can
```

The expected values are `nero gs_usb 0`. Unlike SocketCAN, the Windows
backend discovers the device when the CAN bus is opened. Test a real hardware
connection with:

```powershell
lerobot-nero --no-camera
```

For a second CAN module, pass its zero-based device index, for example
`lerobot-nero --channel 1 --no-camera`.

To monitor raw CAN frames, close NERO Lab and other programs using the adapter,
then run the project's curses-free CAN dump utility. It prints one line per
frame. Stop it with `Ctrl+C`:

```powershell
python can_dump.py
```

For another adapter or bitrate, pass `--channel`, `--interface`, or `--bitrate`.

## Azure task and dataset storage

NERO Lab can use Azure Blob Storage as the shared store while keeping a local
working cache under `~/Nero`. On startup and each refresh, tasks and datasets
are downloaded from Azure. Successful recordings and taught tasks are uploaded
automatically, and GUI deletions are also applied to Azure.

Install the latest project dependencies after pulling these changes:

```powershell
python -m pip install -e .
```

Create an Azure Storage account, then give your Azure user the **Storage Blob
Data Contributor** role on that account. Install the Azure CLI on Windows if it
is not already available:

```powershell
winget install Microsoft.AzureCLI
```

Open a new PowerShell window, activate `.venv`, sign in, and set the storage
account name. The container defaults to `nero` and is created automatically:

```powershell
az login
$env:NERO_AZURE_STORAGE_ACCOUNT = "yourstorageaccount"
$env:NERO_AZURE_STORAGE_CONTAINER = "nero"
python nero_lab.py
```

To upload tasks and datasets that already exist in the local cache, run this
once after setting the environment variables:

```powershell
python azure_sync.py upload
```

To download the Azure contents without opening NERO Lab, run:

```powershell
python azure_sync.py download
```

These variables apply to the current PowerShell window. To persist them for
future terminals, use:

```powershell
setx NERO_AZURE_STORAGE_ACCOUNT "yourstorageaccount"
setx NERO_AZURE_STORAGE_CONTAINER "nero"
```

On Ubuntu, authenticate and configure the same values with:

```bash
az login
export NERO_AZURE_STORAGE_ACCOUNT="yourstorageaccount"
export NERO_AZURE_STORAGE_CONTAINER="nero"
python nero_lab.py
```

No Azure secret is stored in this repository. `DefaultAzureCredential` uses
the Azure CLI login locally and can also use managed identity or standard Azure
identity environment variables on hosted machines. If
`NERO_AZURE_STORAGE_ACCOUNT` is unset, NERO Lab continues in local-only mode.

## Ubuntu 22.04 setup

```bash
sudo apt update
sudo apt install -y can-utils git python3-venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Connect the CAN module, bring up SocketCAN at 1 Mbps, and verify it:

```bash
sudo ip link set can0 up type can bitrate 1000000
lerobot-nero --check-can
```

The expected configuration is `socketcan` with channel `can0`. Test a real
hardware connection with:

```bash
lerobot-nero --no-camera
```

If the interface has another name, use it consistently:

```bash
sudo ip link set can1 up type can bitrate 1000000
lerobot-nero --interface socketcan --channel can1 --no-camera
```

## NERO Lab GUI

With the virtual environment active, launch the dataset workbench on either
platform:

```powershell
python nero_lab.py
```

The GUI can control the arm, record task-labeled episodes, replay taught
trajectories, inspect datasets, launch LeRobot/Rerun, and run SmolVLA policies.
The **Arm control** tab includes platform-specific CAN controls. On Windows,
**Start CAN** opens GS-USB channel `0`. On Linux, it configures and brings up
SocketCAN `can0` at 1 Mbps with automatic bus-off restart; Polkit may display an
authorization dialog. **Refresh status** reports the adapter or kernel-link
state and configured bitrate. Disconnect the arm before stopping CAN.
Data is stored below the current user's home directory:

- Windows: `%USERPROFILE%\Nero\datasets` and `%USERPROFILE%\Nero\tasks`.
- Linux: `~/Nero/datasets` and `~/Nero/tasks`.

LeRobot commands such as `lerobot-train` and `lerobot-dataset-viz` are resolved
from the active virtual environment's `PATH` on both platforms.

For a no-teleoperation workflow, use the **Teach then replay** controls in the Record and train tab:

1. Click **Teach Task** and move the arm manually in Teach mode. Press `q` in the teach window to save the joint trajectory.
2. Return the arm to **Safe Bicep Reset**.
3. Click **Replay Trained Task (record dataset)**. The saved trajectory is replayed while joint data, gripper actions, wrist video, and overview webcam video are recorded into a new `nero_replayed__*` dataset.

SmolVLA training can be launched for a complete dataset with 9-value state and action vectors: 7 joint angles, gripper width in meters, and gripper force. State contains measured gripper feedback; action contains the commanded width and force. The GUI launches `run_smolvla_nero.py` for guarded hardware execution with explicit confirmation and action-size, width, and force checks.

The installed LeRobot policy stack must be able to import SmolVLA before
training or execution. Diagnose an import failure with:

```powershell
python -c "from lerobot.policies.factory import make_policy_config"
```

## Troubleshooting

- `Activate.ps1 cannot be loaded because running scripts is disabled`: run
	`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in the same
	PowerShell window, then run the activation command again.
- `No matching distribution found for pyAgxArm`: pull the latest project
  changes and rerun `python -m pip install -e .`; `pyAgxArm` is installed from
  its official GitHub repository because it is not published on every package
  index.
- `No backend available` from PyUSB on Windows: activate `.venv` and rerun
	`python -m pip install -e .` to install `libusb-package`.
- Python-CAN viewer/curses errors on Windows: use `python can_dump.py` instead.
- `detected 0 device(s)` or no Windows CAN device: close NERO Lab and any CAN
	viewer, reconnect the official Agilex USB-CAN module, and confirm that the
	adapter appears without an error in Windows Device Manager. If Windows does
	not list it, try another USB port and reinstall the adapter's Windows driver.
	Once detected, try channel indices `0`, `1`, and `2`. Windows does not use
	`ip link` or `can0`.
- `SocketCAN interface 'can0' is not available` on Linux: run `ip link show
	can0`, reconnect the adapter, and repeat the `sudo ip link set` command.
- Camera errors: install the Intel RealSense runtime, reconnect the D405, and
	confirm that no other application is using it.
- OpenCV reports `The function is not implemented` from `cvNamedWindow`: replace
	the headless build with the Windows GUI build, then restart NERO Lab:
	`python -m pip uninstall -y opencv-python opencv-python-headless`, followed by
	`python -m pip install opencv-python`.
- Azure reports `CredentialUnavailableError`: run `az login` in the same user
	session that launches NERO Lab.
- `az` is not recognized after installation: open a new PowerShell window, or
	run `$env:Path = "C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin;$env:Path"`
	to refresh the current window.
- Azure reports HTTP `403`: assign the signed-in user the **Storage Blob Data
	Contributor** role on the configured Storage account and allow a few minutes
	for the role assignment to take effect. Close and reopen NERO Lab after 5–10
	minutes so it obtains a fresh access token.

Upstream transport details are in the
[`pyAgxArm` CAN module manual](https://github.com/agilexrobotics/pyAgxArm/blob/master/docs/can_user.md).
