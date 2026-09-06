# lerobot-nero

LeRobot hardware plugin for the Agilex NERO 7-DOF arm with an Agilex Piper-style gripper and an Intel RealSense D405 camera mounted on the wrist/gripper assembly, exposed through the `pyAgxArm` SDK over SocketCAN.

## Install

```bash
python3 -m pip install -e .
```

## Verify the plugin

```bash
python3 -c "from lerobot_robot_nero import NeroConfig; print(NeroConfig(id='demo', can_channel='can0', bitrate=1000000, firmware_version='v121').type)"
```

## Connect via SocketCAN

```bash
sudo ip link set can0 up type can bitrate 1000000
lerobot-nero --channel can0 --bitrate 1000000 --firmware-version v121
```

## Quick hardware check

```bash
lerobot-nero --channel can0 --check-can
```

## NERO Lab GUI

Launch the local dataset workbench:

```bash
python3 nero_lab.py
```

The GUI can record task-labeled episodes, inspect whether gripper data is present, replay the latest episode, open a selected episode in LeRobot/Rerun, and delete datasets. Recordings are stored under `/home/adrian/Nero/datasets`.

For a no-teleoperation workflow, use the **Teach then replay** controls in the Record and train tab:

1. Click **Teach Task** and move the arm manually in Teach mode. Press `q` in the teach window to save the joint trajectory.
2. Return the arm to **Safe Bicep Reset**.
3. Click **Replay Trained Task (record dataset)**. The saved trajectory is replayed while joint data, gripper actions, wrist video, and overview webcam video are recorded into a new `nero_replayed__*` dataset.

SmolVLA training can be launched for a complete dataset with an 8-value action (7 joint angles plus gripper width). The GUI launches `run_smolvla_nero.py` for guarded hardware execution with an explicit confirmation, action-size check, joint-step limit, and gripper clamp.

The installed LeRobot policy stack must be able to import SmolVLA before training or execution. If import fails in the environment, run `python3 -c "from lerobot.policies.factory import make_policy_config"` to see the dependency error.
