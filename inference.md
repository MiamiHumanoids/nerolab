# NERO Policy Inference

Run policy inference from Windows so the NERO arm, GS-USB CAN adapter, RealSense
D405, and overview camera use their native drivers.

## Before starting

1. Activate the project virtual environment.
2. Confirm CUDA detects the GPU:

   ```powershell
   python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
   ```

3. Use NERO Lab to verify arm feedback and both camera views.
4. Put the arm, block, and bin in a setup represented by the training data.
5. Disconnect the arm and close NERO Lab so the CLI process can acquire the CAN
   adapter.
6. Keep the physical emergency stop within reach and the workspace clear.

Pass the checkpoint directory containing the policy configuration, processor
files, normalization data, and model weights. Do not pass an individual
`.safetensors` file. Select the same dataset used to fine-tune the checkpoint;
the runner uses its feature metadata and normalization statistics.

## GR00T

Install the GR00T policy dependencies once:

```powershell
pip install "lerobot[groot]==0.6.1"
```

Run live GR00T inference:

```powershell
python run_smolvla_nero.py `
  --policy-type groot `
  --checkpoint "C:\path\to\groot-checkpoint\pretrained_model" `
  --dataset-root "C:\path\to\nero_replayed__cube-in-bin__30fps" `
  --task "Pick up the red block and place it in the bin" `
  --device cuda `
  --max-gripper-force 1 `
  --confirm
```

The current LeRobot integration expects a GR00T N1.7 checkpoint.

## SmolVLA

Run live SmolVLA inference:

```powershell
python run_smolvla_nero.py `
  --policy-type smolvla `
  --checkpoint "C:\path\to\smolvla-checkpoint\pretrained_model" `
  --dataset-root "C:\path\to\nero_replayed__cube-in-bin__30fps" `
  --task "Pick up the red block and place it in the bin" `
  --device cuda `
  --max-gripper-force 1 `
  --confirm
```

`smolvla` is the default policy type, but specifying it explicitly prevents an
architecture mismatch when switching checkpoints.

## Why use the CLI?

NERO Lab currently launches `run_smolvla_nero.py` without a policy-type
argument, so it defaults to SmolVLA. The GUI therefore cannot safely identify
and load a GR00T checkpoint. The CLI makes the architecture explicit with
`--policy-type groot` or `--policy-type smolvla`.

The inference subprocess also owns the CAN adapter while it runs. Closing NERO
Lab avoids two processes competing for the same hardware, and the foreground
terminal provides model-loading errors and every predicted joint/gripper action.
Use `Ctrl+C` in that terminal to stop the process. The runner does not currently
open a policy camera window, so pressing `q` is not a dependable stop method.

## Expected output

After model-loading messages, the runner repeatedly prints actions similar to:

```text
Task: Pick up the red block and place it in the bin
joints=[0.012, -1.642, 0.038, 2.031, -0.021, 0.081, 1.487] gripper_width_m=0.0980 max_gripper_force=1.00N
```

The terminal does not display camera images or object-detection confidence.
Verify camera framing in NERO Lab before closing it.

## Safety

These commands send actions to the physical robot because they include
`--confirm`. Do not add `--fast-motion`; that uses instantaneous `move_js`
commands and can cause mechanical shock. The current runner validates the 8D
action shape and clamps gripper width, but it does not impose a per-command joint
step limit. Stop immediately with the physical emergency stop if motion is
unexpected.
