#!/usr/bin/env python3
"""Local NERO dataset and policy workbench."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
import torch
import traceback
from dataclasses import dataclass
from pathlib import Path
import re
from tkinter import filedialog, messagebox, ttk

from lerobot_robot_nero import Nero, NeroConfig
from task_trajectory import SAFE_BICEP_JOINTS, is_safe_bicep_pose, prepare_replay_samples

APP_BUILD = "2026-09-06-supported-teach-exit-38"
DATASET_BASE = Path.home() / "Nero" / "datasets"
TASK_BASE = Path.home() / "Nero" / "tasks"
CONTROL_PRIME_POSE = [-0.4, 0.0, 0.4, -1.57, 0.0, -3.14]
UPRIGHT_RESET_JOINTS = [0.0] * 7
SAFE_BICEP_RESET_JOINTS = SAFE_BICEP_JOINTS.copy()
RESET_SPEED_PERCENT = 25
SLIDER_DEBOUNCE_MS = 100
RESET_STREAM_INTERVAL_S = 0.02
RESET_STREAM_SPEED_RAD_S = 0.4
RESET_LIMIT_MARGIN = 0.005
COMMAND_JOINT_LIMITS = [
    (-2.705261, 2.705261),
    (-1.74533, 1.74533),
    (-2.757621, 2.757621),
    (-1.012291, 2.146755),
    (-2.757621, 2.757621),
    (-0.733039, 0.959932),
    (-1.570797, 1.570797),
]
PROJECT_ROOT = Path(__file__).resolve().parent
RECORDER = PROJECT_ROOT / "manual_record_dataset.py"
REPLAYER = PROJECT_ROOT / "replay_latest_dataset.py"
TASK_TEACHER = PROJECT_ROOT / "teach_task.py"
TASK_REPLAY_RECORDER = PROJECT_ROOT / "replay_record_task.py"
TASK_REPLAYER = PROJECT_ROOT / "replay_task.py"
POLICY_RUNNER = PROJECT_ROOT / "run_smolvla_nero.py"
EPISODE_DELETER = PROJECT_ROOT / "delete_dataset_episode.py"
RERUN = Path.home() / ".local" / "bin" / "lerobot-dataset-viz"
LE_ROBOT_TRAIN = Path.home() / ".local" / "bin" / "lerobot-train"
LE_ROBOT_EVAL = Path.home() / ".local" / "bin" / "lerobot-eval"


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class DatasetInfo:
    root: Path
    task: str
    episodes: int
    frames: int
    action_dim: int | None
    gripper: bool
    complete: bool


def read_dataset_info(root: Path) -> DatasetInfo:
    info_path = root / "meta" / "info.json"
    tasks_path = root / "meta" / "tasks.parquet"
    complete = info_path.exists() and tasks_path.exists()
    task = root.name
    for prefix in ("nero_manual__", "nero_replayed__"):
        if task.startswith(prefix):
            task = task.removeprefix(prefix)
            break
    task = task.replace("-", " ")
    episodes = frames = 0
    action_dim = None
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text())
            episodes = int(info.get("total_episodes", 0))
            frames = int(info.get("total_frames", 0))
            action = info.get("features", {}).get("action", {})
            shape = action.get("shape")
            action_dim = int(shape[0]) if shape else None
            gripper = action_dim == 8
        except (OSError, ValueError, TypeError, KeyError):
            gripper = False
    else:
        gripper = False
    return DatasetInfo(root, task, episodes, frames, action_dim, gripper, complete)


class NeroLab(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("NERO Lab")
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        window_width = int(screen_width * 0.95)
        window_height = int(screen_height * 0.95)
        window_x = (screen_width - window_width) // 2
        window_y = (screen_height - window_height) // 2
        self.geometry(
            f"{window_width}x{window_height}+{window_x}+{window_y}"
        )
        self.minsize(1300, 850)
        self.process: subprocess.Popen[str] | None = None
        self.robot: Nero | None = None
        self.activity_trace: list[str] = []
        self.safe_bicep_position_reached = False
        self.slider_motion_job: str | None = None
        self.gripper_motion_job: str | None = None
        self.pending_slider_joint: int | None = None
        self.pending_slider_value: float | None = None
        self.suppress_slider_motion = False
        self.joint_vars = [tk.DoubleVar(value=0.0) for _ in range(7)]
        self.gripper_var = tk.DoubleVar(value=0.1)
        self.joint_enable_vars = [tk.BooleanVar(value=False) for _ in range(7)]
        self.joint_disable_vars = [tk.BooleanVar(value=True) for _ in range(7)]
        self.speed_var = tk.DoubleVar(value=100.0)
        self.datasets: list[DatasetInfo] = []
        self.selected_root: Path | None = None
        self.selected_episode = 0
        self.taught_task_file: Path | None = None
        self.taught_task_files: list[Path] = []
        self.task_var = tk.StringVar(value="pick up the banana")
        self.policy_var = tk.StringVar(value="")
        self.steps_var = tk.StringVar(value="5000")
        self.training_estimate_var = tk.StringVar(value="Recommended: 5,000 steps | Estimate: about 7 hours")
        self.batch_size_var = tk.StringVar(value="8")
        self.workers_var = tk.StringVar(value="2")
        self.compile_model_var = tk.BooleanVar(value=False)
        self.fast_motion_var = tk.BooleanVar(value=False)
        self.device_var = tk.StringVar(value="auto")
        self.arm_status_var = tk.StringVar(value="Arm status: not connected")
        self.arm_status_label: ttk.Label | None = None
        self.status_var = tk.StringVar(value="Ready")
        self.protocol("WM_DELETE_WINDOW", self.close_application)
        self._build_ui()
        self.log_message(f"NERO Lab build {APP_BUILD} running from {Path(__file__).resolve()}")
        self.refresh_datasets()
        self.refresh_tasks()

    def close_application(self) -> None:
        if self.robot is not None and self.robot.is_connected:
            self.disconnect_robot()
            if self.robot is not None:
                return
        self.destroy()

    def report_callback_exception(self, exc_type, exc_value, exc_traceback) -> None:
        if self.robot is not None and self.robot.is_connected:
            try:
                self.robot.disconnect()
                self.robot = None
            except Exception:
                pass
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        messagebox.showerror("Application error", str(exc_value))

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        style.configure("Title.TLabel", font=("DejaVu Sans", 20, "bold"))
        style.configure("Section.TLabel", font=("DejaVu Sans", 12, "bold"))

        header = ttk.Frame(self, padding=(22, 18, 22, 8))
        header.pack(fill="x")
        ttk.Label(header, text="NERO Lab", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="Dataset, review, and policy workspace", padding=(16, 7)).pack(side="left")
        ttk.Button(header, text="Refresh", command=self.refresh_datasets).pack(side="right")

        tabs = ttk.Notebook(self)
        tabs.pack(fill="both", expand=True, padx=18, pady=8)
        arm_tab = ttk.Frame(tabs, padding=16)
        record_tab = ttk.Frame(tabs, padding=12)
        inference_tab = ttk.Frame(tabs, padding=16)
        tabs.add(arm_tab, text="Arm control")
        tabs.add(record_tab, text="Record and train")
        tabs.add(inference_tab, text="Fine Tune and Inference")

        self._build_arm_tab(arm_tab)

        body = ttk.PanedWindow(record_tab, orient="horizontal")
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body, padding=12)
        right = ttk.Frame(body, padding=16)
        body.add(left, weight=1)
        body.add(right, weight=2)

        ttk.Label(left, text="Taught tasks", style="Section.TLabel").pack(anchor="w", pady=(12, 0))
        self.task_list = tk.Listbox(left, exportselection=False, height=6)
        self.task_list.pack(fill="x", pady=(6, 8))
        self.task_list.bind("<<ListboxSelect>>", self._select_taught_task)
        task_buttons = ttk.Frame(left)
        task_buttons.pack(fill="x")
        ttk.Button(task_buttons, text="Delete task", command=self.delete_taught_task).pack(side="left", expand=True, fill="x", padx=(0, 4))
        ttk.Button(task_buttons, text="Replay task", command=self.replay_task).pack(side="left", expand=True, fill="x", padx=(4, 0))

        ttk.Label(left, text="Saved datasets", style="Section.TLabel").pack(anchor="w", pady=(12, 0))
        self.dataset_list = tk.Listbox(left, exportselection=False, height=18)
        self.dataset_list.pack(fill="both", expand=True, pady=(8, 8))
        self.dataset_list.bind("<<ListboxSelect>>", self._select_dataset)
        ttk.Label(left, text="Episodes", style="Section.TLabel").pack(anchor="w", pady=(12, 0))
        self.episode_list = tk.Listbox(left, exportselection=False, height=6)
        self.episode_list.pack(fill="x", pady=(6, 8))
        self.episode_list.bind("<<ListboxSelect>>", self._select_episode)
        ttk.Button(left, text="Delete selected episode", command=self.delete_episode).pack(fill="x", pady=(0, 6))
        ttk.Button(left, text="Delete selected dataset", command=self.delete_dataset).pack(fill="x")

        self.details = tk.Text(left, height=10, wrap="word", state="disabled", relief="flat", background="#f3f3f3")
        self.details.pack(fill="x", pady=(12, 0))

        record = ttk.LabelFrame(right, text="Task Description", padding=12)
        record.pack(fill="x")
        ttk.Label(record, text="Task instruction").grid(row=0, column=0, sticky="w")
        ttk.Entry(record, textvariable=self.task_var).grid(row=1, column=0, sticky="ew", padx=(0, 8), pady=(5, 0))
        record.columnconfigure(0, weight=1)

        task_flow = ttk.LabelFrame(right, text="Teach then replay (no teleop device)", padding=12)
        task_flow.pack(fill="x", pady=(14, 0))
        ttk.Button(task_flow, text="Teach Task", command=self.teach_task).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(task_flow, text="Record Dataset with Selected Trained Task", command=self.replay_trained_task).grid(row=0, column=1)
        ttk.Label(task_flow, text="Teach manually, reset to Safe Bicep, then replay to capture training data.", foreground="#555555").grid(row=1, column=0, columnspan=3, sticky="w", pady=(8, 0))

        review = ttk.LabelFrame(right, text="Review", padding=12)
        review.pack(fill="x", pady=(14, 0))
        ttk.Button(review, text="Replay in NERO review window", command=self.replay_dataset).pack(side="left")
        ttk.Button(review, text="Open in LeRobot / Rerun", command=self.open_rerun).pack(side="left", padx=8)

        log_frame = ttk.LabelFrame(right, text="Activity", padding=8)
        log_frame.pack(fill="both", expand=True, pady=(14, 0))
        self.log = tk.Text(log_frame, wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True)
        footer = ttk.Frame(self, relief="sunken", padding=5)
        footer.pack(fill="x", side="bottom")
        ttk.Label(footer, textvariable=self.status_var, anchor="w").pack(side="left", fill="x", expand=True)
        ttk.Button(
            footer,
            text="Copy Activity Trace to Clipboard",
            command=self.copy_activity_log,
        ).pack(side="right", padx=(8, 0))

        self._build_inference_tab(inference_tab)

    def _build_arm_tab(self, parent: ttk.Frame) -> None:
        connection = ttk.Frame(parent)
        connection.pack(fill="x")
        ttk.Button(connection, text="Connect arm", command=self.connect_robot).pack(side="left")
        ttk.Button(connection, text="Disconnect", command=self.disconnect_robot).pack(side="left", padx=8)
        ttk.Button(connection, text="Read joint angles", command=self.read_joint_angles).pack(side="left")
        ttk.Button(connection, text="Check arm status", command=self.check_arm_status).pack(side="left", padx=8)
        ttk.Button(connection, text="Re-enable Arm", command=self.reenable_arm).pack(side="left", padx=8)
        ttk.Button(connection, text="Upright Reset", command=self.upright_reset).pack(side="right")
        self.arm_status_label = ttk.Label(parent, textvariable=self.arm_status_var, foreground="#555555")
        self.arm_status_label.pack(anchor="w", pady=(8, 0))
        shutdown = ttk.LabelFrame(parent, text="Safe Shutdown Sequence", padding=10)
        shutdown.pack(fill="x", pady=(10, 0))
        ttk.Button(shutdown, text="Safe Bicep Reset", command=self.safe_bicep_reset).pack(side="left")
        ttk.Button(shutdown, text="Emergency Brake", command=self.emergency_brake).pack(side="left", padx=8)
        speed = ttk.LabelFrame(parent, text="Arm speed (%)", padding=12)
        speed.pack(fill="x", pady=(10, 0))
        tk.Scale(speed, from_=0, to=100, resolution=1, orient="horizontal", variable=self.speed_var, showvalue=True, length=700, command=lambda _value: self.set_arm_speed()).pack(fill="x")

        joints = ttk.LabelFrame(parent, text="Joint angles (radians)", padding=12)
        joints.pack(fill="x", pady=(16, 10))
        for index, (variable, enable_var, disable_var, (actual_lower, actual_upper)) in enumerate(zip(self.joint_vars, self.joint_enable_vars, self.joint_disable_vars, COMMAND_JOINT_LIMITS), 1):
            slider_limit = max(abs(actual_lower), abs(actual_upper))
            row = ttk.Frame(joints)
            row.pack(fill="x", pady=3)
            ttk.Label(row, text=f"Joint {index}", width=10).pack(side="left")
            scale = tk.Scale(row, from_=-slider_limit, to=slider_limit, resolution=0.001, orient="horizontal", variable=variable, showvalue=True, length=560, command=lambda _value, joint=index: self.slider_motion(joint))
            scale.pack(side="left", fill="x", expand=True)
            ttk.Checkbutton(row, text="Enable", variable=enable_var, command=lambda joint=index: self.toggle_joint(joint, True)).pack(side="right", padx=(8, 0))
            ttk.Checkbutton(row, text="Disable", variable=disable_var, command=lambda joint=index: self.toggle_joint(joint, False)).pack(side="right")

        gripper = ttk.LabelFrame(parent, text="Gripper width (meters)", padding=12)
        gripper.pack(fill="x", pady=10)
        tk.Scale(gripper, from_=0.0, to=0.1, resolution=0.001, orient="horizontal", variable=self.gripper_var, showvalue=True, length=700, command=lambda _value: self.slider_gripper()).pack(fill="x")
        ttk.Label(parent, text=f"Upright Reset sends joints {UPRIGHT_RESET_JOINTS} and opens the gripper to 0.1 m.", foreground="#555555").pack(anchor="w", pady=12)

    def _build_policy_options(self, parent: ttk.Frame) -> ttk.LabelFrame:
        policy = ttk.LabelFrame(parent, text="Fine Tune", padding=12)
        policy.pack(fill="x", pady=(14, 0))
        ttk.Label(policy, text="Device").grid(row=0, column=0, sticky="w")
        ttk.Combobox(policy, textvariable=self.device_var, values=("auto", "cuda", "xpu", "mps", "cpu"), state="readonly", width=12).grid(row=0, column=1, sticky="e")
        ttk.Button(policy, text="Train SmolVLA", command=self.train_policy).grid(row=0, column=2, padx=(12, 0))
        ttk.Label(policy, text="Training steps").grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(policy, textvariable=self.steps_var, width=12).grid(row=1, column=1, sticky="e", pady=(10, 0))
        self.steps_var.trace_add("write", self._update_training_estimate)
        ttk.Label(policy, textvariable=self.training_estimate_var, wraplength=650, foreground="#555555").grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Label(policy, text="Batch size").grid(row=3, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(policy, textvariable=self.batch_size_var, width=12).grid(row=3, column=1, sticky="e", pady=(10, 0))
        ttk.Label(policy, text="Data-loader workers").grid(row=4, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(policy, textvariable=self.workers_var, width=12).grid(row=4, column=1, sticky="e", pady=(10, 0))
        ttk.Checkbutton(policy, text="Compile model", variable=self.compile_model_var).grid(row=5, column=0, columnspan=2, sticky="w", pady=(10, 0))
        policy.columnconfigure(0, weight=1)
        return policy

    def _build_inference_tab(self, parent: ttk.Frame) -> None:
        self._build_policy_options(parent)
        inference = ttk.LabelFrame(parent, text="Inference", padding=12)
        inference.pack(fill="both", expand=True, pady=(14, 0))
        ttk.Label(inference, text="Task prompt", style="Section.TLabel").pack(anchor="w")
        ttk.Entry(inference, textvariable=self.task_var).pack(fill="x", pady=(6, 14))
        checkpoint = ttk.LabelFrame(inference, text="Trained checkpoint", padding=12)
        checkpoint.pack(fill="x")
        ttk.Entry(checkpoint, textvariable=self.policy_var).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(checkpoint, text="Browse", command=self.browse_policy).pack(side="right")
        actions = ttk.Frame(inference)
        actions.pack(fill="x", pady=18)
        ttk.Button(actions, text="Run policy on NERO", command=self.run_policy).pack(side="left")
        ttk.Button(actions, text="Cancel inference", command=self.cancel_inference).pack(side="left", padx=10)
        ttk.Checkbutton(inference, text="Fast joint motion (move_js; may cause shock)", variable=self.fast_motion_var).pack(anchor="w", pady=(10, 0))
        ttk.Label(inference, text="Prompt examples: pick up the banana; move the owl to the green notebook.", foreground="#555555").pack(anchor="w")

    def connect_robot(self) -> None:
        if self.robot is not None and self.robot.is_connected:
            self.log_message("Arm is already connected")
            return
        self.robot = Nero(NeroConfig(
            id="nero_lab",
            can_channel="can0",
            bitrate=1_000_000,
            firmware_version="v121",
            speed_percent=100,
            has_gripper=True,
            has_camera=False,
            reset_on_connect=False,
        ))
        self.robot.connect(calibrate=False)
        self.robot._arm.enable()
        self.robot._arm.set_speed_percent(100)
        current_joints = self.robot.get_joint_angles()
        self.set_joint_slider_values(current_joints)
        for enable_var, disable_var in zip(self.joint_enable_vars, self.joint_disable_vars):
            enable_var.set(True)
            disable_var.set(False)
        self.safe_bicep_position_reached = self.is_safe_bicep_position(current_joints)
        status = self.robot.get_arm_status()
        status_text = str(getattr(status, "msg", status))
        if "EMERGENCY_STOP" in status_text or "EMERGENCY STOP" in status_text:
            self.set_arm_status_display(
                "Arm status: EMERGENCY STOP | Click Re-enable Arm to release motor brakes",
                color="#008000",
            )
        else:
            self.set_arm_status_display("Arm status: connected")
        self.log_message("Arm connected")
        self.log_arm_debug("connect complete")

    def disconnect_robot(self, emergency_brake: bool = True, disable_arm: bool = True) -> None:
        self.cancel_slider_motion()
        if self.robot is not None:
            try:
                current = self.robot.get_joint_angles()
                safe_position = self.safe_bicep_position_reached or self.is_safe_bicep_position(current)
            except Exception as exc:
                safe_position = self.safe_bicep_position_reached
                self.log_message(f"Could not verify safe position: {exc}")
            if not safe_position and not self.confirm_dangerous_disconnect():
                self.log_message("Disconnect cancelled; arm is not in Safe Bicep Position")
                return
            if emergency_brake:
                try:
                    self.robot.engage_brakes()
                    self.log_message("Emergency-stop resting pose settled before disconnect")
                except Exception as exc:
                    self.log_message(f"Emergency brake before disconnect failed: {exc}")
                    return
            self.robot.disconnect(disable_arm=disable_arm and not emergency_brake)
            self.robot = None
        self.arm_status_var.set("Arm status: not connected")
        self.log_message("Arm disconnected")

    def confirm_dangerous_disconnect(self) -> bool:
        dialog = tk.Toplevel(self)
        dialog.title("Dangerous disconnect")
        dialog.transient(self)
        dialog.grab_set()
        dialog.resizable(False, False)
        ttk.Label(
            dialog,
            text="DANGER: Arm is not in a safe position, it may fall down violently.\nCheck safe shutdown sequence. Move robot to Safe Reset position then click Emergency Brake to lock motor.",
            justify="center",
            padding=18,
        ).pack()
        result = {"proceed": False}
        buttons = ttk.Frame(dialog, padding=(12, 0, 12, 12))
        buttons.pack(fill="x")

        def proceed() -> None:
            result["proceed"] = True
            dialog.destroy()

        ttk.Button(buttons, text="Proceed (DANGER)", command=proceed).pack(side="left", expand=True, fill="x", padx=(0, 6))
        ttk.Button(buttons, text="OK", command=dialog.destroy).pack(side="left", expand=True, fill="x", padx=(6, 0))
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        self.wait_window(dialog)
        return result["proceed"]

    def is_safe_bicep_position(self, values: list[float], tolerance: float = 0.1) -> bool:
        return is_safe_bicep_pose(values, target_tolerance=tolerance)

    def require_robot(self) -> Nero | None:
        if self.robot is None or not self.robot.is_connected:
            messagebox.showwarning("Arm not connected", "Connect the arm first.")
            return None
        return self.robot

    def set_arm_status_display(self, text: str, emergency: bool = False, color: str | None = None) -> None:
        self.arm_status_var.set(text)
        if self.arm_status_label is not None:
            self.arm_status_label.configure(foreground=color or ("#cc0000" if emergency else "#555555"))

    def set_arm_speed(self) -> None:
        robot = self.require_robot()
        if robot is not None:
            speed = int(self.speed_var.get())
            result = robot._arm.set_speed_percent(speed)
            self.log_message(f"COMMAND set_speed_percent({speed}) returned {result!r}")

    def arm_debug_text(self, robot: Nero) -> str:
        status = robot.get_arm_status()
        message = getattr(status, "msg", status)
        fields = " ".join(
            f"{name}={getattr(message, name, '?')}"
            for name in ("ctrl_mode", "arm_status", "mode_feedback", "teach_status", "motion_status", "trajectory_num")
        )
        try:
            joints = [round(float(value), 6) for value in robot.get_joint_angles()]
        except Exception as exc:
            joints = f"ERROR: {exc}"
        try:
            enabled = robot._arm.get_joints_enable_status_list()
        except Exception as exc:
            enabled = f"ERROR: {exc}"
        return f"{fields} enabled={enabled} joints={joints}"

    def log_arm_debug(self, label: str) -> None:
        robot = self.robot
        if robot is None or not robot.is_connected:
            self.log_message(f"DEBUG {label}: arm not connected")
            return
        try:
            self.log_message(f"DEBUG {label}: {self.arm_debug_text(robot)}")
        except Exception as exc:
            self.log_message(f"DEBUG {label}: status read failed: {exc}")

    def set_motion_mode_and_wait(self, robot: Nero, mode, expected: str, label: str, timeout: float = 2.0):
        result = robot._arm.set_motion_mode(mode)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = robot.get_arm_status()
            message = getattr(status, "msg", status)
            if expected in str(getattr(message, "mode_feedback", "")):
                self.log_message(f"COMMAND {label} mode confirmed as {expected}; set_motion_mode returned {result!r}")
                return result
            time.sleep(0.02)
        raise RuntimeError(f"{label} did not enter {expected}: {self.arm_debug_text(robot)}")

    def joint_motion_block_reason(self, robot: Nero) -> str | None:
        status = robot.get_arm_status()
        message = getattr(status, "msg", status)
        arm_status = str(getattr(message, "arm_status", ""))
        if "EMERGENCY_STOP" in arm_status:
            return "emergency stop is active; click Re-enable Arm"
        if "BRAKE_NOT_RELEASED" in arm_status:
            return "motor brake is not released; click Re-enable Arm"
        enabled = robot._arm.get_joints_enable_status_list()
        if not all(enabled):
            return f"one or more joints are disabled: {enabled}"
        return None

    def joints_outside_command_limits(self, robot: Nero) -> bool:
        current = robot.get_joint_angles()
        return any(
            float(value) < lower or float(value) > upper
            for value, (lower, upper) in zip(current, COMMAND_JOINT_LIMITS)
        )

    def prepare_reset_motion(self, robot: Nero, label: str) -> None:
        block_reason = self.joint_motion_block_reason(robot)
        if block_reason is not None:
            raise RuntimeError(f"{label} cannot start because {block_reason}")
        status = robot.get_arm_status()
        message = getattr(status, "msg", status)
        arm_status = str(getattr(message, "arm_status", ""))
        outside_limits = self.joints_outside_command_limits(robot)
        if "NO_SOLUTION" in arm_status or "SINGULARITY" in arm_status or outside_limits:
            reason = "current joints outside command limits" if outside_limits else arm_status
            self.log_message(f"DEBUG {label} running P-to-J recovery because {reason}")
            self.set_motion_mode_and_wait(
                robot, robot._arm.OPTIONS.MOTION_MODE.P, "MOVE_P", f"{label} P recovery"
            )
            recovery_nudges = ((1, 0.12), (3, 0.12))
            next_nudge = 0
            for attempt in range(1, 5):
                start = [float(value) for value in robot.get_joint_angles()]
                move_result = robot._arm.move_p(CONTROL_PRIME_POSE)
                self.log_message(
                    f"COMMAND {label} P recovery {attempt}/4 move_p={move_result!r} "
                    f"target={CONTROL_PRIME_POSE}"
                )
                moved = False
                saw_in_progress = False
                did_not_start = False
                started_at = time.monotonic()
                deadline = started_at + 10.0
                while time.monotonic() < deadline:
                    current = [float(value) for value in robot.get_joint_angles()]
                    moved = moved or any(
                        abs(value - initial) > 0.002 for value, initial in zip(current, start)
                    )
                    status = robot.get_arm_status()
                    message = getattr(status, "msg", status)
                    arm_status = str(getattr(message, "arm_status", ""))
                    motion_status = str(getattr(message, "motion_status", ""))
                    saw_in_progress = saw_in_progress or "FAILED" in motion_status
                    if (
                        moved
                        and saw_in_progress
                        and "NORMAL" in arm_status
                        and "SUCCESSFULLY" in motion_status
                    ):
                        self.log_arm_debug(f"{label} P recovery completed")
                        break
                    if not moved and time.monotonic() - started_at >= 1.5:
                        self.log_message(
                            f"DEBUG {label} P recovery {attempt}/4 did not start within 1.5s"
                        )
                        did_not_start = True
                        break
                    time.sleep(0.05)
                else:
                    progress = "partial encoder movement" if moved else "no encoder movement"
                    self.log_message(
                        f"DEBUG {label} P recovery {attempt}/4 ended with {progress}; retrying"
                    )
                    continue
                if moved and saw_in_progress and "NORMAL" in arm_status and "SUCCESSFULLY" in motion_status:
                    break
                if did_not_start and not outside_limits and next_nudge < len(recovery_nudges):
                    joint_index, offset = recovery_nudges[next_nudge]
                    next_nudge += 1
                    self.set_motion_mode_and_wait(
                        robot,
                        robot._arm.OPTIONS.MOTION_MODE.J,
                        "MOVE_J",
                        f"{label} recovery nudge",
                    )
                    nudge_target = [float(value) for value in robot.get_joint_angles()]
                    lower, upper = COMMAND_JOINT_LIMITS[joint_index]
                    nudge_target[joint_index] = min(
                        max(nudge_target[joint_index] + offset, lower + RESET_LIMIT_MARGIN),
                        upper - RESET_LIMIT_MARGIN,
                    )
                    nudge_result = robot._arm.move_j(nudge_target)
                    self.log_message(
                        f"COMMAND {label} recovery nudge joint={joint_index + 1} "
                        f"move_j={nudge_result!r} target={nudge_target[joint_index]:.3f}"
                    )
                    self.wait_for_joint_target(
                        robot,
                        nudge_target,
                        f"{label} recovery nudge joint {joint_index + 1}",
                        timeout=4.0,
                        tolerance=0.01,
                    )
                    self.set_motion_mode_and_wait(
                        robot,
                        robot._arm.OPTIONS.MOTION_MODE.P,
                        "MOVE_P",
                        f"{label} resume P recovery",
                    )
            else:
                raise RuntimeError(
                    f"{label} clean-connect P recovery did not complete: {self.arm_debug_text(robot)}"
                )
        self.set_motion_mode_and_wait(
            robot, robot._arm.OPTIONS.MOTION_MODE.J, "MOVE_J", f"{label} joint control"
        )

    def move_joint_path(self, robot: Nero, target: list[float], label: str) -> None:
        start = [float(value) for value in robot.get_joint_angles()]
        largest_delta = max(abs(goal - value) for value, goal in zip(start, target))
        duration = max(0.75, largest_delta / RESET_STREAM_SPEED_RAD_S)
        step_count = max(1, int(duration / RESET_STREAM_INTERVAL_S) + 1)
        motion_command = getattr(robot._arm, "move_js", None)
        if motion_command is None:
            raise RuntimeError("Installed pyAgxArm does not provide move_js for smooth reset motion")
        self.log_message(
            f"COMMAND {label} smooth joint stream steps={step_count} duration={duration:.2f}s "
            f"start={start} target={target}"
        )
        stream_started = time.monotonic()
        for step_index in range(1, step_count + 1):
            progress = step_index / step_count
            fraction = progress * progress * (3.0 - 2.0 * progress)
            raw_waypoint = [
                value + (goal - value) * fraction
                for value, goal in zip(start, target)
            ]
            waypoint = [
                min(max(value, lower + RESET_LIMIT_MARGIN), upper - RESET_LIMIT_MARGIN)
                for value, (lower, upper) in zip(raw_waypoint, COMMAND_JOINT_LIMITS)
            ]
            motion_command(waypoint)
            next_step_at = stream_started + step_index * duration / step_count
            remaining = next_step_at - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
        self.wait_for_joint_target(robot, target, label)

    def emergency_brake(self) -> None:
        robot = self.require_robot()
        if robot is None:
            return
        try:
            robot.engage_brakes()
            self.set_arm_status_display("Arm status: EMERGENCY STOP | Click Re-enable Arm to release motor brakes", color="#008000")
            self.log_message("Emergency brake activated; resting pose settled")
        except Exception as exc:
            self.set_arm_status_display(f"Emergency brake error: {exc}")
            self.log_message(f"Emergency brake failed: {exc}")

    def reenable_arm(self) -> None:
        robot = self.require_robot()
        if robot is None:
            return
        self.safe_bicep_position_reached = False
        try:
            self.log_arm_debug("Re-enable before recovery")
            robot.set_teach_mode(False)
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                status = robot.get_arm_status()
                message = getattr(status, "msg", status)
                arm_status = str(getattr(message, "arm_status", ""))
                controller_ready = any(
                    state in arm_status for state in ("NORMAL", "NO_SOLUTION", "SINGULARITY")
                )
                if controller_ready:
                    break
                time.sleep(0.05)
            else:
                raise RuntimeError(
                    f"Arm brakes did not release after re-enable: {self.arm_debug_text(robot)}"
                )
            speed_result = robot._arm.set_speed_percent(100)
            self.speed_var.set(100.0)
            self.set_arm_status_display("Arm status: re-enabled")
            self.log_message(f"COMMAND set_speed_percent(100) returned {speed_result!r}")
            self.log_arm_debug("Re-enable after recovery")
        except Exception as exc:
            self.set_arm_status_display(f"Re-enable error: {exc}")
            self.log_message(f"Arm re-enable failed: {exc}")

    def slider_motion(self, joint: int) -> None:
        if self.suppress_slider_motion:
            return
        robot = self.require_robot()
        if robot is None or not self.joint_enable_vars[joint - 1].get():
            return
        self.pending_slider_joint = joint
        self.pending_slider_value = float(self.joint_vars[joint - 1].get())
        if self.slider_motion_job is not None:
            self.after_cancel(self.slider_motion_job)
        self.slider_motion_job = self.after(SLIDER_DEBOUNCE_MS, self._send_slider_motion)

    def _send_slider_motion(self) -> None:
        self.slider_motion_job = None
        robot = self.require_robot()
        if robot is None:
            return
        try:
            joint = self.pending_slider_joint
            desired = self.pending_slider_value
            if joint is None or desired is None:
                return
            self.pending_slider_joint = None
            self.pending_slider_value = None
            self.safe_bicep_position_reached = False
            self.log_arm_debug(f"slider joint={joint} requested={desired}")
            block_reason = self.joint_motion_block_reason(robot)
            if block_reason is not None or self.joints_outside_command_limits(robot):
                self.set_joint_slider_values(robot.get_joint_angles())
                reason = block_reason or "current joints are outside command limits"
                self.log_message(f"COMMAND slider blocked: {reason}")
                return
            targets = [float(value) for value in robot.get_joint_angles()]
            targets[joint - 1] = desired
            self.set_joint_slider_values(targets)
            mode_result = self.set_motion_mode_and_wait(
                robot, robot._arm.OPTIONS.MOTION_MODE.J, "MOVE_J", "slider"
            )
            move_result = robot._arm.move_j(targets)
            self.log_message(
                f"COMMAND slider joint={joint} requested={desired} mode_j={mode_result!r} "
                f"move_j={move_result!r} full_target={targets}"
            )
            self.after(300, lambda: self.log_arm_debug("slider 300ms after move_j"))
        except Exception as exc:
            self.log_message(f"COMMAND slider failed: {exc}")
            self.log_arm_debug("slider failure")

    def cancel_slider_motion(self) -> None:
        if self.slider_motion_job is not None:
            self.after_cancel(self.slider_motion_job)
            self.slider_motion_job = None
        if self.gripper_motion_job is not None:
            self.after_cancel(self.gripper_motion_job)
            self.gripper_motion_job = None
        self.pending_slider_joint = None
        self.pending_slider_value = None

    def set_joint_slider_values(self, values: list[float]) -> None:
        self.suppress_slider_motion = True
        try:
            for variable, value in zip(self.joint_vars, values):
                variable.set(value)
        finally:
            self.suppress_slider_motion = False

    def slider_gripper(self) -> None:
        if self.gripper_motion_job is not None:
            self.after_cancel(self.gripper_motion_job)
        self.gripper_motion_job = self.after(SLIDER_DEBOUNCE_MS, self._send_gripper_motion)

    def _send_gripper_motion(self) -> None:
        self.gripper_motion_job = None
        robot = self.require_robot()
        if robot is not None:
            width = self.gripper_var.get()
            try:
                effector = robot._get_gripper_effector()
                result = effector.move_gripper_m(value=width, force=30.0)
                self.log_message(f"COMMAND gripper width={width:.3f} returned {result!r}")
            except Exception as exc:
                self.log_message(f"COMMAND gripper width={width:.3f} failed: {exc}")

    def toggle_joint(self, joint: int, enabled: bool) -> None:
        robot = self.require_robot()
        if robot is None:
            return
        self.joint_enable_vars[joint - 1].set(enabled)
        self.joint_disable_vars[joint - 1].set(not enabled)
        try:
            result = (robot._arm.enable if enabled else robot._arm.disable)(joint)
            self.log_message(f"COMMAND joint {joint} {'enable' if enabled else 'disable'} returned {result!r}")
            self.log_arm_debug(f"joint {joint} {'enable' if enabled else 'disable'} complete")
        except Exception as exc:
            self.log_message(f"COMMAND joint {joint} {'enable' if enabled else 'disable'} failed: {exc}")

    def read_joint_angles(self) -> None:
        robot = self.require_robot()
        if robot is None:
            return
        values = robot.get_joint_angles()
        self.set_joint_slider_values(values)
        self.log_message(f"Joint angles read: {values}")

    def check_arm_status(self) -> None:
        robot = self.require_robot()
        if robot is None:
            return
        try:
            status = robot.get_arm_status()
            status_text = str(getattr(status, "msg", status))
            self.arm_status_var.set(f"Arm status: {status_text}")
            self.log_message(f"Arm status: {status_text}")
        except Exception as exc:
            self.arm_status_var.set(f"Arm status error: {exc}")
            self.log_message(f"Arm status error: {exc}")

    def upright_reset(self) -> None:
        robot = self.require_robot()
        if robot is None:
            return
        self.cancel_slider_motion()
        self.safe_bicep_position_reached = False
        try:
            self.log_arm_debug(f"Upright Reset before recovery target={UPRIGHT_RESET_JOINTS}")
            speed_result = robot._arm.set_speed_percent(RESET_SPEED_PERCENT)
            self.log_message(f"COMMAND Upright Reset speed={speed_result!r}")
            self.prepare_reset_motion(robot, "Upright Reset")
            self.move_joint_path(robot, UPRIGHT_RESET_JOINTS, "Upright Reset")
            self.log_arm_debug("Upright Reset target reached")
            robot._get_gripper_effector().move_gripper_m(value=0.1, force=30.0)
            robot._arm.set_speed_percent(100)
        except Exception as exc:
            try:
                robot._arm.set_speed_percent(100)
            except Exception:
                pass
            self.log_message(f"Upright Reset failed: {exc}")
            return
        self.set_joint_slider_values(UPRIGHT_RESET_JOINTS)
        self.gripper_var.set(0.1)
        self.log_message(f"Upright Reset reached: joints {UPRIGHT_RESET_JOINTS}; gripper 0.1 m")

    def wait_for_joint_target(
        self,
        robot: Nero,
        target: list[float],
        label: str,
        timeout: float = 8.0,
        tolerance: float = 0.002,
    ) -> None:
        start = time.monotonic()
        deadline = time.monotonic() + timeout
        early_snapshot_logged = False
        while time.monotonic() < deadline:
            current = robot.get_joint_angles()
            if all(abs(float(value) - goal) <= tolerance for value, goal in zip(current, target)):
                return
            if not early_snapshot_logged and time.monotonic() - start >= 0.25:
                self.log_arm_debug(f"{label} 250ms after move_j")
                early_snapshot_logged = True
            time.sleep(0.02)
        raise RuntimeError(
            f"{label} did not reach target {target}; current joints={robot.get_joint_angles()}; "
            f"status={robot._arm.get_arm_status()}"
        )

    def safe_bicep_reset(self) -> None:
        robot = self.require_robot()
        if robot is None:
            return
        self.cancel_slider_motion()
        try:
            self.log_arm_debug(f"Safe Bicep before recovery target={SAFE_BICEP_RESET_JOINTS}")
            speed_result = robot._arm.set_speed_percent(RESET_SPEED_PERCENT)
            self.log_message(f"COMMAND Safe Bicep speed={speed_result!r}")
            self.prepare_reset_motion(robot, "Safe Bicep")
            self.move_joint_path(robot, SAFE_BICEP_RESET_JOINTS, "Safe Bicep")
            self.log_arm_debug("Safe Bicep target reached")
            robot._get_gripper_effector().move_gripper_m(value=0.1, force=30.0)
            robot._arm.set_speed_percent(100)
        except Exception as exc:
            try:
                robot._arm.set_speed_percent(100)
            except Exception:
                pass
            self.log_message(f"Safe Bicep Reset failed: {exc}")
            return
        self.set_joint_slider_values(SAFE_BICEP_RESET_JOINTS)
        self.safe_bicep_position_reached = True
        self.gripper_var.set(0.1)
        self.log_message("Safe Bicep Reset sent; gripper 0.1 m")

    def cancel_inference(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.log_message("Inference cancelled")
        else:
            self.log_message("No inference process is running")

    def log_message(self, message: str) -> None:
        entry = message.rstrip() + "\n"
        self.activity_trace.append(entry)
        self.log.configure(state="normal")
        self.log.insert("end", entry)
        self.log.see("end")
        self.log.configure(state="disabled")
        self.status_var.set(message.splitlines()[-1][:140])

    def clear_activity_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def copy_activity_log(self) -> None:
        activity = "".join(self.activity_trace).rstrip("\n")
        self.clipboard_clear()
        self.clipboard_append(activity)
        self.update_idletasks()
        self.status_var.set("Full session activity trace copied to clipboard")

    def refresh_datasets(self) -> None:
        DATASET_BASE.mkdir(parents=True, exist_ok=True)
        dataset_paths = list(DATASET_BASE.glob("nero_manual__*")) + list(DATASET_BASE.glob("nero_replayed__*"))
        self.datasets = [read_dataset_info(path) for path in sorted(set(dataset_paths)) if path.is_dir()]
        self.dataset_list.delete(0, "end")
        for item in self.datasets:
            marker = "OK" if item.complete else "INCOMPLETE"
            self.dataset_list.insert("end", f"[{marker}] {item.root.name}")
        if self.datasets:
            self.dataset_list.selection_set(0)
            self._select_dataset()
        else:
            self.selected_root = None
            self.episode_list.delete(0, "end")
            self.selected_episode = 0
            self._show_details(None)
        self.log_message(f"Found {len(self.datasets)} dataset(s) in {DATASET_BASE}")

    def _select_dataset(self, _event: object = None) -> None:
        selection = self.dataset_list.curselection()
        if not selection:
            return
        item = self.datasets[selection[0]]
        self.selected_root = item.root
        self.episode_list.delete(0, "end")
        for episode_index in range(item.episodes):
            self.episode_list.insert("end", f"Episode {episode_index}")
        if item.episodes:
            self.episode_list.selection_set(0)
            self.selected_episode = 0
        self._show_details(item)
        self._update_training_estimate()

    def refresh_tasks(self) -> None:
        TASK_BASE.mkdir(parents=True, exist_ok=True)
        self.taught_task_files = sorted(
            TASK_BASE.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        self.task_list.delete(0, "end")
        for path in self.taught_task_files:
            try:
                recording = json.loads(path.read_text())
                task = str(recording.get("task", path.stem))
                if self._recording_replay_error(recording):
                    task += " [RECORDING ONLY]"
            except (OSError, ValueError, TypeError):
                task = path.stem
            self.task_list.insert("end", task)
        if self.taught_task_file in self.taught_task_files:
            index = self.taught_task_files.index(self.taught_task_file)
            self.task_list.selection_set(index)
            self.task_list.see(index)

    def _select_taught_task(self, _event: object = None) -> None:
        selection = self.task_list.curselection()
        if not selection:
            return
        self.taught_task_file = self.taught_task_files[selection[0]]
        try:
            task = str(json.loads(self.taught_task_file.read_text()).get("task", self.taught_task_file.stem))
            self.task_var.set(task)
            self.log_message(f"Selected taught task: {task}")
        except (OSError, ValueError, TypeError) as exc:
            self.log_message(f"Could not load taught task: {exc}")

    def delete_taught_task(self) -> None:
        selection = self.task_list.curselection()
        if not selection:
            messagebox.showwarning("No taught task", "Select a taught task first.")
            return
        task_file = self.taught_task_files[selection[0]]
        if not messagebox.askyesno("Delete taught task", f"Delete this taught task permanently?\n\n{task_file}"):
            return
        task_file.unlink(missing_ok=True)
        if self.taught_task_file == task_file:
            self.taught_task_file = None
        self.refresh_tasks()
        self.log_message(f"Deleted taught task: {task_file.stem}")

    def _recommended_steps(self) -> int:
        if self.selected_root is None:
            return 5000
        item = next((entry for entry in self.datasets if entry.root == self.selected_root), None)
        if item is None or item.frames <= 0:
            return 5000
        return max(5000, min(20000, item.frames * 20))

    def _update_training_estimate(self, *_args: object) -> None:
        try:
            steps = max(0, int(self.steps_var.get().replace(",", "").strip()))
        except ValueError:
            self.training_estimate_var.set(f"Recommended: {self._recommended_steps():,} steps | Enter a whole number")
            return
        hours = steps * 5.0 / 3600.0
        estimate = f"about {hours:.1f} hours" if hours < 10 else f"about {hours:.0f} hours"
        self.training_estimate_var.set(
            f"Recommended: {self._recommended_steps():,} steps | Estimate: {estimate} at ~5 sec/step"
        )

    def _select_episode(self, _event: object = None) -> None:
        selection = self.episode_list.curselection()
        if selection:
            self.selected_episode = int(selection[0])

    def _show_details(self, item: DatasetInfo | None) -> None:
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        if item:
            self.details.insert("end", f"Task: {item.task}\nEpisodes: {item.episodes}\nFrames: {item.frames}\nAction width: {item.action_dim or 'unknown'}\nGripper: {'present' if item.gripper else 'missing'}\nPath: {item.root}")
        self.details.configure(state="disabled")

    def selected_complete(self) -> DatasetInfo | None:
        if self.selected_root is None:
            messagebox.showwarning("No dataset", "Select a dataset first.")
            return None
        item = next((entry for entry in self.datasets if entry.root == self.selected_root), None)
        if item is None or not item.complete:
            messagebox.showwarning("Incomplete dataset", "This dataset has no complete LeRobot metadata.")
            return None
        return item

    def start_process(self, command: list[str], label: str, environment: dict[str, str] | None = None) -> None:
        if self.process and self.process.poll() is None:
            messagebox.showwarning("Busy", "A NERO Lab command is already running.")
            return
        self.log_message("$ " + " ".join(command))
        process_environment = os.environ.copy()
        if environment:
            process_environment.update(environment)
        self.process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=process_environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        threading.Thread(target=self._read_process, args=(self.process, label), daemon=True).start()

    def _read_process(self, process: subprocess.Popen[str], label: str) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            self.after(0, self.log_message, line)
            if label == "Teach task" and line.startswith("Saved taught task:"):
                self.after(0, self.refresh_tasks)
        code = process.wait()
        self.after(0, self.log_message, f"{label} exited with code {code}")
        if label == "Teach task" and code == 0:
            self.after(0, self.refresh_tasks)
            self.after(0, self.connect_robot)
        if label in {"Episode deletion", "Replay trained task", "Recorder"} and code == 0:
            self.after(0, self.refresh_datasets)

    def record_dataset(self) -> None:
        task = self.task_var.get().strip()
        if not task:
            messagebox.showwarning("Task required", "Enter a task instruction first.")
            return
        self.start_process([sys.executable, str(RECORDER), "--task", task], "Recorder")

    @staticmethod
    def _task_slug(task: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", task.lower()).strip("-")[:50] or "task"

    @staticmethod
    def _recording_replay_error(recording: dict[str, object]) -> str | None:
        try:
            prepare_replay_samples(recording)
        except (KeyError, TypeError, ValueError) as exc:
            return str(exc)
        return None

    @classmethod
    def _task_replay_error(cls, task_file: Path) -> str | None:
        try:
            recording = json.loads(task_file.read_text())
        except (OSError, ValueError, TypeError) as exc:
            return f"Could not read the taught task: {exc}"
        return cls._recording_replay_error(recording)

    def _prepare_task_process(self, emergency_brake: bool = True) -> bool:
        if self.robot is None or not self.robot.is_connected:
            try:
                self.connect_robot()
            except Exception as exc:
                messagebox.showerror("Arm connection failed", str(exc))
                return False
        if self.robot is None or not self.robot.is_connected:
            return False
        try:
            status = self.robot.get_arm_status()
            message = getattr(status, "msg", status)
            arm_status = str(getattr(message, "arm_status", ""))
            if "EMERGENCY_STOP" in arm_status or "EMERGENCY STOP" in arm_status:
                self.log_message("Re-enabling brake-settled arm for task handoff")
                self.reenable_arm()
        except Exception as exc:
            self.log_message(f"Could not prepare arm for task handoff: {exc}")
            return False
        try:
            current = self.robot.get_joint_angles()
            safe = self.is_safe_bicep_position(current)
        except Exception as exc:
            self.log_message(f"Could not verify task start pose: {exc}")
            safe = False
        if not safe:
            messagebox.showwarning("Safe reset required", "Move the arm to Safe Bicep Reset before starting this task flow.")
            return False
        self.disconnect_robot(emergency_brake=emergency_brake, disable_arm=False)
        return self.robot is None

    def teach_task(self) -> None:
        task = self.task_var.get().strip()
        if not task:
            messagebox.showwarning("Task required", "Enter a task instruction first.")
            return
        follower_anchor = SAFE_BICEP_RESET_JOINTS.copy()
        if self.robot is not None and self.robot.is_connected:
            try:
                follower_anchor = [float(value) for value in self.robot.get_joint_angles()]
            except Exception:
                pass
        if not self._prepare_task_process(emergency_brake=False):
            return
        TASK_BASE.mkdir(parents=True, exist_ok=True)
        output = TASK_BASE / f"{self._task_slug(task)}.json"
        self.taught_task_file = output
        self.start_process([
            sys.executable,
            str(TASK_TEACHER),
            "--task",
            task,
            "--output",
            str(output),
            "--follower-anchor",
            *[str(value) for value in follower_anchor],
        ], "Teach task")

    def replay_trained_task(self) -> None:
        task = self.task_var.get().strip()
        task_file = self.taught_task_file or TASK_BASE / f"{self._task_slug(task)}.json"
        if not task_file.exists():
            messagebox.showwarning("No taught task", "Click Teach Task and save a motion before replaying it.")
            return
        replay_error = self._task_replay_error(task_file)
        if replay_error:
            messagebox.showwarning("Task cannot be replayed", replay_error)
            return
        self.clear_activity_log()
        if not self._prepare_task_process(emergency_brake=False):
            return
        dataset_root = DATASET_BASE / f"nero_replayed__{self._task_slug(task)}"
        self.start_process(
            [sys.executable, str(TASK_REPLAY_RECORDER), "--task-file", str(task_file), "--dataset-root", str(dataset_root)],
            "Replay trained task",
        )

    def replay_task(self) -> None:
        task = self.task_var.get().strip()
        task_file = self.taught_task_file or TASK_BASE / f"{self._task_slug(task)}.json"
        if not task_file.exists():
            messagebox.showwarning("No taught task", "Select or teach a task before replaying it.")
            return
        replay_error = self._task_replay_error(task_file)
        if replay_error:
            messagebox.showwarning("Task cannot be replayed", replay_error)
            return
        self.clear_activity_log()
        if not self._prepare_task_process(emergency_brake=False):
            return
        self.start_process(
            [sys.executable, str(TASK_REPLAYER), "--task-file", str(task_file)],
            "Replay task",
        )

    def replay_dataset(self) -> None:
        item = self.selected_complete()
        if item:
            self.start_process([sys.executable, str(REPLAYER), "--dataset-root", str(item.root), "--episode-index", str(self.selected_episode)], "Replay")

    def open_rerun(self) -> None:
        item = self.selected_complete()
        if item:
            command = [str(RERUN), "--repo-id", "adrian/nero_manual", "--root", str(item.root), "--episode-index", str(self.selected_episode), "--mode", "local"]
            self.start_process(command, "Rerun")

    def delete_dataset(self) -> None:
        item = self.selected_complete() or next((entry for entry in self.datasets if entry.root == self.selected_root), None)
        if item is None:
            return
        if not messagebox.askyesno("Delete dataset", f"Delete this dataset permanently?\n\n{item.root}"):
            return
        shutil.rmtree(item.root)
        self.log_message(f"Deleted {item.root}")
        self.refresh_datasets()

    def delete_episode(self) -> None:
        item = self.selected_complete()
        if item is None:
            return
        if item.episodes <= 0:
            messagebox.showwarning("No episodes", "The selected dataset contains no saved episodes.")
            return
        if not messagebox.askyesno(
            "Delete episode",
            f"Delete episode {self.selected_episode} permanently?\n\n{item.root}",
        ):
            return
        command = [
            sys.executable,
            str(EPISODE_DELETER),
            "--dataset-root", str(item.root),
            "--episode-index", str(self.selected_episode),
        ]
        self.start_process(command, "Episode deletion")

    def browse_policy(self) -> None:
        selected = filedialog.askdirectory(title="Select policy checkpoint directory")
        if selected:
            self.policy_var.set(selected)

    def train_policy(self) -> None:
        item = self.selected_complete()
        if item is None:
            return
        if not item.gripper:
            messagebox.showwarning("Gripper data missing", "Select an 8D dataset containing gripper width before training.")
            return
        try:
            steps = int(self.steps_var.get().replace(",", "").strip())
        except ValueError:
            messagebox.showwarning("Invalid steps", "Training steps must be a whole number.")
            return
        if steps <= 0:
            messagebox.showwarning("Invalid steps", "Training steps must be greater than zero.")
            return
        try:
            batch_size = int(self.batch_size_var.get().strip())
            workers = int(self.workers_var.get().strip())
        except ValueError:
            messagebox.showwarning("Invalid performance settings", "Batch size and data-loader workers must be whole numbers.")
            return
        if batch_size <= 0 or workers < 0:
            messagebox.showwarning("Invalid performance settings", "Batch size must be positive and workers cannot be negative.")
            return
        output = PROJECT_ROOT / "policies" / item.root.name / "smolvla"
        device = auto_device() if self.device_var.get() == "auto" else self.device_var.get()
        if device == "cpu" and self.device_var.get() == "auto":
            self.log_message("No supported GPU backend detected; using CPU.")
        compile_model = "true" if self.compile_model_var.get() else "false"
        repo_id = "adrian/nero_replayed" if item.root.name.startswith("nero_replayed__") else "adrian/nero_manual"
        command = [str(LE_ROBOT_TRAIN), "--policy.type=smolvla", "--policy.push_to_hub=false", f"--policy.device={device}", "--policy.use_amp=true", f"--policy.compile_model={compile_model}", f"--batch_size={batch_size}", f"--num_workers={workers}", f"--steps={steps}", f"--dataset.repo_id={repo_id}", f"--dataset.root={item.root}", f"--output_dir={output}"]
        self.start_process(
            command,
            "SmolVLA training",
            {"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1"},
        )

    def run_policy(self) -> None:
        item = self.selected_complete()
        checkpoint = self.policy_var.get().strip()
        if item is None or not checkpoint:
            messagebox.showwarning("Policy required", "Select a complete gripper-enabled dataset and choose a policy checkpoint.")
            return
        if not item.gripper:
            messagebox.showwarning("Gripper data missing", "The selected dataset does not contain the required 8D action.")
            return
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            messagebox.showwarning("Checkpoint not found", "Choose a local SmolVLA checkpoint directory first.")
            return
        task = self.task_var.get().strip()
        if not task:
            messagebox.showwarning("Task required", "Enter the language instruction for this run.")
            return
        if not messagebox.askyesno(
            "Confirm robot motion",
            "This will send SmolVLA actions directly to the physical NERO arm without a joint step limit. Continue?",
        ):
            return
        command = [
            sys.executable,
            str(POLICY_RUNNER),
            "--checkpoint", str(checkpoint_path),
            "--dataset-root", str(item.root),
            "--task", task,
            "--confirm",
        ]
        if self.fast_motion_var.get():
            command.append("--fast-motion")
        self.start_process(command, "SmolVLA NERO run")


if __name__ == "__main__":
    NeroLab().mainloop()
