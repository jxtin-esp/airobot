#!/usr/bin/env python3
"""NEXUS Arm v4 AI: Unified Controller & Local AI Interface.

Key Features:
1. Dual Webcam Tracking:
   - Camera 1 (Side): Tracks J2 Shoulder, J3 Elbow, J4 Wrist pitch in the lateral plane.
   - Camera 2 (Overhead): Tracks J1 Base yaw looking down at base and turret.
   - Automatically excludes the built-in/integrated webcam (index 0).
2. Local AI Integration (Ollama qwen2.5:3b):
   - Natural language motion planning with structured JSON schema.
   - Whitelist safety validation: clamps to soft limits, bounds speeds, checks tracking health.
   - Step-by-step closed-loop execution queue with interactive progress & abort controls.
   - Quick preset shortcuts (Home, Wave, Reach & Grab, Release, Rest).
3. 3D Digital Twin:
   - Real-time software 3D model (pure Tkinter + NumPy) with interactive dragging, orbit, and zoom.
4. Embedded Hardware Interface:
   - Full 115200 baud serial protocol with Arduino Uno firmware.
   - Watchdog heartbeat, dynamic pin remapping (pins 2-13, 16-19), dual FSR touch sensing.
"""
import json
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import serial
import serial.tools.list_ports

import arm_dual_tracker as dual_trk
from arm_ai_bridge import ArmAIBridge, DEFAULT_OLLAMA_URL, DEFAULT_MODEL, PRESET_PLANS
from arm_control import clamp, decide_motion, servo_angle, update_speed_estimate, DEG_PER_PCT
from arm_model import ArmView, JOINT_NAMES

BAUD = 115200
RESET_WAIT = 2.0
TICK_MS = 30
VIEW_MS = 50
CAM_MS = 66
AI_POLL_MS = 2000
FRESH_S = 0.40
LOST_S = 0.55
CONT = (0, 1, 2)
POS = (3, 4, 5)
ALLOWED_PINS = list(range(2, 14)) + [16, 17, 18, 19]
SERVO_LABELS = [
    "J1 base", "J2 shoulder A", "J2 shoulder B (mirror)",
    "J3 elbow", "J4 wrist", "J5 roll", "J6 gripper"
]
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arm_config.json")

DEFAULT_BODY_LINKS = ["base", "turret", "upper_arm", "forearm", "wrist", "gripper"]

DEFAULT_BODY_MARKERS = [
    {"id": 0, "link": "base", "side": "both", "pos": 0.50, "locked": True},
    {"id": 1, "link": "upper_arm", "side": "left", "pos": 0.65, "locked": True},
    {"id": 2, "link": "forearm", "side": "left", "pos": 0.50, "locked": True},
    {"id": 3, "link": "wrist", "side": "left", "pos": 0.60, "locked": True},
    {"id": 5, "link": "turret", "side": "both", "pos": 0.50, "locked": True},
]

DEFAULT_CONFIG = {
    "pins": [3, 4, 5, 6, 7, 8, 9],
    "limits": [[-90, 90], [-60, 90], [-90, 90], [-90, 90], [-90, 90], [-90, 90]],
    "tracking": [
        {"camera": "overhead", "parent": 0, "child": 5, "invert": False, "offset": 0.0, "offset_overhead": 0.0, "offset_side": 0.0, "offset_left": 0.0, "offset_right": 0.0},
        {"camera": "side", "parent": 0, "child": 1, "invert": False, "offset": 0.0, "offset_overhead": 0.0, "offset_side": 0.0, "offset_left": 0.0, "offset_right": 0.0},
        {"camera": "side", "parent": 1, "child": 2, "invert": False, "offset": 0.0, "offset_overhead": 0.0, "offset_side": 0.0, "offset_left": 0.0, "offset_right": 0.0},
        {"camera": "side", "parent": 2, "child": 3, "invert": False, "offset": 0.0, "offset_overhead": 0.0, "offset_side": 0.0, "offset_left": 0.0, "offset_right": 0.0},
        {"camera": "side", "parent": None, "child": None, "invert": False, "offset": 0.0, "offset_overhead": 0.0, "offset_side": 0.0, "offset_left": 0.0, "offset_right": 0.0},
        {"camera": "side", "parent": None, "child": None, "invert": False, "offset": 0.0, "offset_overhead": 0.0, "offset_side": 0.0, "offset_left": 0.0, "offset_right": 0.0},
    ],
    "body_markers": DEFAULT_BODY_MARKERS,
    "webcam_trackers": {"overhead_points": [], "overhead_limbs": [], "side_points": [], "side_limbs": []},
    "camera_overhead_index": 0,
    "camera_side_index": 1,
    "camera_left_index": 0,
    "camera_right_index": 1,
    "ollama_url": DEFAULT_OLLAMA_URL,
    "ollama_model": DEFAULT_MODEL,
    "margin": 4.0,
    "limits_on": True,
    "speed": 50,
    "pos_on": False,
}


def load_config(path=CONFIG_PATH):
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    try:
        if os.path.exists(path):
            with open(path) as fh:
                saved = json.load(fh)
            for key, value in saved.items():
                if key == "tracking" and isinstance(value, list):
                    cfg["tracking"] = value
                elif key == "body_markers" and isinstance(value, list):
                    cfg["body_markers"] = value
                elif key == "webcam_trackers" and isinstance(value, dict):
                    cfg["webcam_trackers"] = value
                elif key in cfg:
                    cfg[key] = value
    except Exception:
        pass
    return cfg


def save_config(cfg, path=CONFIG_PATH):
    try:
        with open(path, "w") as fh:
            json.dump(cfg, fh, indent=2)
    except OSError:
        pass


def to_int_or_none(text):
    try:
        return int(str(text).strip())
    except ValueError:
        return None


def parse_marker_id(val):
    """Parse marker ID from an integer or string like '0', 'M0', 'M0 (base)', or 'None'."""
    if val is None:
        return None
    val = str(val).strip()
    if not val or val.lower() == "none" or val == "-":
        return None
    if val.startswith(("M", "m")):
        val = val[1:]
    val = val.split("(")[0].strip()
    try:
        return int(val)
    except ValueError:
        return None


class App(tk.Tk):
    def __init__(self, config_path=CONFIG_PATH):
        super().__init__()
        self.title("NEXUS Arm v4 AI  -  Dual-Camera Tracking & Local AI (qwen2.5:3b)")
        self.geometry("1240x840")
        self.minsize(1080, 720)
        self.config_path = config_path
        self.cfg = load_config(config_path)

        # Hardware & Tracker
        self.ser = None
        self.dual_tracker = dual_trk.DualArmTracker()
        self.ai_bridge = ArmAIBridge(self.cfg.get("ollama_url", DEFAULT_OLLAMA_URL),
                                     self.cfg.get("ollama_model", DEFAULT_MODEL))

        # Joint telemetry & state
        self.actual = [0.0] * 6
        self.target = [0.0] * 6
        self.meas = [None] * 6
        self.meas_t = [0.0] * 6
        self.vel = [0.0] * 6
        self.cmd = [0.0] * 3
        self.pos_dirty = [True] * 3
        self.held = {}
        self.seek = [None] * 3
        self.armed = [False] * 3
        self.cur_dir = [0] * 3
        self.k_est = [DEG_PER_PCT] * 3
        self.banner_until = 0.0

        # AI Execution State
        self.active_plan = []
        self.plan_step_idx = 0
        self.plan_step_start = 0.0
        self.plan_running = False
        self.ai_querying = False

        # Camera previews (Dual Side Views: Left & Right)
        self.blank_img = tk.PhotoImage(width=320, height=240)
        self.img_left = None
        self.img_right = None
        self.last_frame_left = -1
        self.last_frame_right = -1

        # Tk Variables
        c = self.cfg
        self.port_var = tk.StringVar()
        self.speed = tk.IntVar(value=c.get("speed", 50))
        self.pos_on = tk.BooleanVar(value=c.get("pos_on", False))
        self.limits_on = tk.BooleanVar(value=c.get("limits_on", True))
        self.margin_var = tk.StringVar(value=str(c.get("margin", 4.0)))

        # Cameras (Webcam 1: Overhead, Webcam 2: Side)
        self.cam_overhead_var = tk.StringVar(value=str(c.get("camera_overhead_index", c.get("camera_left_index", 0))))
        self.cam_side_var = tk.StringVar(value=str(c.get("camera_side_index", c.get("camera_right_index", 1))))
        self.cam_overhead_info = tk.StringVar(value="Overhead: Off")
        self.cam_side_info = tk.StringVar(value="Side: Off")
        # Aliases for backwards compatibility
        self.cam_left_var = self.cam_overhead_var
        self.cam_right_var = self.cam_side_var
        self.cam_left_info = self.cam_overhead_info
        self.cam_right_info = self.cam_side_info

        # Webcam AI Point & Limb Tracking State
        self.webcam_track_mode = tk.BooleanVar(value=False)
        self.webcam_trackers = c.get("webcam_trackers", {"overhead_points": [], "overhead_limbs": [], "side_points": [], "side_limbs": []})

        # Telemetry & Status
        self.banner = tk.StringVar(value="Ready. Connect Uno, ensure Ollama is running, and start Overhead & Side Cameras.")
        self.status = tk.StringVar(value="Not connected")
        self.ai_status_text = tk.StringVar(value="Ollama checking...")
        self.lo_vars = [tk.StringVar(value=str(c["limits"][j][0])) for j in range(6)]
        self.hi_vars = [tk.StringVar(value=str(c["limits"][j][1])) for j in range(6)]
        self.meas_text = [tk.StringVar(value="-") for _ in range(6)]
        self.target_text = [tk.StringVar(value="0") for _ in range(6)]
        self.slider_vars = [tk.DoubleVar(value=0.0) for _ in range(3)]
        self.fsr_vars = [tk.StringVar(value="0"), tk.StringVar(value="0")]
        self.pin_vars = [tk.StringVar(value=str(p)) for p in c["pins"]]

        # Body Markers & Joint Trackers
        self.body_markers = [dict(m) for m in self.cfg.get("body_markers", DEFAULT_BODY_MARKERS)]
        self.sel_marker_id_var = tk.StringVar(value="0")
        self.sel_marker_link_var = tk.StringVar(value="upper_arm")
        self.sel_marker_side_var = tk.StringVar(value="left")
        self.sel_marker_pos_var = tk.DoubleVar(value=0.5)
        self.sel_marker_locked_var = tk.BooleanVar(value=True)
        self._cam_cycle = 0

        # Tracking config variables
        self.trk_cam_vars = [tk.StringVar(value=t.get("camera", "side" if j > 0 else "overhead")) for j, t in enumerate(c["tracking"])]
        self.parent_vars = [tk.StringVar(value="" if t["parent"] is None else f"M{t['parent']}") for t in c["tracking"]]
        self.child_vars = [tk.StringVar(value="" if t["child"] is None else f"M{t['child']}") for t in c["tracking"]]
        self.invert_vars = [tk.BooleanVar(value=t.get("invert", False)) for t in c["tracking"]]
        self.offset_text = [tk.StringVar() for _ in range(6)]

        # Visual Joint Limits (set from live camera angle readings)
        # Each entry: {"enabled": BooleanVar, "min": DoubleVar, "max": DoubleVar,
        #              "auto_stop": BooleanVar, "status": StringVar}
        self.cam_limits = []
        for j in range(6):
            lo_def, hi_def = self.cfg["limits"][j]
            self.cam_limits.append({
                "enabled":   tk.BooleanVar(value=False),
                "min":       tk.DoubleVar(value=float(lo_def)),
                "max":       tk.DoubleVar(value=float(hi_def)),
                "auto_stop": tk.BooleanVar(value=True),
                "status":    tk.StringVar(value="—"),
            })

        # AI tab variables
        self.ai_prompt_var = tk.StringVar()
        self.ai_plan_summary = tk.StringVar(value="No active AI plan.")
        self.ollama_url_var = tk.StringVar(value=c.get("ollama_url", DEFAULT_OLLAMA_URL))
        self.ollama_model_var = tk.StringVar(value=c.get("ollama_model", DEFAULT_MODEL))

        self._build_ui()
        self._refresh_offsets()
        self.refresh_ports()
        self.refresh_camera_lists()

        # Start periodic tasks
        self.after(TICK_MS, self.tick)
        self.after(VIEW_MS, self.view_tick)
        self.after(CAM_MS, self.cam_tick)
        self.after(500, self.check_ollama_status)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Keyboard Manual Jogging shortcuts (active when not typing in text boxes)
        self.bind("<Left>", lambda e: self._kb_press(0, -1))
        self.bind("<KeyRelease-Left>", lambda e: self._kb_release(0))
        self.bind("<Right>", lambda e: self._kb_press(0, 1))
        self.bind("<KeyRelease-Right>", lambda e: self._kb_release(0))
        self.bind("<Up>", lambda e: self._kb_press(1, 1))
        self.bind("<KeyRelease-Up>", lambda e: self._kb_release(1))
        self.bind("<Down>", lambda e: self._kb_press(1, -1))
        self.bind("<KeyRelease-Down>", lambda e: self._kb_release(1))
        self.bind("<Prior>", lambda e: self._kb_press(2, -1))
        self.bind("<KeyRelease-Prior>", lambda e: self._kb_release(2))
        self.bind("<Next>", lambda e: self._kb_press(2, 1))
        self.bind("<KeyRelease-Next>", lambda e: self._kb_release(2))
        self.bind("<bracketleft>", lambda e: None if self._is_typing() else self.set_target(5, -90.0))
        self.bind("<bracketright>", lambda e: None if self._is_typing() else self.set_target(5, 90.0))
        self.bind("<space>", lambda e: None if self._is_typing() else self.stop_all())

    def _is_typing(self):
        focused = self.focus_get()
        return isinstance(focused, (tk.Entry, ttk.Entry, tk.Text))

    def _kb_press(self, j, direction):
        if not self._is_typing() and j not in self.held:
            self.press(j, direction)

    def _kb_release(self, j):
        if not self._is_typing():
            self.release(j)

    # ================================================================== UI Layout
    def _build_ui(self):
        # Top Global Toolbar
        top = ttk.Frame(self)
        top.pack(side="top", fill="x", padx=10, pady=6)

        ttk.Label(top, text="Port:").pack(side="left", padx=(0, 4))
        self.port_box = ttk.Combobox(top, textvariable=self.port_var, width=12, state="readonly")
        self.port_box.pack(side="left")
        ttk.Button(top, text="Refresh", width=7, command=self.refresh_ports).pack(side="left", padx=3)
        self.connect_btn = ttk.Button(top, text="Connect Uno", command=self.toggle_connection)
        self.connect_btn.pack(side="left", padx=4)

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=8)

        # Mode Switcher: Manual Control vs AI Assistant
        mode_frame = ttk.Frame(top)
        mode_frame.pack(side="left", padx=4)
        ttk.Label(mode_frame, text="Mode:", font=("TkDefaultFont", 9, "bold")).pack(side="left", padx=2)
        self.btn_mode_manual = tk.Button(mode_frame, text="🎮 Manual Control", bg="#238636", fg="white",
                                         font=("TkDefaultFont", 9, "bold"), relief="raised", padx=8, pady=2,
                                         command=lambda: self.switch_mode("manual"))
        self.btn_mode_manual.pack(side="left", padx=2)
        self.btn_mode_ai = tk.Button(mode_frame, text="🤖 AI Assistant", bg="#21262d", fg="#8b949e",
                                     font=("TkDefaultFont", 9, "bold"), relief="flat", padx=8, pady=2,
                                     command=lambda: self.switch_mode("ai"))
        self.btn_mode_ai.pack(side="left", padx=2)

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=8)

        ttk.Label(top, text="Speed %").pack(side="left", padx=(4, 2))
        tk.Scale(top, from_=10, to=100, orient="horizontal", length=110, variable=self.speed,
                 command=lambda _v: self._speed_changed()).pack(side="left")

        ttk.Checkbutton(top, text="Positional Servos ON", variable=self.pos_on,
                        command=self.on_pos_toggle).pack(side="left", padx=10)

        # Red STOP ALL button
        tk.Button(top, text="STOP ALL", bg="#d93025", fg="white", font=("TkDefaultFont", 11, "bold"),
                  padx=10, pady=2, command=self.stop_all).pack(side="left", padx=10)

        # Ollama Status Badge on far right
        self.ai_badge = tk.Label(top, textvariable=self.ai_status_text, bg="#2d333b", fg="#adbac7",
                                 font=("TkDefaultFont", 9, "bold"), padx=10, pady=4, relief="ridge")
        self.ai_badge.pack(side="right", padx=6)
        self.ai_badge.bind("<Button-1>", lambda e: self.check_ollama_status())

        # Main Body: Left (3D + Sensors) and Right (Dual Cams + Tabs)
        body = ttk.PanedWindow(self, orient="horizontal")
        body.pack(side="top", fill="both", expand=True, padx=8, pady=4)

        left_frame = ttk.Frame(body)

        # Scrollable right panel using Canvas + Scrollbar
        right_scroll_outer = ttk.Frame(body)
        right_canvas = tk.Canvas(right_scroll_outer, highlightthickness=0)
        right_vscroll = ttk.Scrollbar(right_scroll_outer, orient="vertical", command=right_canvas.yview)
        right_hscroll = ttk.Scrollbar(right_scroll_outer, orient="horizontal", command=right_canvas.xview)
        right_canvas.configure(yscrollcommand=right_vscroll.set, xscrollcommand=right_hscroll.set)

        right_vscroll.pack(side="right", fill="y")
        right_hscroll.pack(side="bottom", fill="x")
        right_canvas.pack(side="left", fill="both", expand=True)

        right_frame = ttk.Frame(right_canvas)
        _right_window = right_canvas.create_window((0, 0), window=right_frame, anchor="nw")

        def _on_right_configure(event):
            right_canvas.configure(scrollregion=right_canvas.bbox("all"))
        def _on_canvas_configure(event):
            right_canvas.itemconfig(_right_window, width=event.width)
        right_frame.bind("<Configure>", _on_right_configure)
        right_canvas.bind("<Configure>", _on_canvas_configure)

        # Mouse-wheel scrolling (Windows + Linux)
        def _on_mousewheel(event):
            right_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        def _on_mousewheel_linux(event):
            right_canvas.yview_scroll(-1 if event.num == 4 else 1, "units")
        right_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        right_canvas.bind_all("<Button-4>", _on_mousewheel_linux)
        right_canvas.bind_all("<Button-5>", _on_mousewheel_linux)

        body.add(left_frame, weight=4)
        body.add(right_scroll_outer, weight=6)

        # ---- Left Panel ----
        # 3D Arm Digital Twin
        self.view = ArmView(left_frame, width=470, height=350, on_drag=self._on_drag, on_drag_end=self._on_drag_end)
        self.view.canvas.pack(side="top", fill="both", expand=True, padx=4, pady=2)
        self.view.actual, self.view.target = self.actual, self.target
        self.view.tracked = [False] * 6
        self.view.body_markers = self.body_markers
        self.view.on_marker_select = self._on_3d_marker_clicked

        # Persistent Manual Quick-Jog Panel (Hold buttons to move continuous joints)
        jog_frame = ttk.LabelFrame(left_frame, text="Manual Quick-Jog (Hold buttons to move)")
        jog_frame.pack(side="top", fill="x", padx=4, pady=3)

        # Row 1: J1 Base and J2 Shoulder
        jr1 = ttk.Frame(jog_frame)
        jr1.pack(side="top", fill="x", padx=4, pady=2)
        ttk.Label(jr1, text="J1 Base:", font=("TkDefaultFont", 8, "bold"), width=8).pack(side="left")
        b1_l = tk.Button(jr1, text="◀ Left", width=6, bg="#2d333b", fg="white")
        b1_l.pack(side="left", padx=1)
        b1_l.bind("<ButtonPress-1>", lambda e: self.press(0, -1))
        b1_l.bind("<ButtonRelease-1>", lambda e: self.release(0))
        b1_r = tk.Button(jr1, text="Right ▶", width=6, bg="#2d333b", fg="white")
        b1_r.pack(side="left", padx=1)
        b1_r.bind("<ButtonPress-1>", lambda e: self.press(0, 1))
        b1_r.bind("<ButtonRelease-1>", lambda e: self.release(0))

        ttk.Label(jr1, text=" J2 Shldr:", font=("TkDefaultFont", 8, "bold"), width=9).pack(side="left", padx=(6, 0))
        b2_b = tk.Button(jr1, text="◀ Back", width=6, bg="#2d333b", fg="white")
        b2_b.pack(side="left", padx=1)
        b2_b.bind("<ButtonPress-1>", lambda e: self.press(1, 1))   # reversed
        b2_b.bind("<ButtonRelease-1>", lambda e: self.release(1))
        b2_f = tk.Button(jr1, text="Fwd ▶", width=6, bg="#2d333b", fg="white")
        b2_f.pack(side="left", padx=1)
        b2_f.bind("<ButtonPress-1>", lambda e: self.press(1, -1))  # reversed
        b2_f.bind("<ButtonRelease-1>", lambda e: self.release(1))

        # Row 2: J3 Elbow and J6 Gripper
        jr2 = ttk.Frame(jog_frame)
        jr2.pack(side="top", fill="x", padx=4, pady=2)
        ttk.Label(jr2, text="J3 Elbow:", font=("TkDefaultFont", 8, "bold"), width=8).pack(side="left")
        b3_u = tk.Button(jr2, text="▲ Up", width=6, bg="#2d333b", fg="white")
        b3_u.pack(side="left", padx=1)
        b3_u.bind("<ButtonPress-1>", lambda e: self.press(2, -1))
        b3_u.bind("<ButtonRelease-1>", lambda e: self.release(2))
        b3_d = tk.Button(jr2, text="Down ▼", width=6, bg="#2d333b", fg="white")
        b3_d.pack(side="left", padx=1)
        b3_d.bind("<ButtonPress-1>", lambda e: self.press(2, 1))
        b3_d.bind("<ButtonRelease-1>", lambda e: self.release(2))

        ttk.Label(jr2, text=" Gripper:", font=("TkDefaultFont", 8, "bold"), width=9).pack(side="left", padx=(6, 0))
        tk.Button(jr2, text="Open",  width=4, command=lambda: self.set_target(5, -90.0)).pack(side="left", padx=1)
        tk.Button(jr2, text="Mid",   width=3, command=lambda: self.set_target(5,   0.0)).pack(side="left", padx=1)
        tk.Button(jr2, text="Close", width=4, command=lambda: self.set_target(5,  90.0)).pack(side="left", padx=1)

        # FSR Gripper Pressure Frame
        fsr_box = ttk.LabelFrame(left_frame, text="Gripper Contact Sensors (FSR Raw 0-1023)")
        fsr_box.pack(side="top", fill="x", padx=4, pady=4)
        self.fsr_bars = []
        for i, pin in enumerate(("A0", "A1")):
            sub = ttk.Frame(fsr_box)
            sub.pack(side="left", fill="x", expand=True, padx=6, pady=4)
            ttk.Label(sub, text=f"FSR {i + 1} ({pin})").pack(side="left", padx=4)
            bar = ttk.Progressbar(sub, length=120, maximum=1023)
            bar.pack(side="left", padx=4)
            ttk.Label(sub, textvariable=self.fsr_vars[i], width=5).pack(side="left")
            self.fsr_bars.append(bar)

        # AI Execution Progress Frame
        ai_exec_box = ttk.LabelFrame(left_frame, text="Active AI Plan Execution")
        ai_exec_box.pack(side="top", fill="x", padx=4, pady=4)
        ttk.Label(ai_exec_box, textvariable=self.ai_plan_summary, wraplength=440,
                  font=("TkDefaultFont", 9, "italic")).pack(side="top", anchor="w", padx=6, pady=2)
        exec_btn_row = ttk.Frame(ai_exec_box)
        exec_btn_row.pack(side="top", fill="x", padx=6, pady=3)
        self.abort_plan_btn = tk.Button(exec_btn_row, text="Abort AI Plan", bg="#8b1111", fg="white",
                                        font=("TkDefaultFont", 9, "bold"), state="disabled", command=self.abort_plan)
        self.abort_plan_btn.pack(side="right", padx=4)
        self.run_plan_btn = ttk.Button(exec_btn_row, text="Run Plan", state="disabled", command=self.start_plan_execution)
        self.run_plan_btn.pack(side="right", padx=4)
        self.step_plan_btn = ttk.Button(exec_btn_row, text="Step Once", state="disabled", command=self.step_plan_execution)
        self.step_plan_btn.pack(side="right", padx=4)

        # Status & Safety Banner
        self.banner_label = ttk.Label(left_frame, textvariable=self.banner, wraplength=460,
                                      font=("TkDefaultFont", 10, "bold"))
        self.banner_label.pack(side="top", fill="x", padx=6, pady=3)
        ttk.Label(left_frame, textvariable=self.status, foreground="#57606a", wraplength=460).pack(
            side="top", fill="x", padx=6, pady=1
        )

        # ---- Right Panel ----
        # Top: Dual Webcams View (Left Side & Right Side)
        dual_cam_box = ttk.LabelFrame(right_frame, text="Dual-Camera Real-Time Vision Tracking (Overhead + Side View)")
        dual_cam_box.pack(side="top", fill="x", padx=4, pady=2)

        feeds_row = ttk.Frame(dual_cam_box)
        feeds_row.pack(side="top", fill="x", padx=4, pady=4)

                # Overhead Camera Feed (Webcam 1)
        left_sub = ttk.Frame(feeds_row)
        left_sub.pack(side="left", padx=4)
        ttk.Label(left_sub, text="Webcam 1: Overhead View (Top-Down)", font=("TkDefaultFont", 9, "bold")).pack()
        self.lbl_overhead_cam = tk.Label(left_sub, image=self.blank_img, text="Overhead Camera Off",
                                         compound="center", bg="#0d1117", fg="#8b949e", width=320, height=240)
        self.lbl_overhead_cam.pack(pady=2)
        self.lbl_overhead_cam.bind("<Button-1>", lambda e: self._on_webcam_clicked("overhead", e.x, e.y))
        ttk.Label(left_sub, textvariable=self.cam_overhead_info, foreground="gray").pack()
        self.lbl_left_cam = self.lbl_overhead_cam

        # Side Camera Feed (Webcam 2)
        right_sub = ttk.Frame(feeds_row)
        right_sub.pack(side="left", padx=4)
        ttk.Label(right_sub, text="Webcam 2: Side View (Lateral)", font=("TkDefaultFont", 9, "bold")).pack()
        self.lbl_side_cam = tk.Label(right_sub, image=self.blank_img, text="Side Camera Off",
                                     compound="center", bg="#0d1117", fg="#8b949e", width=320, height=240)
        self.lbl_side_cam.pack(pady=2)
        self.lbl_side_cam.bind("<Button-1>", lambda e: self._on_webcam_clicked("side", e.x, e.y))
        ttk.Label(right_sub, textvariable=self.cam_side_info, foreground="gray").pack()
        self.lbl_right_cam = self.lbl_side_cam

                # Camera Controls Row
        cam_ctl_row = ttk.Frame(dual_cam_box)
        cam_ctl_row.pack(side="top", fill="x", padx=6, pady=2)

        ttk.Label(cam_ctl_row, text="Overhead Cam:").pack(side="left", padx=2)
        self.cam_overhead_combo = ttk.Combobox(cam_ctl_row, textvariable=self.cam_overhead_var, width=4)
        self.cam_overhead_combo.pack(side="left", padx=2)
        self.cam_left_combo = self.cam_overhead_combo

        ttk.Label(cam_ctl_row, text="Side Cam:").pack(side="left", padx=(8, 2))
        self.cam_side_combo = ttk.Combobox(cam_ctl_row, textvariable=self.cam_side_var, width=4)
        self.cam_side_combo.pack(side="left", padx=2)
        self.cam_right_combo = self.cam_side_combo

        ttk.Button(cam_ctl_row, text="Scan Cameras", command=lambda: self.refresh_camera_lists(interactive=True)).pack(side="left", padx=5)
        self.btn_toggle_cams = ttk.Button(cam_ctl_row, text="Start Dual Cameras", command=self.toggle_dual_cameras)
        self.btn_toggle_cams.pack(side="left", padx=6)

        ttk.Button(cam_ctl_row, text="Calibrate Zero Pose", command=self.calibrate_zero).pack(side="left", padx=3)
        ttk.Button(cam_ctl_row, text="Save Marker Sheet", command=self.save_markers).pack(side="left", padx=3)

        # Webcam AI Point & Limb Tracking Action Bar
        wb_bar = ttk.Frame(dual_cam_box)
        wb_bar.pack(side="top", fill="x", padx=6, pady=(3, 5))
        self.btn_webcam_add_point = tk.Button(wb_bar, text="🎯 Click Cam to Add Tracker", bg="#1f6feb", fg="white",
                                             font=("TkDefaultFont", 8, "bold"), relief="raised", padx=6, pady=1,
                                             command=self._toggle_webcam_add_mode)
        self.btn_webcam_add_point.pack(side="left", padx=3)
        ttk.Button(wb_bar, text="⚡ Auto-Link Limbs", command=self._auto_link_webcam_limbs).pack(side="left", padx=2)
        ttk.Button(wb_bar, text="🔄 Reset Points", command=self._reset_webcam_trackers).pack(side="left", padx=2)
        ttk.Button(wb_bar, text="🗑️ Clear Trackers", command=self._clear_webcam_trackers).pack(side="left", padx=2)
        ttk.Label(wb_bar, text="💡 Click joint on video to track via AI optical flow",
                  foreground="#57606a", font=("TkDefaultFont", 8, "italic")).pack(side="left", padx=6)

        # Visual Joint Limits panel (set min/max from live camera angle readings)
        self._build_visual_limits_panel(dual_cam_box)

        # Bottom: Notebook Tabs (Manual Joints & Limits, AI Assistant, Markers & Trackers, Servo Pins)
        self.nb = ttk.Notebook(right_frame)
        self.nb.pack(side="top", fill="both", expand=True, padx=4, pady=4)

        self.tab_joints = self._build_tab_joints(self.nb)
        self.tab_ai = self._build_tab_ai(self.nb)
        self.tab_markers_trackers = self._build_tab_markers_and_trackers(self.nb)
        self.tab_pins = self._build_tab_pins(self.nb)

        self.nb.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def switch_mode(self, mode):
        if mode == "manual":
            self.nb.select(self.tab_joints)
            self.btn_mode_manual.config(bg="#238636", fg="white", relief="raised")
            self.btn_mode_ai.config(bg="#21262d", fg="#8b949e", relief="flat")
            self.set_banner("Manual Control Mode: Hold buttons to jog, use sliders, or drag 3D handles.", "info", 4.0)
        else:
            self.nb.select(self.tab_ai)
            self.btn_mode_ai.config(bg="#1f6feb", fg="white", relief="raised")
            self.btn_mode_manual.config(bg="#21262d", fg="#8b949e", relief="flat")
            self.set_banner("AI Assistant Mode: Prompt qwen2.5:3b or use quick preset actions.", "info", 4.0)

    def _on_tab_changed(self, event=None):
        selected = self.nb.select()
        if hasattr(self, "tab_joints") and selected == str(self.tab_joints):
            self.btn_mode_manual.config(bg="#238636", fg="white", relief="raised")
            self.btn_mode_ai.config(bg="#21262d", fg="#8b949e", relief="flat")
        elif hasattr(self, "tab_ai") and selected == str(self.tab_ai):
            self.btn_mode_ai.config(bg="#1f6feb", fg="white", relief="raised")
            self.btn_mode_manual.config(bg="#21262d", fg="#8b949e", relief="flat")
        else:
            self.btn_mode_manual.config(bg="#21262d", fg="#8b949e", relief="flat")
            self.btn_mode_ai.config(bg="#21262d", fg="#8b949e", relief="flat")

    # ------------------------------------------------------------- Tab: AI Assistant
    def _build_tab_ai(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="AI Assistant (Ollama qwen2.5:3b)")

        # Presets Row
        preset_row = ttk.LabelFrame(tab, text="Quick Presets & Demonstrations")
        preset_row.pack(side="top", fill="x", padx=6, pady=4)
        ttk.Button(preset_row, text="Home Pose", command=lambda: self.load_preset("home")).pack(side="left", padx=4, pady=3)
        ttk.Button(preset_row, text="Wave Greeting", command=lambda: self.load_preset("wave")).pack(side="left", padx=4, pady=3)
        ttk.Button(preset_row, text="Reach & Grab", command=lambda: self.load_preset("reach_grab")).pack(side="left", padx=4, pady=3)
        ttk.Button(preset_row, text="Release Object", command=lambda: self.load_preset("release")).pack(side="left", padx=4, pady=3)
        ttk.Button(preset_row, text="Rest Pose", command=lambda: self.load_preset("rest")).pack(side="left", padx=4, pady=3)

        # AI Thought & Conversation History Box
        log_box = ttk.LabelFrame(tab, text="AI Reasoning & Plan Steps")
        log_box.pack(side="top", fill="both", expand=True, padx=6, pady=4)
        self.ai_text = tk.Text(log_box, height=8, wrap="word", bg="#0d1117", fg="#c9d1d9",
                               font=("Consolas", 10), insertbackground="white")
        self.ai_text.pack(side="left", fill="both", expand=True, padx=4, pady=4)
        sc = ttk.Scrollbar(log_box, command=self.ai_text.yview)
        sc.pack(side="right", fill="y")
        self.ai_text["yscrollcommand"] = sc.set
        self.ai_text.insert("end", "Antigravity AI Assistant Ready.\nEnter a request below or select a preset to start.\n")

        # Instruction Input Row
        inp_row = ttk.Frame(tab)
        inp_row.pack(side="top", fill="x", padx=6, pady=4)
        ttk.Label(inp_row, text="Ask AI:").pack(side="left", padx=2)
        self.ai_entry = ttk.Entry(inp_row, textvariable=self.ai_prompt_var)
        self.ai_entry.pack(side="left", fill="x", expand=True, padx=4)
        self.ai_entry.bind("<Return>", lambda e: self.send_ai_query())
        self.btn_send_ai = ttk.Button(inp_row, text="Send to qwen2.5:3b", command=self.send_ai_query)
        self.btn_send_ai.pack(side="left", padx=4)

        # Ollama Server Config Collapsible Row
        srv_row = ttk.Frame(tab)
        srv_row.pack(side="top", fill="x", padx=6, pady=2)
        ttk.Label(srv_row, text="Ollama URL:", font=("TkDefaultFont", 8)).pack(side="left", padx=2)
        ttk.Entry(srv_row, textvariable=self.ollama_url_var, width=22, font=("TkDefaultFont", 8)).pack(side="left", padx=2)
        ttk.Label(srv_row, text="Model:", font=("TkDefaultFont", 8)).pack(side="left", padx=(8, 2))
        ttk.Entry(srv_row, textvariable=self.ollama_model_var, width=14, font=("TkDefaultFont", 8)).pack(side="left", padx=2)
        ttk.Button(srv_row, text="Test Connection", command=self.check_ollama_status).pack(side="left", padx=8)
        return tab

    # ------------------------------------------------------------- Tab: Joints & Limits
    def _build_tab_joints(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="Manual Joints & Limits")

        for col, text in enumerate(["Joint", "Move Controls", "", "Measured", "Target", "", "Min°", "Max°", "Status"]):
            ttk.Label(tab, text=text, font=("TkDefaultFont", 9, "bold")).grid(row=0, column=col, padx=4, pady=4)

        self.dots = []
        for j in range(6):
            r = j + 1
            ttk.Label(tab, text=JOINT_NAMES[j], width=14).grid(row=r, column=0, padx=4, pady=2, sticky="w")
            if j in CONT:
                left = tk.Button(tab, text="\u25c0 Left", width=7)
                left.grid(row=r, column=1, padx=2)
                left.bind("<ButtonPress-1>", lambda e, j=j: self.press(j, -1))
                left.bind("<ButtonRelease-1>", lambda e, j=j: self.release(j))
                right = tk.Button(tab, text="Right \u25b6", width=7)
                right.grid(row=r, column=2, padx=2)
                right.bind("<ButtonPress-1>", lambda e, j=j: self.press(j, 1))
                right.bind("<ButtonRelease-1>", lambda e, j=j: self.release(j))
            else:
                k = j - 3
                tk.Scale(tab, from_=-90, to=90, orient="horizontal", length=140, showvalue=False,
                         variable=self.slider_vars[k],
                         command=lambda v, j=j: self._slider_moved(j)
                         ).grid(row=r, column=1, columnspan=2, padx=2)

            ttk.Label(tab, textvariable=self.meas_text[j], width=8).grid(row=r, column=3)
            ttk.Entry(tab, textvariable=self.target_text[j], width=6).grid(row=r, column=4, padx=2)
            ttk.Button(tab, text="Go", width=3, command=lambda j=j: self.go(j)).grid(row=r, column=5, padx=2)

            for col, var in ((6, self.lo_vars[j]), (7, self.hi_vars[j])):
                sp = ttk.Spinbox(tab, from_=-180, to=180, width=5, textvariable=var,
                                 command=self._read_limits)
                sp.grid(row=r, column=col, padx=2)
                sp.bind("<Return>", lambda e: self._read_limits())
                sp.bind("<FocusOut>", lambda e: self._read_limits())

            dot = ttk.Label(tab, text="\u25cf", foreground="#7d8590", font=("TkDefaultFont", 12))
            dot.grid(row=r, column=8, padx=4)
            self.dots.append(dot)

        opts = ttk.Frame(tab)
        opts.grid(row=7, column=0, columnspan=9, sticky="w", pady=8, padx=4)
        ttk.Checkbutton(opts, text="Enforce Software Limits (from Vision)", variable=self.limits_on,
                        command=self._read_limits).pack(side="left", padx=4)
        ttk.Label(opts, text="Buffer Margin (degrees):").pack(side="left", padx=(14, 2))
        sp = ttk.Spinbox(opts, from_=0, to=30, width=4, textvariable=self.margin_var,
                         command=self._read_limits)
        sp.pack(side="left")
        sp.bind("<Return>", lambda e: self._read_limits())
        sp.bind("<FocusOut>", lambda e: self._read_limits())
        return tab

    # ------------------------------------------------------------- Tab: Markers & Joint Trackers
    def _build_tab_markers_and_trackers(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="🎯 Markers & Joint Trackers")

        # Top Section: Body Markers on 3D Twin (Move, Lock & Visualize)
        marker_frame = ttk.LabelFrame(tab, text="Body Markers on Arm (Move, Lock & Visualize on 3D Twin)")
        marker_frame.pack(side="top", fill="both", expand=True, padx=6, pady=4)

        # Left subframe: Treeview of markers
        tree_frame = ttk.Frame(marker_frame)
        tree_frame.pack(side="left", fill="both", expand=True, padx=4, pady=4)

        cols = ("id", "link", "side", "pos", "locked", "status")
        self.marker_tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=5, selectmode="browse")
        self.marker_tree.heading("id", text="Marker")
        self.marker_tree.heading("link", text="Link")
        self.marker_tree.heading("side", text="Side")
        self.marker_tree.heading("pos", text="Pos%")
        self.marker_tree.heading("locked", text="Lock")
        self.marker_tree.heading("status", text="Vision")

        self.marker_tree.column("id", width=60, anchor="center")
        self.marker_tree.column("link", width=80, anchor="w")
        self.marker_tree.column("side", width=50, anchor="center")
        self.marker_tree.column("pos", width=55, anchor="center")
        self.marker_tree.column("locked", width=70, anchor="center")
        self.marker_tree.column("status", width=55, anchor="center")

        tree_sc = ttk.Scrollbar(tree_frame, orient="vertical", command=self.marker_tree.yview)
        self.marker_tree.configure(yscrollcommand=tree_sc.set)
        self.marker_tree.pack(side="left", fill="both", expand=True)
        tree_sc.pack(side="right", fill="y")
        self.marker_tree.bind("<<TreeviewSelect>>", self._on_marker_tree_select)

        # Right subframe: Selected Marker Configuration & Visual Positioning Slider
        edit_frame = ttk.LabelFrame(marker_frame, text="Selected Marker Configuration")
        edit_frame.pack(side="right", fill="y", padx=4, pady=4)

        r1 = ttk.Frame(edit_frame)
        r1.pack(side="top", fill="x", padx=4, pady=2)
        ttk.Label(r1, text="ID:").pack(side="left", padx=2)
        self.sp_marker_id = ttk.Spinbox(r1, from_=0, to=49, width=4, textvariable=self.sel_marker_id_var)
        self.sp_marker_id.pack(side="left", padx=2)
        self.sp_marker_id.bind("<Return>", lambda e: self._on_marker_field_edited())
        self.sp_marker_id.bind("<FocusOut>", lambda e: self._on_marker_field_edited())

        ttk.Label(r1, text="Link:").pack(side="left", padx=(4, 2))
        self.cb_marker_link = ttk.Combobox(r1, values=DEFAULT_BODY_LINKS, textvariable=self.sel_marker_link_var,
                                           width=9, state="readonly")
        self.cb_marker_link.pack(side="left", padx=2)
        self.cb_marker_link.bind("<<ComboboxSelected>>", lambda e: self._on_marker_field_edited())

        ttk.Label(r1, text="Side:").pack(side="left", padx=(4, 2))
        self.cb_marker_side = ttk.Combobox(r1, values=["left", "right", "both"], textvariable=self.sel_marker_side_var,
                                           width=5, state="readonly")
        self.cb_marker_side.pack(side="left", padx=2)
        self.cb_marker_side.bind("<<ComboboxSelected>>", lambda e: self._on_marker_field_edited())

        # Position slider: Moving this slider immediately moves the marker badge on the 3D twin!
        r2 = ttk.Frame(edit_frame)
        r2.pack(side="top", fill="x", padx=4, pady=3)
        ttk.Label(r2, text="Position on Link:").pack(side="left", padx=2)
        self.slider_marker_pos = tk.Scale(r2, from_=0.0, to=1.0, resolution=0.01, orient="horizontal",
                                          length=120, showvalue=True, variable=self.sel_marker_pos_var,
                                          command=self._on_marker_slider_moved)
        self.slider_marker_pos.pack(side="left", padx=4)

        # Lock toggle button
        r3 = ttk.Frame(edit_frame)
        r3.pack(side="top", fill="x", padx=4, pady=2)
        self.btn_marker_lock = tk.Button(r3, text="🔒 Locked to Link", font=("TkDefaultFont", 9, "bold"),
                                         bg="#1a472a", fg="#7ee787", padx=6, pady=2,
                                         command=self._toggle_marker_lock)
        self.btn_marker_lock.pack(side="left", fill="x", expand=True, padx=2)

        # Action buttons row
        r4 = ttk.Frame(edit_frame)
        r4.pack(side="top", fill="x", padx=4, pady=2)
        ttk.Button(r4, text="➕ Add", width=6, command=self._add_body_marker).pack(side="left", padx=1)
        ttk.Button(r4, text="🗑️ Del", width=6, command=self._remove_body_marker).pack(side="left", padx=1)
        ttk.Button(r4, text="🔄 Reset", width=7, command=self._reset_body_markers).pack(side="left", padx=1)

        # Bottom Section: Joint Kinematic Trackers
        jt_frame = ttk.LabelFrame(tab, text="Joint Kinematic Trackers (Pairs of Markers Tracking Joint Angles)")
        jt_frame.pack(side="top", fill="both", expand=True, padx=6, pady=4)

        headers = ["Joint", "Camera", "Parent Marker", "Child Marker", "Invert", "Zero Offset", "Measured", "Zero"]
        for col, h in enumerate(headers):
            ttk.Label(jt_frame, text=h, font=("TkDefaultFont", 9, "bold")).grid(row=0, column=col, padx=3, pady=2)

        self.cb_jt_parents = []
        self.cb_jt_children = []
        self.cb_jt_cams = []

        for j in range(6):
            r = j + 1
            ttk.Label(jt_frame, text=JOINT_NAMES[j], width=11).grid(row=r, column=0, sticky="w", padx=3, pady=1)

            cam_box = ttk.Combobox(jt_frame, textvariable=self.trk_cam_vars[j], values=["auto", "overhead", "side"],
                                   width=5, state="readonly")
            cam_box.grid(row=r, column=1, padx=2)
            cam_box.bind("<<ComboboxSelected>>", lambda e, j=j: self._on_jt_changed(j))
            self.cb_jt_cams.append(cam_box)

            p_box = ttk.Combobox(jt_frame, textvariable=self.parent_vars[j], width=13, state="readonly")
            p_box.grid(row=r, column=2, padx=2)
            p_box.bind("<<ComboboxSelected>>", lambda e, j=j: self._on_jt_changed(j))
            self.cb_jt_parents.append(p_box)

            c_box = ttk.Combobox(jt_frame, textvariable=self.child_vars[j], width=13, state="readonly")
            c_box.grid(row=r, column=3, padx=2)
            c_box.bind("<<ComboboxSelected>>", lambda e, j=j: self._on_jt_changed(j))
            self.cb_jt_children.append(c_box)

            ttk.Checkbutton(jt_frame, variable=self.invert_vars[j], command=lambda j=j: self._on_jt_changed(j)
                            ).grid(row=r, column=4, padx=2)
            ttk.Label(jt_frame, textvariable=self.offset_text[j], width=9).grid(row=r, column=5, padx=2)
            ttk.Label(jt_frame, textvariable=self.meas_text[j], width=7, font=("Consolas", 9, "bold")).grid(row=r, column=6, padx=2)

            ttk.Button(jt_frame, text="Zero", width=5, command=lambda j=j: self._zero_single_joint(j)
                       ).grid(row=r, column=7, padx=2)

        jt_bot = ttk.Frame(jt_frame)
        jt_bot.grid(row=7, column=0, columnspan=8, sticky="w", padx=4, pady=5)
        ttk.Button(jt_bot, text="⚡ Auto-Bind Standard Trackers", command=self._auto_bind_joint_trackers).pack(side="left", padx=4)
        ttk.Button(jt_bot, text="Calibrate Zero All", command=self.calibrate_zero).pack(side="left", padx=4)
        ttk.Button(jt_bot, text="Save Printable ArUco Sheet", command=self.save_markers).pack(side="left", padx=4)

        self._refresh_marker_tree()
        self._refresh_tracker_combos()
        return tab

    _build_tab_tracking = _build_tab_markers_and_trackers

    def _refresh_marker_tree(self):
        if not hasattr(self, "marker_tree"):
            return
        for item in self.marker_tree.get_children():
            self.marker_tree.delete(item)

        for m in self.body_markers:
            mid = m.get("id", 0)
            link = m.get("link", "upper_arm")
            side = m.get("side", "left")
            pos_pct = f"{int(float(m.get('pos', 0.5)) * 100)}%"
            lock_sym = "🔒 Locked" if m.get("locked", True) else "🔓 Free"
            seen = mid in self.view.active_markers
            status = "● Seen" if seen else "○ Lost"

            self.marker_tree.insert("", "end", iid=str(mid),
                                   values=(f"M{mid}", link, side, pos_pct, lock_sym, status))

        sel_id = self.view.selected_marker_id
        if sel_id is not None and self.marker_tree.exists(str(sel_id)):
            self.marker_tree.selection_set(str(sel_id))
            self.marker_tree.see(str(sel_id))

    def _on_marker_tree_select(self, event=None):
        sel = self.marker_tree.selection()
        if not sel:
            return
        try:
            mid = int(sel[0])
        except ValueError:
            return

        for m in self.body_markers:
            if m["id"] == mid:
                self.view.selected_marker_id = mid
                self.sel_marker_id_var.set(str(m["id"]))
                self.sel_marker_link_var.set(m.get("link", "upper_arm"))
                self.sel_marker_side_var.set(m.get("side", "left"))
                self.sel_marker_pos_var.set(float(m.get("pos", 0.5)))
                self.sel_marker_locked_var.set(bool(m.get("locked", True)))

                is_locked = m.get("locked", True)
                if is_locked:
                    self.btn_marker_lock.config(text="🔒 Locked (Click to Unlock)", bg="#1a472a", fg="#7ee787")
                else:
                    self.btn_marker_lock.config(text="🔓 Unlocked (Click to Lock)", bg="#593a00", fg="#f0883e")
                break
        self.view.render()

    def _on_3d_marker_clicked(self, marker_id):
        self.view.selected_marker_id = marker_id
        if hasattr(self, "marker_tree") and self.marker_tree.exists(str(marker_id)):
            self.marker_tree.selection_set(str(marker_id))
            self.marker_tree.see(str(marker_id))
            self._on_marker_tree_select()
        self.set_banner(f"Selected Marker M{marker_id} on 3D Twin. Adjust position with slider.", "ok", 3.0)

    def _on_marker_slider_moved(self, val):
        sel_id = self.view.selected_marker_id
        if sel_id is None:
            return
        for m in self.body_markers:
            if m["id"] == sel_id:
                m["pos"] = round(float(val), 2)
                if hasattr(self, "marker_tree") and self.marker_tree.exists(str(sel_id)):
                    vals = list(self.marker_tree.item(str(sel_id), "values"))
                    if len(vals) >= 4:
                        vals[3] = f"{int(m['pos'] * 100)}%"
                        self.marker_tree.item(str(sel_id), values=vals)
                break
        self.view.render()
        self._save()

    def _on_marker_field_edited(self):
        sel_id = self.view.selected_marker_id
        if sel_id is None:
            return
        for m in self.body_markers:
            if m["id"] == sel_id:
                try:
                    new_id = int(self.sel_marker_id_var.get().strip())
                except ValueError:
                    new_id = m["id"]
                m["id"] = new_id
                m["link"] = self.sel_marker_link_var.get().strip()
                m["side"] = self.sel_marker_side_var.get().strip()
                self.view.selected_marker_id = new_id
                break
        self._refresh_marker_tree()
        self._refresh_tracker_combos()
        self.view.render()
        self._save()

    def _toggle_marker_lock(self):
        sel_id = self.view.selected_marker_id
        if sel_id is None:
            return
        for m in self.body_markers:
            if m["id"] == sel_id:
                m["locked"] = not m.get("locked", True)
                is_locked = m["locked"]
                if is_locked:
                    self.btn_marker_lock.config(text="🔒 Locked (Click to Unlock)", bg="#1a472a", fg="#7ee787")
                else:
                    self.btn_marker_lock.config(text="🔓 Unlocked (Click to Lock)", bg="#593a00", fg="#f0883e")
                break
        self._refresh_marker_tree()
        self.view.render()
        self._save()
        status_msg = "🔒 LOCKED to link" if is_locked else "🔓 UNLOCKED (free to move)"
        self.set_banner(f"Marker M{sel_id} is now {status_msg}.", "ok", 3.0)

    def _add_body_marker(self):
        existing_ids = [m["id"] for m in self.body_markers]
        new_id = 0
        while new_id in existing_ids:
            new_id += 1
        new_m = {"id": new_id, "link": "upper_arm", "side": "left", "pos": 0.50, "locked": False}
        self.body_markers.append(new_m)
        self.view.selected_marker_id = new_id
        self._refresh_marker_tree()
        self._refresh_tracker_combos()
        self.view.render()
        self._save()
        self.set_banner(f"Added Marker M{new_id}. Adjust position with the slider, then click Lock.", "ok", 4.0)

    def _remove_body_marker(self):
        sel_id = self.view.selected_marker_id
        if sel_id is None:
            self.set_banner("Select a marker from the table or 3D view first.", "warn")
            return
        self.body_markers = [m for m in self.body_markers if m["id"] != sel_id]
        self.view.body_markers = self.body_markers
        self.view.selected_marker_id = None
        self._refresh_marker_tree()
        self._refresh_tracker_combos()
        self.view.render()
        self._save()
        self.set_banner(f"Removed Marker M{sel_id}.", "ok", 3.0)

    def _reset_body_markers(self):
        if not messagebox.askyesno("Reset Markers", "Reset all body markers back to default layout?"):
            return
        self.body_markers = [dict(m) for m in DEFAULT_BODY_MARKERS]
        self.view.body_markers = self.body_markers
        self.view.selected_marker_id = None
        self._refresh_marker_tree()
        self._refresh_tracker_combos()
        self.view.render()
        self._save()
        self.set_banner("Body markers reset to standard configuration.", "ok", 3.0)

    def _refresh_tracker_combos(self):
        marker_opts = ["None"] + [f"M{m['id']} ({m.get('link', 'link')})" for m in self.body_markers]
        for j in range(6):
            if hasattr(self, "cb_jt_parents") and j < len(self.cb_jt_parents):
                self.cb_jt_parents[j]["values"] = marker_opts
                self.cb_jt_children[j]["values"] = marker_opts

                p = self.cfg["tracking"][j]["parent"]
                c = self.cfg["tracking"][j]["child"]

                p_text = "None"
                for m in self.body_markers:
                    if m["id"] == p:
                        p_text = f"M{m['id']} ({m.get('link', '')})"
                        break
                self.parent_vars[j].set(p_text)

                c_text = "None"
                for m in self.body_markers:
                    if m["id"] == c:
                        c_text = f"M{m['id']} ({m.get('link', '')})"
                        break
                self.child_vars[j].set(c_text)

    def _on_jt_changed(self, j):
        t = self.cfg["tracking"][j]
        t["camera"] = self.trk_cam_vars[j].get()
        t["parent"] = parse_marker_id(self.parent_vars[j].get())
        t["child"] = parse_marker_id(self.child_vars[j].get())
        t["invert"] = bool(self.invert_vars[j].get())
        self.meas[j] = None
        self._save()

    def _auto_bind_joint_trackers(self):
        markers_by_link = {}
        for m in self.body_markers:
            markers_by_link.setdefault(m["link"], []).append(m["id"])

        base_id = markers_by_link.get("base", [0])[0] if "base" in markers_by_link else 0
        turret_id = markers_by_link.get("turret", [5])[0] if "turret" in markers_by_link else 5
        upper_id = markers_by_link.get("upper_arm", [1])[0] if "upper_arm" in markers_by_link else 1
        fore_id = markers_by_link.get("forearm", [2])[0] if "forearm" in markers_by_link else 2
        wrist_id = markers_by_link.get("wrist", [3])[0] if "wrist" in markers_by_link else 3

        pairs = [
            (base_id, turret_id),
            (base_id, upper_id),
            (upper_id, fore_id),
            (fore_id, wrist_id),
            (None, None),
            (None, None),
        ]

        for j, (p, c) in enumerate(pairs):
            t = self.cfg["tracking"][j]
            t["parent"] = p
            t["child"] = c

        self._refresh_tracker_combos()
        self._read_tracking()
        self.set_banner("Auto-bound standard joint trackers (J1-J4) from body markers.", "ok", 4.0)

    def _zero_single_joint(self, j):
        snap = self.dual_tracker.snapshot()
        t = self.cfg["tracking"][j]
        p, c = t["parent"], t["child"]
        if p is None or c is None:
            self.set_banner(f"Configure Parent and Child markers for J{j+1} first.", "warn", 3.0)
            return

        rel_l = dual_trk.raw_relative(snap["left"]["markers"], p, c)
        rel_r = dual_trk.raw_relative(snap["right"]["markers"], p, c)

        calibrated = False
        if rel_l is not None:
            t["offset_left"] = float(rel_l)
            t["offset"] = float(rel_l)
            calibrated = True
        if rel_r is not None:
            t["offset_right"] = float(rel_r)
            if rel_l is None:
                t["offset"] = float(rel_r)
            calibrated = True

        if calibrated:
            self.meas[j] = None
            self._refresh_offsets()
            self._save()
            self.set_banner(f"Zero offset calibrated for J{j+1}: {t['offset']:+.1f}°", "ok", 4.0)
        else:
            self.set_banner(f"Markers M{p} & M{c} for J{j+1} are not visible in cameras.", "warn", 4.0)

    # ------------------------------------------------------------- Tab: Servo Pins
    def _build_tab_pins(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="Servo Pins")
        ttk.Label(tab, text="Logical Servo", font=("TkDefaultFont", 9, "bold")).grid(row=0, column=0, padx=4, pady=3)
        ttk.Label(tab, text="Arduino Pin", font=("TkDefaultFont", 9, "bold")).grid(row=0, column=1, padx=4, pady=3)

        for i in range(7):
            ttk.Label(tab, text=SERVO_LABELS[i], width=22).grid(row=i + 1, column=0, sticky="w", padx=4, pady=2)
            ttk.Combobox(tab, textvariable=self.pin_vars[i], values=[str(p) for p in ALLOWED_PINS],
                         width=5, state="readonly").grid(row=i + 1, column=1, padx=4)

        ttk.Button(tab, text="Apply Pins to Uno", command=self.apply_pins).grid(row=8, column=0, sticky="w", padx=4, pady=8)
        ttk.Label(tab, foreground="#57606a", wraplength=480, justify="left", text=(
            "Allowed pins: 2-13 and 16-19 (A2-A5). Pins 0/1 are USB serial, A0/A1 are FSR inputs.\n"
            "Each servo must have a unique pin. Shoulder B mirrors Shoulder A automatically."
        )).grid(row=9, column=0, columnspan=2, sticky="w", padx=4)
        return tab


    # ============================================================= Webcam AI Point & Limb Tracking
    def _toggle_webcam_add_mode(self):
        val = not self.webcam_track_mode.get()
        self.webcam_track_mode.set(val)
        if val:
            self.btn_webcam_add_point.config(bg="#238636", text="🎯 Click Joint on Video (Active)")
            self.lbl_overhead_cam.config(cursor="crosshair")
            self.lbl_side_cam.config(cursor="crosshair")
            self.set_banner("Click on any joint or feature in the Overhead or Side video to place an AI tracker.", "ok", 5.0)
        else:
            self.btn_webcam_add_point.config(bg="#1f6feb", text="🎯 Click Cam to Add Tracker")
            self.lbl_overhead_cam.config(cursor="")
            self.lbl_side_cam.config(cursor="")
            self.set_banner("Webcam tracker placement deactivated.", "info", 2.0)

    def _on_webcam_clicked(self, cam_name, x, y):
        self._prompt_add_webcam_tracker(cam_name, x, y)

    def _prompt_add_webcam_tracker(self, cam_name, x, y):
        if cam_name == "side":
            roles = [
                ("J2 Shoulder Pivot", "sh_pivot"),
                ("J3 Elbow Pivot", "el_pivot"),
                ("J4 Wrist Pivot", "wr_pivot"),
                ("Gripper Tip", "grip_tip"),
                ("Custom Point", "custom"),
            ]
        else:
            roles = [
                ("J1 Base Center", "base_center"),
                ("J1 Turret Pivot", "turret_ref"),
                ("Workspace Target", "target"),
                ("Custom Point", "custom"),
            ]

        top = tk.Toplevel(self)
        top.title(f"Add AI Tracker - {cam_name.capitalize()} Cam")
        top.geometry("340x260")
        top.transient(self)
        top.grab_set()

        ttk.Label(top, text=f"Place Tracker on {cam_name.capitalize()} Cam", font=("TkDefaultFont", 10, "bold")).pack(pady=(10, 4))
        ttk.Label(top, text=f"Clicked Coordinates: ({x}, {y})", foreground="gray").pack()

        f_role = ttk.Frame(top)
        f_role.pack(fill="x", padx=16, pady=8)
        ttk.Label(f_role, text="Joint / Role:").pack(side="left", padx=4)
        role_var = tk.StringVar(value=roles[0][0])
        role_cb = ttk.Combobox(f_role, textvariable=role_var, values=[r[0] for r in roles], state="readonly", width=18)
        role_cb.pack(side="left", padx=4)

        f_name = ttk.Frame(top)
        f_name.pack(fill="x", padx=16, pady=4)
        ttk.Label(f_name, text="Tracker Name:").pack(side="left", padx=4)
        name_var = tk.StringVar(value=roles[0][0])
        name_entry = ttk.Entry(f_name, textvariable=name_var, width=18)
        name_entry.pack(side="left", padx=4)

        def _on_role_sel(ev):
            name_var.set(role_var.get())
        role_cb.bind("<<ComboboxSelected>>", _on_role_sel)

        snap_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="⚡ AI Auto-Snap to Nearest Bolt/Corner", variable=snap_var).pack(pady=4)

        btn_row = ttk.Frame(top)
        btn_row.pack(fill="x", padx=16, pady=12)

        def _confirm():
            name = name_var.get().strip() or "Tracker"
            role_choice = role_var.get()
            pt_id = "custom_" + str(int(time.time()))
            for r in roles:
                if r[0] == role_choice:
                    pt_id = r[1]
                    break

            worker = self.dual_tracker.overhead_cam if cam_name == "overhead" else self.dual_tracker.side_cam
            sx, sy = worker.add_point_tracker(pt_id, name, x, y, snap_feature=snap_var.get())

            cfg_key = f"{cam_name}_points"
            pts_list = self.webcam_trackers.setdefault(cfg_key, [])
            pts_list = [p for p in pts_list if p.get("id") != pt_id]
            pts_list.append({"id": pt_id, "name": name, "x": sx, "y": sy})
            self.webcam_trackers[cfg_key] = pts_list
            self.cfg["webcam_trackers"] = self.webcam_trackers
            self._save()

            top.destroy()
            self.set_banner(f"AI Tracker '{name}' locked onto ({sx:.0f}, {sy:.0f}) on {cam_name} camera.", "ok", 4.0)

        ttk.Button(btn_row, text="Add Tracker", command=_confirm).pack(side="right", padx=4)
        ttk.Button(btn_row, text="Cancel", command=top.destroy).pack(side="right", padx=4)

    def _auto_link_webcam_limbs(self):
        side_worker = self.dual_tracker.side_cam
        overhead_worker = self.dual_tracker.overhead_cam
        side_pts = side_worker.point_trackers
        linked = []

        if "sh_pivot" in side_pts and "el_pivot" in side_pts:
            side_worker.add_limb_tracker("upper_arm", "Upper Arm", "sh_pivot", "el_pivot", color=(0, 255, 0))
            linked.append("Upper Arm")
        if "el_pivot" in side_pts and "wr_pivot" in side_pts:
            side_worker.add_limb_tracker("forearm", "Forearm", "el_pivot", "wr_pivot", color=(0, 255, 255))
            linked.append("Forearm")
        if "wr_pivot" in side_pts and "grip_tip" in side_pts:
            side_worker.add_limb_tracker("wrist", "Wrist", "wr_pivot", "grip_tip", color=(255, 0, 255))
            linked.append("Wrist")

        overhead_pts = overhead_worker.point_trackers
        if "base_center" in overhead_pts and "turret_ref" in overhead_pts:
            overhead_worker.add_limb_tracker("base_yaw", "Base Turret", "base_center", "turret_ref", color=(0, 255, 0))
            linked.append("Base Turret")

        if linked:
            self.set_banner(f"Auto-linked limbs on webcam: {', '.join(linked)}! Live vector angles active.", "ok", 5.0)
        else:
            self.set_banner("Please add at least 2 joint points (e.g. Shoulder and Elbow) on the webcam first.", "warn", 5.0)

    def _reset_webcam_trackers(self):
        for pt in self.webcam_trackers.get("overhead_points", []):
            self.dual_tracker.overhead_cam.add_point_tracker(pt["id"], pt["name"], pt["x"], pt["y"])
        for pt in self.webcam_trackers.get("side_points", []):
            self.dual_tracker.side_cam.add_point_tracker(pt["id"], pt["name"], pt["x"], pt["y"])
        self._auto_link_webcam_limbs()
        self.set_banner("Webcam AI point trackers re-anchored at reference coordinates.", "ok", 3.0)

    def _clear_webcam_trackers(self):
        self.dual_tracker.overhead_cam.clear_point_trackers()
        self.dual_tracker.overhead_cam.clear_limb_trackers()
        self.dual_tracker.side_cam.clear_point_trackers()
        self.dual_tracker.side_cam.clear_limb_trackers()
        self.webcam_trackers = {"overhead_points": [], "overhead_limbs": [], "side_points": [], "side_limbs": []}
        self.cfg["webcam_trackers"] = self.webcam_trackers
        self._save()
        self.set_banner("Cleared all webcam point & limb trackers.", "info", 3.0)

    # ============================================================= Camera Management
    def refresh_camera_lists(self, interactive=False):
        cams = dual_trk.scan_available_cameras(max_tested=8, exclude_integrated=False)
        values = [str(c) for c in cams]
        self.cam_overhead_combo["values"] = values if values else ["0", "1", "2"]
        self.cam_side_combo["values"] = values if values else ["0", "1", "2"]

        if cams:
            if len(cams) >= 2:
                self.cam_overhead_var.set(str(cams[0]))
                self.cam_side_var.set(str(cams[1]))
            elif len(cams) == 1:
                self.cam_overhead_var.set(str(cams[0]))
            msg = f"Found {len(cams)} working camera(s) at indices: {cams}"
            self.set_banner(msg, "ok", 4.0)
            if interactive:
                messagebox.showinfo("Cameras Detected", f"{msg}\n\nAssigned: Overhead Cam (Index {self.cam_overhead_var.get()}) & Side Cam (Index {self.cam_side_var.get()}).")
        else:
            msg = "No working cameras detected. Check USB connections and Windows Camera Privacy permissions."
            self.set_banner(msg, "warn", 6.0)
            if interactive:
                messagebox.showwarning("No Cameras Found",
                                       f"{msg}\n\nTroubleshooting:\n"
                                       "1. Ensure both USB webcams are firmly plugged in.\n"
                                       "2. Open Windows Settings -> Privacy & security -> Camera and ensure 'Let desktop apps access your camera' is ON.\n"
                                       "3. Run 'py find_cameras.py' from terminal for visual diagnosis.")

    def toggle_dual_cameras(self):
        if self.dual_tracker.is_running():
            self.dual_tracker.stop()
            self.btn_toggle_cams.config(text="Start Dual Cameras")
            self.cam_overhead_info.set("Overhead: Off")
            self.cam_side_info.set("Side: Off")
            self.lbl_overhead_cam.config(image=self.blank_img, text="Overhead Camera Off")
            self.lbl_side_cam.config(image=self.blank_img, text="Side Camera Off")
            self.img_left = None
            self.img_right = None
            self.set_banner("Cameras stopped.", "info")
            return

        try:
            o_idx = int(self.cam_overhead_var.get())
            s_idx = int(self.cam_side_var.get())
        except ValueError:
            self.set_banner("Invalid camera index numbers.", "error")
            return

        self.cfg["camera_overhead_index"] = o_idx
        self.cfg["camera_side_index"] = s_idx
        self.cfg["camera_left_index"] = o_idx
        self.cfg["camera_right_index"] = s_idx
        self._save()

        ok, err = self.dual_tracker.start(overhead_index=o_idx, side_index=s_idx)
        if not ok:
            self.set_banner(f"Could not start cameras: {err}", "error", 6.0)
            return

        # Restore saved webcam trackers if any
        if hasattr(self, "webcam_trackers"):
            for pt in self.webcam_trackers.get("overhead_points", []):
                self.dual_tracker.overhead_cam.add_point_tracker(pt["id"], pt["name"], pt["x"], pt["y"])
            for limb in self.webcam_trackers.get("overhead_limbs", []):
                self.dual_tracker.overhead_cam.add_limb_tracker(limb["id"], limb["name"], limb["p1"], limb["p2"])

            for pt in self.webcam_trackers.get("side_points", []):
                self.dual_tracker.side_cam.add_point_tracker(pt["id"], pt["name"], pt["x"], pt["y"])
            for limb in self.webcam_trackers.get("side_limbs", []):
                self.dual_tracker.side_cam.add_limb_tracker(limb["id"], limb["name"], limb["p1"], limb["p2"])

        self.btn_toggle_cams.config(text="Stop Dual Cameras")
        self.lbl_overhead_cam.config(text="Connecting...")
        self.lbl_side_cam.config(text="Connecting...")
        self.set_banner("Dual webcams started (Overhead & Side). Click video to add AI joint trackers.", "ok")

    def calibrate_zero(self):
        snap = self.dual_tracker.snapshot()
        overhead_snap = snap["overhead"]
        side_snap = snap["side"]
        done = []

        for j, t in enumerate(self.cfg["tracking"]):
            calibrated = False

            # Check webcam limb tracking zero calibration
            wb_ang = self.dual_tracker.get_webcam_joint_angle(j, invert=False, offset=0.0)
            if wb_ang is not None:
                if j == 0:
                    t["offset_overhead"] = float(wb_ang)
                    t["offset_left"] = float(wb_ang)
                else:
                    t["offset_side"] = float(wb_ang)
                    t["offset_right"] = float(wb_ang)
                t["offset"] = float(wb_ang)
                calibrated = True

            # Check ArUco marker zero calibration
            p, c = t.get("parent"), t.get("child")
            if p is not None and c is not None:
                rel_o = dual_trk.raw_relative(overhead_snap["markers"], p, c)
                rel_s = dual_trk.raw_relative(side_snap["markers"], p, c)
                if rel_o is not None:
                    t["offset_overhead"] = float(rel_o)
                    t["offset_left"] = float(rel_o)
                    t["offset"] = float(rel_o)
                    calibrated = True
                if rel_s is not None:
                    t["offset_side"] = float(rel_s)
                    t["offset_right"] = float(rel_s)
                    if rel_o is None:
                        t["offset"] = float(rel_s)
                    calibrated = True

            if calibrated:
                self.meas[j] = None
                done.append(f"J{j + 1}")

        self._refresh_offsets()
        self._save()
        if done:
            self.set_banner(f"Zero calibration stored for {', '.join(done)} across Overhead & Side cams.", "ok", 5.0)
        else:
            self.set_banner("No trackers visible. Add webcam point trackers or ArUco markers first.", "warn", 5.0)

    def save_markers(self):
        path = filedialog.asksaveasfilename(defaultextension=".png", initialfile="arm_markers.png",
                                            title="Save Printable ArUco Marker Sheet")
        if not path:
            return
        try:
            dual_trk.make_marker_sheet(path, ids=range(0, 6))
            self.set_banner(f"Marker sheet saved to {path}.", "ok", 6.0)
        except Exception as err:
            self.set_banner(f"Failed to generate marker sheet: {err}", "error", 6.0)

    # ============================================================= Ollama AI Integration
    def check_ollama_status(self):
        url = self.ollama_url_var.get().strip()
        model = self.ollama_model_var.get().strip()
        self.ai_bridge.ollama_url = url
        self.ai_bridge.model = model

        def _bg_check():
            ok, models, has_model, err = self.ai_bridge.check_connection()
            if ok and has_model:
                self.ai_status_text.set(f"● Ollama Connected ({model})")
                self.ai_badge.config(bg="#1a472a", fg="#7ee787")
            elif ok:
                self.ai_status_text.set(f"▲ Model '{model}' not found in Ollama")
                self.ai_badge.config(bg="#593a00", fg="#f0883e")
            else:
                self.ai_status_text.set("○ Ollama Offline (Click to check)")
                self.ai_badge.config(bg="#3b1b1b", fg="#f85149")

        threading.Thread(target=_bg_check, daemon=True).start()

    def send_ai_query(self):
        prompt = self.ai_prompt_var.get().strip()
        if not prompt:
            return
        if self.ai_querying:
            return

        self.ai_querying = True
        self.btn_send_ai.config(state="disabled")
        self.ai_text.insert("end", f"\n> USER: {prompt}\n[AI thinking via qwen2.5:3b...]\n")
        self.ai_text.see("end")

        current_angles = list(self.actual)
        limits = [self._limits(j) for j in range(6)]
        now = time.time()
        tracking_status = [self._fresh(j, now) for j in range(6)]
        fsr_vals = (int(self.fsr_vars[0].get()), int(self.fsr_vars[1].get()))

        def _worker():
            ok, res, err = self.ai_bridge.plan_instruction(
                prompt, current_angles, limits, tracking_status, fsr_vals
            )
            self.after(0, lambda: self._handle_ai_response(ok, res, err))

        threading.Thread(target=_worker, daemon=True).start()

    def _handle_ai_response(self, ok, res, err):
        self.ai_querying = False
        self.btn_send_ai.config(state="normal")
        self.ai_prompt_var.set("")

        if not ok:
            self.ai_text.insert("end", f"Error from Ollama: {err}\n")
            self.ai_text.see("end")
            self.set_banner(f"AI Plan Error: {err}", "error", 5.0)
            return

        thought = res.get("thought", "No thought specified.")
        explanation = res.get("explanation", "Plan generated.")
        plan = res.get("plan", [])
        warnings = res.get("warnings", [])

        self.ai_text.insert("end", f"THOUGHT: {thought}\n")
        self.ai_text.insert("end", f"SUMMARY: {explanation}\n")
        for i, step in enumerate(plan):
            self.ai_text.insert("end", f"  Step {i+1}: {step}\n")
        if warnings:
            for w in warnings:
                self.ai_text.insert("end", f"  [Safety Notice]: {w}\n")
        self.ai_text.insert("end", "-" * 40 + "\n")
        self.ai_text.see("end")

        self.active_plan = plan
        self.plan_step_idx = 0
        if plan:
            self.ai_plan_summary.set(f"Ready: {explanation} ({len(plan)} steps)")
            self.run_plan_btn.config(state="normal")
            self.step_plan_btn.config(state="normal")
            self.abort_plan_btn.config(state="normal")
        else:
            self.ai_plan_summary.set("AI did not produce any movement steps.")

    def load_preset(self, preset_key):
        if preset_key not in PRESET_PLANS:
            return
        preset = PRESET_PLANS[preset_key]
        self._handle_ai_response(True, preset, "")

    def start_plan_execution(self):
        if not self.active_plan:
            return
        self.plan_running = True
        self.plan_step_idx = 0
        self.run_plan_btn.config(state="disabled")
        self.step_plan_btn.config(state="disabled")
        self.abort_plan_btn.config(state="normal")
        self._execute_current_step()

    def step_plan_execution(self):
        if not self.active_plan:
            return
        self.plan_running = False
        self._execute_current_step()
        self.plan_step_idx += 1
        if self.plan_step_idx >= len(self.active_plan):
            self.ai_plan_summary.set("Completed all plan steps.")
            self.run_plan_btn.config(state="disabled")
            self.step_plan_btn.config(state="disabled")

    def _execute_current_step(self):
        if self.plan_step_idx >= len(self.active_plan):
            self.plan_running = False
            self.ai_plan_summary.set("AI Plan Finished Successfully.")
            self.run_plan_btn.config(state="disabled")
            self.step_plan_btn.config(state="disabled")
            return

        step = self.active_plan[self.plan_step_idx]
        self.plan_step_start = time.time()
        self.ai_plan_summary.set(f"Executing Step {self.plan_step_idx+1}/{len(self.active_plan)}: {step}")

        action = step.get("action")
        if action == "move":
            joints = step.get("joints", {})
            for j_name, val in joints.items():
                j_idx = int(j_name[1]) - 1
                self.set_target(j_idx, float(val))
            if "speed" in step:
                self.speed.set(int(step["speed"]))
                self._speed_changed()

        elif action == "gripper":
            angle = float(step.get("angle", 0.0))
            self.set_target(5, angle)

        elif action == "stop":
            self.stop_all()

    def abort_plan(self):
        self.plan_running = False
        self.active_plan = []
        self.plan_step_idx = 0
        self.stop_all()
        self.ai_plan_summary.set("Plan aborted by operator.")
        self.run_plan_btn.config(state="disabled")
        self.step_plan_btn.config(state="disabled")
        self.abort_plan_btn.config(state="disabled")
        self.set_banner("AI plan aborted.", "warn")

    # ============================================================= Motion & Serial
    def _limits(self, j):
        lo, hi = self.cfg["limits"][j]
        return float(lo), float(hi)

    def _margin(self):
        try:
            return max(0.0, float(self.margin_var.get()))
        except ValueError:
            return 4.0

    def _fresh(self, j, now):
        return self.meas[j] is not None and (now - self.meas_t[j]) <= FRESH_S

    def set_banner(self, text, level="info", seconds=3.0):
        colors = {"info": "#adbac7", "ok": "#57ab5a", "warn": "#c69026", "error": "#e5534b"}
        self.banner.set(text)
        self.banner_label.config(foreground=colors.get(level, "#adbac7"))
        self.banner_until = time.time() + seconds

    def _refresh_offsets(self):
        for j, t in enumerate(self.cfg["tracking"]):
            self.offset_text[j].set(f"{t['offset']:+.1f}°")

    def _save(self):
        self.cfg["body_markers"] = self.body_markers
        save_config(self.cfg, self.config_path)

    def _read_limits(self):
        for j in range(6):
            try:
                lo, hi = float(self.lo_vars[j].get()), float(self.hi_vars[j].get())
            except ValueError:
                continue
            if lo < hi:
                self.cfg["limits"][j] = [lo, hi]
        self.cfg["margin"] = self._margin()
        self.cfg["limits_on"] = bool(self.limits_on.get())
        self._save()

    def _read_tracking(self):
        for j in range(6):
            t = self.cfg["tracking"][j]
            t["camera"] = self.trk_cam_vars[j].get()
            t["parent"] = parse_marker_id(self.parent_vars[j].get())
            t["child"] = parse_marker_id(self.child_vars[j].get())
            t["invert"] = bool(self.invert_vars[j].get())
            self.meas[j] = None
        self._save()

    def _speed_changed(self):
        self.cfg["speed"] = int(self.speed.get())
        self._send(f"V,{int(self.speed.get())}")

    def apply_pins(self):
        try:
            pins = [int(v.get()) for v in self.pin_vars]
        except ValueError:
            self.set_banner("Invalid pin number entered.", "error")
            return False
        if any(p not in ALLOWED_PINS for p in pins):
            self.set_banner("Pins must be 2-13 or 16-19 (A0/A1 are FSRs, 0/1 are USB).", "error")
            return False
        if len(set(pins)) != len(pins):
            self.set_banner("Each servo must have a unique pin.", "error")
            return False
        self.cfg["pins"] = pins
        self._save()
        self.held.clear()
        self.seek = [None] * 3
        self._send("Q," + ",".join(str(p) for p in pins))
        self.set_banner("Servo pins applied to Arduino.", "ok", 4.0)
        return True

    def refresh_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_box["values"] = ports
        if ports and not self.port_var.get():
            self.port_var.set(ports[0])

    def toggle_connection(self):
        if self.ser:
            self.disconnect()
            return
        port = self.port_var.get()
        if not port:
            self.set_banner("Select an Uno COM port first.", "warn")
            return
        try:
            self.ser = serial.Serial(port, BAUD, timeout=0.05)
        except serial.SerialException as err:
            self.set_banner(f"Cannot open {port}: {err}", "error", 5.0)
            return

        self.set_banner("Connecting. Uno is resetting (servos limp briefly)...", "warn")
        self.update()
        time.sleep(RESET_WAIT)
        self.ser.reset_input_buffer()
        self.connect_btn.config(text="Disconnect")
        self.status.set(f"Connected to {port}")
        self.push_settings()
        self.set_banner("Connected to Uno.", "ok")

    def push_settings(self):
        self._send(f"V,{int(self.speed.get())}")
        self._send("Q," + ",".join(str(p) for p in self.cfg["pins"]))
        for k in range(3):
            self.pos_dirty[k] = True
        if self.pos_on.get():
            self._flush_positional(force=True)
            self._send("E,1")

    def disconnect(self):
        self.held.clear()
        self.seek = [None] * 3
        if self.ser:
            try:
                self.ser.write(b"S\n")
                self.ser.close()
            except Exception:
                pass
        self.ser = None
        self.cur_dir = [0] * 3
        self.connect_btn.config(text="Connect Uno")
        self.status.set("Disconnected. Servos stopped.")

    def _send(self, text):
        if not self.ser:
            return
        try:
            self.ser.write((text + "\n").encode())
        except Exception as err:
            self.set_banner(f"Serial communication error: {err}", "error", 5.0)
            self.disconnect()

    def _read_serial(self):
        if not self.ser:
            return
        try:
            while self.ser and self.ser.in_waiting:
                line = self.ser.readline().decode(errors="ignore").strip()
                if line:
                    self._handle_serial_line(line)
        except Exception:
            self.disconnect()

    def _handle_serial_line(self, line):
        if line.startswith("F,"):
            parts = line.split(",")
            if len(parts) == 3:
                try:
                    vals = [int(p) for p in parts[1:]]
                    for i, v in enumerate(vals):
                        self.fsr_vars[i].set(str(v))
                        self.fsr_bars[i]["value"] = v
                except ValueError:
                    pass
        elif line == "READY":
            self.set_banner("Uno restarted. Resending configuration...", "warn", 4.0)
            self.push_settings()
        elif line.startswith("ERR"):
            self.set_banner(f"Uno Firmware: {line}", "error", 5.0)
        else:
            self.status.set(line)

    def press(self, j, direction):
        if not self.ser:
            self.set_banner("Connect to Uno first.", "warn")
            return
        now = time.time()
        self.held[j] = direction
        self.seek[j] = None
        fresh = self._fresh(j, now)
        self.armed[j] = fresh
        if not fresh:
            # No camera signal — send motor command directly (open-loop manual jog)
            speed = int(self.speed.get())
            self._send(f"M,{j},{direction},{speed}")
            self.cur_dir[j] = direction
            self.set_banner(f"{JOINT_NAMES[j]}: manual jog (no camera — limits disabled).", "warn")

    def release(self, j):
        if j in self.held:
            del self.held[j]
            self._send(f"M,{j},0")
            self.cur_dir[j] = 0

    def stop_all(self):
        self.held.clear()
        self.seek = [None] * 3
        self.cur_dir = [0] * 3
        self._send("S")
        self.set_banner("All continuous joints stopped.", "info")

    def on_pos_toggle(self):
        self.cfg["pos_on"] = bool(self.pos_on.get())
        self._save()
        if not self.ser:
            self.pos_on.set(False)
            self.set_banner("Connect to Uno first.", "warn")
            return
        for k in range(3):
            self.pos_dirty[k] = True
        self._flush_positional(force=True)
        self._send("E,1" if self.pos_on.get() else "E,0")

    def set_target(self, j, angle):
        lo, hi = self._limits(j)
        if self.limits_on.get():
            angle = clamp(angle, lo, hi)
        self.target[j] = angle

        if j in CONT:
            if not self.ser:
                self.set_banner("Connect to Uno first.", "warn")
                return
            if self._fresh(j, time.time()):
                if self.seek[j] is None:
                    self.armed[j] = True
                self.seek[j] = angle
                self.held.pop(j, None)
            else:
                self.set_banner(f"{JOINT_NAMES[j]} not visible in camera: cannot seek closed-loop target.", "warn")
        else:
            k = j - 3
            self.cmd[k] = clamp(angle, -90.0, 90.0)
            self.pos_dirty[k] = True
            self.slider_vars[k].set(self.cmd[k])
            if not self.pos_on.get():
                self.set_banner("Enable 'Positional Servos ON' to move J4-J6.", "info")

    def go(self, j):
        try:
            angle = float(self.target_text[j].get())
        except ValueError:
            self.set_banner("Enter numeric angle in degrees.", "warn")
            return
        self.set_target(j, angle)

    def _slider_moved(self, j):
        self.set_target(j, float(self.slider_vars[j - 3].get()))
        self.target_text[j].set(f"{self.target[j]:.0f}")

    def _on_drag(self, j, angle):
        self.set_target(j, angle)
        self.target_text[j].set(f"{self.target[j]:.0f}")

    def _on_drag_end(self, j):
        if j in CONT and self.seek[j] is None:
            self.target[j] = self.actual[j]

    # ============================================================= Main Loops
    def tick(self):
        now = time.time()
        self._read_serial()
        self._update_dual_measurements(now)
        self._check_visual_limits(now)   # enforce camera-set soft limits
        self._control_continuous(now)
        self._flush_positional()
        self._update_poses(now)
        self._check_plan_progress(now)

        if self.banner_until and now > self.banner_until:
            self.banner_until = 0.0
            self.banner.set("Ready.")
            self.banner_label.config(foreground="#adbac7")

        self.after(TICK_MS, self.tick)

    def _update_dual_measurements(self, now):
        snap = self.dual_tracker.snapshot()
        overhead_snap = snap["overhead"]
        side_snap = snap["side"]
        overhead_markers = overhead_snap["markers"]
        side_markers = side_snap["markers"]

        self.view.active_markers = set(overhead_markers.keys()) | set(side_markers.keys())

        for j, t in enumerate(self.cfg["tracking"]):
            cam_src = t.get("camera", "auto").lower()
            invert = t.get("invert", False)
            off_overhead = t.get("offset_overhead", t.get("offset_left", t.get("offset", 0.0)))
            off_side = t.get("offset_side", t.get("offset_right", t.get("offset", 0.0)))

            ang = None
            seen_at = 0.0

            # 1. First check if a Webcam AI Limb/Point Tracker is active for this joint
            webcam_ang = self.dual_tracker.get_webcam_joint_angle(
                j, invert=invert, offset=(off_overhead if j == 0 else off_side)
            )
            if webcam_ang is not None:
                ang = webcam_ang
                seen_at = now

            # 2. If no webcam limb angle, fallback to ArUco markers:
            if ang is None:
                p, c = t.get("parent"), t.get("child")
                if cam_src in ("overhead", "top", "left", "auto"):
                    ang_o = dual_trk.joint_angle(overhead_markers, p, c, invert, off_overhead)
                    if ang_o is not None:
                        ang = ang_o
                        seen_at = min(overhead_markers[p][2], overhead_markers[c][2])

                if ang is None and cam_src in ("side", "right", "auto"):
                    ang_s = dual_trk.joint_angle(side_markers, p, c, invert, off_side)
                    if ang_s is not None:
                        ang = ang_s
                        seen_at = min(side_markers[p][2], side_markers[c][2])

            if ang is None or seen_at <= self.meas_t[j]:
                continue

            if self.meas[j] is None or seen_at - self.meas_t[j] > 1.0:
                smooth, self.vel[j] = ang, 0.0
            else:
                dt = max(1e-3, seen_at - self.meas_t[j])
                smooth = 0.6 * ang + 0.4 * self.meas[j]
                self.vel[j] = 0.5 * self.vel[j] + 0.5 * (smooth - self.meas[j]) / dt

            self.meas[j], self.meas_t[j] = smooth, seen_at

    def _control_continuous(self, now):
        limits_on = bool(self.limits_on.get())
        margin = self._margin()
        speed = int(self.speed.get())

        for j in CONT:
            lo, hi = self._limits(j)
            fresh = self._fresh(j, now)
            angle = self.meas[j] if fresh else None
            held, seek = self.held.get(j, 0), self.seek[j]

            if self.armed[j] and (held or seek is not None) and not fresh and now - self.meas_t[j] > LOST_S:
                # Was armed (had camera) but signal lost mid-motion → safety stop
                direction, pct, note, done = 0, speed, "lost", False
            elif not self.armed[j] and held:
                # Open-loop manual jog (no camera): keep sending at current speed
                direction, pct, note, done = held, speed, "open_loop", False
            else:
                direction, pct, note, done = decide_motion(angle, fresh, self.vel[j], held, seek,
                                                           lo, hi, margin, speed, limits_on, self.k_est[j])

            if note == "lost":
                self.seek[j] = None
                if held or seek is not None:
                    self.set_banner(f"Vision lost {JOINT_NAMES[j]}: safety stop triggered.", "error", 4.0)
                    self.held.pop(j, None)
            elif note.startswith("limit"):
                self.seek[j] = None
                edge = hi if note == "limit_max" else lo
                self.set_banner(f"{JOINT_NAMES[j]} reached limit ({edge:+.0f}°): stopped.", "warn", 2.5)
            elif done:
                self.seek[j] = None

            if direction:
                self._send(f"M,{j},{direction},{pct}")
                self.cur_dir[j] = direction
                if fresh:
                    self.k_est[j] = update_speed_estimate(self.k_est[j], self.vel[j], pct)
            elif self.cur_dir[j]:
                self._send(f"M,{j},0")
                self.cur_dir[j] = 0

            if not (self.held.get(j) or self.seek[j] is not None):
                self.armed[j] = False

    def _flush_positional(self, force=False):
        if not self.ser or not (self.pos_on.get() or force):
            return
        for k in range(3):
            if self.pos_dirty[k]:
                self._send(f"A,{k},{servo_angle(self.cmd[k])}")
                self.pos_dirty[k] = False

    def _update_poses(self, now):
        dragging = self.view.drag["j"] if self.view.drag else None
        for j in range(6):
            fresh = self._fresh(j, now)
            if fresh:
                self.actual[j] = self.meas[j]
            elif j in POS:
                self.actual[j] = self.cmd[j - 3]

            if j in CONT and self.seek[j] is None and j not in self.held and j != dragging:
                self.target[j] = self.actual[j]
            if j in POS and j != dragging:
                self.target[j] = self.cmd[j - 3]

        self.view.tracked = [self._fresh(j, now) for j in range(6)]

    def _check_plan_progress(self, now):
        if not self.plan_running or self.plan_step_idx >= len(self.active_plan):
            return

        step = self.active_plan[self.plan_step_idx]
        wait_time = float(step.get("wait", 1.0))
        elapsed = now - self.plan_step_start

        # Check if all continuous seeks are finished
        all_arrived = all(self.seek[j] is None for j in CONT)

        if elapsed >= wait_time and all_arrived:
            self.plan_step_idx += 1
            if self.plan_step_idx < len(self.active_plan):
                self._execute_current_step()
            else:
                self.plan_running = False
                self.ai_plan_summary.set("AI Plan Completed Successfully!")
                self.run_plan_btn.config(state="disabled")
                self.step_plan_btn.config(state="disabled")

    def view_tick(self):
        now = time.time()
        for j in range(6):
            fresh = self._fresh(j, now)
            if fresh:
                lo, hi = self._limits(j)
                self.meas_text[j].set(f"{self.meas[j]:+.1f}°")
                inside = lo <= self.meas[j] <= hi or not self.limits_on.get()
                self.dots[j].config(foreground="#2ecc71" if inside else "#e5534b")
            else:
                self.meas_text[j].set("-" if j in CONT else f"{self.cmd[j - 3]:+.0f}°")
                self.dots[j].config(foreground="#7d8590")

        self.view.render()
        self.after(VIEW_MS, self.view_tick)

    # ─────────────────────────── Visual Joint Limits Panel ───────────────────
    def _build_visual_limits_panel(self, parent):
        """Panel under the webcam feeds to set per-joint limits from live camera angles."""
        lim_frame = ttk.LabelFrame(parent, text="📐 Visual Joint Limits  (set soft-stop angles from live camera readings)")
        lim_frame.pack(side="top", fill="x", padx=6, pady=(2, 5))

        # Header row
        hdr = ttk.Frame(lim_frame)
        hdr.pack(side="top", fill="x", padx=4, pady=(2, 0))
        for col, (txt, w) in enumerate([
            ("Joint", 10), ("Enabled", 7), ("Min °", 7), ("Set Min", 9),
            ("Max °", 7), ("Set Max", 9), ("AutoStop", 8), ("Live Angle", 11), ("Status", 14)
        ]):
            ttk.Label(hdr, text=txt, width=w, font=("TkDefaultFont", 8, "bold"),
                      anchor="center").grid(row=0, column=col, padx=2)

        # One row per joint
        self._lim_status_labels = []
        for j in range(6):
            cl = self.cam_limits[j]
            row = ttk.Frame(lim_frame)
            row.pack(side="top", fill="x", padx=4, pady=1)

            ttk.Label(row, text=JOINT_NAMES[j], width=10).grid(row=0, column=0, padx=2)

            ttk.Checkbutton(row, variable=cl["enabled"]).grid(row=0, column=1, padx=2)

            ttk.Entry(row, textvariable=cl["min"], width=7).grid(row=0, column=2, padx=2)

            ttk.Button(row, text="📷 Set Min", width=9,
                       command=lambda j=j: self._cam_limit_set(j, "min")
                       ).grid(row=0, column=3, padx=2)

            ttk.Entry(row, textvariable=cl["max"], width=7).grid(row=0, column=4, padx=2)

            ttk.Button(row, text="📷 Set Max", width=9,
                       command=lambda j=j: self._cam_limit_set(j, "max")
                       ).grid(row=0, column=5, padx=2)

            ttk.Checkbutton(row, variable=cl["auto_stop"],
                            text="auto").grid(row=0, column=6, padx=2)

            ttk.Label(row, textvariable=self.meas_text[j], width=11,
                      font=("Consolas", 9, "bold")).grid(row=0, column=7, padx=2)

            lbl_status = ttk.Label(row, textvariable=cl["status"], width=14)
            lbl_status.grid(row=0, column=8, padx=2)
            self._lim_status_labels.append(lbl_status)

        # Bottom row: quick actions
        bot = ttk.Frame(lim_frame)
        bot.pack(side="top", fill="x", padx=6, pady=(2, 4))
        ttk.Button(bot, text="Set ALL Min from Camera",
                   command=lambda: [self._cam_limit_set(j, "min") for j in range(6)]
                   ).pack(side="left", padx=4)
        ttk.Button(bot, text="Set ALL Max from Camera",
                   command=lambda: [self._cam_limit_set(j, "max") for j in range(6)]
                   ).pack(side="left", padx=4)
        ttk.Button(bot, text="Reset to Config Defaults",
                   command=self._cam_limits_reset).pack(side="left", padx=4)
        ttk.Label(bot,
                  text="Tip: Move arm to its physical min/max position, then click Set Min/Max",
                  foreground="#57606a", font=("TkDefaultFont", 8, "italic")).pack(side="left", padx=8)

    def _cam_limit_set(self, joint_idx, which):
        """Record the current measured angle as the visual min or max for a joint."""
        ang = self.meas[joint_idx]
        if ang is None:
            self.set_banner(
                f"No live angle for {JOINT_NAMES[joint_idx]} — ensure camera tracking is active.",
                "warn", 3.0)
            return
        cl = self.cam_limits[joint_idx]
        cl[which].set(round(float(ang), 1))
        cl["enabled"].set(True)
        self.set_banner(
            f"{JOINT_NAMES[joint_idx]} {which} limit set to {ang:+.1f}°", "info", 2.5)

    def _cam_limits_reset(self):
        """Reset all visual limits back to the config file defaults."""
        for j in range(6):
            lo, hi = self.cfg["limits"][j]
            self.cam_limits[j]["min"].set(float(lo))
            self.cam_limits[j]["max"].set(float(hi))
            self.cam_limits[j]["enabled"].set(False)
            self.cam_limits[j]["status"].set("—")
        self.set_banner("Visual limits reset to config defaults.", "info", 2.5)

    def _check_visual_limits(self, now):
        """Called from tick(). Checks live angles against visual limits and stops arm if breached."""
        for j in range(6):
            cl = self.cam_limits[j]
            if not cl["enabled"].get():
                cl["status"].set("off")
                continue
            ang = self.meas[j]
            if ang is None:
                cl["status"].set("no signal")
                continue
            lo = cl["min"].get()
            hi = cl["max"].get()
            margin = 3.0  # °  warning zone before hard stop
            if ang <= lo:
                cl["status"].set(f"⛔ AT MIN ({lo:+.0f}°)")
                if cl["auto_stop"].get() and j in CONT:
                    self._send(f"M,{j},0")
                    self.held.pop(j, None)
            elif ang < lo + margin:
                cl["status"].set(f"⚠ Near MIN ({lo:+.0f}°)")
            elif ang >= hi:
                cl["status"].set(f"⛔ AT MAX ({hi:+.0f}°)")
                if cl["auto_stop"].get() and j in CONT:
                    self._send(f"M,{j},0")
                    self.held.pop(j, None)
            elif ang > hi - margin:
                cl["status"].set(f"⚠ Near MAX ({hi:+.0f}°)")
            else:
                pct = int(100 * (ang - lo) / max(0.1, hi - lo))
                cl["status"].set(f"OK  {pct}%  [{lo:+.0f}…{hi:+.0f}]")

    def cam_tick(self):
        snap = self.dual_tracker.snapshot()
        overhead_snap = snap["overhead"]
        side_snap = snap["side"]

        # Update Overhead Camera (Webcam 1)
        if overhead_snap["running"] and overhead_snap["png"] and overhead_snap["frame"] != self.last_frame_left:
            self.last_frame_left = overhead_snap["frame"]
            try:
                self.img_left = tk.PhotoImage(data=overhead_snap["png"])
                self.lbl_overhead_cam.config(image=self.img_left, text="")
            except tk.TclError:
                pass
        if overhead_snap["running"]:
            o_ids = sorted(overhead_snap["markers"])
            o_pts = len(overhead_snap.get("points", {}))
            pts_str = f" | {o_pts} Point Trk" if o_pts else ""
            self.cam_overhead_info.set(f"Overhead: {overhead_snap['fps']:.0f} FPS | Markers: {o_ids if o_ids else 'none'}{pts_str}")
        elif overhead_snap["error"]:
            self.cam_overhead_info.set(f"Overhead Error: {overhead_snap['error']}")

        # Update Side Camera (Webcam 2)
        if side_snap["running"] and side_snap["png"] and side_snap["frame"] != self.last_frame_right:
            self.last_frame_right = side_snap["frame"]
            try:
                self.img_right = tk.PhotoImage(data=side_snap["png"])
                self.lbl_side_cam.config(image=self.img_right, text="")
            except tk.TclError:
                pass
        if side_snap["running"]:
            s_ids = sorted(side_snap["markers"])
            s_pts = len(side_snap.get("points", {}))
            pts_str = f" | {s_pts} Point Trk" if s_pts else ""
            self.cam_side_info.set(f"Side: {side_snap['fps']:.0f} FPS | Markers: {s_ids if s_ids else 'none'}{pts_str}")
        elif side_snap["error"]:
            self.cam_side_info.set(f"Side Error: {side_snap['error']}")

        # Periodically refresh marker vision status in treeview
        self._cam_cycle += 1
        if self._cam_cycle % 12 == 0 and hasattr(self, "marker_tree"):
            for m in self.body_markers:
                mid_str = str(m["id"])
                if self.marker_tree.exists(mid_str):
                    seen = m["id"] in self.view.active_markers
                    vals = list(self.marker_tree.item(mid_str, "values"))
                    if len(vals) >= 6:
                        new_status = "● Seen" if seen else "○ Lost"
                        if vals[5] != new_status:
                            vals[5] = new_status
                            self.marker_tree.item(mid_str, values=vals)

        self.after(CAM_MS, self.cam_tick)

    def on_close(self):
        self._read_limits()
        self._read_tracking()
        self.dual_tracker.stop()
        self.disconnect()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
