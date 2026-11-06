#!/usr/bin/env python3
"""NEXUS Arm v4: 3D arm you can drag, webcam tracking of the real arm, limits, remappable pins.

Setup (Windows, PowerShell):
    py -m pip install pyserial opencv-python numpy
Run:
    py nexus_arm_v4.py
Upload nexus_arm_v4_fw/nexus_arm_v4_fw.ino to the Uno first, and close the Serial Monitor.

How it fits together
    * The 3D arm shows the REAL arm: its joint angles come from the webcam.
      The orange outline is where you want it to be. Drag a numbered handle to move a joint.
    * Joints J1-J3 are continuous-rotation servos. They have no feedback, so the webcam
      measures them. A drag, or the Go button, makes the joint turn until the webcam sees the
      target angle. Hold Left / Right to move by hand.
    * Limits are checked against the webcam angle. When a joint gets close to a limit it
      slows down and stops, and only moves back inward.
    * J4-J6 are positional servos (angle 0-180 on the servo, shown as -90 to +90).
"""
import json
import os
import time
import tkinter as tk
from tkinter import ttk, filedialog

import serial
import serial.tools.list_ports

import arm_tracker as trk_mod
from arm_control import clamp, decide_motion, servo_angle, update_speed_estimate, DEG_PER_PCT
from arm_model import ArmView, JOINT_NAMES

BAUD = 115200
RESET_WAIT = 2.0                 # the Uno resets when the port opens
TICK_MS, VIEW_MS, CAM_MS = 30, 50, 66
FRESH_S = 0.35                   # a webcam reading older than this is not trusted
LOST_S = 0.5                     # moving with a tracked joint and no reading for this long: stop
CONT = (0, 1, 2)                 # continuous joints: J1 base, J2 shoulder, J3 elbow
POS = (3, 4, 5)                  # positional joints: J4 wrist, J5 roll, J6 gripper
ALLOWED_PINS = list(range(2, 14)) + [16, 17, 18, 19]       # A0 and A1 are the FSRs
SERVO_LABELS = ["J1 base", "J2 shoulder A", "J2 shoulder B (mirror)", "J3 elbow",
                "J4 wrist", "J5 roll", "J6 gripper"]
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arm_config.json")

DEFAULT_CONFIG = {
    "pins": [3, 4, 5, 6, 7, 8, 9],
    "limits": [[-90, 90], [-60, 90], [-90, 90], [-90, 90], [-90, 90], [-90, 90]],
    "tracking": [
        {"parent": 0, "child": 5, "invert": False, "offset": 0.0},     # J1: needs a view from above
        {"parent": 0, "child": 1, "invert": False, "offset": 0.0},     # J2 shoulder
        {"parent": 1, "child": 2, "invert": False, "offset": 0.0},     # J3 elbow
        {"parent": 2, "child": 3, "invert": False, "offset": 0.0},     # J4 wrist
        {"parent": None, "child": None, "invert": False, "offset": 0.0},
        {"parent": None, "child": None, "invert": False, "offset": 0.0},
    ],
    "camera_index": 0, "margin": 4.0, "limits_on": True, "speed": 50, "pos_on": False,
}


def load_config(path=CONFIG_PATH):
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    try:
        with open(path) as fh:
            saved = json.load(fh)
        for key, value in saved.items():
            if key in cfg:
                cfg[key] = value
    except (OSError, ValueError):
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


class App(tk.Tk):
    def __init__(self, config_path=CONFIG_PATH):
        super().__init__()
        self.title("NEXUS Arm v4  -  3D view, webcam tracking, limits")
        self.config_path = config_path
        self.cfg = load_config(config_path)
        self.ser = None
        self.tracker = trk_mod.ArmTracker()
        self.snap = self.tracker.snapshot()
        self.cam_img = None
        self.last_cam_frame = -1

        # joint state
        self.actual = [0.0] * 6            # what the webcam sees (or the last command)
        self.target = [0.0] * 6            # where you want the joint
        self.meas = [None] * 6             # smoothed webcam angle
        self.meas_t = [0.0] * 6            # when it was observed
        self.vel = [0.0] * 6               # degrees per second
        self.cmd = [0.0] * 3               # commanded angle of the positional joints
        self.pos_dirty = [True] * 3
        self.held = {}                     # continuous joint -> direction while a button is held
        self.seek = [None] * 3             # closed-loop target of each continuous joint
        self.armed = [False] * 3           # webcam was tracking this joint when the move started
        self.cur_dir = [0] * 3
        self.k_est = [DEG_PER_PCT] * 3
        self.banner_until = 0.0

        # tk variables
        c = self.cfg
        self.port_var = tk.StringVar()
        self.speed = tk.IntVar(value=c["speed"])
        self.pos_on = tk.BooleanVar(value=c["pos_on"])
        self.limits_on = tk.BooleanVar(value=c["limits_on"])
        self.margin_var = tk.StringVar(value=str(c["margin"]))
        self.cam_index = tk.StringVar(value=str(c["camera_index"]))
        self.banner = tk.StringVar(value="Ready. Connect the Uno, then start the camera.")
        self.status = tk.StringVar(value="Not connected")
        self.lo_vars = [tk.StringVar(value=str(c["limits"][j][0])) for j in range(6)]
        self.hi_vars = [tk.StringVar(value=str(c["limits"][j][1])) for j in range(6)]
        self.parent_vars = [tk.StringVar(value="" if t["parent"] is None else str(t["parent"])) for t in c["tracking"]]
        self.child_vars = [tk.StringVar(value="" if t["child"] is None else str(t["child"])) for t in c["tracking"]]
        self.invert_vars = [tk.BooleanVar(value=t["invert"]) for t in c["tracking"]]
        self.offset_text = [tk.StringVar() for _ in range(6)]
        self.meas_text = [tk.StringVar(value="-") for _ in range(6)]
        self.target_text = [tk.StringVar(value="0") for _ in range(6)]
        self.pin_vars = [tk.StringVar(value=str(p)) for p in c["pins"]]
        self.fsr_vars = [tk.StringVar(value="0"), tk.StringVar(value="0")]
        self.cam_info = tk.StringVar(value="Camera off")
        self.slider_vars = [tk.DoubleVar(value=0.0) for _ in range(3)]

        self._build_ui()
        self._refresh_offsets()
        self.refresh_ports()
        self.banner_label.config(foreground="#6e7781")
        self.after(TICK_MS, self.tick)
        self.after(VIEW_MS, self.view_tick)
        self.after(CAM_MS, self.cam_tick)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ================================================================== UI
    def _build_ui(self):
        top = ttk.Frame(self)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=6)
        ttk.Label(top, text="Port").pack(side="left", padx=(0, 4))
        self.port_box = ttk.Combobox(top, textvariable=self.port_var, width=14, state="readonly")
        self.port_box.pack(side="left")
        ttk.Button(top, text="Refresh", command=self.refresh_ports).pack(side="left", padx=4)
        self.connect_btn = ttk.Button(top, text="Connect", command=self.toggle_connection)
        self.connect_btn.pack(side="left", padx=4)
        ttk.Label(top, text="   Speed %").pack(side="left")
        tk.Scale(top, from_=5, to=100, orient="horizontal", length=130, variable=self.speed,
                 command=lambda _v: self._speed_changed()).pack(side="left")
        tk.Button(top, text="STOP ALL", bg="#d93025", fg="white", font=("TkDefaultFont", 11, "bold"),
                  command=self.stop_all).pack(side="left", padx=12)
        ttk.Checkbutton(top, text="Positional servos ON", variable=self.pos_on,
                        command=self.on_pos_toggle).pack(side="left", padx=4)

        left = ttk.Frame(self)
        left.grid(row=1, column=0, sticky="n", padx=(8, 4))
        self.view = ArmView(left, 460, 340, on_drag=self._on_drag, on_drag_end=self._on_drag_end)
        self.view.canvas.grid(row=0, column=0)
        self.view.actual, self.view.target = self.actual, self.target
        self.view.tracked = [False] * 6

        fsr = ttk.LabelFrame(left, text="FSR sensors (raw 0-1023)")
        fsr.grid(row=1, column=0, sticky="ew", pady=4)
        self.fsr_bars = []
        for i, pin in enumerate(("A0", "A1")):
            ttk.Label(fsr, text=f"FSR {i + 1} {pin}").grid(row=0, column=i * 3, padx=4)
            bar = ttk.Progressbar(fsr, length=110, maximum=1023)
            bar.grid(row=0, column=i * 3 + 1, padx=2, pady=4)
            ttk.Label(fsr, textvariable=self.fsr_vars[i], width=5).grid(row=0, column=i * 3 + 2)
            self.fsr_bars.append(bar)
        self.banner_label = ttk.Label(left, textvariable=self.banner, wraplength=450,
                                      font=("TkDefaultFont", 10, "bold"))
        self.banner_label.grid(row=2, column=0, sticky="w", pady=2)
        ttk.Label(left, textvariable=self.status, foreground="gray", wraplength=450
                  ).grid(row=3, column=0, sticky="w")

        right = ttk.Frame(self)
        right.grid(row=1, column=1, sticky="n", padx=(4, 8))
        self._build_camera(right).grid(row=0, column=0, sticky="ew")
        nb = ttk.Notebook(right)
        nb.grid(row=1, column=0, sticky="ew", pady=6)
        self._build_joints_tab(nb)
        self._build_tracking_tab(nb)
        self._build_pins_tab(nb)

    def _build_camera(self, parent):
        box = ttk.LabelFrame(parent, text="Webcam: tracks the real arm")
        self.blank = tk.PhotoImage(width=320, height=240)
        self.cam_label = tk.Label(box, image=self.blank, text="Camera off", compound="center",
                                  bg="black", fg="#888888")
        self.cam_label.grid(row=0, column=0, rowspan=6, padx=4, pady=4)
        ttk.Label(box, text="Camera index").grid(row=0, column=1, sticky="w", padx=6)
        ttk.Spinbox(box, from_=0, to=9, width=4, textvariable=self.cam_index
                    ).grid(row=1, column=1, sticky="w", padx=6)
        self.cam_btn = ttk.Button(box, text="Start camera", command=self.toggle_camera)
        self.cam_btn.grid(row=2, column=1, sticky="ew", padx=6, pady=2)
        ttk.Button(box, text="Calibrate zero pose", command=self.calibrate_zero
                   ).grid(row=3, column=1, sticky="ew", padx=6, pady=2)
        ttk.Button(box, text="Save marker sheet", command=self.save_markers
                   ).grid(row=4, column=1, sticky="ew", padx=6, pady=2)
        ttk.Label(box, textvariable=self.cam_info, wraplength=190, foreground="gray"
                  ).grid(row=5, column=1, sticky="nw", padx=6)
        return box

    def _build_joints_tab(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="Joints & limits")
        for col, text in enumerate(["Joint", "Move", "", "Webcam", "Target", "", "Min", "Max", ""]):
            ttk.Label(tab, text=text).grid(row=0, column=col, padx=3)
        self.dots = []
        for j in range(6):
            r = j + 1
            ttk.Label(tab, text=JOINT_NAMES[j], width=12).grid(row=r, column=0, padx=3, pady=2, sticky="w")
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
                tk.Scale(tab, from_=-90, to=90, orient="horizontal", length=150, showvalue=False,
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
            dot = ttk.Label(tab, text="\u25cf", foreground="#7d8590")
            dot.grid(row=r, column=8, padx=3)
            self.dots.append(dot)
        opts = ttk.Frame(tab)
        opts.grid(row=7, column=0, columnspan=9, sticky="w", pady=4)
        ttk.Checkbutton(opts, text="Stop at limits (webcam)", variable=self.limits_on,
                        command=self._read_limits).pack(side="left", padx=4)
        ttk.Label(opts, text="Stop this many degrees early").pack(side="left", padx=(12, 2))
        sp = ttk.Spinbox(opts, from_=0, to=30, width=4, textvariable=self.margin_var,
                         command=self._read_limits)
        sp.pack(side="left")
        sp.bind("<Return>", lambda e: self._read_limits())
        sp.bind("<FocusOut>", lambda e: self._read_limits())

    def _build_tracking_tab(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="Marker tracking")
        for col, text in enumerate(["Joint", "Parent marker ID", "Child marker ID", "Invert", "Zero offset"]):
            ttk.Label(tab, text=text).grid(row=0, column=col, padx=4)
        for j in range(6):
            r = j + 1
            ttk.Label(tab, text=JOINT_NAMES[j], width=12).grid(row=r, column=0, sticky="w", padx=4, pady=2)
            for col, var in ((1, self.parent_vars[j]), (2, self.child_vars[j])):
                e = ttk.Entry(tab, textvariable=var, width=6)
                e.grid(row=r, column=col, padx=4)
                e.bind("<Return>", lambda ev: self._read_tracking())
                e.bind("<FocusOut>", lambda ev: self._read_tracking())
            ttk.Checkbutton(tab, variable=self.invert_vars[j], command=self._read_tracking
                            ).grid(row=r, column=3)
            ttk.Label(tab, textvariable=self.offset_text[j], width=10).grid(row=r, column=4)
        ttk.Label(tab, foreground="gray", wraplength=470, justify="left", text=(
            "Stick a printed marker on each link. A joint angle is the child marker's rotation "
            "minus the parent's. Leave an ID blank for a joint you do not track. Put the arm in the "
            "neutral pose and press Calibrate zero pose. If a joint moves the wrong way in the 3D "
            "view, tick Invert. A side view sees J2-J4; the base (J1) needs a view from above."
        )).grid(row=7, column=0, columnspan=5, sticky="w", padx=4, pady=6)

    def _build_pins_tab(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="Servo pins")
        ttk.Label(tab, text="Servo").grid(row=0, column=0, padx=4)
        ttk.Label(tab, text="Arduino pin").grid(row=0, column=1, padx=4)
        for i in range(7):
            ttk.Label(tab, text=SERVO_LABELS[i], width=22).grid(row=i + 1, column=0, sticky="w", padx=4, pady=1)
            ttk.Combobox(tab, textvariable=self.pin_vars[i], values=[str(p) for p in ALLOWED_PINS],
                         width=5, state="readonly").grid(row=i + 1, column=1, padx=4)
        ttk.Button(tab, text="Apply pins", command=self.apply_pins).grid(row=8, column=0, sticky="w", padx=4, pady=6)
        ttk.Label(tab, foreground="gray", wraplength=470, justify="left", text=(
            "Pins 2-13 and 16-19 (A2-A5) are allowed. Every servo needs its own pin. A0 and A1 "
            "are used by the FSR sensors, and 0 and 1 by USB. Move the signal wire to match, then "
            "press Apply. Shoulder B always mirrors shoulder A."
        )).grid(row=9, column=0, columnspan=3, sticky="w", padx=4)

    # =============================================================== helpers
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
        colors = {"info": "#6e7781", "ok": "#1a7f37", "warn": "#b35900", "error": "#cf222e"}
        self.banner.set(text)
        self.banner_label.config(foreground=colors.get(level, "#6e7781"))
        self.banner_until = time.time() + seconds

    def _refresh_offsets(self):
        for j, t in enumerate(self.cfg["tracking"]):
            self.offset_text[j].set(f"{t['offset']:+.1f}\u00b0")

    def _save(self):
        save_config(self.cfg, self.config_path)

    # ============================================================== settings
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
            t["parent"] = to_int_or_none(self.parent_vars[j].get())
            t["child"] = to_int_or_none(self.child_vars[j].get())
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
            self.set_banner("Pick a pin for every servo", "error")
            return False
        if any(p not in ALLOWED_PINS for p in pins):
            self.set_banner("Pin not allowed (A0 and A1 are the FSRs, 0 and 1 are USB)", "error")
            return False
        if len(set(pins)) != len(pins):
            self.set_banner("Two servos use the same pin. Every servo needs its own pin.", "error")
            return False
        self.cfg["pins"] = pins
        self._save()
        self.held.clear()
        self.seek = [None] * 3
        self._send("Q," + ",".join(str(p) for p in pins))
        self.set_banner("Pins applied: " + ", ".join(f"{SERVO_LABELS[i].split(' (')[0]}=D{p}" if p < 14 else
                                                       f"{SERVO_LABELS[i].split(' (')[0]}=A{p - 14}"
                                                       for i, p in enumerate(pins)), "ok", 6.0)
        return True

    # ============================================================== camera
    def toggle_camera(self):
        if self.snap["running"]:
            self.tracker.stop()
            self.cam_btn.config(text="Start camera")
            self.cam_info.set("Camera off")
            self.cam_label.config(image=self.blank, text="Camera off")
            self.cam_img = None
            return
        try:
            idx = int(self.cam_index.get())
        except ValueError:
            idx = 0
        self.cfg["camera_index"] = idx
        self._save()
        if not self.tracker.start(idx):
            self.set_banner(self.tracker.error, "error", 6.0)
            return
        self.cam_btn.config(text="Stop camera")
        self.cam_label.config(text="Starting...")

    def calibrate_zero(self):
        markers = self.tracker.snapshot()["markers"]
        done = []
        for j, t in enumerate(self.cfg["tracking"]):
            rel = trk_mod.raw_relative(markers, t["parent"], t["child"])
            if rel is not None:
                t["offset"] = float(rel)
                self.meas[j] = None
                done.append(f"J{j + 1}")
        self._refresh_offsets()
        self._save()
        if done:
            self.set_banner("Zero pose saved for " + ", ".join(done) + ". Those joints now read 0.", "ok", 5.0)
        else:
            self.set_banner("No joint has both markers in view. Start the camera and show the markers.", "warn", 5.0)

    def save_markers(self):
        path = filedialog.asksaveasfilename(defaultextension=".png", initialfile="arm_markers.png",
                                            title="Save printable marker sheet")
        if not path:
            return
        try:
            trk_mod.make_marker_sheet(path, ids=range(0, 6))
            self.set_banner(f"Marker sheet saved: {path}. Print it, cut out the markers, one per link.", "ok", 8.0)
        except Exception as err:                       # OpenCV missing or the file cannot be written
            self.set_banner(f"Could not save the marker sheet: {err}", "error", 6.0)

    # ============================================================== serial
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
            self.set_banner("Pick a port first", "warn")
            return
        try:
            self.ser = serial.Serial(port, BAUD, timeout=0.05)
        except serial.SerialException as err:
            self.set_banner(f"Could not open {port}: {err}", "error", 6.0)
            return
        self.set_banner("Connecting. The Uno resets now, so the positional servos go limp briefly.", "warn")
        self.update()
        time.sleep(RESET_WAIT)
        self.ser.reset_input_buffer()
        self.connect_btn.config(text="Disconnect")
        self.status.set(f"Connected to {port}")
        self.push_settings()
        self.set_banner("Connected.", "ok")

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
            except (serial.SerialException, OSError):
                pass
        self.ser = None
        self.cur_dir = [0] * 3
        self.connect_btn.config(text="Connect")
        self.status.set("Disconnected. Continuous joints stopped.")

    def _send(self, text):
        if not self.ser:
            return
        try:
            self.ser.write((text + "\n").encode())
        except (serial.SerialException, OSError) as err:
            self.set_banner(f"Serial error: {err}", "error", 6.0)
            self.disconnect()

    def _read_serial(self):
        if not self.ser:
            return
        try:
            while self.ser and self.ser.in_waiting:
                line = self.ser.readline().decode(errors="ignore").strip()
                if line:
                    self._handle_line(line)
        except (serial.SerialException, OSError) as err:
            self.set_banner(f"Serial error: {err}", "error", 6.0)
            self.disconnect()

    def _handle_line(self, line):
        if line.startswith("F,"):
            parts = line.split(",")
            if len(parts) == 3:
                try:
                    values = [int(p) for p in parts[1:]]
                except ValueError:
                    return
                for i, v in enumerate(values):
                    self.fsr_vars[i].set(str(v))
                    self.fsr_bars[i]["value"] = v
        elif line == "READY":
            self.set_banner("The Uno restarted (power dip?). Settings were sent again.", "warn", 5.0)
            self.push_settings()
        elif line.startswith("ERR"):
            self.set_banner("Uno: " + line, "error", 6.0)
        else:
            self.status.set(line)

    # ============================================================== moving
    def press(self, j, direction):
        if not self.ser:
            self.set_banner("Connect the Uno first", "warn")
            return
        now = time.time()
        self.held[j] = direction
        self.seek[j] = None
        self.armed[j] = self._fresh(j, now)
        if not self.armed[j]:
            self.set_banner(f"{JOINT_NAMES[j]} is not seen by the webcam: moving WITHOUT limit protection.", "warn")

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
        self.set_banner("Stopped.", "info")

    def on_pos_toggle(self):
        self.cfg["pos_on"] = bool(self.pos_on.get())
        self._save()
        if not self.ser:
            self.pos_on.set(False)
            self.set_banner("Connect the Uno first", "warn")
            return
        for k in range(3):
            self.pos_dirty[k] = True
        self._flush_positional(force=True)
        self._send("E,1" if self.pos_on.get() else "E,0")

    def set_target(self, j, angle):
        """Ask a joint to go to an angle. Continuous joints need the webcam to do it."""
        lo, hi = self._limits(j)
        if self.limits_on.get():
            angle = clamp(angle, lo, hi)
        self.target[j] = angle
        if j in CONT:
            if not self.ser:
                self.set_banner("Connect the Uno first", "warn")
                return
            if self._fresh(j, time.time()):
                if self.seek[j] is None:
                    self.armed[j] = True
                self.seek[j] = angle
                self.held.pop(j, None)
            else:
                self.set_banner(f"{JOINT_NAMES[j]} is not seen by the webcam, so it cannot go to an "
                                f"angle. Use Left / Right, or show its markers to the camera.", "warn")
        else:
            k = j - 3
            self.cmd[k] = clamp(angle, -90.0, 90.0)
            self.pos_dirty[k] = True
            self.slider_vars[k].set(self.cmd[k])
            if not self.pos_on.get():
                self.set_banner("Tick 'Positional servos ON' so this servo follows.", "info")

    def go(self, j):
        try:
            angle = float(self.target_text[j].get())
        except ValueError:
            self.set_banner("Type an angle in degrees", "warn")
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

    # ============================================================ the loop
    def tick(self):
        now = time.time()
        self.snap = self.tracker.snapshot()
        self._read_serial()
        self._update_measurements(now)
        self._control_continuous(now)
        self._flush_positional()
        self._update_poses(now)
        if self.banner_until and now > self.banner_until:
            self.banner_until = 0.0
            self.banner.set("Ready.")
            self.banner_label.config(foreground="#6e7781")
        self.after(TICK_MS, self.tick)

    def _update_measurements(self, now):
        markers = self.snap["markers"]
        for j, t in enumerate(self.cfg["tracking"]):
            ang = trk_mod.joint_angle(markers, t["parent"], t["child"], t["invert"], t["offset"])
            if ang is None:
                continue
            seen_at = min(markers[t["parent"]][2], markers[t["child"]][2])
            if seen_at <= self.meas_t[j]:
                continue                                       # no new camera frame for this joint
            if self.meas[j] is None or seen_at - self.meas_t[j] > 1.0:
                smooth, self.vel[j] = ang, 0.0
            else:
                dt = max(1e-3, seen_at - self.meas_t[j])
                smooth = 0.6 * ang + 0.4 * self.meas[j]
                self.vel[j] = 0.5 * self.vel[j] + 0.5 * (smooth - self.meas[j]) / dt
            self.meas[j], self.meas_t[j] = smooth, seen_at

    def _control_continuous(self, now):
        limits_on, margin, speed = bool(self.limits_on.get()), self._margin(), int(self.speed.get())
        for j in CONT:
            lo, hi = self._limits(j)
            fresh = self._fresh(j, now)
            angle = self.meas[j] if fresh else None
            held, seek = self.held.get(j, 0), self.seek[j]
            if self.armed[j] and (held or seek is not None) and not fresh and now - self.meas_t[j] > LOST_S:
                direction, pct, note, done = 0, speed, "lost", False
            else:
                direction, pct, note, done = decide_motion(angle, fresh, self.vel[j], held, seek,
                                                           lo, hi, margin, speed, limits_on, self.k_est[j])
            if note == "lost":
                self.seek[j] = None
                if held or seek is not None:
                    self.set_banner(f"Webcam lost {JOINT_NAMES[j]}: stopped.", "error", 4.0)
                    self.held.pop(j, None)
            elif note.startswith("limit"):
                self.seek[j] = None
                edge = hi if note == "limit_max" else lo
                self.set_banner(f"{JOINT_NAMES[j]} at its limit ({edge:+.0f}\u00b0): blocked that way.", "warn", 2.5)
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

    # ============================================================ displays
    def view_tick(self):
        now = time.time()
        for j in range(6):
            fresh = self._fresh(j, now)
            if fresh:
                lo, hi = self._limits(j)
                self.meas_text[j].set(f"{self.meas[j]:+.1f}\u00b0")
                inside = lo <= self.meas[j] <= hi or not self.limits_on.get()
                self.dots[j].config(foreground="#2ecc71" if inside else "#e5534b")
            else:
                self.meas_text[j].set("-" if j in CONT else f"{self.cmd[j - 3]:+.0f}\u00b0")
                self.dots[j].config(foreground="#7d8590")
        self.view.render()
        self.after(VIEW_MS, self.view_tick)

    def cam_tick(self):
        snap = self.snap
        if snap["running"] and snap["png"] and snap["frame"] != self.last_cam_frame:
            self.last_cam_frame = snap["frame"]
            try:
                self.cam_img = tk.PhotoImage(data=snap["png"])
                self.cam_label.config(image=self.cam_img, text="")
            except tk.TclError:
                pass
        if snap["running"]:
            ids = sorted(snap["markers"])
            self.cam_info.set(f"{snap['fps']:.0f} fps. Markers seen: {ids if ids else 'none'}")
        elif snap["error"]:
            self.cam_info.set(snap["error"])
            self.cam_btn.config(text="Start camera")
        self.after(CAM_MS, self.cam_tick)

    def on_close(self):
        self._read_limits()
        self._read_tracking()
        self.tracker.stop()
        self.disconnect()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
