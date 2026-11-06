"""3D model of the NEXUS arm for the GUI. Pure tkinter + numpy, no graphics card needed.

Joint angles (degrees), all zero in the neutral pose: upper arm straight up,
forearm pointing forward, wrist straight, gripper half open.
    J1 base yaw      positive = turn left (counter-clockwise seen from above)
    J2 shoulder      positive = lean the arm forward
    J3 elbow         positive = swing the forearm downward
    J4 wrist pitch   positive = tip the gripper downward
    J5 wrist roll    positive = roll about the gripper axis
    J6 gripper       -90 = closed, +90 = fully open (maps to servo 0-180)
"""
import math
import tkinter as tk

import numpy as np

JOINT_NAMES = ["J1 Base", "J2 Shoulder", "J3 Elbow", "J4 Wrist", "J5 Roll", "J6 Gripper"]

# Link sizes in millimetres, approximated from the arm photos
SHOULDER_Z = 90.0
L_UPPER = 120.0
L_FORE = 115.0
L_WRIST = 38.0

BOX_FACES = ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0))
BOX_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7))
LIGHT = np.array([0.45, -0.35, 0.82])
LIGHT = LIGHT / np.linalg.norm(LIGHT)


def wrap180(angle):
    return (angle + 180.0) % 360.0 - 180.0


def rot_x(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.0]])


def rot_y(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1.0]])


def rot_z(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


def trans(x, y, z):
    t = np.eye(4)
    t[:3, 3] = (x, y, z)
    return t


def box_local(x0, x1, y0, y1, z0, z1):
    return np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                     [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)


def apply(T, pts):
    return pts @ T[:3, :3].T + T[:3, 3]


def grip_gap(grip_deg):
    """Gripper angle -90..+90 to the gap between the fingers in mm."""
    return 4.0 + (max(-90.0, min(90.0, grip_deg)) + 90.0) / 180.0 * 36.0


def build_arm(angles):
    """Return (parts, handles) for the given six joint angles.

    parts   list of {"name", "verts" (8x3 world points), "color"}
    handles list of {"joint", "point", "pivot", "axis", "mode"}: the draggable spots
    """
    yaw, sh, el, wr, roll, grip = [float(a) for a in angles]
    T_yaw = rot_z(yaw)
    T_sh = T_yaw @ trans(0, 0, SHOULDER_Z) @ rot_y(sh)
    T_el = T_sh @ trans(0, 0, L_UPPER) @ rot_y(el + 90.0)
    T_wr = T_el @ trans(0, 0, L_FORE) @ rot_y(wr)
    T_rl = T_wr @ trans(0, 0, L_WRIST) @ rot_z(roll)
    half = grip_gap(grip) / 2.0

    def part(name, T, box, color):
        return {"name": name, "verts": apply(T, box_local(*box)), "color": color}

    dark, mid, light, finger = (104, 114, 132), (128, 139, 158), (150, 162, 182), (186, 198, 216)
    parts = [
        part("base", np.eye(4), (-65, 65, -65, 65, 0, 45), (92, 101, 116)),
        part("turret", T_yaw, (-38, 38, -38, 38, 45, 80), light),
        part("shoulder_block", T_yaw, (-26, 26, -44, 44, SHOULDER_Z - 22, SHOULDER_Z + 22), mid),
        part("upper_arm", T_sh, (-16, 16, -13, 13, 0, L_UPPER), dark),
        part("elbow_block", T_el, (-20, 20, -22, 22, -16, 16), mid),
        part("forearm", T_el, (-14, 14, -12, 12, 0, L_FORE), dark),
        part("wrist", T_wr, (-17, 17, -14, 14, 0, L_WRIST), mid),
        part("gripper_body", T_rl, (-24, 24, -15, 15, 0, 28), light),
        part("finger_left", T_rl, (-half - 7, -half, -5, 5, 26, 76), finger),
        part("finger_right", T_rl, (half, half + 7, -5, 5, 26, 76), finger),
    ]

    def pt(T, x, y, z):
        return (T @ np.array([x, y, z, 1.0]))[:3]

    handles = [
        {"joint": 0, "point": pt(T_yaw, 62, 0, 64), "pivot": pt(T_yaw, 0, 0, 60),
         "axis": np.array([0.0, 0.0, 1.0]), "mode": "rot"},
        {"joint": 1, "point": pt(T_el, 0, 0, 0), "pivot": pt(T_sh, 0, 0, 0),
         "axis": T_sh[:3, 1].copy(), "mode": "rot"},
        {"joint": 2, "point": pt(T_wr, 0, 0, 0), "pivot": pt(T_el, 0, 0, 0),
         "axis": T_el[:3, 1].copy(), "mode": "rot"},
        {"joint": 3, "point": pt(T_rl, 0, 0, 30), "pivot": pt(T_wr, 0, 0, 0),
         "axis": T_wr[:3, 1].copy(), "mode": "rot"},
        {"joint": 4, "point": pt(T_rl, 36, 0, 6), "pivot": pt(T_rl, 0, 0, 0),
         "axis": T_rl[:3, 2].copy(), "mode": "lin"},
        {"joint": 5, "point": pt(T_rl, 0, 0, 80), "pivot": pt(T_rl, 0, 0, 0),
         "axis": T_rl[:3, 0].copy(), "mode": "lin"},
    ]
    return parts, handles


def get_marker_3d_point(marker, angles):
    """Calculate the 3D world coordinate of a marker locked/positioned on the robot body."""
    yaw, sh, el, wr, roll, grip = [float(a) for a in angles]
    T_yaw = rot_z(yaw)
    T_sh = T_yaw @ trans(0, 0, SHOULDER_Z) @ rot_y(sh)
    T_el = T_sh @ trans(0, 0, L_UPPER) @ rot_y(el + 90.0)
    T_wr = T_el @ trans(0, 0, L_FORE) @ rot_y(wr)
    T_rl = T_wr @ trans(0, 0, L_WRIST) @ rot_z(roll)

    link = str(marker.get("link", "upper_arm")).lower()
    side = str(marker.get("side", "left")).lower()
    pos = max(0.0, min(1.0, float(marker.get("pos", 0.5))))

    # x offset: negative is Left side, positive is Right side
    x_sign = 1.0 if side == "right" else (-1.0 if side == "left" else 0.0)

    if link == "base":
        T = np.eye(4)
        local_pt = np.array([x_sign * 68.0, 0.0, 15.0 + pos * 25.0, 1.0])
    elif link == "turret":
        T = T_yaw
        local_pt = np.array([x_sign * 42.0, 0.0, 48.0 + pos * 30.0, 1.0])
    elif link == "upper_arm":
        T = T_sh
        local_pt = np.array([x_sign * 20.0, 0.0, pos * L_UPPER, 1.0])
    elif link == "forearm":
        T = T_el
        local_pt = np.array([x_sign * 18.0, 0.0, pos * L_FORE, 1.0])
    elif link == "wrist":
        T = T_wr
        local_pt = np.array([x_sign * 20.0, 0.0, pos * L_WRIST, 1.0])
    elif link == "gripper":
        T = T_rl
        local_pt = np.array([x_sign * 26.0, 0.0, 10.0 + pos * 25.0, 1.0])
    else:
        T = T_sh
        local_pt = np.array([x_sign * 20.0, 0.0, pos * L_UPPER, 1.0])

    return (T @ local_pt)[:3]


class Camera:
    """Orbit camera looking at a target point."""

    def __init__(self):
        self.az = -50.0
        self.el = 20.0
        self.dist = 560.0
        self.target = np.array([60.0, 0.0, 120.0])

    def basis(self):
        az, el = math.radians(self.az), math.radians(self.el)
        offset = self.dist * np.array([math.cos(el) * math.cos(az),
                                       math.cos(el) * math.sin(az),
                                       math.sin(el)])
        pos = self.target + offset
        f = -offset / np.linalg.norm(offset)
        r = np.cross(f, np.array([0.0, 0.0, 1.0]))
        r = r / np.linalg.norm(r)
        u = np.cross(r, f)
        return pos, f, r, u

    def project(self, pts, w, h):
        pos, f, r, u = self.basis()
        pts = np.atleast_2d(pts)
        rel = pts - pos
        xc, yc, zc = rel @ r, rel @ u, np.maximum(rel @ f, 1.0)
        focal = 1.05 * h
        sx = w / 2.0 + focal * xc / zc
        sy = h / 2.0 - focal * yc / zc
        return np.stack([sx, sy], axis=1), zc


def _hex(rgb, shade=1.0):
    r, g, b = [int(max(0, min(255, c * shade))) for c in rgb]
    return f"#{r:02x}{g:02x}{b:02x}"


class ArmView:
    """A tkinter canvas that draws the arm and lets you drag its joints."""

    HIT_PX = 16

    def __init__(self, parent, width=460, height=340, on_drag=None, on_drag_end=None):
        self.w, self.h = width, height
        self.canvas = tk.Canvas(parent, width=width, height=height, bg="#161a20",
                                highlightthickness=0)
        self.cam = Camera()
        self.actual = [0.0] * 6        # shown as the solid arm (what the webcam sees)
        self.target = [0.0] * 6        # shown as the orange outline (where you want it)
        self.tracked = [False] * 6     # joints currently measured by the webcam
        self.on_drag = on_drag
        self.on_drag_end = on_drag_end
        self.handles = []              # screen positions from the last render
        self.body_markers = []         # list of {"id", "link", "side", "pos", "locked"}
        self.active_markers = set()    # ids currently detected by cameras
        self.selected_marker_id = None
        self.marker_screen_handles = []
        self.on_marker_select = None
        self.hover = None
        self.drag = None
        self.orbit = None

        c = self.canvas
        c.bind("<ButtonPress-1>", self._press)
        c.bind("<B1-Motion>", self._motion)
        c.bind("<ButtonRelease-1>", self._release)
        c.bind("<Motion>", self._hover_move)
        c.bind("<ButtonPress-3>", self._press_orbit)
        c.bind("<B3-Motion>", self._motion_orbit)
        c.bind("<MouseWheel>", lambda e: self._zoom(0.92 if e.delta > 0 else 1.08))
        c.bind("<Button-4>", lambda e: self._zoom(0.92))
        c.bind("<Button-5>", lambda e: self._zoom(1.08))
        self.render()

    # ------------------------------------------------------------- drawing
    def render(self):
        c = self.canvas
        c.delete("all")
        w, h = self.w, self.h
        self._draw_floor()

        parts, handles3d = build_arm(self.actual)
        pos, _f, _r, _u = self.cam.basis()
        faces = []
        for p in parts:
            v = p["verts"]
            center = v.mean(axis=0)
            xy, depth = self.cam.project(v, w, h)
            for f in BOX_FACES:
                idx = list(f)
                p3 = v[idx]
                n = np.cross(p3[1] - p3[0], p3[2] - p3[1])
                n = n / (np.linalg.norm(n) + 1e-9)
                fc = p3.mean(axis=0)
                if np.dot(n, fc - center) < 0:
                    n = -n
                if np.dot(n, fc - pos) >= 0:
                    continue                      # facing away from the camera
                shade = 0.38 + 0.62 * max(0.0, float(np.dot(n, LIGHT)))
                coords = xy[idx].flatten().tolist()
                faces.append((float(depth[idx].mean()), coords,
                              _hex(p["color"], shade), _hex(p["color"], shade * 0.6)))
        for _d, coords, fill, edge in sorted(faces, key=lambda t: -t[0]):
            c.create_polygon(coords, fill=fill, outline=edge)

        if any(abs(a - b) > 1.0 for a, b in zip(self.actual, self.target)):
            self._draw_target_outline()

        self._draw_handles(handles3d)
        self._draw_body_markers()
        c.create_text(8, h - 8, anchor="sw", fill="#7d8590", font=("TkDefaultFont", 8),
                      text="Drag handles = move joints   |   click markers = inspect   |   wheel = zoom")

    def _draw_floor(self):
        c = self.canvas
        for k in range(-200, 201, 50):
            for a, b in (((k, -200, 0), (k, 200, 0)), ((-200, k, 0), (200, k, 0))):
                xy, _ = self.cam.project(np.array([a, b], float), self.w, self.h)
                c.create_line(xy[0, 0], xy[0, 1], xy[1, 0], xy[1, 1], fill="#323a46")

    def _draw_target_outline(self):
        parts, _ = build_arm(self.target)
        c = self.canvas
        for p in parts:
            xy, _d = self.cam.project(p["verts"], self.w, self.h)
            for a, b in BOX_EDGES:
                c.create_line(xy[a, 0], xy[a, 1], xy[b, 0], xy[b, 1],
                              fill="#ffb020", dash=(3, 3))

    def _draw_handles(self, handles3d):
        c = self.canvas
        pts = np.array([hd["point"] for hd in handles3d])
        piv = np.array([hd["pivot"] for hd in handles3d])
        pxy, _ = self.cam.project(pts, self.w, self.h)
        vxy, _ = self.cam.project(piv, self.w, self.h)
        self.handles = []
        for hd, xy, pv in zip(handles3d, pxy, vxy):
            j = hd["joint"]
            self.handles.append({"joint": j, "xy": (float(xy[0]), float(xy[1])),
                                 "pivot_xy": (float(pv[0]), float(pv[1])),
                                 "point3": np.array(hd["point"], float),
                                 "pivot3": np.array(hd["pivot"], float),
                                 "axis": hd["axis"], "mode": hd["mode"]})
            active = self.drag is not None and self.drag["j"] == j
            fill = "#ff8a00" if active else ("#ffd23f" if self.hover == j else "#2f81f7")
            ring = "#2ecc71" if self.tracked[j] else "#7d8590"
            r = 10
            c.create_oval(xy[0] - r - 3, xy[1] - r - 3, xy[0] + r + 3, xy[1] + r + 3,
                          outline=ring, width=2)
            c.create_oval(xy[0] - r, xy[1] - r, xy[0] + r, xy[1] + r, fill=fill, outline="#0d1117")
            c.create_text(xy[0], xy[1], text=str(j + 1), fill="white", font=("TkDefaultFont", 9, "bold"))
            if active or self.hover == j:
                label = f"{JOINT_NAMES[j]}  {self.target[j]:+.0f}\u00b0"
                c.create_text(xy[0] + 16, xy[1] - 16, anchor="w", text=label, fill="#ffffff",
                              font=("TkDefaultFont", 10, "bold"))

    def _draw_body_markers(self):
        c = self.canvas
        w, h = self.w, self.h
        self.marker_screen_handles = []
        for m in self.body_markers:
            mid = m.get("id", 0)
            pt3d = get_marker_3d_point(m, self.actual)
            (xy,), zc = self.cam.project(pt3d[None, :], w, h)
            if zc[0] <= 10.0:
                continue
            sx, sy = float(xy[0]), float(xy[1])
            self.marker_screen_handles.append({"id": mid, "xy": (sx, sy), "marker": m})

            is_seen = mid in self.active_markers
            is_locked = m.get("locked", True)
            is_sel = (self.selected_marker_id == mid)

            border = "#ffd23f" if is_sel else ("#2ecc71" if is_seen else "#7d8590")
            fill = "#d97706" if is_sel else ("#0f3820" if is_seen else "#161b22")
            sz = 10

            # Outer marker badge
            c.create_rectangle(sx - sz, sy - sz, sx + sz, sy + sz, fill=fill, outline=border, width=2)
            c.create_rectangle(sx - 3, sy - 3, sx + 3, sy + 3, fill=border, outline="")

            # Tag label: M{id} and lock symbol
            lock_sym = " 🔒" if is_locked else ""
            tag_label = f"M{mid}{lock_sym}"
            c.create_text(sx, sy - sz - 7, text=tag_label, fill=border if is_sel else "#ffffff",
                          font=("TkDefaultFont", 8, "bold"))

    def _hit_marker(self, x, y):
        for mh in self.marker_screen_handles:
            if math.hypot(mh["xy"][0] - x, mh["xy"][1] - y) <= 15:
                return mh["id"]
        return None

    # --------------------------------------------------------- interaction
    def _hit(self, x, y):
        best, best_d = None, self.HIT_PX
        for hd in self.handles:
            d = math.hypot(hd["xy"][0] - x, hd["xy"][1] - y)
            if d < best_d:
                best, best_d = hd, d
        return best

    def _hover_move(self, e):
        hd = self._hit(e.x, e.y)
        j = hd["joint"] if hd else None
        if j != self.hover:
            self.hover = j
            self.render()

    def _press(self, e):
        mid = self._hit_marker(e.x, e.y)
        if mid is not None:
            self.selected_marker_id = mid
            if self.on_marker_select:
                self.on_marker_select(mid)
            self.render()
            return

        hd = self._hit(e.x, e.y)
        if hd is None:
            self.orbit = (e.x, e.y)
            return
        self.drag = self.begin_drag(hd, e.x, e.y)
        self.render()

    def begin_drag(self, hd, x, y):
        """Start dragging a handle. Also used by the tests."""
        j = hd["joint"]
        drag = {"j": j, "mode": hd["mode"], "x0": x, "start": self.target[j], "delta": 0.0}
        if hd["mode"] == "rot":
            axis = np.asarray(hd["axis"], float)
            drag["axis"] = axis / (np.linalg.norm(axis) + 1e-9)
            drag["pivot3"] = np.asarray(hd["pivot3"], float)
            drag["point3"] = np.asarray(hd["point3"], float)
        return drag

    def _rotated_points(self, drag, deltas_deg):
        """The handle position after rotating it by each angle in deltas_deg (Rodrigues)."""
        k = drag["axis"]
        v = drag["point3"] - drag["pivot3"]
        th = np.radians(np.asarray(deltas_deg, float))[:, None]
        rotated = (v * np.cos(th) + np.cross(k, v) * np.sin(th)
                   + k * np.dot(k, v) * (1.0 - np.cos(th)))
        return drag["pivot3"] + rotated

    def drag_angle(self, drag, x, y):
        """New joint angle for a mouse position during a drag.

        Rotary joints: find the rotation whose projected handle lands closest to the mouse.
        This stays correct from any camera angle, including slanted views of a circle.
        """
        if drag["mode"] != "rot":
            return drag["start"] + (x - drag["x0"]) * 0.6
        mouse = np.array([x, y], float)

        def best(center, span, step):
            cand = center + np.arange(-span, span + step * 0.5, step)
            xy, _ = self.cam.project(self._rotated_points(drag, cand), self.w, self.h)
            dist = np.hypot(xy[:, 0] - mouse[0], xy[:, 1] - mouse[1])
            return float(cand[int(np.argmin(dist))])

        delta = best(drag["delta"], 45.0, 1.0)       # stay near the last value: no jumping
        delta = best(delta, 1.0, 0.1)
        drag["delta"] = delta
        return drag["start"] + delta

    def _motion(self, e):
        if self.drag is not None:
            angle = self.drag_angle(self.drag, e.x, e.y)
            if self.on_drag:
                self.on_drag(self.drag["j"], angle)
            self.render()
        elif self.orbit is not None:
            self._do_orbit(e.x, e.y)

    def _release(self, _e):
        if self.drag is not None:
            j = self.drag["j"]
            self.drag = None
            if self.on_drag_end:
                self.on_drag_end(j)
            self.render()
        self.orbit = None

    def _press_orbit(self, e):
        self.orbit = (e.x, e.y)

    def _motion_orbit(self, e):
        if self.orbit is not None:
            self._do_orbit(e.x, e.y)

    def _do_orbit(self, x, y):
        ox, oy = self.orbit
        self.cam.az -= (x - ox) * 0.5
        self.cam.el = max(-5.0, min(85.0, self.cam.el + (y - oy) * 0.5))
        self.orbit = (x, y)
        self.render()

    def _zoom(self, factor):
        self.cam.dist = max(250.0, min(1200.0, self.cam.dist * factor))
        self.render()
