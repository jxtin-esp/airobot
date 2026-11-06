"""Dual-Camera Tracking (Overhead + Side) of the NEXUS robotic arm.

Camera Topology:
  1. Overhead Camera (Top-Down View):
     - Placed above the robotic arm looking straight down at the base, turret, and reach.
     - Tracks J1 Base Yaw and global planar workspace positioning.
  2. Side Camera (Lateral View):
     - Placed on the flank/side looking at the shoulder, elbow, and wrist.
     - Tracks J2 Shoulder, J3 Elbow, and J4 Wrist pitch in the elevation plane.

Tracking Methods Supported:
  A. AI Optical Flow Point & Limb Tracking:
     - Click directly on the live webcam feed to place a tracker on any joint, bolt, or limb.
     - AI Auto-Snaps onto high-contrast optical features (screws, horns, brackets) via Shi-Tomasi corners.
     - Tracks sub-pixel movement in real-time at 60+ FPS using Lucas-Kanade optical flow pyramids.
     - Pairs of points form Limb Trackers, computing vector rotations and relative joint angles directly.
  B. Printed ArUco Markers (DICT_4X4_50):
     - Automatic quad detection for printed physical tags.
"""
import math
import sys
import threading
import time

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

DICT_NAME = "DICT_4X4_50"
DISPLAY_SIZE = (320, 240)
MARKER_MAX_AGE = 0.45  # seconds before marker is treated as lost


def wrap180(angle):
    """Normalize angle to -180 .. +180 degrees."""
    return (angle + 180.0) % 360.0 - 180.0


def opencv_ok():
    return cv2 is not None and hasattr(cv2, "aruco")


def _dictionary():
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT_NAME))


def _make_detector():
    """Return detector callable gray_image -> (corners, ids). Compatible with OpenCV 4.x and 5.x."""
    dic = _dictionary()
    if hasattr(cv2.aruco, "ArucoDetector"):
        det = cv2.aruco.ArucoDetector(dic, cv2.aruco.DetectorParameters())
        return lambda gray: det.detectMarkers(gray)[:2]
    params = cv2.aruco.DetectorParameters_create()
    return lambda gray: cv2.aruco.detectMarkers(gray, dic, parameters=params)[:2]


def marker_angle(corners):
    """In-picture rotation of marker in degrees (clockwise positive)."""
    c = np.asarray(corners, float).reshape(4, 2)
    d = c[1] - c[0]
    return math.degrees(math.atan2(d[1], d[0]))


def raw_relative(markers, parent_id, child_id):
    """Relative angle child - parent, or None if either marker is missing."""
    if parent_id is None or child_id is None:
        return None
    if parent_id not in markers or child_id not in markers:
        return None
    return wrap180(markers[child_id][0] - markers[parent_id][0])


def joint_angle(markers, parent_id, child_id, invert=False, offset=0.0):
    """Calibrated joint angle in degrees. None if parent or child is missing."""
    rel = raw_relative(markers, parent_id, child_id)
    if rel is None:
        return None
    return (-1.0 if invert else 1.0) * wrap180(rel - offset)


def snap_feature_point(gray_frame, x, y, radius=18):
    """AI feature snapping: finds the nearest distinct optical corner/bolt/screw within radius."""
    if gray_frame is None or gray_frame.size == 0:
        return float(x), float(y)
    h, w = gray_frame.shape[:2]
    x0, x1 = max(0, int(x - radius)), min(w, int(x + radius))
    y0, y1 = max(0, int(y - radius)), min(h, int(y + radius))
    patch = gray_frame[y0:y1, x0:x1]
    if patch.size == 0:
        return float(x), float(y)
    corners = cv2.goodFeaturesToTrack(patch, maxCorners=1, qualityLevel=0.01, minDistance=3)
    if corners is not None and len(corners):
        return x0 + float(corners[0, 0, 0]), y0 + float(corners[0, 0, 1])
    return float(x), float(y)


def make_marker_sheet(path, ids=range(0, 8), marker_px=360):
    """Generate and save printable ArUco marker sheet."""
    if not opencv_ok():
        raise RuntimeError("OpenCV with the aruco module is not installed")
    dic = _dictionary()
    pad = 60
    tile = marker_px + 2 * pad
    ids = list(ids)
    cols = 4 if len(ids) > 6 else 3
    rows = (len(ids) + cols - 1) // cols
    sheet = np.full((rows * (tile + 50) + 20, cols * tile + 20), 255, np.uint8)
    for n, mid in enumerate(ids):
        if hasattr(cv2.aruco, "generateImageMarker"):
            img = cv2.aruco.generateImageMarker(dic, mid, marker_px)
        else:
            img = cv2.aruco.drawMarker(dic, mid, marker_px)
        r, c = divmod(n, cols)
        y0, x0 = 10 + r * (tile + 50), 10 + c * tile
        sheet[y0 + pad:y0 + pad + marker_px, x0 + pad:x0 + pad + marker_px] = img
        cv2.putText(sheet, f"ID {mid}", (x0 + pad, y0 + pad - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2, cv2.LINE_AA)
        cv2.putText(sheet, "this edge up", (x0 + pad, y0 + pad + marker_px + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 90, 1, cv2.LINE_AA)
    cv2.imwrite(path, sheet)
    return path


def scan_available_cameras(max_tested=8, exclude_integrated=False):
    """Scan and return list of usable camera indices across backends (DSHOW, MSMF, ANY)."""
    if not opencv_ok():
        return []
    found = []
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform.startswith("win") else [0]

    start_idx = 1 if exclude_integrated else 0
    for i in range(start_idx, max_tested):
        opened = False
        for backend in backends:
            try:
                cap = cv2.VideoCapture(i, backend) if backend is not None else cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        found.append(i)
                        opened = True
                    cap.release()
                if opened:
                    break
            except Exception:
                pass
    return found


class CameraWorker:
    """Manages capture loop, ArUco markers, and AI optical point & limb tracking."""

    def __init__(self, name="cam", detector=None):
        self.name = name
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.index = None
        self.error = ""
        self.fps = 0.0
        self._png = None
        self._frame_no = 0
        self._markers = {}  # id -> (angle, (cx, cy), last_seen_time)
        self._detect = detector

        # AI Point & Limb Trackers
        self.point_trackers = {}  # id -> {name, x, y, template, status, color, last_seen}
        self.limb_trackers = []   # list of {id, name, p1, p2, color}
        self.prev_gray = None
        self.last_frame = None

    def start(self, index, detector=None):
        self.stop()
        if detector is not None:
            self._detect = detector
        elif self._detect is None and opencv_ok():
            self._detect = _make_detector()

        self.index = int(index)
        self.error = ""
        self.running = True
        self.thread = threading.Thread(target=self._loop, name=f"Worker-{self.name}", daemon=True)
        self.thread.start()
        return True

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.5)
        self.thread = None
        with self.lock:
            self._markers = {}
            self._png = None
            self.prev_gray = None
            self.last_frame = None

    def add_point_tracker(self, pt_id, name, x, y, snap_feature=True, color=(0, 255, 255)):
        """Add or update an optical flow point tracker at (x, y).

        Stores a 35x35 grayscale template for robust template-matching based
        drift rejection, plus anchor coordinates to guard against catastrophic drift.
        """
        with self.lock:
            snap_x, snap_y = float(x), float(y)
            template = None
            if self.last_frame is not None:
                gray = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
                if snap_feature:
                    snap_x, snap_y = snap_feature_point(gray, snap_x, snap_y, radius=18)
                h, w = gray.shape[:2]
                # 35x35 template for robust matching (was 25x25)
                half = 17
                tx0, tx1 = max(0, int(snap_x - half)), min(w, int(snap_x + half + 1))
                ty0, ty1 = max(0, int(snap_y - half)), min(h, int(snap_y + half + 1))
                if tx1 > tx0 and ty1 > ty0:
                    template = gray[ty0:ty1, tx0:tx1].copy()

                # Sample HSV color fingerprint of the arm at placement point.
                # Used every frame to reject trackers that have drifted onto a
                # person / hand (which will have different skin/clothing colors).
                hsv_mean = None
                hsv_std  = None
                try:
                    hsv_frame = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2HSV)
                    ch = 7   # 15x15 patch for color sampling
                    cxs = max(0, int(snap_x) - ch)
                    cys = max(0, int(snap_y) - ch)
                    cxe = min(w,  int(snap_x) + ch + 1)
                    cye = min(h,  int(snap_y) + ch + 1)
                    if cxe > cxs and cye > cys:
                        hsv_patch = hsv_frame[cys:cye, cxs:cxe].reshape(-1, 3).astype(float)
                        hsv_mean  = hsv_patch.mean(axis=0)
                        # Min std of 10 to avoid over-rejection on uniform surfaces
                        hsv_std   = np.clip(hsv_patch.std(axis=0), 10.0, None)
                except Exception:
                    pass

            self.point_trackers[str(pt_id)] = {
                "id": str(pt_id),
                "name": str(name),
                "x": snap_x,
                "y": snap_y,
                "anchor_x": snap_x,       # Original placement position
                "anchor_y": snap_y,        # Used for max-drift guard
                "template": template,
                "hsv_mean": hsv_mean,      # HSV color fingerprint of the arm
                "hsv_std":  hsv_std,       # Tolerance band for color gating
                "confidence": 1.0,         # Template match correlation
                "status": "tracking",
                "color": color,
                "frame_count": 0,          # Frames since last template refresh
                "last_seen": time.time(),
            }
            return snap_x, snap_y

    def remove_point_tracker(self, pt_id):
        with self.lock:
            self.point_trackers.pop(str(pt_id), None)
            self.limb_trackers = [
                l for l in self.limb_trackers
                if l["p1"] != str(pt_id) and l["p2"] != str(pt_id)
            ]

    def clear_point_trackers(self):
        with self.lock:
            self.point_trackers.clear()
            self.limb_trackers.clear()

    def add_limb_tracker(self, limb_id, name, p1_id, p2_id, color=(0, 255, 0)):
        with self.lock:
            self.limb_trackers = [l for l in self.limb_trackers if l["id"] != str(limb_id)]
            self.limb_trackers.append({
                "id": str(limb_id),
                "name": str(name),
                "p1": str(p1_id),
                "p2": str(p2_id),
                "color": color,
            })

    def remove_limb_tracker(self, limb_id):
        with self.lock:
            self.limb_trackers = [l for l in self.limb_trackers if l["id"] != str(limb_id)]

    def clear_limb_trackers(self):
        with self.lock:
            self.limb_trackers.clear()

    def _loop(self):
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform.startswith("win") else [0]
        cap = None
        for b in backends:
            try:
                c = cv2.VideoCapture(self.index, b) if b is not None else cv2.VideoCapture(self.index)
                if c.isOpened():
                    ret, _ = c.read()
                    if ret:
                        cap = c
                        break
                c.release()
            except Exception:
                pass

        if cap is None:
            self.error = f"Cannot open {self.name} (index {self.index})."
            self.running = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        t_prev = time.time()
        frames = 0

        while self.running:
            ok, raw_frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            # Standardize processing resolution to 320x240 for fast optical flow and direct UI coords
            frame = cv2.resize(raw_frame, DISPLAY_SIZE)
            now = time.time()
            gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv_now = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)   # for color-lock check
            seen    = {}

            # 1. Optical Flow for Point Trackers (with template-verification drift guard)
            #
            # Strategy:
            #   A) Run Lucas-Kanade to get candidate position
            #   B) Template-match at candidate — if correlation >= TMPL_ACCEPT_THRESH:
            #      accept the new position (genuine arm movement)
            #   C) If correlation < TMPL_ACCEPT_THRESH: FREEZE the tracker at its
            #      previous position (occluded by passing object / person)
            #   D) While frozen, search a wider area (±40px) each frame via template
            #      matching to re-find the feature once the occluder passes
            #   E) Max-drift guard: if tracker drifts > MAX_DRIFT_PX from anchor,
            #      snap back and re-verify
            #
            TMPL_ACCEPT_THRESH = 0.50  # Accept optical flow if template correlation >= this
            MAX_DRIFT_PX = 60.0        # Max allowed drift from anchor before snap-back
            TMPL_REFRESH_INTERVAL = 30 # Refresh stored template every N frames (adapt to lighting)
            SEARCH_RADIUS = 40         # Wider search area when frozen (pixels)

            with self.lock:
                active_pts = [p for p in self.point_trackers.values()]

            if len(active_pts) > 0 and self.prev_gray is not None:
                try:
                    p0 = np.array([[p["x"], p["y"]] for p in active_pts], dtype=np.float32).reshape(-1, 1, 2)
                    p1, st, _err = cv2.calcOpticalFlowPyrLK(
                        self.prev_gray, gray, p0, None,
                        winSize=(21, 21), maxLevel=3,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
                    )
                except Exception:
                    p1, st = None, None
                dw, dh = DISPLAY_SIZE
                with self.lock:
                    for i, p_info in enumerate(active_pts):
                        if p1 is None or st is None:
                            p_info["status"] = "lost"
                            p_info["confidence"] = 0.0
                            continue
                        try:
                          tmpl = p_info.get("template")
                          prev_status = p_info.get("status", "tracking")
                        except Exception:
                          continue

                        if prev_status == "frozen":
                            # ── FROZEN: Search nearby first, then escalate to full frame ──
                            frozen_fc = p_info.get("frozen_frames", 0) + 1
                            p_info["frozen_frames"] = frozen_fc
                            recovered = False

                            if tmpl is not None:
                                # First 60 frames frozen: search ±SEARCH_RADIUS
                                # After 60 frames:        search full frame (occluder has passed)
                                if frozen_fc <= 60:
                                    cx, cy = int(p_info["x"]), int(p_info["y"])
                                    rx0 = max(0, cx - SEARCH_RADIUS)
                                    rx1 = min(dw, cx + SEARCH_RADIUS)
                                    ry0 = max(0, cy - SEARCH_RADIUS)
                                    ry1 = min(dh, cy + SEARCH_RADIUS)
                                    if (rx1 - rx0) >= tmpl.shape[1] and (ry1 - ry0) >= tmpl.shape[0]:
                                        search_roi = gray[ry0:ry1, rx0:rx1]
                                        res = cv2.matchTemplate(search_roi, tmpl, cv2.TM_CCOEFF_NORMED)
                                        _, max_val, _, max_loc = cv2.minMaxLoc(res)
                                        if max_val >= TMPL_ACCEPT_THRESH:
                                            p_info["x"] = rx0 + max_loc[0] + tmpl.shape[1] / 2.0
                                            p_info["y"] = ry0 + max_loc[1] + tmpl.shape[0] / 2.0
                                            p_info["status"] = "tracking"
                                            p_info["frozen_frames"] = 0
                                            p_info["confidence"] = float(max_val)
                                            p_info["last_seen"] = now
                                            recovered = True
                                else:
                                    # Full-frame search after prolonged freeze
                                    try:
                                        res_full = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
                                        _, max_val, _, max_loc = cv2.minMaxLoc(res_full)
                                        if max_val >= TMPL_ACCEPT_THRESH:
                                            p_info["x"] = max_loc[0] + tmpl.shape[1] / 2.0
                                            p_info["y"] = max_loc[1] + tmpl.shape[0] / 2.0
                                            p_info["status"] = "tracking"
                                            p_info["frozen_frames"] = 0
                                            p_info["confidence"] = float(max_val)
                                            p_info["last_seen"] = now
                                            p_info["anchor_x"] = p_info["x"]
                                            p_info["anchor_y"] = p_info["y"]
                                            recovered = True
                                    except Exception:
                                        pass

                            if not recovered:
                                p_info["confidence"] = 0.0
                            continue  # Skip optical flow processing for frozen trackers

                        if st[i, 0] == 1:
                            nx, ny = float(p1[i, 0, 0]), float(p1[i, 0, 1])
                            if not (0 <= nx < dw and 0 <= ny < dh):
                                p_info["status"] = "lost"
                                p_info["confidence"] = 0.0
                                continue

                            # ── Template-verify the optical flow candidate ──
                            accepted = False
                            if tmpl is not None:
                                th, tw = tmpl.shape[:2]
                                half_w, half_h = tw // 2, th // 2
                                vx0 = max(0, int(nx) - half_w)
                                vy0 = max(0, int(ny) - half_h)
                                vx1 = min(dw, vx0 + tw)
                                vy1 = min(dh, vy0 + th)
                                patch_w, patch_h = vx1 - vx0, vy1 - vy0
                                if patch_w == tw and patch_h == th:
                                    patch = gray[vy0:vy1, vx0:vx1]
                                    # patch is already 2D (H×W) — matchTemplate requires 2D
                                    res_v = cv2.matchTemplate(patch, tmpl, cv2.TM_CCOEFF_NORMED)
                                    score = float(res_v[0, 0])
                                    if score >= TMPL_ACCEPT_THRESH:
                                        accepted = True
                                        p_info["confidence"] = score
                                    else:
                                        p_info["confidence"] = score
                                else:
                                    # Near edge — accept optical flow with reduced confidence
                                    accepted = True
                                    p_info["confidence"] = 0.5
                            else:
                                # No template stored — accept optical flow blindly
                                accepted = True
                                p_info["confidence"] = 0.5

                            if accepted:
                                # ── Color-lock gate: candidate pixel must match arm HSV ──
                                hmean = p_info.get("hsv_mean")
                                hstd  = p_info.get("hsv_std")
                                if hmean is not None and hstd is not None:
                                    cand_hsv = hsv_now[int(ny), int(nx)].astype(float)
                                    hdiff    = np.abs(cand_hsv - hmean) / hstd
                                    # Hue wraps at 180 — handle the wrap-around
                                    if hdiff[0] > 90:
                                        hdiff[0] = 180.0 - hdiff[0]
                                    # Accept if within 2.5σ on at least S and V channels
                                    # (Hue alone is unreliable on metallic/grey parts)
                                    sv_ok = hdiff[1] <= 2.5 and hdiff[2] <= 2.5
                                    if not sv_ok:
                                        accepted = False  # Color mismatch — freeze

                                # ── Speed gate: arm moves slowly; fast motion = person ──
                                if accepted:
                                    dx_flow = nx - p_info["x"]
                                    dy_flow = ny - p_info["y"]
                                    speed_px = math.hypot(dx_flow, dy_flow)
                                    if speed_px > 18.0:  # > 18px/frame at 30fps ≈ very fast
                                        accepted = False  # Moving object, not the arm

                                # ── Max-drift guard ──
                                ax = p_info.get("anchor_x", nx)
                                ay = p_info.get("anchor_y", ny)
                                drift = math.hypot(nx - ax, ny - ay)
                                if drift > MAX_DRIFT_PX:
                                    # Snap back to anchor and re-verify via template
                                    p_info["x"] = ax
                                    p_info["y"] = ay
                                    p_info["status"] = "frozen"
                                    p_info["confidence"] = 0.0
                                else:
                                    p_info["x"] = nx
                                    p_info["y"] = ny
                                    p_info["status"] = "tracking"
                                    p_info["last_seen"] = now

                                    # Periodically refresh stored template to adapt to slow changes
                                    fc = p_info.get("frame_count", 0) + 1
                                    p_info["frame_count"] = fc
                                    if tmpl is not None and fc % TMPL_REFRESH_INTERVAL == 0:
                                        th, tw = tmpl.shape[:2]
                                        half_w, half_h = tw // 2, th // 2
                                        rx0 = max(0, int(nx) - half_w)
                                        ry0 = max(0, int(ny) - half_h)
                                        rx1 = min(dw, rx0 + tw)
                                        ry1 = min(dh, ry0 + th)
                                        if (rx1 - rx0) == tw and (ry1 - ry0) == th:
                                            new_tmpl = gray[ry0:ry1, rx0:rx1].copy()
                                            # Only update template if it still looks similar
                                            c2 = cv2.matchTemplate(
                                                new_tmpl, tmpl, cv2.TM_CCOEFF_NORMED
                                            )
                                            if float(c2[0, 0]) >= 0.55:
                                                p_info["template"] = new_tmpl
                                                # Also slowly drift the anchor toward current pos
                                                p_info["anchor_x"] = ax * 0.95 + nx * 0.05
                                                p_info["anchor_y"] = ay * 0.95 + ny * 0.05
                            else:
                                # ── FREEZE: optical flow followed wrong object ──
                                p_info["status"] = "frozen"
                                # Keep previous position (don't update x, y)
                        else:
                            # ── Optical flow totally lost (st=0): Full-frame pyramid search ──
                            # Search the ENTIRE frame at multiple scales (pyramid) so the
                            # tracker can re-attach from anywhere after full occlusion.
                            recovered = False
                            if tmpl is not None:
                                th_t, tw_t = tmpl.shape[:2]
                                best_val, best_loc, best_scale = -1.0, None, 1.0

                                # Try three scales: 100%, 90%, 110% to handle minor perspective change
                                for scale in (1.0, 0.9, 1.1):
                                    sw = max(tw_t, int(tw_t * scale))
                                    sh = max(th_t, int(th_t * scale))
                                    if sw > dw or sh > dh:
                                        continue
                                    if scale != 1.0:
                                        search_tmpl = cv2.resize(tmpl, (sw, sh))
                                    else:
                                        search_tmpl = tmpl
                                    if search_tmpl.shape[0] > gray.shape[0] or search_tmpl.shape[1] > gray.shape[1]:
                                        continue
                                    try:
                                        res = cv2.matchTemplate(gray, search_tmpl, cv2.TM_CCOEFF_NORMED)
                                        _, val, _, loc = cv2.minMaxLoc(res)
                                        if val > best_val:
                                            best_val, best_loc, best_scale = val, loc, scale
                                    except Exception:
                                        pass

                                if best_val >= TMPL_ACCEPT_THRESH and best_loc is not None:
                                    # Also verify color-lock at found position (avoid false positives)
                                    hmean = p_info.get("hsv_mean")
                                    hstd  = p_info.get("hsv_std")
                                    color_ok = True
                                    if hmean is not None and hstd is not None:
                                        try:
                                            sw = max(tw_t, int(tw_t * best_scale))
                                            sh = max(th_t, int(th_t * best_scale))
                                            fx = best_loc[0] + sw / 2.0
                                            fy = best_loc[1] + sh / 2.0
                                            cand_hsv = hsv_now[int(fy), int(fx)].astype(float)
                                            hdiff = np.abs(cand_hsv - hmean) / hstd
                                            if hdiff[0] > 90: hdiff[0] = 180.0 - hdiff[0]
                                            color_ok = hdiff[1] <= 3.0 and hdiff[2] <= 3.0
                                        except Exception:
                                            pass

                                    if color_ok:
                                        sw = max(tw_t, int(tw_t * best_scale))
                                        sh = max(th_t, int(th_t * best_scale))
                                        p_info["x"] = best_loc[0] + sw / 2.0
                                        p_info["y"] = best_loc[1] + sh / 2.0
                                        p_info["status"] = "tracking"
                                        p_info["confidence"] = float(best_val)
                                        p_info["last_seen"] = now
                                        # Update anchor to new re-found position
                                        p_info["anchor_x"] = p_info["x"]
                                        p_info["anchor_y"] = p_info["y"]
                                        recovered = True

                            if not recovered:
                                p_info["status"] = "lost"
                                p_info["confidence"] = 0.0

            self.prev_gray = gray.copy()
            self.last_frame = frame.copy()

            # 2. ArUco Marker Detection
            if self._detect is not None:
                corners, ids = self._detect(gray)
                if ids is not None and len(ids):
                    for c, mid in zip(corners, ids.flatten()):
                        pts = np.asarray(c, float).reshape(4, 2)
                        seen[int(mid)] = (
                            marker_angle(pts),
                            (float(pts[:, 0].mean()), float(pts[:, 1].mean())),
                            now,
                        )
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    for mid, (ang, (cx, cy), _t) in seen.items():
                        cv2.putText(
                            frame, f"{ang:+.0f}",
                            (int(cx) - 14, int(cy) + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 1, cv2.LINE_AA,
                        )

            # 3. Draw Limb Trackers (Vectors & Angles)
            with self.lock:
                limbs_copy = list(self.limb_trackers)
                points_copy = {k: dict(v) for k, v in self.point_trackers.items()}

            for limb in limbs_copy:
                p1_id, p2_id = limb["p1"], limb["p2"]
                if p1_id in points_copy and p2_id in points_copy:
                    pt1, pt2 = points_copy[p1_id], points_copy[p2_id]
                    s1, s2 = pt1.get("status", "tracking"), pt2.get("status", "tracking")
                    both_active = s1 in ("tracking", "frozen") and s2 in ("tracking", "frozen")
                    if both_active:
                        x1, y1 = int(round(pt1["x"])), int(round(pt1["y"]))
                        x2, y2 = int(round(pt2["x"])), int(round(pt2["y"]))
                        any_frozen = s1 == "frozen" or s2 == "frozen"
                        col = (140, 140, 140) if any_frozen else limb.get("color", (0, 255, 0))

                        cv2.line(frame, (x1, y1), (x2, y2), (20, 20, 20), 4, cv2.LINE_AA)
                        cv2.line(frame, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)

                        dx = pt2["x"] - pt1["x"]
                        dy = pt2["y"] - pt1["y"]
                        ang = math.degrees(math.atan2(dy, dx))

                        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
                        cv2.circle(frame, (mx, my), 3, (255, 255, 255), -1)
                        frozen_tag = " ⚠️" if any_frozen else ""
                        badge_text = f"{limb['name']}: {ang:+.0f}°{frozen_tag}"
                        cv2.putText(frame, badge_text, (mx - 25, my - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)

            # 4. Draw Point Trackers (with frozen/tracking/lost visual states)
            for _p_id, pt in points_copy.items():
                px, py = int(round(pt["x"])), int(round(pt["y"]))
                status = pt.get("status", "tracking")
                conf = pt.get("confidence", 1.0)

                if status == "tracking":
                    col = pt.get("color", (0, 255, 255))
                    # Bright crosshair for actively tracking
                    cv2.circle(frame, (px, py), 9, (15, 15, 15), 3, cv2.LINE_AA)
                    cv2.circle(frame, (px, py), 9, col, 1, cv2.LINE_AA)
                    cv2.circle(frame, (px, py), 2, col, -1)
                    cv2.drawMarker(frame, (px, py), col, markerType=cv2.MARKER_CROSS,
                                   markerSize=14, thickness=1, line_type=cv2.LINE_AA)
                    label = pt["name"]
                    if conf < 0.65:
                        label += f" ({conf:.0%})"
                    cv2.putText(frame, label, (px + 10, py - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
                elif status == "frozen":
                    # Gray dashed crosshair + "FROZEN" badge for occluded trackers
                    gray_col = (140, 140, 140)
                    cv2.circle(frame, (px, py), 11, (40, 40, 180), 2, cv2.LINE_AA)
                    cv2.circle(frame, (px, py), 2, gray_col, -1)
                    cv2.drawMarker(frame, (px, py), gray_col, markerType=cv2.MARKER_CROSS,
                                   markerSize=14, thickness=1, line_type=cv2.LINE_AA)
                    cv2.putText(frame, f"{pt['name']} [FROZEN]", (px + 10, py - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 100, 255), 1, cv2.LINE_AA)
                else:
                    # Lost — dim marker
                    dim_col = (80, 80, 80)
                    cv2.circle(frame, (px, py), 9, dim_col, 1, cv2.LINE_AA)
                    cv2.drawMarker(frame, (px, py), dim_col, markerType=cv2.MARKER_CROSS,
                                   markerSize=14, thickness=1, line_type=cv2.LINE_AA)
                    cv2.putText(frame, f"{pt['name']} [LOST]", (px + 10, py - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 80, 200), 1, cv2.LINE_AA)

            # 5. Overlay Camera Header
            header_text = "OVERHEAD CAM (TOP-DOWN)" if "overhead" in self.name.lower() else "SIDE CAM (LATERAL)"
            cv2.putText(
                frame, header_text,
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

            ok_enc, png = cv2.imencode(".png", frame)

            with self.lock:
                self._markers.update(seen)
                if ok_enc:
                    self._png = png.tobytes()
                    self._frame_no += 1

            frames += 1
            elapsed = time.time() - t_prev
            if elapsed >= 1.0:
                self.fps = frames / elapsed
                t_prev, frames = time.time(), 0

        cap.release()

    def snapshot(self):
        now = time.time()
        with self.lock:
            fresh_markers = {
                mid: (a, xy, t)
                for mid, (a, xy, t) in self._markers.items()
                if now - t <= MARKER_MAX_AGE
            }
            fresh_points = {}
            for pid, p in self.point_trackers.items():
                pd = dict(p)
                pd.pop("template", None)  # Don't include binary blob in snapshot
                pd.setdefault("confidence", 1.0)
                pd.setdefault("status", "tracking")
                fresh_points[pid] = pd
            limb_data = {}
            for l in self.limb_trackers:
                lid = l["id"]
                p1_id, p2_id = l["p1"], l["p2"]
                if p1_id in fresh_points and p2_id in fresh_points:
                    pt1, pt2 = fresh_points[p1_id], fresh_points[p2_id]
                    s1 = pt1.get("status", "tracking")
                    s2 = pt2.get("status", "tracking")
                    both_usable = s1 in ("tracking", "frozen") and s2 in ("tracking", "frozen")
                    if both_usable:
                        dx = pt2["x"] - pt1["x"]
                        dy = pt2["y"] - pt1["y"]
                        ang = math.degrees(math.atan2(dy, dx))
                        any_frozen = s1 == "frozen" or s2 == "frozen"
                        limb_data[lid] = {
                            "name": l["name"],
                            "p1": p1_id,
                            "p2": p2_id,
                            "angle": ang,
                            "valid": True,
                            "frozen": any_frozen,
                            "vector": (dx, dy),
                        }
                    else:
                        limb_data[lid] = {"name": l["name"], "valid": False, "angle": None}

            return {
                "png": self._png,
                "frame": self._frame_no,
                "markers": fresh_markers,
                "points": fresh_points,
                "limbs": limb_data,
                "running": self.running,
                "fps": self.fps,
                "error": self.error,
                "index": self.index,
            }


class DualArmTracker:
    """Coordinates Overhead (Top-Down) and Side (Lateral) cameras."""

    def __init__(self):
        self.detector = _make_detector() if opencv_ok() else None
        self.overhead_cam = CameraWorker(name="Overhead", detector=self.detector)
        self.side_cam = CameraWorker(name="Side", detector=self.detector)

        # Aliases for backwards compatibility
        self.left_cam = self.overhead_cam
        self.right_cam = self.side_cam

    def start(self, overhead_index=0, side_index=1, left_index=None, right_index=None):
        """Start both cameras."""
        if left_index is not None:
            overhead_index = left_index
        if right_index is not None:
            side_index = right_index

        if not opencv_ok():
            return False, "OpenCV with aruco is not installed (pip install opencv-python)"

        ok_overhead = self.overhead_cam.start(overhead_index, self.detector)
        ok_side = self.side_cam.start(side_index, self.detector)
        return ok_overhead and ok_side, ""

    def stop(self):
        self.overhead_cam.stop()
        self.side_cam.stop()

    def is_running(self):
        return self.overhead_cam.running or self.side_cam.running

    def snapshot(self):
        """Return combined snapshot of both cameras."""
        snap_overhead = self.overhead_cam.snapshot()
        snap_side = self.side_cam.snapshot()

        return {
            "overhead": snap_overhead,
            "side": snap_side,
            # Backwards-compatibility aliases:
            "left": snap_overhead,
            "right": snap_side,
            "running": snap_overhead["running"] or snap_side["running"],
            "errors": [err for err in (snap_overhead["error"], snap_side["error"]) if err],
        }

    def get_webcam_joint_angle(self, j_idx, invert=False, offset=0.0):
        """Calculate joint angle from webcam point & limb trackers.

        j_idx:
          0: J1 Base Yaw -> from Overhead cam (Base Center -> Turret Ref)
          1: J2 Shoulder -> from Side cam (Shoulder -> Elbow)
          2: J3 Elbow -> from Side cam relative angle: (Elbow -> Wrist) relative to (Shoulder -> Elbow)
          3: J4 Wrist -> from Side cam relative angle: (Wrist -> Gripper) relative to (Elbow -> Wrist)
        """
        snap = self.snapshot()
        overhead_limbs = snap["overhead"]["limbs"]
        side_limbs = snap["side"]["limbs"]

        raw_ang = None

        if j_idx == 0:
            limb = overhead_limbs.get("base_yaw") or overhead_limbs.get("turret") or overhead_limbs.get("j1")
            if limb and limb.get("valid"):
                v = limb.get("vector")
                if v:
                    raw_ang = math.degrees(math.atan2(-v[1], v[0]))
                else:
                    raw_ang = limb.get("angle")

        elif j_idx == 1:
            limb = side_limbs.get("upper_arm") or side_limbs.get("shoulder") or side_limbs.get("j2")
            if limb and limb.get("valid"):
                v = limb.get("vector")
                if v:
                    # In neutral pose, upper arm points straight up: (dx=0, dy<0)
                    # angle relative to vertical = atan2(dx, -dy)
                    raw_ang = math.degrees(math.atan2(v[0], -v[1]))
                else:
                    raw_ang = limb.get("angle")

        elif j_idx == 2:
            l_upper = side_limbs.get("upper_arm") or side_limbs.get("j2")
            l_fore = side_limbs.get("forearm") or side_limbs.get("j3")
            if l_upper and l_upper.get("valid") and l_fore and l_fore.get("valid"):
                v1, v2 = l_upper.get("vector"), l_fore.get("vector")
                if v1 and v2:
                    cross = v1[0] * v2[1] - v1[1] * v2[0]
                    dot = v1[0] * v2[0] + v1[1] * v2[1]
                    raw_ang = math.degrees(math.atan2(cross, dot))

        elif j_idx == 3:
            l_fore = side_limbs.get("forearm") or side_limbs.get("j3")
            l_wrist = side_limbs.get("wrist") or side_limbs.get("gripper") or side_limbs.get("j4")
            if l_fore and l_fore.get("valid") and l_wrist and l_wrist.get("valid"):
                v1, v2 = l_fore.get("vector"), l_wrist.get("vector")
                if v1 and v2:
                    cross = v1[0] * v2[1] - v1[1] * v2[0]
                    dot = v1[0] * v2[0] + v1[1] * v2[1]
                    raw_ang = math.degrees(math.atan2(cross, dot))

        if raw_ang is None:
            return None

        return (-1.0 if invert else 1.0) * wrap180(raw_ang - offset)

    def get_joint_angle(self, camera_source, parent_id, child_id, invert=False,
                        offset_left=0.0, offset_right=0.0):
        """Calculate joint angle using Overhead, Side, or Auto (whichever has fresh markers)."""
        snap = self.snapshot()
        overhead_markers = snap["overhead"]["markers"]
        side_markers = snap["side"]["markers"]

        source = str(camera_source).lower()

        if source in ("overhead", "top", "left"):
            return joint_angle(overhead_markers, parent_id, child_id, invert, offset_left)
        elif source in ("side", "right"):
            return joint_angle(side_markers, parent_id, child_id, invert, offset_right)
        else:
            # Auto: check overhead first for J1 or if available, then side
            ang_overhead = joint_angle(overhead_markers, parent_id, child_id, invert, offset_left)
            if ang_overhead is not None:
                return ang_overhead
            return joint_angle(side_markers, parent_id, child_id, invert, offset_right)
