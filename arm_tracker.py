"""Webcam tracking of the real arm using printed ArUco markers.

Stick one square marker on each link (see make_marker_sheet). The camera finds each
marker, and the rotation of a marker in the picture gives the angle of its link.
A joint angle is the angle of the child link's marker minus the parent link's marker.

Needs OpenCV 4.7 or newer:  pip install opencv-python
"""
import math
import sys
import threading
import time

import numpy as np

try:
    import cv2
except ImportError:                      # the rest of the GUI still works without a webcam
    cv2 = None

DICT_NAME = "DICT_4X4_50"
DISPLAY_SIZE = (320, 240)
MARKER_MAX_AGE = 0.4                     # a marker not seen for this long counts as lost


def wrap180(angle):
    return (angle + 180.0) % 360.0 - 180.0


def opencv_ok():
    return cv2 is not None and hasattr(cv2, "aruco")


def _dictionary():
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT_NAME))


def _make_detector():
    """Return a function gray_image -> (corners, ids). Works on old and new OpenCV."""
    dic = _dictionary()
    if hasattr(cv2.aruco, "ArucoDetector"):
        det = cv2.aruco.ArucoDetector(dic, cv2.aruco.DetectorParameters())
        return lambda gray: det.detectMarkers(gray)[:2]
    params = cv2.aruco.DetectorParameters_create()
    return lambda gray: cv2.aruco.detectMarkers(gray, dic, parameters=params)[:2]


def marker_angle(corners):
    """In-picture rotation of a marker in degrees (clockwise is positive on screen)."""
    c = np.asarray(corners, float).reshape(4, 2)
    d = c[1] - c[0]
    return math.degrees(math.atan2(d[1], d[0]))


def raw_relative(markers, parent_id, child_id):
    """Angle of the child marker relative to the parent marker, or None if either is missing."""
    if parent_id is None or child_id is None:
        return None
    if parent_id not in markers or child_id not in markers:
        return None
    return wrap180(markers[child_id][0] - markers[parent_id][0])


def joint_angle(markers, parent_id, child_id, invert=False, offset=0.0):
    """Joint angle in degrees, 0 in the calibrated zero pose. None if a marker is not visible."""
    rel = raw_relative(markers, parent_id, child_id)
    if rel is None:
        return None
    return (-1.0 if invert else 1.0) * wrap180(rel - offset)


def make_marker_sheet(path, ids=range(0, 6), marker_px=360):
    """Save a printable sheet of markers. Print it, cut the markers out, stick one on each link."""
    if not opencv_ok():
        raise RuntimeError("OpenCV with the aruco module is not installed")
    dic = _dictionary()
    pad = 60
    tile = marker_px + 2 * pad
    ids = list(ids)
    cols = 3
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
        cv2.putText(sheet, f"ID {mid}", (x0 + pad, y0 + pad - 18), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, 0, 2, cv2.LINE_AA)
        cv2.putText(sheet, "this edge up", (x0 + pad, y0 + pad + marker_px + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 90, 1, cv2.LINE_AA)
    cv2.imwrite(path, sheet)
    return path


class ArmTracker:
    """Reads the webcam in a background thread and keeps the latest marker angles."""

    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.error = ""
        self.fps = 0.0
        self._png = None
        self._frame_no = 0
        self._markers = {}               # id -> (angle, (cx, cy), time last seen)
        self._detect = None

    # --------------------------------------------------------------- control
    def start(self, index):
        if not opencv_ok():
            self.error = "OpenCV with aruco is not installed (pip install opencv-python)"
            return False
        self.stop()
        self.error = ""
        self.running = True
        self.thread = threading.Thread(target=self._loop, args=(int(index),), daemon=True)
        self.thread.start()
        return True

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self.thread = None
        with self.lock:
            self._markers = {}
            self._png = None

    # ---------------------------------------------------------------- frames
    def process_frame(self, frame, now=None):
        """Detect markers in one BGR frame and update the stored angles. Returns the annotated frame."""
        if self._detect is None:
            self._detect = _make_detector()
        now = time.time() if now is None else now
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = self._detect(gray)
        seen = {}
        if ids is not None and len(ids):
            for c, mid in zip(corners, ids.flatten()):
                pts = np.asarray(c, float).reshape(4, 2)
                seen[int(mid)] = (marker_angle(pts), (float(pts[:, 0].mean()), float(pts[:, 1].mean())), now)
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for mid, (ang, (cx, cy), _t) in seen.items():
                cv2.putText(frame, f"{ang:+.0f}", (int(cx) - 14, int(cy) + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        small = cv2.resize(frame, DISPLAY_SIZE)
        ok, png = cv2.imencode(".png", small)
        with self.lock:
            self._markers.update(seen)
            if ok:
                self._png = png.tobytes()
                self._frame_no += 1
        return frame

    def _loop(self, index):
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else 0
        cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
        if not cap.isOpened():
            self.error = f"Could not open camera {index}"
            self.running = False
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        t_prev, frames = time.time(), 0
        while self.running:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            self.process_frame(frame)
            frames += 1
            if time.time() - t_prev >= 1.0:
                self.fps = frames / (time.time() - t_prev)
                t_prev, frames = time.time(), 0
        cap.release()

    # -------------------------------------------------------------- results
    def snapshot(self):
        """Latest picture (PNG bytes), frame counter and the markers seen recently."""
        now = time.time()
        with self.lock:
            fresh = {mid: (a, xy, t) for mid, (a, xy, t) in self._markers.items()
                     if now - t <= MARKER_MAX_AGE}
            return {"png": self._png, "frame": self._frame_no, "markers": fresh,
                    "running": self.running, "fps": self.fps, "error": self.error}
