"""NEXUS Arm Camera Diagnostic & Index Finder Tool.

Run this script to:
1. Detect all connected video capture devices in Windows Device Manager.
2. Probe all OpenCV camera indices (0 to 6) across backends (DSHOW, MSMF, ANY).
3. Open live preview windows so you can visually identify which index corresponds to which physical camera (Side vs Overhead).
"""
import sys
import subprocess
import time

try:
    import cv2
except ImportError:
    print("OpenCV is not installed. Run: py -m pip install opencv-python")
    sys.exit(1)


def check_windows_pnp_cameras():
    print("=" * 60)
    print("1. CHECKING WINDOWS DEVICE MANAGER (PnP)")
    print("=" * 60)
    cmd = (
        "Get-CimInstance Win32_PnPEntity | "
        "Where-Object { $_.PNPClass -match 'Camera|Image' -or $_.Name -match 'Camera|Webcam|Video' } | "
        "Select-Object Name, PNPClass, Status, ConfigManagerErrorCode | "
        "Format-Table -AutoSize"
    )
    try:
        res = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                             capture_output=True, text=True, timeout=10)
        output = res.stdout.strip()
        if output:
            print(output)
        else:
            print("No camera devices reported by Windows PnP.")
    except Exception as e:
        print("Could not query Windows PnP:", e)


def probe_opencv_cameras(max_indices=6):
    print("\n" + "=" * 60)
    print("2. PROBING OPENCV CAMERA INDICES")
    print("=" * 60)

    backends = [
        ("DirectShow (CAP_DSHOW)", cv2.CAP_DSHOW),
        ("MediaFoundation / Default (CAP_ANY)", cv2.CAP_ANY),
    ]

    working_cameras = []

    for idx in range(max_indices):
        found = False
        for b_name, b_flag in backends:
            try:
                cap = cv2.VideoCapture(idx, b_flag)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        h, w = frame.shape[:2]
                        print(f"  [SUCCESS] Camera Index {idx} FOUND using {b_name} (Resolution: {w}x{h})")
                        working_cameras.append((idx, b_flag, b_name, w, h))
                        found = True
                        cap.release()
                        break
                    cap.release()
            except Exception:
                pass
        if not found:
            print(f"  [--] Index {idx}: No camera responding")

    print("\nSummary of working camera indices:", [c[0] for c in working_cameras])
    return working_cameras


def visual_camera_preview(working_cameras):
    if not working_cameras:
        print("\n" + "!" * 60)
        print("NO WORKING CAMERAS FOUND!")
        print("Troubleshooting checklist:")
        print("1. Are your USB webcams plugged in securely?")
        print("2. In Windows Settings -> Privacy & security -> Camera:")
        print("   Make sure 'Camera access' is ON and 'Let desktop apps access your camera' is ON.")
        print("3. If using a USB hub, try plugging directly into the laptop ports.")
        print("!" * 60)
        return

    print("\n" + "=" * 60)
    print("3. OPENING VISUAL PREVIEW WINDOWS")
    print("=" * 60)
    print("Press 'q' or ESC in any preview window to exit.\n")

    caps = []
    for idx, b_flag, b_name, w, h in working_cameras:
        cap = cv2.VideoCapture(idx, b_flag)
        if cap.isOpened():
            caps.append((idx, cap))

    if not caps:
        print("Could not reopen cameras for preview.")
        return

    try:
        while True:
            for idx, cap in caps:
                ret, frame = cap.read()
                if ret:
                    cv2.putText(frame, f"CAMERA INDEX {idx}", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    cv2.putText(frame, "Press 'q' to close", (20, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow(f"Camera Index {idx}", frame)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == 27:
                break
    finally:
        for _, cap in caps:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    check_windows_pnp_cameras()
    working = probe_opencv_cameras()
    visual_camera_preview(working)
