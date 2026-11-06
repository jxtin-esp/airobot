# NEXUS Arm v4

3D arm you can drag, webcam tracking of the real arm, webcam-based limits, remappable servo pins.

## Files

| File | What it is |
|---|---|
| `nexus_arm_v4.py` | The program. Run this one. |
| `arm_model.py`, `arm_tracker.py`, `arm_control.py` | Parts of the program. Keep them in the same folder. |
| `nexus_arm_v4_fw/nexus_arm_v4_fw.ino` | Arduino firmware. Upload it first. |
| `arm_config.json` | Created automatically. Holds your pins, limits and calibration. |

## 1. Install (internet needed once)

```powershell
py -m pip install pyserial opencv-python numpy
```

OpenCV 4.7 or newer is needed for the marker detection. `py -c "import cv2; print(cv2.__version__)"` shows the version.

## 2. Upload the firmware

Copy this folder to `C:\nexus_arm_v4`, then (replace COM3 with your port):

```powershell
cd C:\nexus_arm_v4
arduino-cli compile --fqbn arduino:avr:uno nexus_arm_v4_fw
arduino-cli upload -p COM3 --fqbn arduino:avr:uno nexus_arm_v4_fw
```

The v4 program needs the v4 firmware. Older firmware does not understand the pin and speed commands.

## 3. Run

```powershell
cd C:\nexus_arm_v4
py nexus_arm_v4.py
```

Close the Arduino Serial Monitor first. Pick the port, press **Connect**, and wait two seconds.
Support the arm before connecting, because the Uno resets and the positional servos go limp for a moment.

## 4. Set up the webcam tracking

1. Press **Save marker sheet**. Print the PNG so each black square is about 4 cm wide (scale it in the print dialog). Cut the markers out.
2. Stick one marker flat on each part, on the side the camera sees:

   | Marker ID | Where |
   |---|---|
   | 0 | The base (does not move) |
   | 1 | Upper arm |
   | 2 | Forearm |
   | 3 | Wrist / gripper body |
   | 5 | Top of the turret (only for a camera looking down, for J1) |

3. Put the camera to the SIDE of the arm, looking straight at it, so the arm moves in the picture's plane. Keep the base turned to its zero position. Good light, no glare.
4. Press **Start camera**. The preview should show green boxes and the markers' IDs.
5. Move the arm to the neutral pose (upper arm vertical, forearm pointing forward, wrist straight) and press **Calibrate zero pose**.
6. Move a joint. If the 3D arm moves the wrong way, tick **Invert** for that joint in the **Marker tracking** tab.

One camera sees one view. A side view gives J2, J3 and J4. J1 (base) needs a view from above, so by default it is moved with Left / Right and has no limit protection.

## 5. Use it

- **Drag a numbered handle** in the 3D view. The orange outline is the target. Joints J1 to J3 turn until the webcam sees the target angle. Joints J4 to J6 follow at once (tick **Positional servos ON**).
- **Left / Right** buttons move a continuous joint while you hold them.
- **Limits:** set Min and Max for each joint. A tracked joint slows down near a limit, stops a few degrees before it (the "stop early" box), and can only move back inward.
- **Servo pins:** open the **Servo pins** tab, choose a pin for each servo, move the signal wire to match, press **Apply pins**.
- **STOP ALL** stops every continuous joint.

## What to expect

- The webcam adds delay (about 0.1 to 0.3 s), so a continuous joint can still coast a few degrees past a target or limit. The program slows it down near the edge to reduce that. Keep the speed at 60% or lower when limits matter.
- Angles are accurate to a few degrees, not fractions of a degree.
- A joint that is not tracked has NO limit protection. The banner warns you each time.
- If the webcam loses a tracked joint while it moves, that joint stops.
- Software limits do not replace mechanical stops. Never rely on them to protect the arm from a hard stall.
- The 3D arm is an approximation of the real arm's shape.
