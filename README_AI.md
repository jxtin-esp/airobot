# NEXUS Arm v4 AI - Dual-Camera Tracking (Overhead + Side) & Local Ollama Interface

Unified robotic arm controller featuring **Overhead (Top-Down) + Side (Lateral) Vision Tracking**, interactive **Webcam AI Point & Limb Tracking**, 3D Digital Twin, complete manual controls, and natural language motion planning via local **Ollama** running `qwen2.5:3b`.

---

## 1. Quick Start

### A. Requirements
Ensure required packages are installed:
```powershell
py -m pip install pyserial opencv-python numpy
```

Ensure Ollama is installed and the model is pulled:
```powershell
ollama pull qwen2.5:3b
```
*(Make sure the Ollama desktop application or service is running in the background).*

### B. Launching the Controller
```powershell
py nexus_arm_v4_ai.py
```

---

## 2. Dual Camera Topology (Overhead + Side View)

The vision system uses two webcams working in harmony:
1. **Webcam 1: Overhead View (Top-Down)**
   - Positioned directly above the arm looking down at the base, turret, and reach area.
   - Tracks **J1 Base Yaw** rotation and planar coordinate reaching.
2. **Webcam 2: Side View (Lateral)**
   - Positioned to the side of the arm looking horizontally across the joints.
   - Tracks **J2 Shoulder Pitch**, **J3 Elbow Pitch**, and **J4 Wrist Pitch** elevation angles.

---

## 3. Webcam AI Point & Limb Tracking (No Markers Required!)

You can track joints and limbs directly through the live webcam video without needing printed paper tags:

### A. Click-to-Track on Live Video
1. Click the **🎯 Click Cam to Add Tracker** button below the video feeds.
2. Click directly on any joint, bolt, or servo horn in the video (e.g. click on the Shoulder pivot, Elbow pivot, Wrist pivot, or Base).
3. **⚡ AI Auto-Snap:**
   - The computer vision engine analyzes the local neighborhood using Shi-Tomasi corner detection and **automatically snaps the tracker onto the exact optical center of the bolt or joint**.
4. The tracker locks onto the feature and tracks it in real-time at 60+ FPS using **Lucas-Kanade optical flow pyramids (`calcOpticalFlowPyrLK`)** with template matching re-anchoring to prevent drift.

### B. Auto-Linking Limbs (Kinematic Vectors)
1. After placing trackers on your joints (e.g. Shoulder and Elbow), click **⚡ Auto-Link Limbs**.
2. The system automatically connects:
   - **Upper Arm Limb:** Vector from `J2 Shoulder` ➔ `J3 Elbow`
   - **Forearm Limb:** Vector from `J3 Elbow` ➔ `J4 Wrist`
   - **Base Turret Limb:** Vector from `Base Center` ➔ `Turret Pivot`
3. A glowing neon skeleton line is rendered directly on the live video, showing the real-time angle of each limb!
4. Relative joint angles (e.g. Elbow angle between Upper Arm and Forearm) are computed dynamically from vector geometry and feed directly into the arm's closed-loop control system.

### C. Tracker Management
* **🔄 Reset Points:** Re-anchors trackers at their starting coordinates if obstructed.
* **🗑️ Clear Trackers:** Clears all active webcam point and limb trackers.

---

## 4. Visual Body Markers & Joint Trackers Tab (🎯)

In addition to webcam point tracking, the system also supports physical printed ArUco tags (`DICT_4X4_50`):

### A. Body Markers (3D Twin Synchronization)
* **Interactive 3D Canvas Click:**
  * Click any marker badge (`M0`, `M1`, `M2`, etc.) directly on the 3D arm model or in the table to inspect it.
  * Selected markers glow in **amber**.
  * When detected live by either camera, markers turn **bright green** (`● Seen`).
* **Moving Markers on the Arm Body:**
  * Adjust the **Position on Link** slider (0% to 100%).
  * The marker badge glides along the robotic arm link in **real-time** on the 3D twin.
* **Locking to Arm Links:**
  * Click **🔒 Lock to Link / 🔓 Unlock**.
  * When locked (`🔒`), the marker is fastened rigidly to that link's coordinate frame (`base`, `turret`, `upper_arm`, `forearm`, `wrist`, `gripper`).
* **Add, Remove & Reset:**
  * **➕ Add Marker:** Creates a new marker with the next available ID.
  * **🗑️ Remove Marker:** Deletes the selected marker.
  * **🔄 Reset:** Restores the standard 5-marker layout.

### B. Joint Kinematic Trackers
* **Relative Tracking Pairs:**
  * Track relative angle between Parent and Child markers or choose **Overhead** / **Side** / **Auto** cameras.
* **⚡ Auto-Bind Standard Trackers:**
  * Automatically sets standard kinematic pairs:
    * J1 Base Yaw: `M0 (base)` ➔ `M5 (turret)`
    * J2 Shoulder: `M0 (base)` ➔ `M1 (upper_arm)`
    * J3 Elbow: `M1 (upper_arm)` ➔ `M2 (forearm)`
    * J4 Wrist: `M2 (forearm)` ➔ `M3 (wrist)`
* **Per-Joint Zero Calibration:**
  * Click the individual **Zero** button beside any joint to zero just that joint from current video feedback.

---

## 5. Manual Control Options

1. **Top Mode Buttons:**
   * Switch between **🎮 Manual Control** and **🤖 AI Assistant** with a single click.
2. **Persistent Quick-Jog Bar (Under the 3D Canvas):**
   * Jog J1 (Base), J2 (Shoulder), J3 (Elbow), or Gripper (Open/Mid/Close) at any time without switching tabs.
3. **Interactive 3D Handles:**
   * Drag any numbered handle (1 to 6) in the 3D viewport to command angles directly.
4. **Keyboard Shortcuts (Active when not typing in text boxes):**
   * `←` / `→` : J1 Base Left / Right
   * `↑` / `↓` : J2 Shoulder Forward / Back
   * `PageUp` / `PageDown` : J3 Elbow Up / Down
   * `[` / `]` : Gripper Close / Open
   * `Spacebar` : Emergency Stop All

---

## 6. Using the AI Assistant (qwen2.5:3b)

1. Ensure the top-right badge shows `● Ollama Connected (qwen2.5:3b)`.
2. In the **AI Assistant** tab:
   * Type any natural language prompt into the box, e.g.:
     * *"Pick up the block in front of you"*
     * *"Wave to the crowd"*
     * *"Turn 45 degrees left and bow down"*
     * *"Open the gripper, reach down, and grip firmly"*
   * Or click any of the preset buttons: **Home Pose**, **Wave Greeting**, **Reach & Grab**, **Release Object**, **Rest Pose**.
3. The AI reasoning (`thought`), user summary, and structured plan steps will appear in the log.
4. Click **Run Plan** to execute the plan in closed-loop, or **Step Once** to advance one step at a time.
5. If at any time you need to stop, click **Abort AI Plan** or the global red **STOP ALL** button.
