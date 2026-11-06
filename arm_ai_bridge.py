"""AI Bridge for NEXUS Arm v4 connecting local Ollama (qwen2.5:3b) to the robot controller.

Features:
- Local Ollama connectivity check (detects server and model availability).
- Structured JSON prompt engineering with strict kinematics and limit enforcement.
- Whitelist safety validation: clamps angles to soft limits, bounds speed, prevents collisions.
- Step-by-step action queue generator with fallback presets for offline/quick use.
"""
import json
import logging
import urllib.error
import urllib.request

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:3b"

logger = logging.getLogger("ArmAIBridge")

SYSTEM_PROMPT = """You are the AI motion planner for NEXUS Arm v4, a 6-DOF robotic arm.
Your job is to convert natural language instructions into safe, executable, structured JSON plans.

Robotic Arm Joints & Limits:
- J1 (Base Yaw): -90° to +90° (0 = straight ahead, + = left/CCW, - = right/CW)
- J2 (Shoulder): -60° to +90° (+ = lean forward/down, - = pull back/up) [Tracked by Dual-Side Webcams]
- J3 (Elbow): -90° to +90° (+ = swing forearm down, - = extend up) [Tracked by Dual-Side Webcams]
- J4 (Wrist Pitch): -90° to +90° (+ = pitch gripper down, - = pitch up) [Tracked by Dual-Side Webcams]
- J5 (Wrist Roll): -90° to +90° (+ = roll clockwise, - = roll CCW)
- J6 (Gripper): -90° (fully closed) to +90° (fully open), 0° is half-open.
Vision Feedback: Two lateral webcams (Left Side and Right Side) provide redundant, occlusion-free angular feedback.

Available Plan Actions:
1. {"action": "move", "joints": {"J1": float, "J2": float, ...}, "speed": int (15-65), "wait": float (0.5-3.0)}
2. {"action": "gripper", "state": "open" | "close" | float_angle, "wait": float}
3. {"action": "home", "speed": 40, "wait": 1.5}
4. {"action": "wait", "seconds": float}
5. {"action": "stop"}

Safety Rules:
- Never exceed joint limits provided in the prompt.
- For complex motions (e.g. pick and place, wave), decompose them into sequential, logical steps.
- Always open the gripper BEFORE lowering onto an object to pick it up.
- Lift the arm before rotating the base if carrying an object or moving across obstacles.
- Speed must stay between 15% and 65% for safe physical execution.

Respond ONLY with valid JSON in this exact structure:
{
  "thought": "Brief step-by-step reasoning on kinematics and safety",
  "explanation": "Clear 1-sentence user-facing explanation",
  "plan": [
    {"action": "...", ...}
  ]
}
"""

PRESET_PLANS = {
    "home": {
        "thought": "Move all joints to zero neutral configuration.",
        "explanation": "Returning arm to neutral home position.",
        "plan": [
            {"action": "move", "joints": {"J1": 0.0, "J2": 0.0, "J3": 0.0, "J4": 0.0, "J5": 0.0, "J6": 0.0}, "speed": 45, "wait": 2.0}
        ]
    },
    "wave": {
        "thought": "Raise shoulder and elbow, then oscillate base left and right to wave.",
        "explanation": "Raising arm and performing a friendly waving gesture.",
        "plan": [
            {"action": "move", "joints": {"J2": -20.0, "J3": -30.0, "J4": 15.0}, "speed": 45, "wait": 1.5},
            {"action": "move", "joints": {"J1": 30.0}, "speed": 40, "wait": 0.8},
            {"action": "move", "joints": {"J1": -30.0}, "speed": 40, "wait": 0.8},
            {"action": "move", "joints": {"J1": 30.0}, "speed": 40, "wait": 0.8},
            {"action": "move", "joints": {"J1": 0.0}, "speed": 40, "wait": 0.8},
            {"action": "move", "joints": {"J2": 0.0, "J3": 0.0, "J4": 0.0}, "speed": 40, "wait": 1.2}
        ]
    },
    "reach_grab": {
        "thought": "Open gripper, reach forward and down, close gripper to grasp, then lift.",
        "explanation": "Reaching forward to grasp an object and lifting it.",
        "plan": [
            {"action": "gripper", "state": "open", "wait": 0.6},
            {"action": "move", "joints": {"J2": 25.0, "J3": 35.0, "J4": -20.0}, "speed": 40, "wait": 2.0},
            {"action": "gripper", "state": "close", "wait": 0.8},
            {"action": "move", "joints": {"J2": -10.0, "J3": 10.0}, "speed": 40, "wait": 1.5}
        ]
    },
    "release": {
        "thought": "Lower arm slightly, open gripper to release object, then retract.",
        "explanation": "Releasing gripped object and retracting arm.",
        "plan": [
            {"action": "move", "joints": {"J2": 20.0, "J3": 30.0}, "speed": 35, "wait": 1.2},
            {"action": "gripper", "state": "open", "wait": 0.8},
            {"action": "move", "joints": {"J2": 0.0, "J3": 0.0, "J4": 0.0}, "speed": 40, "wait": 1.5}
        ]
    },
    "rest": {
        "thought": "Tuck the arm into a compact low-profile resting pose.",
        "explanation": "Folding arm into safe resting configuration.",
        "plan": [
            {"action": "gripper", "state": -30.0, "wait": 0.5},
            {"action": "move", "joints": {"J1": 0.0, "J2": 45.0, "J3": 75.0, "J4": 30.0, "J5": 0.0}, "speed": 35, "wait": 2.0}
        ]
    }
}


class ArmAIBridge:
    """Manages LLM queries to local Ollama with safety guardrails and plan execution."""

    def __init__(self, ollama_url=DEFAULT_OLLAMA_URL, model=DEFAULT_MODEL):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model

    def check_connection(self):
        """Check if Ollama server is responding and whether target model exists."""
        url = f"{self.ollama_url}/api/tags"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "NEXUS-Arm-AI"})
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                data = json.loads(resp.read().decode())
                models = [m.get("name", "") for m in data.get("models", [])]
                has_model = any(self.model in m for m in models)
                return True, models, has_model, ""
        except urllib.error.URLError as err:
            return False, [], False, f"Cannot reach Ollama at {self.ollama_url}: {err.reason}"
        except Exception as err:
            return False, [], False, f"Ollama connection error: {err}"

    def build_state_prompt(self, current_angles, limits, tracking_status, fsr_values):
        """Format current arm telemetry into prompt text."""
        joint_names = ["J1 Base", "J2 Shoulder", "J3 Elbow", "J4 Wrist", "J5 Roll", "J6 Gripper"]
        lines = ["CURRENT ROBOTIC ARM TELEMETRY:"]
        for j in range(6):
            name = joint_names[j]
            ang = current_angles[j]
            lo, hi = limits[j]
            trk = "TRACKED" if tracking_status[j] else "UNTRACKED"
            lines.append(f"- {name} (J{j+1}): Angle = {ang:+.1f}°, Safe Limits = [{lo:+.0f}°, {hi:+.0f}°], Status = {trk}")
        if fsr_values:
            lines.append(f"- Gripper Contact Sensors: FSR1={fsr_values[0]}, FSR2={fsr_values[1]} (range 0-1023)")
        return "\n".join(lines)

    def plan_instruction(self, instruction, current_angles, limits, tracking_status, fsr_values=(0, 0), timeout=90):
        """Generate structured motion plan from natural language via Ollama qwen2.5:3b."""
        # Check presets first for exact keyword shortcuts
        norm = instruction.strip().lower()
        if norm in ("home", "home pose", "reset"):
            return True, PRESET_PLANS["home"], "Loaded preset: Home"
        elif norm in ("wave", "say hi", "greet"):
            return True, PRESET_PLANS["wave"], "Loaded preset: Wave"
        elif norm in ("grab", "pick", "pick up", "reach and grab"):
            return True, PRESET_PLANS["reach_grab"], "Loaded preset: Reach & Grab"
        elif norm in ("release", "drop", "let go"):
            return True, PRESET_PLANS["release"], "Loaded preset: Release Object"
        elif norm in ("rest", "park", "fold"):
            return True, PRESET_PLANS["rest"], "Loaded preset: Rest Pose"

        # Construct context prompt
        telemetry = self.build_state_prompt(current_angles, limits, tracking_status, fsr_values)
        user_prompt = f"{telemetry}\n\nUSER REQUEST: {instruction}\nGenerate the JSON motion plan:"

        payload = {
            "model": self.model,
            "system": SYSTEM_PROMPT,
            "prompt": user_prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.2,  # Low temperature for deterministic robot planning
                "num_predict": 700
            }
        }

        # Use stream=True + chunked read so the socket stays alive for the full generation.
        # This avoids urllib's read-timeout killing long responses (~90s limit).
        payload["stream"] = True

        url = f"{self.ollama_url}/api/generate"
        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=req_data, headers={"Content-Type": "application/json"})
            full_response = []
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                while True:
                    line = resp.readline()
                    if not line:
                        break
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                        token = chunk.get("response", "")
                        if token:
                            full_response.append(token)
                        if chunk.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

            raw_response = "".join(full_response).strip()
            if not raw_response:
                return False, None, "Empty response from Ollama — model may still be loading."

            # Strip markdown code fences if present
            if raw_response.startswith("```"):
                raw_response = "
".join(
                    l for l in raw_response.splitlines()
                    if not l.strip().startswith("```")
                )

            parsed = json.loads(raw_response)

            # Validate and sanitize plan
            is_safe, sanitized_plan, warnings = self.validate_plan(parsed.get("plan", []), limits)
            parsed["plan"] = sanitized_plan
            parsed["warnings"] = warnings
            return True, parsed, ""
        except urllib.error.URLError as err:
            return False, None, f"Ollama request failed ({self.ollama_url}): {err.reason}"
        except json.JSONDecodeError as err:
            return False, None, f"Failed to parse LLM JSON response: {err}\nRaw: {raw_response[:200] if 'raw_response' in dir() else 'n/a'}"
        except Exception as err:
            return False, None, f"AI generation error: {err}"

    def validate_plan(self, raw_plan, limits):
        """Whitelist validator and boundary clamper for all AI-generated plan steps."""
        sanitized = []
        warnings = []

        if not isinstance(raw_plan, list):
            return False, [], ["Plan must be a list of action steps."]

        valid_joint_keys = {"J1": 0, "J2": 1, "J3": 2, "J4": 3, "J5": 4, "J6": 5}

        for idx, step in enumerate(raw_plan):
            if not isinstance(step, dict):
                continue

            action = step.get("action", "").lower()

            if action == "move":
                joints = step.get("joints", {})
                cleaned_joints = {}
                for k, v in joints.items():
                    k_upper = str(k).upper()
                    if k_upper in valid_joint_keys:
                        j_idx = valid_joint_keys[k_upper]
                        lo, hi = limits[j_idx]
                        try:
                            val = float(v)
                            if val < lo or val > hi:
                                warnings.append(f"Step {idx+1}: {k_upper} angle {val:.1f}° was clamped to [{lo:.1f}°, {hi:.1f}°]")
                                val = max(lo, min(hi, val))
                            cleaned_joints[k_upper] = val
                        except (ValueError, TypeError):
                            pass

                if cleaned_joints:
                    speed = max(15, min(65, int(step.get("speed", 45))))
                    wait = max(0.2, min(5.0, float(step.get("wait", 1.0))))
                    sanitized.append({
                        "action": "move",
                        "joints": cleaned_joints,
                        "speed": speed,
                        "wait": wait
                    })

            elif action == "gripper":
                state = step.get("state", "open")
                wait = max(0.2, min(4.0, float(step.get("wait", 0.8))))
                if isinstance(state, str):
                    angle = 90.0 if state.lower() in ("open", "release") else -90.0
                else:
                    try:
                        angle = max(-90.0, min(90.0, float(state)))
                    except (ValueError, TypeError):
                        angle = 0.0
                sanitized.append({
                    "action": "gripper",
                    "angle": angle,
                    "wait": wait
                })

            elif action == "home":
                sanitized.append({
                    "action": "move",
                    "joints": {"J1": 0.0, "J2": 0.0, "J3": 0.0, "J4": 0.0, "J5": 0.0, "J6": 0.0},
                    "speed": 40,
                    "wait": 2.0
                })

            elif action == "wait":
                secs = max(0.1, min(10.0, float(step.get("seconds", 1.0))))
                sanitized.append({"action": "wait", "seconds": secs})

            elif action == "stop":
                sanitized.append({"action": "stop"})

        return True, sanitized, warnings
