[instructions.md](https://github.com/user-attachments/files/34323887/instructions.md)
# NEXUS Arm: Local AI Setup (Windows)

Machine: Windows 10 laptop, PowerShell
Arduino: Uno R3, firmware `nexus_arm_v3_fw.ino`
Project folder: `C:\nexus_arm_v3`

Run every command in **PowerShell**. Right-click the Start button and choose "Terminal" or "Windows PowerShell".

---

## BEFORE YOU START (WINDOWS 10)

**Check your Windows version.** Press Win+R, type `winver`, and press Enter. Ollama needs **Windows 10 version 22H2 or newer**. If you're on an older build, run Windows Update first.

**Check that winget works.** Open PowerShell and run:

```powershell
winget --version
```

If it says the command isn't recognised, install **App Installer** from the Microsoft Store, then try again. If you can't use winget at all, use the manual downloads below instead.

**Manual downloads (use only if winget fails):**
- Python 3.12: python.org/downloads (tick "Add python.exe to PATH")
- Ollama: ollama.com/download/windows
- Arduino IDE 2: arduino.cc/en/software
- Arduino CLI: arduino.cc/en/software, then "Arduino CLI" zip (unzip it and add the folder to your PATH)

---

## PART A: AT HOME (INTERNET REQUIRED)

Do all of Part A while you have internet. Part B then works fully offline.

### A1. Install Python

```powershell
winget install Python.Python.3.12
```

Close PowerShell and open it again, then check:

```powershell
py --version
```

Expected: `Python 3.12.x`.

If `winget` is not available, download the installer from python.org. **Tick "Add python.exe to PATH" and make sure tcl/tk is included** (it is by default).

### A2. Install the Python packages

```powershell
py -m pip install pyserial opencv-python
```

Check:

```powershell
py -c "import serial, cv2, tkinter; print('python OK')"
```

Expected: `python OK`.

### A3. Install Ollama

```powershell
winget install Ollama.Ollama
```

Open the Ollama app from the Start menu once so it runs in the background (it shows an icon in the system tray). Then check:

```powershell
ollama --version
```

### A4. Install the Arduino tools

Option 1, the Arduino IDE (easiest, with a graphical interface):

```powershell
winget install ArduinoSA.IDE.stable
```

Option 2, the command-line tool (used in the steps below):

```powershell
winget install ArduinoSA.CLI
```

Close and reopen PowerShell, then install the Uno's support package. This needs internet once:

```powershell
arduino-cli config init
arduino-cli core update-index
arduino-cli core install arduino:avr
```

Check:

```powershell
arduino-cli core list
```

Expected: `arduino:avr` appears in the list. The Servo library is built into this core, so you don't download it separately.

### A5. Driver for cheap Uno boards

If your Uno is a clone with a **CH340** chip (a small square chip near the USB port), Windows may need its driver. Plug the board in, and if it shows as an unknown device, search for "CH340 driver" and install it. Genuine Uno R3 boards don't need this.

### A6. Download the AI model

```powershell
ollama pull qwen2.5:3b
```

Check:

```powershell
ollama list
```

Expected: `qwen2.5:3b` in the list.

### A7. Back up everything offline

Copy the Ollama model folder somewhere safe, such as a USB drive:

```powershell
Copy-Item -Recurse "$env:USERPROFILE\.ollama\models" "D:\ollama-models-backup"
```

Change `D:` to your USB drive letter. If that folder doesn't exist, check where Ollama stored the models:

```powershell
dir "$env:USERPROFILE\.ollama"
```

Download the Python packages for an offline reinstall (optional):

```powershell
mkdir C:\offline-wheels
py -m pip download pyserial opencv-python -d C:\offline-wheels
```

### A8. Check the Arduino and upload the firmware

Plug in the Uno, then find its COM port:

```powershell
arduino-cli board list
```

Expected: a line such as `COM3` next to `arduino:avr:uno`. Write down the COM number.

Compile and upload the firmware (replace `COM3` with your port):

```powershell
cd C:\nexus_arm_v3
arduino-cli compile --fqbn arduino:avr:uno nexus_arm_v3_fw
arduino-cli upload -p COM3 --fqbn arduino:avr:uno nexus_arm_v3_fw
```

Expected: "Upload done" or similar, with no errors.

---

## PART B: AT SCHOOL (NO INTERNET)

Everything below works offline, once Part A is done.

### B1. Confirm the AI model runs

```powershell
ollama run qwen2.5:3b "Reply with only the word OK"
```

Expected: a short reply containing `OK`. Type `/bye` to leave the chat.

### B2. Confirm the Arduino connects

Plug the Uno into a USB port on the laptop:

```powershell
arduino-cli board list
```

Expected: the same COM port as in A8. If the number changed, use the new one.

### B3. Run the arm GUI

```powershell
cd C:\nexus_arm_v3
py nexus_arm_v3.py
```

In the window:
1. Choose the COM port, then press **Connect**.
2. Wait 2 seconds. The status line should show `READY` or `RX` lines.
3. Support the arm before connecting, because the Uno reset releases the positional servos.

**Close the Arduino IDE's Serial Monitor first.** Only one program can use the COM port at a time. If the GUI says "Could not open", that's usually why.

### B4. Test the servos

- Hold **C1 Left** and **C1 Right**. The status line shows `RX M,0,...`.
- Tick **Positional servos ON**, then move P1. The status shows `RX A,0,...`.
- Watch the FSR bars at rest. They should sit near 0.

### B5. Emergency stop

- Press **STOP JOINTS** in the GUI.
- Untick **Positional servos ON** to release the positional servos.
- If anything goes wrong, switch off the battery first.

---

## PART C: NEXT BUILD STEPS (ALSO OFFLINE)

Do these in order, after Part B works.

1. **Controller script** (`arm_controller.py`): takes a JSON action, checks it against a whitelist of joints and actions, and sends the serial command. Test it with hand-written JSON before adding the model.
2. **Model bridge:** send text to `qwen2.5:3b` through the local Ollama server at `http://localhost:11434`, ask for JSON only, and pass the reply through the controller.
3. **Vision:** OpenCV colour detection on the webcam. Calibrate the camera to the arm before picking anything.

---

## QUICK CHECKLIST BEFORE SCHOOL

- [ ] `ollama list` shows `qwen2.5:3b`
- [ ] `arduino-cli core list` shows `arduino:avr`
- [ ] `py -c "import serial, cv2, tkinter"` runs without errors
- [ ] `arduino-cli board list` shows your Uno's COM port
- [ ] The firmware uploads without errors
- [ ] The model backup exists on a USB drive
- [ ] You ran Part B with Wi-Fi turned off
