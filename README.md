# 🎯 People Counter — Production Edition

Real-time object counting using **YOLOv8n + ByteTrack**.  
Counts objects crossing a configurable line — people, cars, bikes, anything YOLO detects.

**Video sources:** USB/Webcam · IP Camera (RTSP/MJPEG) · Screen Capture · Video File  
**Platform:** Windows · macOS · Linux  
**Hardware:** CPU or NVIDIA GPU

---

## 📋 Table of Contents

1. [Requirements](#1-requirements)
2. [Quick Start](#2-quick-start)
3. [Step-by-Step Setup](#3-step-by-step-setup)
   - [Windows](#windows)
   - [macOS](#macos)
   - [Linux (Ubuntu/Debian)](#linux-ubuntudebian)
4. [Running the App](#4-running-the-app)
5. [Video Source Setup](#5-video-source-setup)
6. [UI Controls Reference](#6-ui-controls-reference)
7. [Detecting Cars or Other Objects](#7-detecting-cars-or-other-objects)
8. [Troubleshooting](#8-troubleshooting)
9. [File Structure](#9-file-structure)

---

## 1. Requirements

| Item | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11 |
| RAM | 4 GB | 8 GB |
| CPU | Any modern | 4+ cores |
| GPU | Not required | NVIDIA (CUDA 12+) |
| OS | Windows 10 / macOS 12 / Ubuntu 20.04 | Latest |
| Camera | USB webcam or IP camera | 1080p USB |

> **GPU note:** The app runs fine on CPU at 10–20 FPS inference.  
> With a NVIDIA GPU (GTX 1060 or newer) you get 30–60+ FPS.

---

## 2. Quick Start

```
# Windows
setup.bat        ← run once
run.bat          ← run every time after

# macOS / Linux
bash setup.sh    ← run once
bash run.sh      ← run every time after
```

---

## 3. Step-by-Step Setup

### Windows

**Step 1 — Install Python 3.11**

1. Go to https://www.python.org/downloads/
2. Download Python **3.11.x** (Windows installer 64-bit)
3. Run the installer
4. ✅ **Tick "Add Python to PATH"** — this is critical
5. Click "Install Now"

Verify it worked — open Command Prompt and type:
```
python --version
```
You should see `Python 3.11.x`

**Step 2 — Download or clone this repo**

Option A — Download ZIP:
- Click the green "Code" button on GitHub → "Download ZIP"
- Extract it anywhere, e.g. `C:\people-counter`

Option B — Git clone (if you have Git):
```
git https://github.com/NorakneaTh34/People-Counter
cd people-counter
```

**Step 3 — Run setup**

Double-click `setup.bat` OR open Command Prompt in the folder and run:
```
setup.bat
```

This will:
- Create a `venv\` virtual environment
- Auto-detect your GPU and install the right PyTorch
- Install all dependencies
- Pre-download the YOLOv8n model (6 MB)

**Step 4 — Launch**

```
run.bat
```

---

### macOS

**Step 1 — Install Python 3.11**

Option A — Homebrew (recommended):
```bash
brew install python@3.11
```

Option B — Download from https://www.python.org/downloads/

Verify:
```bash
python3.11 --version
```

**Step 2 — Get the repo**

```bash
git clone https://github.com/NorakneaTh34/People-Counter
cd people-counter
```

**Step 3 — Run setup**

```bash
bash setup.sh
```

> On macOS with Apple Silicon (M1/M2/M3) PyTorch uses the **MPS** backend automatically — you'll get GPU-accelerated inference without CUDA.

**Step 4 — Launch**

```bash
bash run.sh
```

---

### Linux (Ubuntu/Debian)

**Step 1 — Install Python + tkinter**

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-tk -y
```

Verify:
```bash
python3.11 --version
python3.11 -c "import tkinter; print('tkinter OK')"
```

> If `tkinter` fails: `sudo apt install python3-tk` then retry.

**Step 2 — Get the repo**

```bash
git clone https://github.com/NorakneaTh34/People-Counter
cd people-counter
```

**Step 3 — Run setup**

```bash
bash setup.sh
```

> NVIDIA GPU users: the script auto-installs CUDA 12.1 PyTorch.  
> Make sure your NVIDIA driver is ≥ 525. Check with: `nvidia-smi`

**Step 4 — Launch**

```bash
bash run.sh
```

---

## 4. Running the App

After setup, always launch from the repo folder:

```bash
# Windows
run.bat

# macOS / Linux
bash run.sh
```

On first run the app will print:
```
[INFO] Device  : cpu          ← or "cuda" if GPU found
[INFO] FP16    : False
[INFO] Loading YOLOv8n …
[INFO] Model ready
```

The GUI window opens. Select your video source on the right panel, then click **▶ START**.

---

## 5. Video Source Setup

### 🎥 USB / Webcam Camera (default)

1. Select **USB / Webcam Camera** in the source panel
2. Click **🔍 Scan** to detect all connected cameras
3. Select the camera index from the dropdown (usually `Camera 0`)
4. Optionally set a resolution (native = camera's max res)
5. Click **▶ START**

**Multiple cameras:** If you have more than one USB camera, Scan will list all of them (Camera 0, Camera 1, …).

---

### 📷 IP Camera (RTSP / MJPEG)

1. Select **IP Camera**
2. Enter the camera's **IP address** (e.g. `192.168.1.100`)
3. Select your **camera brand** from the preset dropdown — the port and stream path fill in automatically
4. Enter **username / password** if required
5. The URL preview at the bottom shows you exactly what will be dialed
6. Click **▶ START**

**Common presets:**

| Brand | Default URL built |
|---|---|
| Hikvision | `rtsp://user:pass@ip:554/Streaming/Channels/101` |
| Dahua | `rtsp://user:pass@ip:554/cam/realmonitor?channel=1&subtype=0` |
| Reolink | `rtsp://user:pass@ip:554/h264Preview_01_main` |
| Android IP Webcam app | `http://ip:8080/video` |
| DroidCam | `http://ip:4747/mjpegfeed` |

**Android phone as camera:**
1. Install **IP Webcam** from Play Store
2. Open the app → scroll to bottom → tap **Start server**
3. Note the IP shown (e.g. `192.168.1.55:8080`)
4. In the app select **Android — IP Webcam (video)**, enter the IP, click START

---

### 🖥️ Screen Capture

Captures directly from your monitor — useful for testing with any video playing on screen.

1. Select **Screen Capture**
2. Set monitor number (1 = primary)
3. Click **▶ START**

---

### 🎬 Video File

1. Select **Video File**
2. A file browser opens — select any `.mp4 .avi .mov .mkv` file
3. Click **▶ START**

The video loops automatically.

---

## 6. UI Controls Reference

### 📊 Stats Panel
| Label | Meaning |
|---|---|
| ENTERED | Total objects that crossed IN |
| EXITED | Total objects that crossed OUT |
| INSIDE | ENTERED − EXITED (current occupancy) |
| INFER FPS | YOLO inference frames per second |
| Capture FPS | Camera read frames per second |

### 📏 Counting Line

| Control | Effect |
|---|---|
| **Vertical position Y** | Moves line up/down (10% = near top, 90% = near bottom) |
| **Rotation** | Tilts the line (useful for diagonal corridors) |

### 🔀 IN / OUT Direction

Click the direction button to flip which side counts as IN.  
The **green arrow** always points toward the IN side.

### 🎨 Visualisation

| Control | Effect |
|---|---|
| Line thickness | How thick the counting line is drawn |
| Bounding box thickness | Thickness of detection boxes |
| Motion trail length | How long the movement trails are (0 = off) |
| Show tracker IDs | Show/hide `#ID` labels on each detection |
| Show motion trails | Show/hide movement history lines |

### 🎯 Detection Tuning

| Control | Effect |
|---|---|
| **Confidence threshold** | Lower = detects more (but more false positives). Start at 0.35 |
| **Inference img size** | 320 = fast, 640 = accurate. For demo use 320 or 480 |

### 🔧 CCTV Normalisation

Helps with dark/low-quality camera feeds:

| Control | Effect |
|---|---|
| Enable normalisation | Applies CLAHE contrast enhancement |
| Denoise | Reduces noise — **slows FPS significantly** |
| Sharpen | Sharpens H.264-blurred footage |
| Gamma | < 1.0 = brighter, > 1.0 = darker |

### 🔄 Reset Counters

Resets IN, OUT, INSIDE counts to zero without stopping the stream.

---

## 7. Detecting Cars or Other Objects

Two things to change in `people_counter.py`:

### Change the detection class

Find this line (around line 190):
```python
classes=[0],
```

Change `[0]` to whatever you want to detect:

```python
classes=[0]        # person only (default)
classes=[2]        # car only
classes=[2, 7]     # car + truck
classes=[2, 3, 5, 7]  # car + motorcycle + bus + truck
classes=None       # detect ALL 80 COCO classes
```

**Full COCO class index list:**

| Index | Object | Index | Object |
|---|---|---|---|
| 0 | person | 14 | bird |
| 1 | bicycle | 15 | cat |
| 2 | **car** | 16 | dog |
| 3 | motorcycle | 24 | backpack |
| 4 | airplane | 25 | umbrella |
| 5 | bus | 41 | cup |
| 6 | train | 56 | chair |
| 7 | truck | 57 | couch |
| 8 | boat | 62 | tv |
| 9 | traffic light | 67 | cell phone |
| 11 | stop sign | 73 | laptop |

Full list: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml

### Change the model size

Find this line (line 34):
```python
MODEL_PATH = "yolov8n.pt"
```

| Model | Speed | Accuracy | Size |
|---|---|---|---|
| `yolov8n.pt` | ⚡⚡⚡⚡ Fastest | ⭐⭐ | 6 MB |
| `yolov8s.pt` | ⚡⚡⚡ | ⭐⭐⭐ | 22 MB |
| `yolov8m.pt` | ⚡⚡ | ⭐⭐⭐⭐ | 52 MB |
| `yolov8l.pt` | ⚡ | ⭐⭐⭐⭐⭐ | 87 MB |
| `yolov8x.pt` | 🐢 Slowest | ⭐⭐⭐⭐⭐ | 136 MB |

Models download automatically on first use.  
For demo purposes `yolov8n` or `yolov8s` is recommended.

---

## 8. Troubleshooting

### App won't open / crashes immediately

```
ModuleNotFoundError: No module named 'cv2'
```
→ You're not in the venv. Use `run.bat` / `run.sh` instead of calling `python` directly.

---

### Tkinter not found (Linux)

```
ModuleNotFoundError: No module named '_tkinter'
```
→ Run: `sudo apt install python3-tk` then re-run `bash setup.sh`

---

### Camera won't connect (IP camera)

Checklist:
- [ ] Camera and PC are on the **same network / same WiFi**
- [ ] IP address is correct (check camera's web interface)
- [ ] Username/password are correct (default is often `admin` / `admin`)
- [ ] Selected the correct **brand preset** (wrong path = no stream)
- [ ] No firewall blocking port 554 (RTSP) or 8080 (HTTP)
- [ ] Try pinging the camera: `ping 192.168.1.100`
- [ ] Try the URL directly in VLC: Media → Open Network Stream

---

### USB camera not detected

- Click **🔍 Scan** — it probes indices 0–7
- If still not found, check Device Manager (Windows) or `ls /dev/video*` (Linux)
- Try a different USB port
- On Windows, some cameras need `cv2.CAP_DSHOW` — the scan already uses this

---

### Very low FPS (under 5)

- Lower **Inference img size** to 320
- Turn off **Denoise** in CCTV Normalisation
- On CPU this is normal — add a GPU for 3–5× speedup
- Use `yolov8n.pt` (fastest model)

---

### Detection boxes jittery / too many false positives

- Raise **Confidence threshold** (try 0.5–0.6)
- Raise **Inference img size** to 480 or 640
- Switch to a larger model (`yolov8s.pt`)

---

### Counting fires too early / double-counts

- Increase `MIN_FRAMES_ON_SIDE` in `people_counter.py` (default 4) — requires person to be on one side longer before a cross is counted
- Increase `VOTE_WINDOW` (default 6) — more frames averaged before side is decided

---

### PyTorch CUDA error

```
RuntimeError: expected scalar type Half but found Float
```
→ Already fixed in this version (`HALF = False`). If you see this, make sure you're running the latest `people_counter.py`.

---

## 9. File Structure

```
people-counter/
│
├── people_counter.py     ← Main application (edit this to change class/model)
├── requirements.txt      ← Python dependencies
│
├── setup.bat             ← Windows: one-time setup
├── run.bat               ← Windows: launch app
│
├── setup.sh              ← macOS/Linux: one-time setup
├── run.sh                ← macOS/Linux: launch app
│
├── .gitignore
└── README.md             ← This file
```

After first run, YOLO adds:
```
├── yolov8n.pt            ← Auto-downloaded model weights
└── runs/                 ← YOLO logs (can be deleted)
```

---

## Notes

- Counting accuracy depends on camera angle — **top-down or 45° overhead** works best
- The counting line works best across a **corridor or doorway**, not an open area
- For best results, ensure people/objects are **fully visible** when crossing the line
- The INSIDE counter (`ENTERED − EXITED`) can go negative if people exit before being counted entering — this is normal for edge cases near the line at startup
