"""
People Counter — Production Edition
YOLOv8n + ByteTrack + Full-Resolution Display + Adjustable Line + Direction Toggle
Supports: RTSP · HTTP/MJPEG · USB/Webcam · IP Camera · Screen Capture · Video File

Architecture (3 threads for maximum FPS):
  Thread 1 — Capture   : reads raw frames at native resolution
  Thread 2 — Inference : runs YOLO on downscaled copy, draws on FULL-RES frame
  Main     — UI        : tkinter display at ~60 Hz

Usage:
  python people_counter_desktop.py

Dependencies:
  pip install ultralytics torch torchvision opencv-python pillow mss numpy
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import torch
import time
import math
from collections import defaultdict
import threading
import queue
import mss

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = "yolov8n.pt"
IMG_SIZE   = 320          # inference resolution (320/480/640 — user adjustable)
CONF_DEF   = 0.35
IOU        = 0.45
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
HALF       = False        # FP16 disabled — avoids dtype errors on CPU/some GPUs

RAW_Q_SIZE  = 1
DISP_Q_SIZE = 1

print(f"[INFO] Device  : {DEVICE}")
print(f"[INFO] FP16    : {HALF}")
print("[INFO] Loading YOLOv8n …")
from ultralytics import YOLO
model = YOLO(MODEL_PATH)
model.to(DEVICE)
print("[INFO] Model ready")


# ─────────────────────────────────────────────────────────────────────────────
# INPUT NORMALIZER
# ─────────────────────────────────────────────────────────────────────────────
class CCTVNormalizer:
    def __init__(self):
        self.clahe         = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self.enabled       = True
        self.denoise       = False
        self.sharpen       = True
        self.gamma         = 1.0
        self.infer_width   = 640
        self.infer_height  = 480

    def apply_for_inference(self, frame: np.ndarray) -> np.ndarray:
        """Resize + enhance for YOLO inference only (not display)."""
        h, w  = frame.shape[:2]
        scale = min(self.infer_width / w, self.infer_height / h)
        if scale < 1.0:
            nw, nh = int(w * scale), int(h * scale)
            small  = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        else:
            small = frame.copy()

        if not self.enabled:
            return small

        if self.denoise:
            small = cv2.fastNlMeansDenoisingColored(small, None, 5, 5, 7, 21)

        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        small = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        if self.gamma != 1.0:
            small = self._gamma_correct(small, self.gamma)

        if self.sharpen:
            blur  = cv2.GaussianBlur(small, (0, 0), 2)
            small = cv2.addWeighted(small, 1.4, blur, -0.4, 0)

        return small

    @staticmethod
    def _gamma_correct(frame, gamma):
        inv = 1.0 / gamma
        lut = np.array([((i / 255.0) ** inv) * 255
                        for i in range(256)], dtype=np.uint8)
        return cv2.LUT(frame, lut)


normalizer = CCTVNormalizer()


# ─────────────────────────────────────────────────────────────────────────────
# SCREEN CAPTURE SOURCE
# ─────────────────────────────────────────────────────────────────────────────
class ScreenCapture:
    def __init__(self, monitor_index: int = 1, region: dict = None):
        self.sct           = mss.mss()
        self.monitor_index = monitor_index
        self.region        = region
        self._running      = True

    def _get_mon(self):
        if self.region:
            return self.region
        return self.sct.monitors[self.monitor_index]

    def read(self):
        try:
            img   = self.sct.grab(self._get_mon())
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)
            return True, frame
        except Exception as e:
            print(f"[ScreenCapture] {e}")
            return False, None

    def isOpened(self):
        return self._running

    def release(self):
        self._running = False
        try:
            self.sct.close()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# USB CAMERA ENUMERATOR
# ─────────────────────────────────────────────────────────────────────────────
def enumerate_usb_cameras(max_test: int = 8) -> list:
    """Probe indices 0..max_test-1 and return list of available camera indices."""
    found = []
    for idx in range(max_test):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if hasattr(cv2, 'CAP_DSHOW') else 0)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                found.append(idx)
            cap.release()
    return found if found else [0]   # fallback to 0


# ─────────────────────────────────────────────────────────────────────────────
# IP CAMERA URL BUILDER
# ─────────────────────────────────────────────────────────────────────────────
CAM_PRESETS = {
    "📱 Android — IP Webcam (video)":     ("http", 8080, "/video",                               False, "MJPEG stream from IP Webcam app"),
    "📱 Android — IP Webcam (shot.jpg)":  ("http", 8080, "/shot.jpg",                            False, "JPEG snapshot (lower latency)"),
    "📱 Android — DroidCam":              ("http", 4747, "/mjpegfeed",                            False, "DroidCam MJPEG feed"),
    "📷 Hikvision":                       ("rtsp",  554, "/Streaming/Channels/101",              True,  "Hikvision NVR/DVR main stream"),
    "📷 Dahua":                           ("rtsp",  554, "/cam/realmonitor?channel=1&subtype=0", True,  "Dahua main stream ch1"),
    "📷 Reolink":                         ("rtsp",  554, "/h264Preview_01_main",                 True,  "Reolink main stream"),
    "📷 Axis":                            ("rtsp",  554, "/axis-media/media.amp",                True,  "Axis VAPIX stream"),
    "📷 Amcrest":                         ("rtsp",  554, "/cam/realmonitor?channel=1&subtype=0", True,  "Amcrest/Dahua OEM"),
    "📷 Uniview (UNV)":                   ("rtsp",  554, "/media/video1",                        True,  "UNV main stream"),
    "📷 Hanwha / Samsung":                ("rtsp",  554, "/profile1/media.smp",                  True,  "Hanwha SNO/QNO series"),
    "📷 Bosch":                           ("rtsp",  554, "/rtsp_tunnel",                         True,  "Bosch IP cameras"),
    "📷 ONVIF (profile S)":               ("rtsp",  554, "/onvif1",                              True,  "Generic ONVIF cameras"),
    "📷 Generic RTSP":                    ("rtsp",  554, "/stream1",                             True,  "Try if brand unknown"),
    "✏️  Custom …":                        ("rtsp",  554, "",                                     True,  "Enter path manually"),
}

def build_cam_url(preset_name, ip, port_override, user, password, custom_path):
    proto, default_port, preset_path, needs_auth, _ = CAM_PRESETS[preset_name]
    ip   = ip.strip().rstrip("/")
    path = (custom_path.strip() if preset_name.startswith("✏️") else preset_path)
    if path and not path.startswith("/"):
        path = "/" + path
    try:
        port = int(port_override) if port_override.strip() else default_port
    except ValueError:
        port = default_port
    if proto == "http":
        if user and password:
            return f"http://{user}:{password}@{ip}:{port}{path}"
        return f"http://{ip}:{port}{path}"
    else:
        if user and password:
            return f"rtsp://{user}:{password}@{ip}:{port}{path}"
        elif user:
            return f"rtsp://{user}@{ip}:{port}{path}"
        return f"rtsp://{ip}:{port}{path}"


# ─────────────────────────────────────────────────────────────────────────────
# COUNTER STATE  (robust crossing algorithm)
# ─────────────────────────────────────────────────────────────────────────────
MIN_FRAMES_ON_SIDE = 4
VOTE_WINDOW        = 6
HISTORY_LEN        = 40

class TrackInfo:
    __slots__ = ("dist_history", "xy_history", "status", "flash_timer")
    def __init__(self):
        self.dist_history: list = []
        self.xy_history:   list = []
        self.status:       str  = ""
        self.flash_timer:  int  = 0

class CounterState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count_in    = 0
        self.count_out   = 0
        self.tracks      = defaultdict(TrackInfo)
        self.fps_buf     = []
        self.cap_fps_buf = []
        self.t_last      = time.time()
        self.t_cap_last  = time.time()

state      = CounterState()
state_lock = threading.Lock()


def _signed_dist(px, py, x1, y1, x2, y2, invert: bool) -> float:
    raw = float((x2 - x1) * (py - y1) - (y2 - y1) * (px - x1))
    return -raw if invert else raw

def _majority_side(dist_history: list) -> int:
    recent = dist_history[-VOTE_WINDOW:]
    pos = sum(1 for d in recent if d > 0)
    neg = sum(1 for d in recent if d < 0)
    if pos > neg: return 1
    if neg > pos: return -1
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# CORE PROCESSING
# Key design: inference runs on DOWNSCALED frame, detections are SCALED BACK
# and drawn on the ORIGINAL full-resolution frame for crisp display.
# ─────────────────────────────────────────────────────────────────────────────
def process_frame(frame: np.ndarray,
                  line_pct:          float,
                  angle_deg:         float,
                  line_thickness:    int,
                  box_thickness:     int,
                  trail_length:      int,
                  show_ids:          bool,
                  show_trail:        bool,
                  conf:              float,
                  invert_direction:  bool = False) -> tuple:
    """
    Returns: (display_frame_full_res, infer_fps)
    display_frame_full_res — original resolution with overlays drawn on it
    """

    full_h, full_w = frame.shape[:2]

    # ── Downscale for inference (frame is already display-scaled) ─────────────
    infer_frame = normalizer.apply_for_inference(frame)
    inf_h, inf_w = infer_frame.shape[:2]

    # Scale factors: inference coords → display frame coords
    sx = full_w / inf_w
    sy = full_h / inf_h

    results = model.track(
        infer_frame,
        persist=True,
        classes=[0],
        conf=conf,
        iou=IOU,
        imgsz=IMG_SIZE,
        tracker="bytetrack.yaml",
        device=DEVICE,
        half=HALF,
        verbose=False,
        stream=False,
    )

    # ── Draw on full-resolution frame ─────────────────────────────────────────
    out = frame.copy()

    # Line anchor: center X, user-controlled Y + angle
    cx_img = full_w * 0.5
    cy_img = full_h * line_pct
    theta  = math.radians(angle_deg)
    dx = math.cos(theta) * full_w * 2
    dy = math.sin(theta) * full_w * 2
    x1, y1 = int(cx_img - dx), int(cy_img - dy)
    x2, y2 = int(cx_img + dx), int(cy_img + dy)

    # ── Draw counting line ────────────────────────────────────────────────────
    # Shadow for visibility on any background
    cv2.line(out, (x1, y1), (x2, y2), (0, 0, 0), line_thickness + 3)
    cv2.line(out, (x1, y1), (x2, y2), (0, 220, 255), line_thickness)
    cv2.circle(out, (int(cx_img), int(cy_img)), line_thickness + 3, (0, 0, 0), -1)
    cv2.circle(out, (int(cx_img), int(cy_img)), line_thickness + 2, (0, 220, 255), -1)

    # ── Draw IN arrow ─────────────────────────────────────────────────────────
    nx = -(y2 - y1)
    ny =  (x2 - x1)
    if invert_direction:
        nx, ny = -nx, -ny
    mag = math.hypot(nx, ny)
    if mag > 0:
        arrow_len = max(50, full_h // 10)
        nx, ny = (nx / mag) * arrow_len, (ny / mag) * arrow_len
    arrow_end = (int(cx_img + nx), int(cy_img + ny))
    cv2.arrowedLine(out, (int(cx_img), int(cy_img)), arrow_end,
                    (0, 0, 0), 5, tipLength=0.3)
    cv2.arrowedLine(out, (int(cx_img), int(cy_img)), arrow_end,
                    (0, 230, 0), 3, tipLength=0.3)
    cv2.putText(out, "IN", (arrow_end[0] + 8, arrow_end[1] + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(out, "IN", (arrow_end[0] + 8, arrow_end[1] + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 0), 2)

    current_n = 0

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes_inf = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids       = results[0].boxes.id.cpu().numpy().astype(int)
        current_n = len(boxes_inf)

        with state_lock:
            for (bx1i, by1i, bx2i, by2i), tid in zip(boxes_inf, ids):
                # Scale bounding box back to full-res
                bx1 = int(bx1i * sx); by1 = int(by1i * sy)
                bx2 = int(bx2i * sx); by2 = int(by2i * sy)
                cx  = (bx1 + bx2) // 2
                cy  = (by1 + by2) // 2

                info = state.tracks[tid]

                dist = _signed_dist(cx, cy, x1, y1, x2, y2, invert_direction)
                info.dist_history.append(dist)
                info.xy_history.append((cx, cy))
                if len(info.dist_history) > HISTORY_LEN:
                    info.dist_history.pop(0)
                if len(info.xy_history) > HISTORY_LEN:
                    info.xy_history.pop(0)

                dh = info.dist_history

                # Crossing detection
                if info.status == "" and len(dh) >= MIN_FRAMES_ON_SIDE + 1:
                    anchor_slice = dh[-(MIN_FRAMES_ON_SIDE + VOTE_WINDOW): -VOTE_WINDOW]
                    current_side = _majority_side(dh)
                    if len(anchor_slice) >= MIN_FRAMES_ON_SIDE:
                        ap = sum(1 for d in anchor_slice if d > 0)
                        an = sum(1 for d in anchor_slice if d < 0)
                        anchor_side = 1 if ap > an else (-1 if an > ap else 0)
                        if anchor_side == -1 and current_side == 1:
                            state.count_in  += 1
                            info.status      = "IN"
                            info.flash_timer = 25
                        elif anchor_side == 1 and current_side == -1:
                            state.count_out += 1
                            info.status      = "OUT"
                            info.flash_timer = 25

                # Colour
                if info.flash_timer > 0:
                    t = info.flash_timer / 25.0
                    if info.status == "IN":
                        color = (int(50*(1-t)), 255, int(50*(1-t)))
                    else:
                        color = (255, int(50*(1-t)), int(50*(1-t)))
                    info.flash_timer -= 1
                elif info.status == "IN":
                    color = (0, 200, 0)
                elif info.status == "OUT":
                    color = (60, 60, 220)
                else:
                    cur_side = _majority_side(dh) if dh else 0
                    color    = (180, 100, 0) if cur_side >= 0 else (0, 130, 180)

                # Bounding box (shadow + colour)
                cv2.rectangle(out, (bx1, by1), (bx2, by2), (0, 0, 0), box_thickness + 2)
                cv2.rectangle(out, (bx1, by1), (bx2, by2), color, box_thickness)
                cv2.circle(out, (cx, cy), 5, (0, 0, 0), -1)
                cv2.circle(out, (cx, cy), 4, color, -1)

                if show_ids:
                    tag = f"#{tid}  {info.status}" if info.status else f"#{tid}"
                    cv2.putText(out, tag, (bx1, max(by1 - 8, 18)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
                    cv2.putText(out, tag, (bx1, max(by1 - 8, 18)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

                if show_trail:
                    trail_pts = info.xy_history[-trail_length:]
                    for i in range(1, len(trail_pts)):
                        alpha = i / len(trail_pts)
                        tc = tuple(int(c * alpha) for c in color)
                        cv2.line(out, trail_pts[i-1], trail_pts[i], (0,0,0), 3)
                        cv2.line(out, trail_pts[i-1], trail_pts[i], tc, 2)

    # ── FPS ───────────────────────────────────────────────────────────────────
    now = time.time()
    fps = 1.0 / max(now - state.t_last, 1e-6)
    state.t_last = now
    state.fps_buf.append(fps)
    if len(state.fps_buf) > 20:
        state.fps_buf.pop(0)
    avg_fps = float(np.mean(state.fps_buf))

    _draw_hud(out, current_n, avg_fps, full_w, full_h)

    # Watermark: inference resolution
    res_tag = f"Display: {full_w}×{full_h}  |  Infer: {inf_w}×{inf_h} @ {IMG_SIZE}px"
    cv2.putText(out, res_tag, (full_w - 420, full_h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

    return out, avg_fps


def _draw_hud(frame, current_n, fps, fw, fh):
    scale  = max(0.5, min(fw, fh) / 640)
    pad_x, pad_y = 12, 12
    box_w  = int(280 * scale)
    box_h  = int(175 * scale)
    overlay = frame.copy()
    cv2.rectangle(overlay, (pad_x, pad_y), (pad_x + box_w, pad_y + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.rectangle(frame, (pad_x, pad_y), (pad_x + box_w, pad_y + box_h), (0, 220, 255), 1)

    with state_lock:
        c_in, c_out = state.count_in, state.count_out

    font   = cv2.FONT_HERSHEY_SIMPLEX
    fs     = 0.55 * scale
    fs_big = 0.70 * scale
    lh     = int(26 * scale)

    items = [
        ("PEOPLE COUNTER",               (0, 220, 255), fs,     2),
        (f"IN    : {c_in}",              (0, 210, 80),  fs_big, 2),
        (f"OUT   : {c_out}",             (60, 60, 255), fs_big, 2),
        (f"INSIDE: {max(0,c_in-c_out)}", (255, 200, 0), fs_big, 2),
        (f"NOW   : {current_n}",         (200, 200, 200), fs,   1),
        (f"FPS   : {fps:.1f}",           (160, 160, 160), fs,   1),
    ]
    y = pad_y + lh
    for text, color, fsize, thick in items:
        cv2.putText(frame, text, (pad_x + 10, y), font, fsize, (0,0,0), thick+2)
        cv2.putText(frame, text, (pad_x + 10, y), font, fsize, color, thick)
        y += lh


# ─────────────────────────────────────────────────────────────────────────────
# DESKTOP APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
class PeopleCounterApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("People Counter  ·  Production Edition  ·  YOLOv8n + ByteTrack")
        self.root.configure(bg="#07080d")
        self.root.geometry("1440x860")
        self.root.minsize(1100, 660)

        self._cap           = None
        self._running       = False
        self._t_capture     = None
        self._t_infer       = None
        self._raw_q         = queue.Queue(maxsize=RAW_Q_SIZE)
        self._frame_q       = queue.Queue(maxsize=DISP_Q_SIZE)
        self._video_path    = None
        self._screen_mon    = 1
        self._screen_region = None
        self._usb_cameras   = []
        self._canvas_w      = 1080   # updated on resize; read by infer thread
        self._canvas_h      = 720
        self._canvas_img_id = None   # reuse canvas item instead of delete/recreate

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_ui()

    # ─────────────────────────────────────────────────────────────────────────
    # UI CONSTRUCTION
    # ─────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        PANEL  = "#0c0f18"
        GREEN  = "#00ff88"
        CYAN   = "#00d4e8"
        ORANGE = "#ff9500"
        RED    = "#ff4455"
        YELLOW = "#ffcc00"
        FONT   = ("Courier New", 10)
        FONT_S = ("Courier New", 8)

        # ── Left canvas (video) ───────────────────────────────────────────
        self.canvas = tk.Canvas(self.root, bg="#000", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True,
                         padx=(10, 4), pady=10)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # ── Right panel (controls) ────────────────────────────────────────
        right = tk.Frame(self.root, bg=PANEL, width=360)
        right.pack(side="right", fill="y", padx=(4, 10), pady=10)
        right.pack_propagate(False)

        c2  = tk.Canvas(right, bg=PANEL, highlightthickness=0)
        vsb = ttk.Scrollbar(right, orient="vertical", command=c2.yview)
        self._p = tk.Frame(c2, bg=PANEL)
        self._p.bind("<Configure>", lambda e: c2.configure(scrollregion=c2.bbox("all")))
        c2.create_window((0, 0), window=self._p, anchor="nw")
        c2.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        c2.pack(side="left", fill="both", expand=True)

        p = self._p

        def H(text, color=GREEN):
            f = tk.Frame(p, bg=PANEL)
            f.pack(fill="x", padx=8, pady=(14, 3))
            tk.Label(f, text=text, bg=PANEL, fg=color,
                     font=("Courier New", 11, "bold")).pack(side="left")

        def SEP():
            tk.Frame(p, bg="#1c1f2e", height=1).pack(fill="x", padx=10, pady=8)

        def stat_row(label, color, attr):
            f = tk.Frame(p, bg=PANEL)
            f.pack(fill="x", padx=12, pady=2)
            tk.Label(f, text=label, bg=PANEL, fg="#555", font=FONT,
                     width=10, anchor="w").pack(side="left")
            lbl = tk.Label(f, text="0", bg=PANEL, fg=color,
                           font=("Courier New", 20, "bold"), anchor="e")
            lbl.pack(side="right")
            setattr(self, attr, lbl)

        def slider(label, var, from_, to, res, color, cmd=None):
            tk.Label(p, text=label, bg=PANEL, fg="#666", font=FONT_S
                     ).pack(anchor="w", padx=16, pady=(4, 0))
            s = tk.Scale(p, from_=from_, to=to, resolution=res,
                         orient="horizontal", variable=var,
                         bg=PANEL, fg=color, troughcolor="#151820",
                         highlightthickness=0, font=FONT_S,
                         activebackground=color)
            s.pack(fill="x", padx=12)
            if cmd:
                s.config(command=cmd)
            return s

        # ── STATS ─────────────────────────────────────────────────────────
        H("⬡  PEOPLE COUNTER", GREEN)
        stat_row("ENTERED",  "#00dd66", "_lbl_in")
        stat_row("EXITED",   "#ff4455", "_lbl_out")
        stat_row("INSIDE",   "#ffcc00", "_lbl_now")
        stat_row("INFER FPS","#33b5e5", "_lbl_fps")
        SEP()

        # ── COUNTING LINE POSITION ─────────────────────────────────────────
        H("📏  Counting Line", CYAN)

        self._line_var   = tk.IntVar(value=50)
        self._line_x_var = tk.IntVar(value=50)
        self._angle_var  = tk.DoubleVar(value=0.0)

        slider("Vertical position  (Y, 10=top → 90=bottom)",
               self._line_var, 10, 90, 1, GREEN)
        slider("Horizontal position  (X, 0=left → 100=right)",
               self._line_x_var, 0, 100, 1, GREEN)
        slider("Rotation  (-90° ↔ +90°)",
               self._angle_var, -89, 89, 1, CYAN)
        SEP()

        # ── IN/OUT DIRECTION ──────────────────────────────────────────────
        H("🔀  IN / OUT Direction", CYAN)
        self._invert_var = tk.BooleanVar(value=False)
        self._dir_btn = tk.Button(
            p, text="⬆  Arrow UP  =  IN direction",
            bg="#0d2211", fg=GREEN,
            activebackground="#152e1a", activeforeground=GREEN,
            relief="flat", bd=0, font=("Courier New", 9, "bold"),
            cursor="hand2", pady=7, command=self._toggle_direction)
        self._dir_btn.pack(fill="x", padx=12, pady=(2, 4))
        tk.Label(p, text="Tap to flip which side of the line counts as IN",
                 bg=PANEL, fg="#444", font=FONT_S, justify="center"
                 ).pack(pady=(0, 4))
        SEP()

        # ── VISUALISATION ─────────────────────────────────────────────────
        H("🎨  Visualisation", ORANGE)

        # Line thickness
        self._line_thick_var = tk.IntVar(value=2)
        slider("Line thickness  (px)", self._line_thick_var, 1, 8, 1, ORANGE)

        # Box thickness
        self._box_thick_var = tk.IntVar(value=2)
        slider("Bounding box thickness  (px)", self._box_thick_var, 1, 6, 1, ORANGE)

        # Trail length
        self._trail_len_var = tk.IntVar(value=15)
        slider("Motion trail length  (frames)", self._trail_len_var, 0, 40, 1, ORANGE)

        # Checkboxes
        chk_frame = tk.Frame(p, bg=PANEL)
        chk_frame.pack(fill="x", padx=12, pady=4)
        self._show_ids_var   = tk.BooleanVar(value=True)
        self._show_trail_var = tk.BooleanVar(value=True)

        def chk(parent, text, var, cmd=None):
            tk.Checkbutton(parent, text=text, variable=var, bg=PANEL, fg="#999",
                           selectcolor="#151820", activebackground=PANEL,
                           activeforeground=ORANGE, font=FONT,
                           command=cmd).pack(anchor="w", pady=1)

        chk(chk_frame, "Show tracker IDs", self._show_ids_var)
        chk(chk_frame, "Show motion trails", self._show_trail_var)
        SEP()

        # ── DETECTION TUNING ──────────────────────────────────────────────
        H("🎯  Detection Tuning", ORANGE)

        self._conf_var = tk.DoubleVar(value=CONF_DEF)
        slider("Confidence threshold", self._conf_var, 0.10, 0.90, 0.05, ORANGE)

        self._imgsize_var = tk.IntVar(value=IMG_SIZE)
        def _apply_imgsize(val):
            global IMG_SIZE
            IMG_SIZE = int(float(val))
        slider("Inference img size  (320=fast · 640=accurate)",
               self._imgsize_var, 160, 640, 32, ORANGE, _apply_imgsize)
        SEP()

        # ── CCTV NORMALISATION ────────────────────────────────────────────
        H("🔧  CCTV Normalisation", ORANGE)

        self._norm_en_var = tk.BooleanVar(value=True)
        self._denoise_var = tk.BooleanVar(value=False)
        self._sharpen_var = tk.BooleanVar(value=True)

        norm_chk = tk.Frame(p, bg=PANEL)
        norm_chk.pack(fill="x", padx=12)
        chk(norm_chk, "Enable normalisation (CLAHE + gamma)", self._norm_en_var,
            lambda: setattr(normalizer, "enabled", self._norm_en_var.get()))
        chk(norm_chk, "Denoise — slows FPS", self._denoise_var,
            lambda: setattr(normalizer, "denoise", self._denoise_var.get()))
        chk(norm_chk, "Sharpen (fixes H.264 softness)", self._sharpen_var,
            lambda: setattr(normalizer, "sharpen", self._sharpen_var.get()))

        self._gamma_var = tk.DoubleVar(value=1.0)
        slider("Gamma  (1.0=off  <1=brighter)", self._gamma_var,
               0.4, 2.0, 0.1, ORANGE,
               lambda v: setattr(normalizer, "gamma", float(v)))

        tk.Label(p, text="Normalise to resolution:", bg=PANEL, fg="#666",
                 font=FONT_S).pack(anchor="w", padx=16, pady=(6, 0))
        rf = tk.Frame(p, bg=PANEL); rf.pack(fill="x", padx=12, pady=2)
        self._res_var = tk.StringVar(value="640x480")
        ttk.Combobox(rf, textvariable=self._res_var, width=13, state="readonly",
                     values=["640x480", "1280x720", "1920x1080", "native"]
                     ).pack(side="left")
        tk.Button(rf, text="Apply", bg="#111", fg=ORANGE, relief="flat",
                  font=FONT, command=self._apply_res
                  ).pack(side="left", padx=6)
        SEP()

        # ── PERFORMANCE ───────────────────────────────────────────────────
        H("⚡  Performance", YELLOW)

        self._uifps_var = tk.IntVar(value=60)
        slider("UI refresh rate (Hz)", self._uifps_var, 15, 120, 5, YELLOW)

        # Throughput display
        trow = tk.Frame(p, bg=PANEL); trow.pack(fill="x", padx=12, pady=(4,2))
        tk.Label(trow, text="Capture FPS", bg=PANEL, fg="#555",
                 font=FONT, width=13, anchor="w").pack(side="left")
        self._lbl_cap_fps = tk.Label(trow, text="—", bg=PANEL,
                                     fg=YELLOW, font=("Courier New", 13, "bold"))
        self._lbl_cap_fps.pack(side="right")
        SEP()

        # ── VIDEO SOURCE ──────────────────────────────────────────────────
        H("📡  Video Source", CYAN)

        self._src_var = tk.StringVar(value="usb")
        sources = [
            ("usb",    "🎥 USB / Webcam Camera"),
            ("ip",     "📷 IP Camera  (enter IP below)"),
            ("screen", "🖥️  Screen Capture"),
            ("rtsp",   "📡 Full RTSP URL (manual)"),
            ("video",  "🎬 Video File"),
        ]
        for val, txt in sources:
            tk.Radiobutton(p, text=txt, variable=self._src_var, value=val,
                           bg=PANEL, fg="#aaa", selectcolor=PANEL,
                           activebackground=PANEL, activeforeground=GREEN,
                           font=FONT, command=self._on_src_change
                           ).pack(anchor="w", padx=18, pady=1)

        # ── USB Camera Panel ──────────────────────────────────────────────
        self._usb_frame = tk.Frame(p, bg="#0a1520", relief="flat", bd=1)
        tk.Label(self._usb_frame, text="Select camera index:", bg="#0a1520",
                 fg="#888", font=FONT_S).pack(anchor="w", padx=8, pady=(6, 0))

        usb_row = tk.Frame(self._usb_frame, bg="#0a1520")
        usb_row.pack(fill="x", padx=8, pady=4)

        self._usb_var = tk.IntVar(value=0)
        self._usb_combo = ttk.Combobox(usb_row, textvariable=self._usb_var,
                                       values=["0"], state="readonly", width=6)
        self._usb_combo.pack(side="left")
        tk.Button(usb_row, text="🔍 Scan", bg="#111", fg=CYAN,
                  relief="flat", font=FONT_S,
                  command=self._scan_usb_cameras).pack(side="left", padx=6)

        self._usb_status_lbl = tk.Label(self._usb_frame, text="Click Scan to detect cameras",
                                        bg="#0a1520", fg="#2a6a8a", font=FONT_S,
                                        wraplength=290, justify="left")
        self._usb_status_lbl.pack(padx=8, pady=(0, 6))

        # USB resolution
        tk.Label(self._usb_frame, text="Requested resolution:",
                 bg="#0a1520", fg="#888", font=FONT_S
                 ).pack(anchor="w", padx=8)
        self._usb_res_var = tk.StringVar(value="native")
        ttk.Combobox(self._usb_frame, textvariable=self._usb_res_var, width=16,
                     state="readonly",
                     values=["native", "1920x1080", "1280x720", "640x480"]
                     ).pack(anchor="w", padx=8, pady=(0, 6))

        # ── IP Camera Panel ───────────────────────────────────────────────
        self._ip_frame = tk.Frame(p, bg="#0a1520", relief="flat", bd=1)

        def ip_lbl(text, parent=None):
            tk.Label(parent or self._ip_frame, text=text, bg="#0a1520", fg="#888",
                     font=FONT_S).pack(anchor="w", padx=8, pady=(4, 0))

        ip_lbl("Camera IP Address")
        self._ip_entry = tk.Entry(self._ip_frame, bg="#111", fg=CYAN,
                                  insertbackground=CYAN,
                                  font=("Courier New", 10, "bold"),
                                  relief="flat", bd=2)
        self._ip_entry.insert(0, "192.168.1.100")
        self._ip_entry.pack(fill="x", padx=8, pady=(0, 4))

        ip_lbl("Port  (auto-filled by preset)")
        self._port_entry = tk.Entry(self._ip_frame, bg="#111", fg=CYAN,
                                    insertbackground=CYAN, font=FONT,
                                    relief="flat", bd=2, width=8)
        self._port_entry.insert(0, "554")
        self._port_entry.pack(anchor="w", padx=8, pady=(0, 4))

        ip_lbl("Camera Brand / App")
        self._preset_var = tk.StringVar(value="📱 Android — IP Webcam (video)")
        preset_box = ttk.Combobox(self._ip_frame, textvariable=self._preset_var,
                                  values=list(CAM_PRESETS.keys()),
                                  state="readonly", width=34)
        preset_box.pack(fill="x", padx=8, pady=(0, 2))
        preset_box.bind("<<ComboboxSelected>>", self._on_preset_change)

        self._preset_desc_var = tk.StringVar(value="MJPEG stream from IP Webcam app")
        tk.Label(self._ip_frame, textvariable=self._preset_desc_var,
                 bg="#0a1520", fg="#2a6a8a", font=FONT_S,
                 wraplength=290, justify="left").pack(fill="x", padx=8, pady=(0, 4))

        self._custom_path_frame = tk.Frame(self._ip_frame, bg="#0a1520")
        ip_lbl("Stream Path", parent=self._custom_path_frame)
        self._path_entry = tk.Entry(self._custom_path_frame, bg="#111",
                                    fg=CYAN, insertbackground=CYAN,
                                    font=FONT, relief="flat", bd=2)
        self._path_entry.insert(0, "/stream1")
        self._path_entry.pack(fill="x", padx=8, pady=(0, 4))

        self._auth_frame = tk.Frame(self._ip_frame, bg="#0a1520")
        ip_lbl("Username", parent=self._auth_frame)
        self._ip_user_entry = tk.Entry(self._auth_frame, bg="#111", fg=CYAN,
                                       insertbackground=CYAN, font=FONT,
                                       relief="flat", bd=2)
        self._ip_user_entry.insert(0, "admin")
        self._ip_user_entry.pack(fill="x", padx=8, pady=(0, 4))
        ip_lbl("Password", parent=self._auth_frame)
        self._ip_pass_entry = tk.Entry(self._auth_frame, bg="#111", fg=CYAN,
                                       insertbackground=CYAN, font=FONT,
                                       relief="flat", bd=2, show="●")
        self._ip_pass_entry.pack(fill="x", padx=8, pady=(0, 4))

        self._url_preview_var = tk.StringVar(value="")
        tk.Label(self._ip_frame, textvariable=self._url_preview_var,
                 bg="#0a1520", fg="#446677", font=FONT_S,
                 wraplength=290, justify="left").pack(fill="x", padx=8, pady=(0, 6))

        for w in (self._ip_entry, self._port_entry, self._path_entry,
                  self._ip_user_entry, self._ip_pass_entry):
            w.bind("<KeyRelease>", lambda e: self._refresh_url_preview())
        self._on_preset_change()

        # ── RTSP manual ───────────────────────────────────────────────────
        self._rtsp_entry = tk.Entry(p, bg="#111", fg=CYAN,
                                    insertbackground=CYAN, font=FONT,
                                    relief="flat", bd=2)
        self._rtsp_entry.insert(0, "rtsp://admin:pass@192.168.1.100:554/stream1")

        # ── Screen capture ────────────────────────────────────────────────
        self._scr_frame = tk.Frame(p, bg=PANEL)
        tk.Label(self._scr_frame, text="Monitor #  (1 = primary):",
                 bg=PANEL, fg="#888", font=FONT_S).pack(anchor="w", padx=4)
        mr = tk.Frame(self._scr_frame, bg=PANEL); mr.pack(fill="x", padx=4, pady=2)
        self._mon_var = tk.IntVar(value=1)
        tk.Spinbox(mr, from_=1, to=4, textvariable=self._mon_var, width=4,
                   bg="#111", fg=CYAN, font=FONT).pack(side="left")
        tk.Button(mr, text="Set", bg="#111", fg=ORANGE, relief="flat",
                  font=FONT, command=self._set_monitor).pack(side="left", padx=4)

        # ── File label ────────────────────────────────────────────────────
        self._file_lbl = tk.Label(p, text="No file selected", bg=PANEL,
                                  fg="#555", font=FONT_S, wraplength=300)
        SEP()

        # ── ACTION BUTTONS ─────────────────────────────────────────────────
        def btn(text, cmd, fg, bg="#111"):
            tk.Button(p, text=text, command=cmd,
                      bg=bg, fg=fg,
                      activebackground="#1a1a2e", activeforeground=fg,
                      relief="flat", bd=0,
                      font=("Courier New", 10, "bold"),
                      cursor="hand2", pady=9
                      ).pack(fill="x", padx=12, pady=3)

        btn("▶  START",            self._start_stream,   GREEN)
        btn("⏹  STOP",             self._stop_stream,    RED)
        btn("🔄  RESET COUNTERS",  self._reset_counters, YELLOW)

        self._status_lbl = tk.Label(p, text="● Idle", bg=PANEL, fg="#555",
                                    font=("Courier New", 9))
        self._status_lbl.pack(pady=(8, 16))

        self._on_src_change()

    # ─────────────────────────────────────────────────────────────────────────
    # DIRECTION TOGGLE
    # ─────────────────────────────────────────────────────────────────────────
    def _toggle_direction(self):
        inv = not self._invert_var.get()
        self._invert_var.set(inv)
        if inv:
            self._dir_btn.config(text="⬇  Arrow DOWN  =  IN direction",
                                 bg="#220d0d", fg="#ff4455",
                                 activebackground="#2e1a1a", activeforeground="#ff4455")
        else:
            self._dir_btn.config(text="⬆  Arrow UP  =  IN direction",
                                 bg="#0d2211", fg="#00ff88",
                                 activebackground="#152e1a", activeforeground="#00ff88")
        with state_lock:
            state.reset()
        for lbl in (self._lbl_in, self._lbl_out, self._lbl_now):
            lbl.config(text="0")

    # ─────────────────────────────────────────────────────────────────────────
    # USB CAMERA
    # ─────────────────────────────────────────────────────────────────────────
    def _scan_usb_cameras(self):
        self._usb_status_lbl.config(text="Scanning… (may take a few seconds)", fg="#ffcc00")
        self.root.update_idletasks()

        def _scan():
            cams = enumerate_usb_cameras()
            self._usb_cameras = cams
            labels = [f"Camera {i}" for i in cams]
            self._usb_combo.config(values=labels)
            self._usb_combo.current(0)
            self._usb_var.set(cams[0])
            self._usb_status_lbl.config(
                text=f"Found: {len(cams)} camera(s)  →  indices {cams}",
                fg="#00ff88")

        threading.Thread(target=_scan, daemon=True).start()

    def _get_usb_index(self) -> int:
        try:
            sel = self._usb_combo.get()
            return int(sel.split()[-1])
        except Exception:
            return 0

    # ─────────────────────────────────────────────────────────────────────────
    # IP CAMERA HELPERS
    # ─────────────────────────────────────────────────────────────────────────
    def _on_preset_change(self, _event=None):
        preset = self._preset_var.get()
        proto, default_port, path, needs_auth, desc = CAM_PRESETS[preset]
        self._port_entry.delete(0, tk.END)
        self._port_entry.insert(0, str(default_port))
        self._preset_desc_var.set(desc)
        if preset.startswith("✏️"):
            self._custom_path_frame.pack(fill="x")
        else:
            self._custom_path_frame.pack_forget()
        if needs_auth:
            self._auth_frame.pack(fill="x")
        else:
            self._auth_frame.pack_forget()
        self._refresh_url_preview()

    def _refresh_url_preview(self):
        try:
            url = build_cam_url(self._preset_var.get(), self._ip_entry.get(),
                                self._port_entry.get(), self._ip_user_entry.get(),
                                self._ip_pass_entry.get(), self._path_entry.get())
        except Exception:
            url = "(fill in fields above)"
        self._url_preview_var.set(f"→ {url}")

    def _get_ip_camera_url(self) -> str:
        return build_cam_url(self._preset_var.get(), self._ip_entry.get(),
                             self._port_entry.get(), self._ip_user_entry.get(),
                             self._ip_pass_entry.get(), self._path_entry.get())

    # ─────────────────────────────────────────────────────────────────────────
    def _apply_res(self):
        v = self._res_var.get()
        if v == "native":
            normalizer.infer_width = normalizer.infer_height = 9999
        else:
            w, h = v.split("x")
            normalizer.infer_width, normalizer.infer_height = int(w), int(h)

    # ─────────────────────────────────────────────────────────────────────────
    # SOURCE PANEL SWITCHING
    # ─────────────────────────────────────────────────────────────────────────
    def _on_src_change(self):
        src = self._src_var.get()
        for w in (self._rtsp_entry, self._file_lbl, self._scr_frame,
                  self._ip_frame, self._usb_frame):
            w.pack_forget()

        if src == "usb":
            self._usb_frame.pack(fill="x", padx=12, pady=(4, 0))
        elif src == "ip":
            self._ip_frame.pack(fill="x", padx=12, pady=(4, 0))
        elif src == "rtsp":
            self._rtsp_entry.pack(fill="x", padx=12, pady=(4, 0))
        elif src == "video":
            self._file_lbl.pack(padx=12)
            self._browse_file()
        elif src == "screen":
            self._scr_frame.pack(fill="x", padx=12, pady=(4, 0))

    def _browse_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.flv"),
                       ("All", "*.*")])
        if path:
            self._video_path = path
            self._file_lbl.config(text=path.split("/")[-1], fg="#aaa")
        self._file_lbl.pack(padx=12)

    def _set_monitor(self):
        self._screen_mon = self._mon_var.get()
        self._status(f"● Monitor #{self._screen_mon} selected", "#ffcc00")

    # ─────────────────────────────────────────────────────────────────────────
    # STREAM CONTROL
    # ─────────────────────────────────────────────────────────────────────────
    def _start_stream(self):
        if self._running:
            return
        src = self._src_var.get()

        if src == "screen":
            self._cap = ScreenCapture(monitor_index=self._screen_mon,
                                      region=self._screen_region)

        elif src == "usb":
            idx = self._get_usb_index()
            self._status(f"● Opening camera {idx} …", "#ffcc00")
            self.root.update_idletasks()
            cap = cv2.VideoCapture(idx)
            # Request full resolution from USB camera
            res = self._usb_res_var.get()
            if res != "native":
                w, h = res.split("x")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  int(w))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._cap = cap

        elif src == "webcam":
            self._cap = cv2.VideoCapture(0)

        elif src == "video":
            if not self._video_path:
                messagebox.showerror("No file", "Select a video file first.")
                return
            self._cap = cv2.VideoCapture(self._video_path)

        elif src == "ip":
            url = self._get_ip_camera_url()
            self._status(f"● Connecting…  {url[:50]}…", "#ffcc00")
            self.root.update_idletasks()
            cap = cv2.VideoCapture(url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._cap = cap

        else:  # manual rtsp
            cap = cv2.VideoCapture(self._rtsp_entry.get().strip())
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._cap = cap

        if not self._cap.isOpened():
            self._status("❌  Cannot open source", "#ff4455")
            if src in ("ip", "rtsp"):
                messagebox.showerror(
                    "Connection Failed",
                    "Could not connect to the camera.\n\n"
                    "Check:\n"
                    "  • IP address and port\n"
                    "  • Username / password\n"
                    "  • Stream path matches camera brand\n"
                    "  • Camera on same network\n"
                    "  • Firewall not blocking port")
            return

        # Log actual resolution
        if hasattr(self._cap, "get"):
            aw = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            ah = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if aw and ah:
                self._status(f"● Running  {int(aw)}×{int(ah)}", "#00ff88")

        with state_lock:
            state.reset()

        self._raw_q   = queue.Queue(maxsize=RAW_Q_SIZE)
        self._frame_q = queue.Queue(maxsize=DISP_Q_SIZE)
        self._running = True

        if not getattr(self, "_status_set_by_open", False):
            self._status("● Running", "#00ff88")

        self._t_capture = threading.Thread(target=self._capture_loop, daemon=True)
        self._t_capture.start()
        self._t_infer   = threading.Thread(target=self._infer_loop,   daemon=True)
        self._t_infer.start()

    def _stop_stream(self):
        self._running = False
        if self._cap:
            self._cap.release()
            self._cap = None
        self._status("● Stopped", "#ffcc00")

    def _on_close(self):
        self._stop_stream()
        self.root.destroy()

    # ─────────────────────────────────────────────────────────────────────────
    # THREAD 1 — CAPTURE
    # ─────────────────────────────────────────────────────────────────────────
    def _capture_loop(self):
        src = self._src_var.get()
        while self._running:
            if not self._cap or not self._cap.isOpened():
                break
            ret, frame = self._cap.read()
            if not ret:
                if src == "video":
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            # Keep only latest frame
            if self._raw_q.full():
                try:
                    self._raw_q.get_nowait()
                except queue.Empty:
                    pass
            try:
                self._raw_q.put_nowait(frame)
            except queue.Full:
                pass

            with state_lock:
                now     = time.time()
                cap_fps = 1.0 / max(now - state.t_cap_last, 1e-6)
                state.t_cap_last = now
                state.cap_fps_buf.append(cap_fps)
                if len(state.cap_fps_buf) > 30:
                    state.cap_fps_buf.pop(0)

        self._running = False
        self._status("● Finished", "#33b5e5")

    # ─────────────────────────────────────────────────────────────────────────
    # THREAD 2 — INFERENCE (YOLO + drawing on full-res frame)
    # ─────────────────────────────────────────────────────────────────────────
    def _infer_loop(self):
        while self._running:
            try:
                frame = self._raw_q.get(timeout=1.0)
            except queue.Empty:
                continue

            # Pre-scale display frame to canvas size so main thread does zero work
            cw = self._canvas_w
            ch = self._canvas_h
            if cw > 2 and ch > 2:
                fh, fw = frame.shape[:2]
                scale  = min(cw / fw, ch / fh)
                nw, nh = int(fw * scale), int(fh * scale)
                display_frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
            else:
                display_frame = frame
                nw, nh = frame.shape[1], frame.shape[0]

            display, fps = process_frame(
                display_frame,
                line_pct         = self._line_var.get() / 100.0,
                angle_deg        = self._angle_var.get(),
                line_thickness   = self._line_thick_var.get(),
                box_thickness    = self._box_thick_var.get(),
                trail_length     = self._trail_len_var.get(),
                show_ids         = self._show_ids_var.get(),
                show_trail       = self._show_trail_var.get(),
                conf             = self._conf_var.get(),
                invert_direction = self._invert_var.get(),
            )

            # Convert BGR→RGB and to PIL on the inference thread (off main thread)
            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            cw  = self._canvas_w
            ch  = self._canvas_h
            ox  = max(0, (cw - nw) // 2)
            oy  = max(0, (ch - nh) // 2)

            if self._frame_q.full():
                try:
                    self._frame_q.get_nowait()
                except queue.Empty:
                    pass
            try:
                self._frame_q.put_nowait((pil, fps, ox, oy))
            except queue.Full:
                pass

    # ─────────────────────────────────────────────────────────────────────────
    # UI POLL (main thread ~60 Hz)
    # ─────────────────────────────────────────────────────────────────────────
    def _on_canvas_resize(self, event):
        self._canvas_w = event.width
        self._canvas_h = event.height

    def _poll_ui(self):
        try:
            rgb_img, fps, offset_x, offset_y = self._frame_q.get_nowait()
            self._show_frame(rgb_img, offset_x, offset_y)
            with state_lock:
                c_in    = state.count_in
                c_out   = state.count_out
                cap_fps = float(np.mean(state.cap_fps_buf)) if state.cap_fps_buf else 0.0
            self._lbl_in.config( text=str(c_in))
            self._lbl_out.config(text=str(c_out))
            self._lbl_now.config(text=str(max(0, c_in - c_out)))
            self._lbl_fps.config(text=f"{fps:.1f}")
            self._lbl_cap_fps.config(text=f"{cap_fps:.1f}")
        except queue.Empty:
            pass
        interval_ms = max(8, int(1000 / self._uifps_var.get()))
        self.root.after(interval_ms, self._poll_ui)

    def _show_frame(self, pil_img: Image.Image, ox: int, oy: int):
        """Paint frame — reuse the canvas item to avoid delete/create overhead."""
        photo = ImageTk.PhotoImage(pil_img)
        if self._canvas_img_id is None:
            self._canvas_img_id = self.canvas.create_image(ox, oy, anchor="nw", image=photo)
        else:
            self.canvas.coords(self._canvas_img_id, ox, oy)
            self.canvas.itemconfig(self._canvas_img_id, image=photo)
        # Must keep a reference or GC deletes the image
        self.canvas._img_ref = photo

    # ─────────────────────────────────────────────────────────────────────────
    def _reset_counters(self):
        with state_lock:
            state.reset()
        for lbl in (self._lbl_in, self._lbl_out, self._lbl_now):
            lbl.config(text="0")
        self._lbl_fps.config(text="—")

    def _status(self, text, color="#aaa"):
        self._status_lbl.config(text=text, fg=color)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    PeopleCounterApp(root)
    root.mainloop()
