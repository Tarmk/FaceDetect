#!/usr/bin/env python3
import argparse
import time
import json
import math
import sys
import os
import platform
from typing import Optional, Tuple

import cv2
import requests
from ultralytics import YOLO


# -----------------------------
# Utilities
# -----------------------------

def clamp(value: float, min_v: float, max_v: float) -> float:
    return max(min_v, min(max_v, value))


def ema(prev: float, new: float, a: float) -> float:
    return a * prev + (1.0 - a) * new


def eye_pupil_pos_xy(
    gray_roi: Optional[cv2.UMat],
    min_eye_area_px: int,
    min_eye_area_frac: float,
) -> Optional[Tuple[float, float]]:
    """Return normalized (x, y) centroid of darkest significant blob in ROI if blob area is plausible.
    - min_eye_area_px: minimum contour area in pixels
    - min_eye_area_frac: minimum area fraction relative to ROI (0..1)
    """
    if gray_roi is None or gray_roi.size == 0:
        return None
    h, w = gray_roi.shape[:2]
    roi_area = max(1, h * w)
    blur = cv2.GaussianBlur(gray_roi, (7, 7), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area < float(min_eye_area_px):
        return None
    if (area / float(roi_area)) < float(min_eye_area_frac):
        return None
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    nx = clamp(cx / float(w), 0.0, 1.0)
    ny = clamp(cy / float(h), 0.0, 1.0)
    return (nx, ny)


def safe_roi(frame, x1: int, y1: int, x2: int, y2: int):
    H, W = frame.shape[:2]
    x1 = clamp(int(x1), 0, W)
    x2 = clamp(int(x2), 0, W)
    y1 = clamp(int(y1), 0, H)
    y2 = clamp(int(y2), 0, H)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def color_for_state(state: str):
    if state == "ATTENTIVE":
        return (0, 200, 0)
    if state in ("LEFT", "RIGHT"):
        return (0, 215, 255)
    return (0, 0, 255)


def ensure_weights(path: str) -> str:
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    url = "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.pt"
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return path
    except Exception as e:
        print(json.dumps({"type": "error", "message": f"Auto-download failed: {e}"}), flush=True)
        return path


# -----------------------------
# Main
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Attention Tracking with YOLO (face) + OpenCV")
    p.add_argument("--model", type=str, default=os.path.join("weights", "yolov8n-face-lindevs.pt"), help="Path to a face-capable YOLO checkpoint")
    p.add_argument("--src", type=int, default=0, help="Camera index or video file path (use integer for camera)")
    p.add_argument("--width", type=int, default=0, help="Capture width (0 to skip)")
    p.add_argument("--height", type=int, default=0, help="Capture height (0 to skip)")

    p.add_argument("--start", dest="START_S", type=float, default=0.0)
    p.add_argument("--end", dest="END_S", type=float, default=float("inf"))

    p.add_argument("--px-left", dest="PX_LEFT", type=float, default=0.35)
    p.add_argument("--px-right", dest="PX_RIGHT", type=float, default=0.65)

    p.add_argument("--center-attentive", dest="CENTER_ATTENTIVE", type=float, default=0.08)
    p.add_argument("--center-tol", dest="CENTER_TOL", type=float, default=0.10)
    p.add_argument("--center-distract", dest="CENTER_DISTRACT", type=float, default=0.15)

    p.add_argument("--face-miss-ms", dest="FACE_MISS_MS", type=int, default=400)
    p.add_argument("--popup-after-s", dest="POPUP_AFTER_S", type=float, default=3.0)
    p.add_argument("--popup-cooldown-s", dest="POPUP_COOLDOWN_S", type=float, default=10.0)

    p.add_argument("--smoothing", dest="SMOOTHING", type=float, default=0.7)
    p.add_argument("--min-face-conf", dest="MIN_FACE_CONF", type=float, default=0.3)
    p.add_argument("--min-eye-area", dest="MIN_EYE_AREA", type=int, default=80)
    p.add_argument("--min-eye-area-frac", dest="MIN_EYE_AREA_FRAC", type=float, default=0.002)

    p.add_argument("--fps-target", dest="FPS_TARGET", type=float, default=30.0)
    p.add_argument("--overlay", action="store_true", help="Draw optional overlay window")
    p.add_argument("--no-overlay", dest="overlay", action="store_false")
    p.set_defaults(overlay=True)

    # YOLO inference
    p.add_argument("--imgsz", dest="IMGSZ", type=int, default=640)
    p.add_argument("--device", dest="DEVICE", type=str, default="")
    p.add_argument("--adaptive-conf", dest="ADAPTIVE_CONF", action="store_true")
    p.add_argument("--no-adaptive-conf", dest="ADAPTIVE_CONF", action="store_false")
    p.set_defaults(ADAPTIVE_CONF=True)

    # Eye vertical attention
    p.add_argument("--down-py-min", dest="DOWN_PY_MIN", type=float, default=0.6)

    # Gaze override safety
    p.add_argument("--eye-reliable-frames", dest="EYE_RELIABLE_FRAMES", type=int, default=3)
    p.add_argument("--min-aspect-for-override", dest="MIN_ASPECT_FOR_OVERRIDE", type=float, default=0.55)
    p.add_argument("--yaw-override-max", dest="YAW_OVERRIDE_MAX", type=float, default=0.2)

    # Camera backend
    p.add_argument("--backend", dest="BACKEND", type=str, default="auto", choices=["auto", "avfoundation", "qt", "v4l2", "dshow"])

    return p.parse_args()


def main():
    args = parse_args()

    # Config (from args)
    MODEL_PATH = ensure_weights(args.model)
    START_S = float(args.START_S)
    END_S = float(args.END_S)
    PX_LEFT, PX_RIGHT = float(args.PX_LEFT), float(args.PX_RIGHT)
    CENTER_ATTENTIVE, CENTER_TOL, CENTER_DISTRACT = (
        float(args.CENTER_ATTENTIVE), float(args.CENTER_TOL), float(args.CENTER_DISTRACT)
    )
    FACE_MISS_MS = int(args.FACE_MISS_MS)
    POPUP_AFTER_S = float(args.POPUP_AFTER_S)
    POPUP_COOLDOWN_S = float(args.POPUP_COOLDOWN_S)
    SMOOTHING = float(args.SMOOTHING)
    MIN_FACE_CONF = float(args.MIN_FACE_CONF)
    MIN_EYE_AREA = int(args.MIN_EYE_AREA)
    MIN_EYE_AREA_FRAC = float(args.MIN_EYE_AREA_FRAC)
    FPS_TARGET = float(args.FPS_TARGET)
    OVERLAY = bool(args.overlay)
    IMGSZ = int(args.IMGSZ)
    DEVICE = args.DEVICE.strip()
    ADAPTIVE_CONF = bool(args.ADAPTIVE_CONF)
    DOWN_PY_MIN = float(args.DOWN_PY_MIN)
    EYE_RELIABLE_FRAMES = int(args.EYE_RELIABLE_FRAMES)
    MIN_ASPECT_FOR_OVERRIDE = float(args.MIN_ASPECT_FOR_OVERRIDE)
    YAW_OVERRIDE_MAX = float(args.YAW_OVERRIDE_MAX)
    BACKEND = args.BACKEND

    # Capture
    cap_src = args.src

    def backend_flag(name: str):
        name = (name or "").lower()
        if name == "avfoundation":
            return getattr(cv2, "CAP_AVFOUNDATION", 0)
        if name == "qt":
            return getattr(cv2, "CAP_QT", 0)
        if name == "v4l2":
            return getattr(cv2, "CAP_V4L2", 0)
        if name == "dshow":
            return getattr(cv2, "CAP_DSHOW", 0)
        return 0

    if BACKEND == "auto":
        cap = cv2.VideoCapture(cap_src, backend_flag("avfoundation") if sys.platform == "darwin" else 0)
    else:
        cap = cv2.VideoCapture(cap_src, backend_flag(BACKEND))

    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        try_alt = cv2.VideoCapture(cap_src)
        if try_alt.isOpened():
            cap = try_alt
        else:
            print(json.dumps({"type": "error", "message": "Unable to open capture source"}))
            sys.exit(1)

    # Model
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(json.dumps({"type": "error", "message": f"Failed to load YOLO model: {e}. Try passing --model to a valid face weights .pt"}))
        sys.exit(1)

    # State
    t0 = time.time()
    prev_time = t0
    last_face_ms = None
    ema_center = 0.0
    ema_lx = 0.5
    ema_rx = 0.5
    ema_ly = 0.5
    ema_ry = 0.5

    distraction_acc = 0.0
    last_popup_t = -1e9
    popups_triggered = 0

    stable_state = "AWAY"
    pending_state: Optional[str] = None
    pending_count = 0

    attentive_time = 0.0
    distracted_time = 0.0
    frames = 0
    breakdown = {"LEFT": 0.0, "RIGHT": 0.0, "AWAY": 0.0}

    prev_bbox_h = None
    last_bbox: Optional[Tuple[int, int, int, int]] = None

    no_face_streak = 0
    eye_reliable_streak = 0

    warmup_start = time.time()
    warmup_frames = 0

    while True:
        ok, frame = cap.read()
        now = time.time()
        if not ok:
            if warmup_frames == 0 and (now - warmup_start) > 1.0:
                cap.release()
                alt = cv2.VideoCapture(1, backend_flag(BACKEND if BACKEND != "auto" else ("avfoundation" if sys.platform == "darwin" else "")))
                if alt.isOpened():
                    cap = alt
                    warmup_start = time.time()
                    continue
            break
        warmup_frames += 1

        t = now - t0
        if t < START_S:
            prev_time = now
            if OVERLAY:
                disp = frame.copy()
                H, W = disp.shape[:2]
                cv2.putText(disp, "waiting window...", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                if last_bbox is not None:
                    x, y, w, h = last_bbox
                    cv2.rectangle(disp, (x, y), (x + w, y + h), (128, 128, 128), 2)
                    cv2.putText(disp, "LAST FACE", (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2, cv2.LINE_AA)
                cv2.imshow("attention", disp)
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    break
            continue
        if t > END_S:
            break

        H, W = frame.shape[:2]

        face_present = False
        bbox = None
        current_conf = MIN_FACE_CONF if not (ADAPTIVE_CONF and no_face_streak >= 10) else max(0.15, MIN_FACE_CONF * 0.6)
        try:
            results = model.predict(source=frame, verbose=False, imgsz=IMGSZ, device=DEVICE if DEVICE else None, conf=current_conf)
        except Exception:
            results = []
        best = None
        for r in results:
            if getattr(r, 'boxes', None) is None:
                continue
            for b in r.boxes:
                try:
                    conf = float(b.conf)
                except Exception:
                    try:
                        conf = float(b.conf.item())
                    except Exception:
                        continue
                if conf < current_conf:
                    continue
                try:
                    x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                except Exception:
                    try:
                        x1, y1, x2, y2 = map(int, b.xyxy[0])
                    except Exception:
                        continue
                if best is None or conf > best[0]:
                    best = (conf, (x1, y1, x2 - x1, y2 - y1))

        if best is not None:
            face_present = True
            bbox = best[1]
            last_face_ms = now * 1000.0
            no_face_streak = 0
            last_bbox = bbox
        else:
            if last_face_ms is None:
                last_face_ms = now * 1000.0
            face_present = False
            no_face_streak += 1

        gaze_to_screen: Optional[bool] = None
        gaze_down: Optional[bool] = None
        center_offset = 0.0
        candidate_state = "AWAY"
        missing_face_trigger = False
        # For overlay rendering of eyes/pupils
        left_eye_rect: Optional[Tuple[int, int, int, int]] = None
        right_eye_rect: Optional[Tuple[int, int, int, int]] = None
        lpupil_px: Optional[Tuple[int, int]] = None
        rpupil_px: Optional[Tuple[int, int]] = None

        if face_present and bbox is not None:
            x, y, w, h = bbox
            x = int(clamp(x, 0, W - 1))
            y = int(clamp(y, 0, H - 1))
            w = int(max(1, min(w, W - x)))
            h = int(max(1, min(h, H - y)))

            cx = x + w / 2.0
            center_offset = (cx - (W / 2.0)) / float(W)
            ema_center = ema(ema_center, center_offset, SMOOTHING)

            eye_top = y
            eye_bottom = y + int(h / 3.0)
            eye_mid = x + w // 2

            left_roi = safe_roi(frame, x, eye_top, eye_mid, eye_bottom)
            right_roi = safe_roi(frame, eye_mid, eye_top, x + w, eye_bottom)
            # Save eye rects for overlay
            left_eye_rect = (x, eye_top, max(0, eye_mid - x), max(0, eye_bottom - eye_top))
            right_eye_rect = (eye_mid, eye_top, max(0, (x + w) - eye_mid), max(0, eye_bottom - eye_top))

            def to_gray(roi):
                if roi is None or roi.size == 0:
                    return None
                return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            lxy = eye_pupil_pos_xy(to_gray(left_roi), MIN_EYE_AREA, MIN_EYE_AREA_FRAC) if left_roi is not None else None
            rxy = eye_pupil_pos_xy(to_gray(right_roi), MIN_EYE_AREA, MIN_EYE_AREA_FRAC) if right_roi is not None else None

            if lxy is not None and rxy is not None:
                lx, ly = lxy
                rx, ry = rxy
                ema_lx = ema(ema_lx, float(lx), SMOOTHING)
                ema_rx = ema(ema_rx, float(rx), SMOOTHING)
                ema_ly = ema(ema_ly, float(ly), SMOOTHING)
                ema_ry = ema(ema_ry, float(ry), SMOOTHING)
                gaze_to_screen = (PX_LEFT <= ema_lx <= PX_RIGHT) and (PX_LEFT <= ema_rx <= PX_RIGHT)
                gaze_down = (ema_ly >= DOWN_PY_MIN) and (ema_ry >= DOWN_PY_MIN)
                eye_reliable_streak += 1
                # Compute absolute pupil positions for overlay (EMA-smoothed)
                try:
                    lpx = x + int(ema_lx * max(1, eye_mid - x))
                    lpy = eye_top + int(ema_ly * max(1, eye_bottom - eye_top))
                    rpx = eye_mid + int(ema_rx * max(1, (x + w) - eye_mid))
                    rpy = eye_top + int(ema_ry * max(1, eye_bottom - eye_top))
                    lpupil_px = (lpx, lpy)
                    rpupil_px = (rpx, rpy)
                except Exception:
                    lpupil_px = None
                    rpupil_px = None
            else:
                eye_reliable_streak = 0
                # If only one eye detected, do not assert gaze_to_screen; allow gaze_down as weak hint only
                if lxy is not None:
                    lx, ly = lxy
                    ema_lx = ema(ema_lx, float(lx), SMOOTHING)
                    ema_ly = ema(ema_ly, float(ly), SMOOTHING)
                    gaze_down = (ema_ly >= DOWN_PY_MIN)
                    try:
                        lpx = x + int(ema_lx * max(1, eye_mid - x))
                        lpy = eye_top + int(ema_ly * max(1, eye_bottom - eye_top))
                        lpupil_px = (lpx, lpy)
                    except Exception:
                        lpupil_px = None
                elif rxy is not None:
                    rx, ry = rxy
                    ema_rx = ema(ema_rx, float(rx), SMOOTHING)
                    ema_ry = ema(ema_ry, float(ry), SMOOTHING)
                    gaze_down = (ema_ry >= DOWN_PY_MIN)
                    try:
                        rpx = eye_mid + int(ema_rx * max(1, (x + w) - eye_mid))
                        rpy = eye_top + int(ema_ry * max(1, eye_bottom - eye_top))
                        rpupil_px = (rpx, rpy)
                    except Exception:
                        rpupil_px = None

            away_pitch = False
            if prev_bbox_h is not None:
                if h < 0.6 * prev_bbox_h:
                    away_pitch = True
            if y <= int(0.02 * H):
                away_pitch = True
            prev_bbox_h = h

            aspect_ratio = w / float(h) if h > 0 else 0.0
            override_allowed = (
                (eye_reliable_streak >= EYE_RELIABLE_FRAMES)
                and (aspect_ratio >= MIN_ASPECT_FOR_OVERRIDE)
                and (abs(ema_center) <= YAW_OVERRIDE_MAX)
            )

            gaze_attentive = override_allowed and ((gaze_to_screen is True) or (gaze_down is True))
            if gaze_attentive:
                candidate_state = "ATTENTIVE"
            else:
                # Spec-aligned fallback when no reliable gaze
                if gaze_to_screen is False and not (gaze_down is True):
                    candidate_state = "AWAY"
                elif ema_center < -CENTER_DISTRACT:
                    candidate_state = "LEFT"
                elif ema_center > +CENTER_DISTRACT:
                    candidate_state = "RIGHT"
                else:
                    if abs(ema_center) <= CENTER_ATTENTIVE:
                        candidate_state = "ATTENTIVE"
                    else:
                        candidate_state = "AWAY"

            if away_pitch and not gaze_attentive:
                candidate_state = "AWAY"
        else:
            if (now * 1000.0 - last_face_ms) > FACE_MISS_MS:
                candidate_state = "AWAY"
                missing_face_trigger = True

        if missing_face_trigger:
            stable_state = "AWAY"
            pending_state = None
            pending_count = 0
        else:
            if candidate_state == stable_state:
                pending_state = None
                pending_count = 0
            else:
                if pending_state == candidate_state:
                    pending_count += 1
                else:
                    pending_state = candidate_state
                    pending_count = 1
                if pending_count >= 2:
                    stable_state = candidate_state
                    pending_state = None
                    pending_count = 0

        attention_score = 0
        if face_present:
            attention_score += 60
        if (gaze_to_screen is True) or (gaze_down is True):
            attention_score += 25
        if stable_state == "ATTENTIVE":
            attention_score += 15
        attention_score = int(clamp(attention_score, 0, 100))

        dt = max(0.0, now - prev_time)
        prev_time = now
        frames += 1

        if stable_state == "ATTENTIVE":
            attentive_time += dt
            distraction_acc = 0.0
        else:
            distracted_time += dt
            if stable_state in breakdown:
                breakdown[stable_state] += dt
            if stable_state in ("LEFT", "RIGHT", "AWAY"):
                distraction_acc += dt

        if distraction_acc >= POPUP_AFTER_S and (t - last_popup_t) >= POPUP_COOLDOWN_S:
            popup = {"type": "popup", "message": "Stop being distracted.", "state": stable_state, "at_seconds": round(t, 2)}
            print(json.dumps(popup), flush=True)
            last_popup_t = t
            distraction_acc = 0.0
            popups_triggered += 1

        per_frame = {
            "t": round(t, 2),
            "face_present": bool(face_present),
            "bbox": [int(b) for b in bbox] if bbox is not None else None,
            "gaze_to_screen": (True if gaze_to_screen is True else False if gaze_to_screen is False else None),
            "center_offset": round(float(ema_center), 4),
            "state": stable_state,
            "attention_score": attention_score,
        }
        print(json.dumps(per_frame), flush=True)

        if OVERLAY:
            disp = frame.copy()
            if bbox is not None:
                x, y, w, h = bbox
                col = color_for_state(stable_state)
                cv2.rectangle(disp, (x, y), (x + w, y + h), col, 2)
                cv2.putText(disp, stable_state, (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
            elif last_bbox is not None:
                x, y, w, h = last_bbox
                cv2.rectangle(disp, (x, y), (x + w, y + h), (128, 128, 128), 2)
                cv2.putText(disp, "NO FACE (last)", (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2, cv2.LINE_AA)

            # Draw eye ROIs
            if left_eye_rect is not None:
                ex, ey, ew, eh = left_eye_rect
                if ew > 0 and eh > 0:
                    cv2.rectangle(disp, (ex, ey), (ex + ew, ey + eh), (80, 160, 255), 1)
            if right_eye_rect is not None:
                ex, ey, ew, eh = right_eye_rect
                if ew > 0 and eh > 0:
                    cv2.rectangle(disp, (ex, ey), (ex + ew, ey + eh), (80, 160, 255), 1)

            # Draw pupil markers and direction pointers
            def draw_pupil_with_arrows(img, pt, horiz_bias, vert_bias, color=(0, 255, 255)):
                try:
                    cv2.circle(img, pt, 4, color, -1)
                    # Horizontal arrow: left/right based on bias (<0.5 left)
                    dx = -18 if horiz_bias < 0.5 else 18
                    hx = pt[0] + dx
                    hy = pt[1]
                    cv2.arrowedLine(img, pt, (hx, hy), (0, 255, 200), 2, tipLength=0.35)
                    # Vertical arrow: up/down based on bias (<0.5 up)
                    dy = -14 if vert_bias < 0.5 else 14
                    vx = pt[0]
                    vy = pt[1] + dy
                    cv2.arrowedLine(img, pt, (vx, vy), (255, 0, 255), 2, tipLength=0.35)
                except Exception:
                    pass

            if lpupil_px is not None:
                draw_pupil_with_arrows(disp, lpupil_px, horiz_bias=ema_lx, vert_bias=ema_ly)
            if rpupil_px is not None:
                draw_pupil_with_arrows(disp, rpupil_px, horiz_bias=ema_rx, vert_bias=ema_ry)

            H, W = disp.shape[:2]
            bar_w = int(200 * (attention_score / 100.0))
            cv2.rectangle(disp, (10, H - 20), (10 + 200, H - 10), (60, 60, 60), -1)
            cv2.rectangle(disp, (10, H - 20), (10 + bar_w, H - 10), color_for_state(stable_state), -1)
            cv2.putText(disp, f"score {attention_score}", (10, H - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            elapsed_in_window = max(0.0, t - START_S)
            attn_percent = 0.0 if elapsed_in_window <= 0 else (100.0 * (attentive_time / elapsed_in_window))
            cv2.putText(disp, f"attn% {attn_percent:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            status = [f"conf>={current_conf:.2f}"]
            if gaze_to_screen is True:
                status.append("gaze:center")
            elif gaze_down is True:
                status.append("gaze:down")
            elif gaze_to_screen is False:
                status.append("gaze:away")
            else:
                status.append("gaze:NA")
            status.append(f"yaw:{ema_center:+.2f}")
            aspect_ratio = (w / float(h)) if (bbox is not None and h > 0) else 0.0
            status.append(f"override:{'on' if (eye_reliable_streak >= EYE_RELIABLE_FRAMES and aspect_ratio >= MIN_ASPECT_FOR_OVERRIDE and abs(ema_center) <= YAW_OVERRIDE_MAX) else 'off'}")
            cv2.putText(disp, ", ".join(status), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("attention", disp)
            if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
                break

    elapsed = time.time() - t0
    duration = (min(END_S, elapsed) - START_S) if START_S < END_S else 0.0
    duration = max(0.0, duration)
    attention_percent = 0.0 if duration <= 0 else round(100.0 * (attentive_time / duration), 2)

    print(json.dumps({
        "type": "summary",
        "duration_s": round(duration, 2),
        "frames": frames,
        "attentive_time_s": round(attentive_time, 2),
        "distracted_time_s": round(distracted_time, 2),
        "attention_percent": attention_percent,
        "distracted_breakdown": {k: round(v, 2) for k, v in breakdown.items()},
        "popups_triggered": popups_triggered,
    }), flush=True)

    cap.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


if __name__ == "__main__":
    main() 