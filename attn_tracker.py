#!/usr/bin/env python3
import argparse
import time
import json
import math
import sys
from typing import Optional, Tuple

import cv2
from ultralytics import YOLO


# -----------------------------
# Utilities
# -----------------------------

def clamp(value: float, min_v: float, max_v: float) -> float:
    return max(min_v, min(max_v, value))


def ema(prev: float, new: float, a: float) -> float:
    return a * prev + (1.0 - a) * new


def eye_pupil_pos(gray_roi: Optional[cv2.UMat], min_eye_area: int) -> Optional[float]:
    if gray_roi is None or gray_roi.size == 0:
        return None
    blur = cv2.GaussianBlur(gray_roi, (7, 7), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < min_eye_area:
        return None
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    cx = M["m10"] / M["m00"]
    w = gray_roi.shape[1]
    return clamp(cx / float(w), 0.0, 1.0)


def safe_roi(frame, x1: int, y1: int, x2: int, y2: int):
    H, W = frame.shape[:2]
    x1 = clamp(int(x1), 0, W)
    x2 = clamp(int(x2), 0, W)
    y1 = clamp(int(y1), 0, H)
    y2 = clamp(int(y2), 0, H)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def color_for_state(state: str) -> Tuple[int, int, int]:
    if state == "ATTENTIVE":
        return (0, 200, 0)
    if state in ("LEFT", "RIGHT"):
        return (0, 215, 255)
    return (0, 0, 255)


# -----------------------------
# Main
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Attention Tracking with YOLO (face) + OpenCV")
    p.add_argument("--model", type=str, default="yolov8n-face.pt", help="Path to a face-capable YOLO checkpoint")
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

    p.add_argument("--fps-target", dest="FPS_TARGET", type=float, default=30.0)
    p.add_argument("--overlay", action="store_true", help="Draw optional overlay window")
    p.add_argument("--no-overlay", dest="overlay", action="store_false")
    p.set_defaults(overlay=False)

    # New: control YOLO inference
    p.add_argument("--imgsz", dest="IMGSZ", type=int, default=640, help="YOLO inference size")
    p.add_argument("--device", dest="DEVICE", type=str, default="", help="YOLO device: cpu|mps|cuda")
    p.add_argument("--adaptive-conf", dest="ADAPTIVE_CONF", action="store_true", help="Lower conf if no faces for several frames")
    p.add_argument("--no-adaptive-conf", dest="ADAPTIVE_CONF", action="store_false")
    p.set_defaults(ADAPTIVE_CONF=True)

    return p.parse_args()


def main():
    args = parse_args()

    # Config (from args)
    MODEL_PATH = args.model
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
    FPS_TARGET = float(args.FPS_TARGET)
    OVERLAY = bool(args.overlay)
    IMGSZ = int(args.IMGSZ)
    DEVICE = args.DEVICE.strip()
    ADAPTIVE_CONF = bool(args.ADAPTIVE_CONF)

    # Capture
    cap_src = args.src
    cap = cv2.VideoCapture(cap_src)
    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(json.dumps({"type": "error", "message": "Unable to open capture source"}))
        sys.exit(1)

    # Model
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(json.dumps({"type": "error", "message": f"Failed to load YOLO model: {e}"}))
        sys.exit(1)

    # State
    t0 = time.time()
    prev_time = t0
    last_face_ms = None
    ema_center = 0.0
    ema_lx = 0.5
    ema_rx = 0.5

    distraction_acc = 0.0
    last_popup_t = -1e9
    popups_triggered = 0

    stable_state = "AWAY"  # debounced
    pending_state: Optional[str] = None
    pending_count = 0

    attentive_time = 0.0
    distracted_time = 0.0
    frames = 0
    breakdown = {"LEFT": 0.0, "RIGHT": 0.0, "AWAY": 0.0}

    prev_bbox_h = None

    # Adaptive confidence control
    no_face_streak = 0

    # Main loop
    while True:
        ok, frame = cap.read()
        now = time.time()
        if not ok:
            break

        t = now - t0
        if t < START_S:
            prev_time = now
            # Render overlay but do not process/emit JSON before window start
            if OVERLAY:
                cv2.imshow("attention", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    break
            continue
        if t > END_S:
            break

        H, W = frame.shape[:2]

        # Detect face (YOLO)
        face_present = False
        bbox = None
        current_conf = MIN_FACE_CONF
        if ADAPTIVE_CONF and no_face_streak >= 10:
            current_conf = max(0.15, MIN_FACE_CONF * 0.6)
        try:
            results = model.predict(
                source=frame,
                verbose=False,
                imgsz=IMGSZ,
                device=DEVICE if DEVICE else None,
                conf=current_conf,
            )
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
        else:
            if last_face_ms is None:
                last_face_ms = now * 1000.0
            face_present = False
            no_face_streak += 1

        # Eye/gaze heuristic
        gaze_to_screen: Optional[bool] = None
        center_offset = 0.0
        candidate_state = "AWAY"
        missing_face_trigger = False

        if face_present and bbox is not None:
            x, y, w, h = bbox
            # Clamp bbox to frame
            x = int(clamp(x, 0, W - 1))
            y = int(clamp(y, 0, H - 1))
            w = int(max(1, min(w, W - x)))
            h = int(max(1, min(h, H - y)))

            cx = x + w / 2.0
            center_offset = (cx - (W / 2.0)) / float(W)
            ema_center = ema(ema_center, center_offset, SMOOTHING)

            # Eye ROIs: upper third, split halves
            eye_top = y
            eye_bottom = y + int(h / 3.0)
            eye_mid = x + w // 2

            left_roi = safe_roi(frame, x, eye_top, eye_mid, eye_bottom)
            right_roi = safe_roi(frame, eye_mid, eye_top, x + w, eye_bottom)

            def to_gray(roi):
                if roi is None or roi.size == 0:
                    return None
                return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            lx = eye_pupil_pos(to_gray(left_roi), MIN_EYE_AREA) if left_roi is not None else None
            rx = eye_pupil_pos(to_gray(right_roi), MIN_EYE_AREA) if right_roi is not None else None

            if lx is not None and rx is not None:
                ema_lx = ema(ema_lx, float(lx), SMOOTHING)
                ema_rx = ema(ema_rx, float(rx), SMOOTHING)
                gaze_to_screen = (PX_LEFT <= ema_lx <= PX_RIGHT) and (PX_LEFT <= ema_rx <= PX_RIGHT)

            # Pitch proxy (away up/down or far)
            away_pitch = False
            if prev_bbox_h is not None:
                if h < 0.6 * prev_bbox_h:
                    away_pitch = True
            if y <= int(0.02 * H):
                away_pitch = True
            prev_bbox_h = h

            # State rules
            if gaze_to_screen is False:
                candidate_state = "AWAY"
            else:
                if ema_center < -CENTER_DISTRACT:
                    candidate_state = "LEFT"
                elif ema_center > +CENTER_DISTRACT:
                    candidate_state = "RIGHT"
                else:
                    if abs(ema_center) <= CENTER_ATTENTIVE:
                        candidate_state = "ATTENTIVE"
                    else:
                        candidate_state = "AWAY"

            if away_pitch:
                candidate_state = "AWAY"
        else:
            # Face missing
            if (now * 1000.0 - last_face_ms) > FACE_MISS_MS:
                candidate_state = "AWAY"
                missing_face_trigger = True

        # Debounce (2 consecutive frames) except when face becomes missing
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

        # Score
        attention_score = 0
        if face_present:
            attention_score += 60
        if gaze_to_screen is True:
            attention_score += 25
        if stable_state == "ATTENTIVE":
            attention_score += 15
        attention_score = int(clamp(attention_score, 0, 100))

        # Timers
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

        # Popup logic
        if distraction_acc >= POPUP_AFTER_S and (t - last_popup_t) >= POPUP_COOLDOWN_S:
            popup = {
                "type": "popup",
                "message": "Stop being distracted.",
                "state": stable_state,
                "at_seconds": round(t, 2),
            }
            print(json.dumps(popup), flush=True)
            last_popup_t = t
            distraction_acc = 0.0
            popups_triggered += 1

        # Per-frame JSON
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

        # Overlay (optional)
        if OVERLAY:
            disp = frame.copy()
            if bbox is not None:
                x, y, w, h = bbox
                col = color_for_state(stable_state)
                cv2.rectangle(disp, (x, y), (x + w, y + h), col, 2)
                cv2.putText(
                    disp,
                    stable_state,
                    (x, max(0, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    col,
                    2,
                    cv2.LINE_AA,
                )

            # Attention score bar
            bar_w = int(200 * (attention_score / 100.0))
            cv2.rectangle(disp, (10, H - 20), (10 + 200, H - 10), (60, 60, 60), -1)
            cv2.rectangle(disp, (10, H - 20), (10 + bar_w, H - 10), color_for_state(stable_state), -1)
            cv2.putText(
                disp,
                f"score {attention_score}",
                (10, H - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Running attention %
            elapsed_in_window = max(0.0, t - START_S)
            attn_percent = 0.0 if elapsed_in_window <= 0 else (100.0 * (attentive_time / elapsed_in_window))
            cv2.putText(
                disp,
                f"attn% {attn_percent:.1f}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Show conf status
            cv2.putText(
                disp,
                f"conf>={current_conf:.2f}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("attention", disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

        # FPS pacing (best-effort)
        if FPS_TARGET > 0:
            frame_time = time.time() - now
            target = 1.0 / FPS_TARGET
            sleep_s = target - frame_time
            if sleep_s > 0:
                time.sleep(sleep_s)

    # Summary
    elapsed = time.time() - t0
    duration = (min(END_S, elapsed) - START_S) if START_S < END_S else 0.0
    duration = max(0.0, duration)
    attention_percent = 0.0 if duration <= 0 else round(100.0 * (attentive_time / duration), 2)

    print(
        json.dumps(
            {
                "type": "summary",
                "duration_s": round(duration, 2),
                "frames": frames,
                "attentive_time_s": round(attentive_time, 2),
                "distracted_time_s": round(distracted_time, 2),
                "attention_percent": attention_percent,
                "distracted_breakdown": {k: round(v, 2) for k, v in breakdown.items()},
                "popups_triggered": popups_triggered,
            }
        ),
        flush=True,
    )

    cap.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


if __name__ == "__main__":
    main() 