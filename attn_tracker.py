#!/usr/bin/env python3
"""Attention tracking from a webcam or a video file.

Prints one JSON line per frame, a {"type": "popup"} line when the viewer has
been distracted for too long, and a {"type": "summary"} line at the end.
Run with --help for every option.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import IO, Optional

import cv2

from attention.config import Config
from attention.detector import DEFAULT_WEIGHTS, FaceDetector, ensure_weights
from attention.overlay import draw
from attention.state import AttentionTracker
from attention.vision import clip_rect, crop, eye_rects, pupil_position, to_gray

BACKENDS = {
    "auto": None,
    "avfoundation": "CAP_AVFOUNDATION",
    "qt": "CAP_QT",
    "v4l2": "CAP_V4L2",
    "dshow": "CAP_DSHOW",
}


def parse_args(argv=None) -> argparse.Namespace:
    d = Config()
    p = argparse.ArgumentParser(description="Attention tracking with a YOLO face detector and OpenCV pupil finding.")
    src = p.add_argument_group("input")
    src.add_argument("--src", default="0", help="camera index (0, 1, ...) or path to a video file (default 0)")
    src.add_argument("--model", default=DEFAULT_WEIGHTS, help="face-capable YOLO weights; downloaded if missing")
    src.add_argument("--width", type=int, default=0, help="camera capture width (0 = leave as is)")
    src.add_argument("--height", type=int, default=0, help="camera capture height (0 = leave as is)")
    src.add_argument("--backend", default="auto", choices=list(BACKENDS), help="camera backend")
    src.add_argument("--start", type=float, default=0.0, help="seconds to skip at the beginning")
    src.add_argument("--end", type=float, default=float("inf"), help="stop after this many seconds")
    src.add_argument("--fps-target", type=float, default=30.0, help="frame-rate cap for cameras (0 = no cap)")

    out = p.add_argument_group("output")
    out.add_argument(
        "--overlay", dest="overlay", action="store_true", default=True, help="show the debug window (default)"
    )
    out.add_argument("--no-overlay", dest="overlay", action="store_false", help="run without a window")
    out.add_argument("--jsonl", metavar="FILE", help="write the per-frame JSON lines to FILE instead of stdout")
    out.add_argument(
        "--quiet", action="store_true", help="do not print per-frame lines (popups and summary still print)"
    )

    det = p.add_argument_group("detection")
    det.add_argument("--min-face-conf", type=float, default=0.3)
    det.add_argument(
        "--adaptive-conf",
        dest="adaptive_conf",
        action="store_true",
        default=True,
        help="lower the face threshold after 10 frames without a face (default)",
    )
    det.add_argument("--no-adaptive-conf", dest="adaptive_conf", action="store_false")
    det.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    det.add_argument("--device", default="", help="YOLO device, e.g. cpu, 0, mps (default: auto)")
    det.add_argument("--min-eye-area", type=int, default=80, help="smallest pupil blob in pixels")
    det.add_argument(
        "--min-eye-area-frac", type=float, default=0.002, help="smallest pupil blob as a share of the eye box"
    )

    th = p.add_argument_group("thresholds")
    th.add_argument("--px-left", type=float, default=d.px_left)
    th.add_argument("--px-right", type=float, default=d.px_right)
    th.add_argument("--center-distract", type=float, default=d.center_distract)
    th.add_argument("--pupil-center-tol", type=float, default=d.pupil_center_tol)
    th.add_argument("--pupil-still-max-s", type=float, default=d.pupil_still_max_s)
    th.add_argument("--pupil-still-speed-max", type=float, default=d.pupil_still_speed_max)
    th.add_argument("--down-py-min", type=float, default=d.down_py_min)
    th.add_argument("--face-miss-ms", type=int, default=d.face_miss_ms)
    th.add_argument("--popup-after-s", type=float, default=d.popup_after_s)
    th.add_argument("--popup-cooldown-s", type=float, default=d.popup_cooldown_s)
    th.add_argument("--smoothing", type=float, default=d.smoothing)
    th.add_argument("--eye-reliable-frames", type=int, default=d.eye_reliable_frames)
    th.add_argument("--min-aspect-for-override", type=float, default=d.min_aspect_for_override)
    th.add_argument("--yaw-override-max", type=float, default=d.yaw_override_max)
    # Accepted for compatibility with older command lines; they never affected the result.
    th.add_argument("--center-attentive", type=float, help=argparse.SUPPRESS)
    th.add_argument("--center-tol", type=float, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        px_left=args.px_left,
        px_right=args.px_right,
        center_distract=args.center_distract,
        pupil_center_tol=args.pupil_center_tol,
        pupil_still_max_s=args.pupil_still_max_s,
        pupil_still_speed_max=args.pupil_still_speed_max,
        down_py_min=args.down_py_min,
        face_miss_ms=args.face_miss_ms,
        popup_after_s=args.popup_after_s,
        popup_cooldown_s=args.popup_cooldown_s,
        smoothing=args.smoothing,
        eye_reliable_frames=args.eye_reliable_frames,
        min_aspect_for_override=args.min_aspect_for_override,
        yaw_override_max=args.yaw_override_max,
    )


def open_capture(args: argparse.Namespace) -> tuple[cv2.VideoCapture, bool]:
    """Returns (capture, is_camera)."""
    is_camera = args.src.isdigit()
    source = int(args.src) if is_camera else args.src
    backend_name = BACKENDS[args.backend]
    if args.backend == "auto" and is_camera and sys.platform == "darwin":
        backend_name = "CAP_AVFOUNDATION"
    backend = getattr(cv2, backend_name, cv2.CAP_ANY) if backend_name else cv2.CAP_ANY

    cap = cv2.VideoCapture(source, backend)
    if not cap.isOpened() and backend != cv2.CAP_ANY:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(json.dumps({"type": "error", "message": f"Unable to open capture source {args.src!r}"}))
    if is_camera:
        if args.width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        if args.height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    elif args.start > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000.0)
    return cap, is_camera


def emit(line: dict, stream: IO[str]) -> None:
    stream.write(json.dumps(line) + "\n")
    stream.flush()


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        detector = FaceDetector(ensure_weights(args.model), imgsz=args.imgsz, device=args.device.strip())
    except Exception as exc:  # bad weights, no network on first run, missing ultralytics...
        emit(
            {"type": "error", "message": f"Failed to load the face model: {exc}. Pass --model to a valid .pt"},
            sys.stdout,
        )
        return 1

    cap, is_camera = open_capture(args)
    frames_out: IO[str] = open(args.jsonl, "w", encoding="utf-8") if args.jsonl else sys.stdout
    t0 = time.time()
    tracker = AttentionTracker(config_from_args(args), start_t=args.start if not is_camera else 0.0)
    no_face_streak = 0
    last_face: Optional[tuple] = None
    last_view = None
    paused = False
    frame_budget = 1.0 / args.fps_target if is_camera and args.fps_target > 0 else 0.0
    loop_start = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if is_camera:
                t = time.time() - t0
            else:
                # Use the video's own clock, so --start/--end and timings mean video time.
                t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if t > args.end:
                break
            if is_camera and t < args.start:
                if args.overlay:
                    cv2.imshow("attention", frame)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
                continue

            height, width = frame.shape[:2]
            min_conf = args.min_face_conf
            if args.adaptive_conf and no_face_streak >= 10:
                min_conf = max(0.15, args.min_face_conf * 0.6)
            try:
                face = detector.detect(frame, min_conf)
            except Exception as exc:  # keep going on a bad frame
                print(f"face detection failed on a frame: {exc}", file=sys.stderr)
                face = None

            left_box = right_box = None
            left = right = None
            if face:
                face = clip_rect(face, width, height)
                last_face, no_face_streak = face, 0
                left_box, right_box = eye_rects(face)
                left = pupil_position(to_gray(crop(frame, left_box)), args.min_eye_area, args.min_eye_area_frac)
                right = pupil_position(to_gray(crop(frame, right_box)), args.min_eye_area, args.min_eye_area_frac)
            else:
                no_face_streak += 1

            result = tracker.update(t, (width, height), face, left, right, left_box, right_box)
            if not args.quiet or args.jsonl:
                emit(result.to_json(), frames_out)
            if result.popup:
                emit(
                    {
                        "type": "popup",
                        "message": "Stop being distracted.",
                        "state": result.state,
                        "at_seconds": round(result.t, 2),
                    },
                    sys.stdout,
                )

            if args.overlay:
                now = time.time()
                fps = 1.0 / max(1e-6, now - loop_start)
                boxes = [b for b in (left_box, right_box) if b]
                last_view = draw(frame, result, tracker, boxes, last_face, fps)
                cv2.imshow("attention", last_view)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key in (ord("s"), ord("S")):
                    tracker.start_session()
                elif key in (ord("e"), ord("E")):
                    ended = tracker.end_session()
                    if ended:
                        emit(ended, sys.stdout)
                elif key in (ord("p"), ord("P"), 32):
                    paused = True
                while paused:
                    view = last_view.copy()
                    cv2.putText(
                        view,
                        "PAUSED - press P to resume",
                        (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("attention", view)
                    key = cv2.waitKey(30) & 0xFF
                    if key in (ord("p"), ord("P"), 32):
                        paused = False
                    elif key in (27, ord("q")):
                        raise KeyboardInterrupt

            if frame_budget:
                spare = frame_budget - (time.time() - loop_start)
                if spare > 0:
                    time.sleep(spare)
            loop_start = time.time()
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if args.overlay:
            cv2.destroyAllWindows()
        if frames_out is not sys.stdout:
            frames_out.close()

    emit(tracker.summary(), sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
