"""Debug overlay drawn on top of the camera image."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import cv2
import numpy as np

from .state import AttentionTracker, EyeTrack, FrameResult
from .vision import Rect

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREY = (128, 128, 128)
WHITE = (255, 255, 255)
STATE_COLOURS = {"ATTENTIVE": (0, 200, 0), "LEFT": (0, 215, 255), "RIGHT": (0, 215, 255), "AWAY": (0, 0, 255)}
CONTROLS = "[S]tart  [E]nd session  [P]ause  [Q]/[Esc] quit"


def _text(img, text: str, org: Tuple[int, int], scale: float = 0.5, colour=WHITE, thickness: int = 1) -> None:
    # Dark outline first so the text stays readable on bright backgrounds.
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, colour, thickness, cv2.LINE_AA)


def _trail(img, eye: EyeTrack, colour) -> None:
    points = list(eye.trail)
    for i in range(1, len(points)):
        fade = i / len(points)
        cv2.line(
            img,
            points[i - 1][:2],
            points[i][:2],
            tuple(int(c * fade) for c in colour),
            2 if fade > 0.7 else 1,
            cv2.LINE_AA,
        )


def _pupil(img, point: Tuple[int, int], eye: EyeTrack) -> None:
    cv2.circle(img, point, 4, (0, 255, 255), -1)
    dx = -18 if eye.x < 0.5 else 18
    dy = -14 if eye.y < 0.5 else 14
    cv2.arrowedLine(img, point, (point[0] + dx, point[1]), (0, 255, 200), 2, tipLength=0.35)
    cv2.arrowedLine(img, point, (point[0], point[1] + dy), (255, 0, 255), 2, tipLength=0.35)


def draw(
    frame: np.ndarray,
    result: FrameResult,
    tracker: AttentionTracker,
    eye_boxes: Iterable[Rect] = (),
    last_face: Optional[Rect] = None,
    fps: float = 0.0,
) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]
    colour = STATE_COLOURS[result.state]

    if result.bbox:
        x, y, bw, bh = result.bbox
        cv2.rectangle(img, (x, y), (x + bw, y + bh), colour, 2)
        _text(img, result.state, (x, max(12, y - 8)), 0.6, colour, 2)
    elif last_face:
        x, y, bw, bh = last_face
        cv2.rectangle(img, (x, y), (x + bw, y + bh), GREY, 2)
        _text(img, "NO FACE (last seen)", (x, max(12, y - 8)), 0.6, GREY, 2)

    for ex, ey, ew, eh in eye_boxes:
        if ew > 0 and eh > 0:
            cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (80, 160, 255), 1)
    _trail(img, tracker.left, (0, 255, 200))
    _trail(img, tracker.right, (255, 0, 255))
    if result.left_pupil:
        _pupil(img, result.left_pupil, tracker.left)
    if result.right_pupil:
        _pupil(img, result.right_pupil, tracker.right)

    lines = [
        f"attention {tracker.attention_percent:5.1f}%   state {result.state}   score {result.attention_score}",
        f"t {result.t:6.2f}s   frames {tracker.frames}   fps {fps:4.1f}",
        f"on screen {result.gaze_to_screen}   looking down {result.gaze_down}   centre {result.center_offset:+.3f}",
        f"pupils L ({tracker.left.x:.2f}, {tracker.left.y:.2f})  R ({tracker.right.x:.2f}, {tracker.right.y:.2f})"
        f"   gaze trusted {result.override_allowed}",
        f"eye speed L {result.eye_speed_l:.0f}  R {result.eye_speed_r:.0f} px/s   still {result.pupil_still_time:.1f}s",
        f"away L {tracker.breakdown['LEFT']:.1f}s  R {tracker.breakdown['RIGHT']:.1f}s"
        f"  AWAY {tracker.breakdown['AWAY']:.1f}s   popups {tracker.popups}",
    ]
    for i, line in enumerate(lines):
        _text(img, line, (10, 22 + i * 20), 0.5, (255, 255, 0) if i else WHITE)

    if tracker.session_active:
        pct = 100.0 * tracker.session_inattentive / tracker.session_time if tracker.session_time else 0.0
        _text(
            img, f"session {tracker.session_time:.1f}s   not attentive {pct:.1f}%", (10, h - 60), 0.5, (160, 250, 160)
        )

    bar = int(200 * result.attention_score / 100.0)
    cv2.rectangle(img, (10, h - 20), (210, h - 10), (60, 60, 60), -1)
    cv2.rectangle(img, (10, h - 20), (10 + bar, h - 10), colour, -1)
    _text(img, CONTROLS, (10, h - 32), 0.45, (200, 255, 200))
    return img
