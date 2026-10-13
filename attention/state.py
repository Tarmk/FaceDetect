"""Turn per-frame detections into an attention state, timers and popup events.

Nothing in here touches OpenCV or YOLO: it takes the face box and the pupil
positions found in a frame, plus the frame's timestamp, so it can be tested
with plain numbers.

States:
    ATTENTIVE  both eyes found reliably, pupils centred and looking at the screen
    LEFT/RIGHT face turned away from the centre of the frame
    AWAY       anything else: no face, looking off-screen or down, head tilted
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

from .config import Config
from .vision import Rect, ema

Point = Tuple[float, float]
STATES = ("ATTENTIVE", "LEFT", "RIGHT", "AWAY")


@dataclass
class EyeTrack:
    """Smoothed pupil position (0..1 inside the eye box) and its on-screen speed."""

    x: float = 0.5
    y: float = 0.5
    speed: float = 0.0
    trail: Deque[Tuple[int, int, float]] = field(default_factory=lambda: deque(maxlen=90))

    def update(self, pupil: Point, box: Rect, t: float, smoothing: float) -> Tuple[int, int]:
        self.x = ema(self.x, pupil[0], smoothing)
        self.y = ema(self.y, pupil[1], smoothing)
        bx, by, bw, bh = box
        px, py = bx + int(self.x * max(1, bw)), by + int(self.y * max(1, bh))
        if self.trail:
            ax, ay, at = self.trail[-1]
            self.speed = ema(self.speed, math.hypot(px - ax, py - ay) / max(1e-3, t - at), 0.7)
        self.trail.append((px, py, t))
        return px, py


@dataclass
class FrameResult:
    t: float
    face_present: bool
    bbox: Optional[Rect]
    gaze_to_screen: Optional[bool]
    gaze_down: Optional[bool]
    center_offset: float
    state: str
    attention_score: int
    left_pupil: Optional[Tuple[int, int]]
    right_pupil: Optional[Tuple[int, int]]
    eye_speed_l: float
    eye_speed_r: float
    pupil_still_time: float
    override_allowed: bool
    popup: bool

    def to_json(self) -> dict:
        return {
            "t": round(self.t, 2),
            "face_present": self.face_present,
            "bbox": list(self.bbox) if self.bbox else None,
            "gaze_to_screen": self.gaze_to_screen,
            "center_offset": round(self.center_offset, 4),
            "state": self.state,
            "attention_score": self.attention_score,
            "lpupil_px": list(self.left_pupil) if self.left_pupil else None,
            "rpupil_px": list(self.right_pupil) if self.right_pupil else None,
            "eye_speed_l_pxps": round(self.eye_speed_l, 2),
            "eye_speed_r_pxps": round(self.eye_speed_r, 2),
            "pupil_still_time_s": round(self.pupil_still_time, 2),
        }


class AttentionTracker:
    def __init__(self, config: Optional[Config] = None, start_t: float = 0.0) -> None:
        self.cfg = config or Config()
        self.start_t = start_t
        self.prev_t: Optional[float] = None
        self.last_face_t: Optional[float] = None
        self.prev_face_h: Optional[int] = None

        self.center = 0.0
        self.left = EyeTrack()
        self.right = EyeTrack()
        self.eye_streak = 0
        self.pupil_still_time = 0.0

        self.state = "AWAY"
        self._pending: Optional[str] = None
        self._pending_count = 0

        self.frames = 0
        self.attentive_time = 0.0
        self.distracted_time = 0.0
        self.breakdown: Dict[str, float] = {"LEFT": 0.0, "RIGHT": 0.0, "AWAY": 0.0}
        self.distraction = 0.0
        self.last_popup_t = -math.inf
        self.popups = 0

        self.session_active = False
        self.session_time = 0.0
        self.session_inattentive = 0.0

    # ------------------------------------------------------------------ frames

    def update(
        self,
        t: float,
        frame_size: Tuple[int, int],
        face: Optional[Rect],
        left_pupil: Optional[Point] = None,
        right_pupil: Optional[Point] = None,
        left_box: Optional[Rect] = None,
        right_box: Optional[Rect] = None,
    ) -> FrameResult:
        """Feed one frame. ``t`` is seconds (camera wall clock or video timestamp),
        ``frame_size`` is (width, height) and pupils are (x, y) in 0..1 inside their eye box."""
        cfg = self.cfg
        frame_width, frame_height = frame_size
        dt = 0.0 if self.prev_t is None else max(0.0, t - self.prev_t)
        self.prev_t = t

        gaze_to_screen: Optional[bool] = None
        gaze_down: Optional[bool] = None
        left_px = right_px = None
        candidate = "AWAY"
        face_lost = False
        override_allowed = False

        if face is not None:
            self.last_face_t = t
            x, y, w, h = face
            self.center = ema(self.center, (x + w / 2.0 - frame_width / 2.0) / float(frame_width), cfg.smoothing)

            if left_pupil and left_box:
                left_px = self.left.update(left_pupil, left_box, t, cfg.smoothing)
            if right_pupil and right_box:
                right_px = self.right.update(right_pupil, right_box, t, cfg.smoothing)

            pupils_centred = False
            if left_px and right_px:
                self.eye_streak += 1
                in_band = (cfg.px_left <= self.left.x <= cfg.px_right) and (cfg.px_left <= self.right.x <= cfg.px_right)
                gaze_to_screen = in_band
                pupils_centred = (
                    abs(self.left.x - 0.5) <= cfg.pupil_center_tol and abs(self.right.x - 0.5) <= cfg.pupil_center_tol
                )
                gaze_down = self.left.y >= cfg.down_py_min and self.right.y >= cfg.down_py_min
            else:
                # One eye (or none): never claim the viewer is looking at the screen.
                self.eye_streak = 0
                if left_px:
                    gaze_down = self.left.y >= cfg.down_py_min
                elif right_px:
                    gaze_down = self.right.y >= cfg.down_py_min

            # Head pitched away: the face box suddenly shrinks, or slides off the top of the frame.
            pitched_away = (self.prev_face_h is not None and h < 0.6 * self.prev_face_h) or y <= int(
                0.02 * frame_height
            )
            self.prev_face_h = h

            override_allowed = (
                self.eye_streak >= cfg.eye_reliable_frames
                and (w / float(h) if h else 0.0) >= cfg.min_aspect_for_override
                and abs(self.center) <= cfg.yaw_override_max
            )
            attentive = (
                override_allowed
                and gaze_to_screen is True
                and pupils_centred
                and self.pupil_still_time < cfg.pupil_still_max_s
            )
            if attentive:
                candidate = "ATTENTIVE"
            elif gaze_to_screen is False or pitched_away:
                candidate = "AWAY"
            elif self.center < -cfg.center_distract:
                candidate = "LEFT"
            elif self.center > cfg.center_distract:
                candidate = "RIGHT"
        else:
            if self.last_face_t is None:
                self.last_face_t = t
            face_lost = (t - self.last_face_t) * 1000.0 > cfg.face_miss_ms

        self._settle(candidate, force=face_lost)

        score = (
            (60 if face is not None else 0) + (25 if gaze_to_screen else 0) + (15 if self.state == "ATTENTIVE" else 0)
        )

        if self.left.speed <= cfg.pupil_still_speed_max and self.right.speed <= cfg.pupil_still_speed_max:
            self.pupil_still_time += dt
        else:
            self.pupil_still_time = 0.0

        popup = self._accumulate(t, dt)
        self.frames += 1
        return FrameResult(
            t=t - self.start_t,
            face_present=face is not None,
            bbox=face,
            gaze_to_screen=gaze_to_screen,
            gaze_down=gaze_down,
            center_offset=self.center,
            state=self.state,
            attention_score=score,
            left_pupil=left_px,
            right_pupil=right_px,
            eye_speed_l=self.left.speed,
            eye_speed_r=self.right.speed,
            pupil_still_time=self.pupil_still_time,
            override_allowed=override_allowed,
            popup=popup,
        )

    def _settle(self, candidate: str, force: bool) -> None:
        """Debounce: a new state has to show up ``confirm_frames`` times in a row (or be forced)."""
        if force:
            self.state, self._pending, self._pending_count = candidate, None, 0
            return
        if candidate == self.state:
            self._pending, self._pending_count = None, 0
            return
        if candidate == self._pending:
            self._pending_count += 1
        else:
            self._pending, self._pending_count = candidate, 1
        if self._pending_count >= self.cfg.confirm_frames:
            self.state, self._pending, self._pending_count = candidate, None, 0

    def _accumulate(self, t: float, dt: float) -> bool:
        if self.session_active:
            self.session_time += dt
            if self.state != "ATTENTIVE":
                self.session_inattentive += dt

        if self.state == "ATTENTIVE":
            self.attentive_time += dt
            self.distraction = 0.0
            return False

        self.distracted_time += dt
        self.breakdown[self.state] += dt
        self.distraction += dt
        if self.distraction >= self.cfg.popup_after_s and (t - self.last_popup_t) >= self.cfg.popup_cooldown_s:
            self.last_popup_t = t
            self.distraction = 0.0
            self.popups += 1
            return True
        return False

    # ---------------------------------------------------------------- sessions

    def start_session(self) -> None:
        self.session_active = True
        self.session_time = 0.0
        self.session_inattentive = 0.0

    def end_session(self) -> Optional[dict]:
        if not self.session_active:
            return None
        self.session_active = False
        pct = 100.0 * self.session_inattentive / self.session_time if self.session_time > 0 else 0.0
        return {
            "type": "session_end",
            "session_elapsed_s": round(self.session_time, 2),
            "non_attentive_percent": round(pct, 2),
        }

    # ----------------------------------------------------------------- summary

    @property
    def duration(self) -> float:
        return self.attentive_time + self.distracted_time

    @property
    def attention_percent(self) -> float:
        return 100.0 * self.attentive_time / self.duration if self.duration > 0 else 0.0

    def summary(self) -> dict:
        return {
            "type": "summary",
            "duration_s": round(self.duration, 2),
            "frames": self.frames,
            "attentive_time_s": round(self.attentive_time, 2),
            "distracted_time_s": round(self.distracted_time, 2),
            "attention_percent": round(self.attention_percent, 2),
            "distracted_breakdown": {k: round(v, 2) for k, v in self.breakdown.items()},
            "popups_triggered": self.popups,
        }
