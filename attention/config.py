"""Tunable thresholds. Defaults match the command-line defaults in ``attn_tracker.py``."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    # Pupil x position (0 = left edge of the eye box, 1 = right edge) that still counts as "on screen".
    px_left: float = 0.35
    px_right: float = 0.65
    # How far the face centre may sit from the frame centre (fraction of width) before it counts as LEFT/RIGHT.
    center_distract: float = 0.15
    # Both pupils must be within this distance of the middle of their eye box to count as attentive.
    pupil_center_tol: float = 0.10
    # Pupils that stay still (below this speed, px/s) for longer than this many seconds read as a stare.
    pupil_still_max_s: float = 2.0
    pupil_still_speed_max: float = 8.0
    # Pupil y position at or below which (towards the bottom of the eye box) the viewer is looking down.
    down_py_min: float = 0.6

    # How long the face may be missing before the state drops straight to AWAY.
    face_miss_ms: int = 400
    # Raise a popup after this long distracted, at most once per cooldown.
    popup_after_s: float = 3.0
    popup_cooldown_s: float = 10.0

    # Exponential smoothing factor for positions (higher = smoother, slower).
    smoothing: float = 0.7

    # Gaze is only trusted after both eyes were found for this many frames in a row,
    # the face box is not too narrow (turned head) and the face is roughly centred.
    eye_reliable_frames: int = 3
    min_aspect_for_override: float = 0.55
    yaw_override_max: float = 0.2

    # A new state must be seen this many frames in a row before it replaces the current one.
    confirm_frames: int = 2
