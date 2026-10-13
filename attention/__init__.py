"""Attention tracking: face box + pupil positions -> attention state and timers."""

from .config import Config
from .state import AttentionTracker, FrameResult

__all__ = ["AttentionTracker", "Config", "FrameResult"]
