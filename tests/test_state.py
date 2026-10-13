import pytest

from attention.config import Config
from attention.state import AttentionTracker
from attention.vision import eye_rects

FPS = 30
FRAME = (640, 480)
CENTRED_FACE = (270, 140, 100, 140)  # x, y, w, h: middle of the frame


def run(tracker, seconds, face=CENTRED_FACE, pupils=((0.5, 0.5), (0.5, 0.5)), t0=0.0, jitter=False):
    """Feed `seconds` of identical frames; return the last result."""
    result = None
    frames = int(seconds * FPS)
    for i in range(frames):
        t = t0 + i / FPS
        if face is None:
            result = tracker.update(t, FRAME, None)
            continue
        left_box, right_box = eye_rects(face)
        lp, rp = pupils
        if jitter:  # real eyes never sit perfectly still
            d = 0.04 if i % 2 else -0.04
            lp = (lp[0] + d, lp[1]) if lp else None
            rp = (rp[0] + d, rp[1]) if rp else None
        result = tracker.update(t, FRAME, face, lp, rp, left_box, right_box)
    return result


def test_no_face_means_away_with_zero_score():
    tracker = AttentionTracker()
    result = run(tracker, 1.0, face=None)
    assert result.state == "AWAY"
    assert result.attention_score == 0
    assert not result.face_present


def test_centred_face_with_centred_pupils_becomes_attentive():
    tracker = AttentionTracker()
    result = run(tracker, 1.0, jitter=True)
    assert result.state == "ATTENTIVE"
    assert result.attention_score == 100
    assert result.gaze_to_screen is True


def test_a_long_motionless_stare_is_not_attentive():
    tracker = AttentionTracker()
    assert run(tracker, 1.0).state == "ATTENTIVE"
    assert run(tracker, 2.5, t0=1.0).state == "AWAY"


def test_pupils_off_to_the_side_mean_looking_away():
    tracker = AttentionTracker()
    result = run(tracker, 1.0, pupils=((0.85, 0.5), (0.85, 0.5)), jitter=True)
    assert result.gaze_to_screen is False
    assert result.state == "AWAY"


def test_face_at_the_edge_of_the_frame_reads_as_left_or_right():
    tracker = AttentionTracker()
    one_eye_only = ((0.5, 0.5), None)  # gaze is not trusted, so the head position decides
    assert run(tracker, 1.0, face=(20, 140, 100, 140), pupils=one_eye_only).state == "LEFT"
    assert run(tracker, 1.5, face=(520, 140, 100, 140), pupils=one_eye_only, t0=1.0).state == "RIGHT"


def test_a_missing_face_needs_two_frames_or_the_grace_period():
    tracker = AttentionTracker(Config(face_miss_ms=400))
    assert run(tracker, 1.0, jitter=True).state == "ATTENTIVE"
    assert tracker.update(1.0, FRAME, None).state == "ATTENTIVE"  # one frame is not enough to switch
    assert tracker.update(1.5, FRAME, None).state == "AWAY"  # 500 ms without a face: forced straight to AWAY


def test_popup_after_three_seconds_distracted_then_cooldown():
    tracker = AttentionTracker(Config(popup_after_s=3.0, popup_cooldown_s=10.0))
    popups = []
    for i in range(int(20 * FPS)):
        if tracker.update(i / FPS, FRAME, None).popup:
            popups.append(i / FPS)
    assert len(popups) == 2
    assert popups[0] == pytest.approx(3.0, abs=0.1)
    assert popups[1] - popups[0] == pytest.approx(10.0, abs=0.1)


def test_summary_adds_up():
    tracker = AttentionTracker()
    run(tracker, 2.0, jitter=True)
    run(tracker, 2.0, face=None, t0=2.0)
    summary = tracker.summary()
    assert summary["type"] == "summary"
    assert summary["duration_s"] == pytest.approx(summary["attentive_time_s"] + summary["distracted_time_s"], abs=0.02)
    assert 30 < summary["attention_percent"] < 60
    assert summary["frames"] == 4 * FPS


def test_sessions_report_the_inattentive_share():
    tracker = AttentionTracker()
    assert tracker.end_session() is None
    tracker.start_session()
    run(tracker, 2.0, jitter=True)
    run(tracker, 2.0, face=None, t0=2.0)
    ended = tracker.end_session()
    assert ended["type"] == "session_end"
    assert 40 < ended["non_attentive_percent"] < 70


def test_frame_json_has_the_documented_keys():
    tracker = AttentionTracker()
    data = run(tracker, 0.2, jitter=True).to_json()
    assert set(data) == {
        "t",
        "face_present",
        "bbox",
        "gaze_to_screen",
        "center_offset",
        "state",
        "attention_score",
        "lpupil_px",
        "rpupil_px",
        "eye_speed_l_pxps",
        "eye_speed_r_pxps",
        "pupil_still_time_s",
    }
