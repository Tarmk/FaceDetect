import cv2
import numpy as np
import pytest

from attention.vision import clip_rect, crop, ema, eye_rects, pupil_position, to_gray


def eye_image(cx, cy, w=60, h=30, radius=6):
    img = np.full((h, w), 220, np.uint8)  # light sclera/skin
    cv2.circle(img, (cx, cy), radius, 20, -1)  # dark pupil
    return img


@pytest.mark.parametrize("cx, cy", [(30, 15), (12, 10), (48, 20)])
def test_pupil_position_finds_the_dark_blob(cx, cy):
    x, y = pupil_position(eye_image(cx, cy), min_area_px=20)
    assert x == pytest.approx(cx / 60, abs=0.03)
    assert y == pytest.approx(cy / 30, abs=0.05)


def test_pupil_position_ignores_tiny_blobs_and_empty_input():
    assert pupil_position(eye_image(30, 15, radius=1), min_area_px=20) is None
    assert pupil_position(None) is None
    assert pupil_position(np.zeros((0, 0), np.uint8)) is None


def test_eye_rects_split_the_top_third_of_the_face():
    left, right = eye_rects((100, 50, 90, 120))
    assert left == (100, 50, 45, 40)
    assert right == (145, 50, 45, 40)


def test_clip_rect_keeps_boxes_inside_the_frame():
    assert clip_rect((-10, -5, 50, 50), 640, 480) == (0, 0, 50, 50)
    assert clip_rect((620, 470, 50, 50), 640, 480) == (620, 470, 20, 10)


def test_crop_and_gray():
    frame = np.zeros((100, 100, 3), np.uint8)
    assert crop(frame, (10, 10, 20, 30)).shape == (30, 20, 3)
    assert crop(frame, (200, 200, 10, 10)) is None
    assert to_gray(crop(frame, (0, 0, 5, 5))).shape == (5, 5)


def test_ema_weights_the_previous_value():
    assert ema(1.0, 0.0, 0.7) == pytest.approx(0.7)


def test_overlay_draws_on_a_copy():
    from attention.overlay import draw
    from attention.state import AttentionTracker

    frame = np.zeros((480, 640, 3), np.uint8)
    tracker = AttentionTracker()
    face = (270, 140, 100, 140)
    left_box, right_box = eye_rects(face)
    result = tracker.update(0.0, (640, 480), face, (0.5, 0.5), (0.5, 0.5), left_box, right_box)
    tracker.start_session()
    view = draw(frame, result, tracker, [left_box, right_box], face, fps=30.0)
    assert view.shape == frame.shape
    assert view.any() and not frame.any()
