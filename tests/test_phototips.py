import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import phototips as pt  # noqa: E402

# A made-up "face": a 200x240 box in the middle of a 640x480 picture, with some texture so it can be sharp or blurry
BOX = (220, 120, 420, 360)


def picture(level=130, left=None, right=None):
    img = np.full((480, 640, 3), 60, np.uint8)
    rng = np.random.RandomState(0)
    left, right = left or level, right or level
    face = rng.randint(-40, 40, (240, 200, 1)) + np.concatenate(
        [np.full((240, 100, 1), left), np.full((240, 100, 1), right)], axis=1)
    img[120:360, 220:420] = np.clip(face, 0, 255).astype(np.uint8)
    return img


def landmarks(box=BOX):
    left, top, right, bottom = box
    return [left, right, left, right], [top, top, bottom, bottom]


def measure(img, angles=(0.0, 0.0), box=BOX):
    return pt.measure(img, *landmarks(box), angles)


def test_good_photo_has_no_tips_and_gets_the_positive_line():
    m = measure(picture())
    assert pt.pick_tips(m) == []
    assert pt.tip_lines(m) == [pt.GREAT]


def test_dark_and_bright():
    assert pt.pick_tips(measure(picture(level=40))) == ["dark"]
    assert pt.pick_tips(measure(picture(level=235))) == ["bright"]


def test_one_side_brighter():
    m = measure(picture(left=190, right=90))
    assert m["imbalance"] > pt.UNEVEN_LIGHT_ABOVE
    assert pt.pick_tips(m) == ["uneven"]
    assert pt.light_imbalance(pt.to_gray(picture()), BOX) < 0.05


def test_blurry():
    sharp = measure(picture())
    blurry = measure(cv2.GaussianBlur(picture(), (0, 0), 6))
    assert blurry["sharpness"] < sharp["sharpness"] / 4
    assert pt.pick_tips(blurry) == ["blurry"]


def test_head_turn_and_tilt():
    assert pt.pick_tips(measure(picture(), angles=(-20.0, 0.0))) == ["turn"]
    assert pt.pick_tips(measure(picture(), angles=(3.0, -20.0))) == ["tilt"]
    # Unknown angles give no head tip
    assert pt.pick_tips(pt.measure(picture(), *landmarks(), None)) == []


def test_face_size():
    # (the made-up texture gets smoothed when the box is resized, so it may also count as blurry)
    assert "close" in pt.pick_tips(measure(picture(), box=(60, 20, 600, 470)), maxTips=9)
    assert "far" in pt.pick_tips(measure(picture(), box=(300, 200, 340, 250)), maxTips=9)
    assert pt.face_size(BOX, (480, 640)) < pt.TOO_CLOSE_ABOVE


def test_at_most_two_tips_most_important_first():
    m = measure(cv2.GaussianBlur(picture(level=40), (0, 0), 6), angles=(30.0, 0.0))
    assert pt.pick_tips(m) == ["turn", "blurry"]
    assert pt.pick_tips(m, maxTips=1) == ["turn"]
    assert len(pt.tip_lines(m)) == 2


def test_gray_and_see_through_pictures_and_edge_boxes_work():
    img = picture()
    assert pt.measure(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), *landmarks(), None)["brightness"] > 0
    assert pt.measure(cv2.cvtColor(img, cv2.COLOR_BGR2BGRA), *landmarks(), None)["brightness"] > 0
    # A box partly outside the picture is cut to fit
    assert pt.measure(img, [-50, 700], [-50, 500], None)["size"] > 1
    big = np.zeros((2000, 3000, 3), np.uint8)
    assert max(pt.shrink(big).shape[:2]) == pt.MEASURE_MAX_SIZE
