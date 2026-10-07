import os
import sys

import cv2
import numpy as np
import pytest

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

import headpose as hp  # noqa: E402


# ---------- head_angles from the head matrix ----------

def matrix_of_head(turn, tilt, size=1.0, move=(3.0, -2.0, -30.0)):
    # A 4x4 head matrix like MediaPipe gives (its camera has x to the right, y up and z towards the viewer; the
    # standard head looks along +z): the head turned `turn` degrees to the right and `tilt` degrees down,
    # resized and moved
    t, p = np.radians(turn), np.radians(tilt)
    turning = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
    tilting = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    matrix = np.eye(4)
    matrix[:3, :3] = size * turning @ tilting
    matrix[:3, 3] = move
    return matrix


def raw(turn, tilt):
    # What the matrix of a head that is turned and tilted this much *as the standard head* gives: the tilt that is
    # taken off (TILT_OFFSET) is added, so the answer is the angles that were asked for
    return matrix_of_head(turn, tilt + hp.TILT_OFFSET)


def test_straight_head_is_about_zero():
    turn, tilt = hp.head_angles(raw(0, 0))
    assert abs(turn) < 0.01 and abs(tilt) < 0.01


def test_the_standard_head_reads_the_offset_lower():
    # The offset is the one thing that is taken off, so a head turned like the standard head reads -TILT_OFFSET
    turn, tilt = hp.head_angles(np.eye(4))
    assert turn == pytest.approx(0) and tilt == pytest.approx(-hp.TILT_OFFSET)


@pytest.mark.parametrize("turn", [-40, -20, -5, 5, 20, 40])
def test_turning_to_the_right_is_positive_and_to_the_left_negative(turn):
    # (turned like the standard head otherwise, so only the offset is taken off the tilt)
    got_turn, got_tilt = hp.head_angles(matrix_of_head(turn, 0))
    assert got_turn == pytest.approx(turn, abs=0.01)
    assert got_tilt == pytest.approx(-hp.TILT_OFFSET, abs=0.01)


@pytest.mark.parametrize("tilt", [-40, -20, -5, 5, 20, 40])
def test_looking_down_is_positive_and_looking_up_negative(tilt):
    got_turn, got_tilt = hp.head_angles(raw(0, tilt))
    assert got_tilt == pytest.approx(tilt, abs=0.01)
    assert got_turn == pytest.approx(0, abs=0.01)


def test_turn_and_tilt_together():
    turn, tilt = hp.head_angles(raw(10, 8))
    assert turn == pytest.approx(10, abs=1.5) and tilt == pytest.approx(8, abs=1.5)


def test_size_and_place_of_the_head_do_not_matter():
    assert hp.head_angles(matrix_of_head(15, 20, size=1.0)) == pytest.approx(hp.head_angles(matrix_of_head(15, 20, size=3.7, move=(-40, 9, -120))))
    # The 3x3 turning part alone works as well
    assert hp.head_angles(matrix_of_head(15, 20)[:3, :3]) == pytest.approx(hp.head_angles(matrix_of_head(15, 20)))


@pytest.mark.parametrize("matrix", [None, "head", [], np.zeros((4, 4)), np.full((4, 4), np.nan), np.eye(2), np.zeros(9)],
                         ids=["none", "text", "empty", "zeros", "nan", "2x2", "flat"])
def test_a_matrix_that_is_no_use_gives_no_angles(matrix):
    assert hp.head_angles(matrix) is None


@pytest.mark.parametrize("turn", [-35, 35])
def test_a_head_turned_far_is_refused(turn):
    assert hp.facing_problem(hp.head_angles(raw(turn, 0))) == "side"


def test_a_head_tilted_far_down_or_up_is_refused():
    assert hp.facing_problem(hp.head_angles(raw(0, 35))) == "down"
    assert hp.facing_problem(hp.head_angles(raw(0, -35))) == "up"


def test_a_slightly_turned_head_is_still_rated():
    assert hp.facing_problem(hp.head_angles(raw(10, 8))) is None


def test_no_angles_means_nothing_is_refused():
    assert hp.facing_problem(None) is None


# ---------- head_angles on the two model faces ----------

def model_pictures():
    landmarkdetect = pytest.importorskip("landmarkdetect")
    if not os.path.isfile(landmarkdetect.LANDMARKER_PATH):
        pytest.skip("face_landmarker.task is missing")
    return landmarkdetect, [landmarkdetect.read_image(os.path.join(REPO_DIR, name)) for name in ("perBoy.jpg", "perGirl.jpg")]


def test_the_model_faces_read_as_facing_the_camera():
    # The old measurement (20 landmarks and solvePnP) gave about (0, -4.7) and (-1, -0.3) for these two pictures
    landmarkdetect, pictures = model_pictures()
    boy, girl = (landmarkdetect.detect_face(p).angles for p in pictures)
    assert abs(boy[0]) < 2 and -7 < boy[1] < -3
    assert abs(girl[0]) < 2 and -2 < girl[1] < 2


def test_a_mirrored_picture_turns_the_other_way_and_tilts_the_same():
    landmarkdetect, pictures = model_pictures()
    for picture in pictures:
        turn, tilt = landmarkdetect.detect_face(picture).angles
        flippedTurn, flippedTilt = landmarkdetect.detect_face(cv2.flip(picture, 1)).angles
        assert flippedTurn == pytest.approx(-turn, abs=1.5)
        assert flippedTilt == pytest.approx(tilt, abs=1.5)


@pytest.mark.parametrize("degrees", [-15, 15])
def test_a_picture_turned_in_its_own_plane_barely_changes_the_angles(degrees):
    # Rolling the head sideways is not turning or tilting it
    landmarkdetect, pictures = model_pictures()
    for picture in pictures:
        h, w = picture.shape[:2]
        rolled = cv2.warpAffine(picture, cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1), (w, h),
                                borderMode=cv2.BORDER_REPLICATE)
        before, after = landmarkdetect.detect_face(picture).angles, landmarkdetect.detect_face(rolled).angles
        assert abs(after[0] - before[0]) < 6 and abs(after[1] - before[1]) < 4
        assert landmarkdetect.facing_problem(after) is None


# ---------- HeadTracker ----------

def feed(tracker, angles, start=0.0, step=0.07):
    return [tracker.update(turn, tilt, now=start + i * step) for i, (turn, tilt) in enumerate(angles)]


def test_one_wild_picture_gives_no_tip():
    tracker = hp.HeadTracker()
    results = feed(tracker, [(1, 0)] * 5 + [(40, 0)] + [(1, 0)] * 5)
    assert results == [None] * 11


def test_a_turned_head_gets_a_tip():
    results = feed(hp.HeadTracker(), [(20, 0)] * 12)
    assert results[-1] == "side"


def test_tip_stays_between_the_limits_and_goes_when_clearly_back():
    tracker = hp.HeadTracker()
    feed(tracker, [(20, 0)] * 12)
    assert tracker.problem == "side"
    # 13 degrees: under the limit that makes it come (15), but not clearly back, so it stays
    assert feed(tracker, [(13, 0)] * 12, start=2.0)[-1] == "side"
    assert feed(tracker, [(5, 0)] * 12, start=4.0)[-1] is None
    # and 13 degrees on its own doesn't make it come
    assert feed(tracker, [(13, 0)] * 12, start=6.0)[-1] is None


def test_chin_tips_say_which_way():
    assert feed(hp.HeadTracker(), [(0, 20)] * 12)[-1] == "down"
    assert feed(hp.HeadTracker(), [(0, -20)] * 12)[-1] == "up"


def test_old_angles_are_forgotten():
    tracker = hp.HeadTracker()
    feed(tracker, [(25, 0)] * 12)
    assert tracker.update(0, 0, now=10.0) is None


# ---------- RecentFrames ----------

def picture(sharp, seed=0):
    # A picture with more or less sharp noise in it
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (48, 64), dtype=np.uint8)
    if not sharp:
        base = np.full((48, 64), 128, np.uint8) + base // 64
    return np.dstack([base] * 3)


def test_the_most_frontal_recent_picture_wins():
    recent = hp.RecentFrames()
    frames = {name: picture(True, i) for i, name in enumerate("abcd")}
    recent.add(frames["a"], 8, 3, now=0.0)
    recent.add(frames["b"], 2, 1, now=0.1)  # best
    recent.add(frames["c"], 6, 0, now=0.2)
    recent.add(frames["d"], 9, 5, now=0.3)
    assert recent.best(now=0.4) is frames["b"]


def test_the_sharper_of_equally_frontal_pictures_wins():
    recent = hp.RecentFrames()
    blurry, sharp = picture(False), picture(True)
    recent.add(sharp, 3, 3, now=0.0)
    recent.add(blurry, 3, 3.5, now=0.1)  # a bit different angle, but the same within SAME_ANGLE
    assert recent.best(now=0.2) is sharp


def test_a_picture_that_is_too_old_is_not_used():
    recent = hp.RecentFrames(seconds=1.0)
    old, new = picture(True, 1), picture(True, 2)
    recent.add(old, 0, 0, now=0.0)
    recent.add(new, 10, 10, now=1.5)
    assert recent.best(now=1.6) is new
    assert recent.best(now=3.0) is None


def test_only_pictures_that_can_still_win_are_kept():
    recent = hp.RecentFrames(seconds=1.2)
    rng = np.random.default_rng(3)
    for i in range(200):  # 200 pictures at 60 fps
        recent.add(picture(True, i), *rng.normal(0, 8, 2), now=i / 60)
    assert len(recent.entries) < 30
    recent.clear()
    assert recent.best(now=4.0) is None
