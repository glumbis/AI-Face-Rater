import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import headpose as hp  # noqa: E402


# ---------- head_angles on a made-up head ----------

def landmarks_of_head(turn, tilt, width=640):
    # Photographs HEAD_MODEL turned/tilted by the given degrees with a pinhole camera, and returns 68-long lists
    # (only the landmarks the model uses are filled in)
    t, p = np.radians(turn), np.radians(tilt)
    turnMatrix = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
    tiltMatrix = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    # The model has y up and z towards the camera, the camera has y down and z away
    points = hp.HEAD_MODEL * [1, -1, -1]
    centre = np.array([0.0, 0.0, 300.0])
    points = (tiltMatrix @ turnMatrix @ (points - centre).T).T + centre + [0, 0, 3000]
    xList, yList = [0.0] * 68, [0.0] * 68
    for n, (x, y, z) in zip(hp.HEAD_LANDMARKS, points):
        xList[n] = 320 + width * x / z
        yList[n] = 240 + width * y / z
    return xList, yList


def test_straight_head_is_about_zero():
    turn, tilt = hp.head_angles(*landmarks_of_head(0, 0), (480, 640))
    assert abs(turn) < 1 and abs(tilt) < 1


@pytest.mark.parametrize("turn", [-35, 35])
def test_a_head_turned_far_is_refused(turn):
    xList, yList = landmarks_of_head(turn, 0)
    assert hp.facing_problem(xList, yList, (480, 640)) == "side"


def test_a_head_tilted_far_down_or_up_is_refused():
    assert hp.facing_problem(*landmarks_of_head(0, 35), (480, 640)) == "down"
    assert hp.facing_problem(*landmarks_of_head(0, -35), (480, 640)) == "up"


def test_a_slightly_turned_head_is_still_rated():
    assert hp.facing_problem(*landmarks_of_head(10, 8), (480, 640)) is None


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
