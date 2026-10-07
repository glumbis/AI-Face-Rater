import time
from collections import deque

import cv2
import numpy as np

# Which way the head is facing, and what to do about it. Everything about head direction lives in this file, and the
# rest of the program only needs head_angles(), so a better way of measuring the angles can be swapped in there.

# ---------- Limits ----------
# How many degrees the head may be turned to the side, or tilted up or down. MediaPipe's head angles are steady (the
# angle of a head that doesn't move wobbles by only about 0.15 degrees, and the two model faces read 0 to 5 degrees),
# so a picture can be refused for a head that is only a little off. The rating turns the head back in 3D before
# comparing it (see TILT_SEARCH in landmarkdetect.py), so a slightly turned or tilted head costs (almost) nothing.
# Turning is kept to 12 degrees, because then the far side of the face starts to hide behind the nose and its
# landmarks are guesses. Tilt is allowed much more, because people looking at the screen under the camera tilt down,
# and a tilted face hides little. Over 25 degrees the forehead or chin is so foreshortened that it gets less reliable
MAX_TURN = 12
MAX_TILT = 25
# The live preview gives a gentle tip earlier. It doesn't stop anything. The tip comes when the (smoothed) angle
# is over HINT_TURN / HINT_TILT, and goes away again when it's HINT_MARGIN degrees back under, so it doesn't flicker
# (the margin is wider than the smoothed angles wobble, but small enough to be easy to hit when you straighten up)
HINT_TURN = 8
HINT_TILT = 18
HINT_MARGIN = 2

# The tip in the live preview, and the message when a picture is refused, for each way of facing
HINT_MESSAGES = {
    "side": "Turn a little towards the camera.",
    "down": "Lift your chin a little.",
    "up": "Lower your chin a little.",
}
REFUSED_MESSAGES = {
    "side": "Head turned too far. Look at the camera.",
    "down": "Head tilted too far down. Lift your chin.",
    "up": "Head tilted too far up. Lower your chin.",
}

# ---------- Measuring the angles ----------
# MediaPipe's Face Landmarker also says how the head is placed in 3D: a 4x4 matrix that moves a standard head into
# the camera's view, where the head is moved, turned and resized. Only the turning part is used here. Which way the
# front of the face points follows from it, and from that the turn and the tilt.
# Raw, a head looking straight into the camera reads about 9 to 14 degrees down, because the standard head's front
# doesn't point the way a head looking at the camera does. This is taken off the tilt, so a face looking straight
# into the camera reads as about 0 degrees. 14 makes the two model faces (perBoy.jpg and perGirl.jpg) read
# about the same as they did with the old measurement from landmarks (a few degrees up, and level)
TILT_OFFSET = 14.0


def head_angles(matrix):
    # Returns (turn, tilt) in degrees from the head matrix of a MediaPipe face (4x4, or just the 3x3 turning part),
    # or None if it can't be worked out.
    # turn is to the side (positive when the face points to the right in the picture), tilt is positive when looking
    # down and negative when looking up. This is the one place that works out the angles
    try:
        rotation = np.asarray(matrix, dtype=np.float64)[:3, :3]
    except (TypeError, ValueError, IndexError):
        return None
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        return None
    # The matrix also resizes the head, so make each column length 1 to be left with only the turning
    lengths = np.linalg.norm(rotation, axis=0)
    if np.any(lengths < 1e-9):
        return None
    # Which way the front of the face points. MediaPipe's camera has x to the right, y up and z towards the viewer
    forward = (rotation / lengths) @ np.array([0.0, 0.0, 1.0])
    turn = np.degrees(np.arctan2(forward[0], forward[2]))
    tilt = np.degrees(np.arctan2(-forward[1], forward[2])) - TILT_OFFSET
    return float(turn), float(tilt)


def problem_for(turn, tilt, maxTurn=MAX_TURN, maxTilt=MAX_TILT):
    # "side", "down" or "up" if the angles are over the limits, otherwise None
    if abs(turn) > maxTurn:
        return "side"
    if tilt > maxTilt:
        return "down"
    if tilt < -maxTilt:
        return "up"
    return None


def facing_problem(angles):
    # Returns None if the face is facing the camera well enough to be rated, otherwise "side", "down" or "up".
    # angles is (turn, tilt) from head_angles, or None when they couldn't be worked out (then the picture is rated)
    if angles is None:
        return None
    return problem_for(*angles)


# ---------- Live preview: steady angles ----------
# How many pictures (and at most how many seconds) the live preview averages over. One landmark that wobbles
# can throw one picture's angle off by 10 degrees, but not the middle value of several
SMOOTH_FRAMES = 9
SMOOTH_SECONDS = 0.6


class HeadTracker:
    # Keeps the last few angles of the live preview and says which tip to show, if any.
    # Uses the median, so a single bad picture doesn't count, and a margin so the tip doesn't flicker on and off

    def __init__(self, frames=SMOOTH_FRAMES, seconds=SMOOTH_SECONDS):
        self.seconds = seconds
        self.history = deque(maxlen=frames)  # (time, turn, tilt)
        self.problem = None

    def reset(self):
        self.history.clear()
        self.problem = None

    def update(self, turn, tilt, now=None):
        # Adds the angles of the newest picture. Returns the problem to give a tip about ("side", "down", "up"),
        # or None when the head is facing the camera
        now = time.monotonic() if now is None else now
        self.history.append((now, turn, tilt))
        while self.history and now - self.history[0][0] > self.seconds:
            self.history.popleft()
        smoothTurn = float(np.median([h[1] for h in self.history]))
        smoothTilt = float(np.median([h[2] for h in self.history]))
        self.problem = self.next_problem(smoothTurn, smoothTilt)
        return self.problem

    def next_problem(self, turn, tilt):
        # The tip comes at the hint limit and only goes at the limit minus the margin. While one is showing it keeps
        # showing as long as that same one is still over the lower limit
        if self.problem is None:
            return problem_for(turn, tilt, HINT_TURN, HINT_TILT)
        stay = problem_for(turn, tilt, HINT_TURN - HINT_MARGIN, HINT_TILT - HINT_MARGIN)
        if stay == self.problem:
            return stay
        return problem_for(turn, tilt, HINT_TURN, HINT_TILT)


# ---------- Taking a photo: the best of the last moments ----------
# How long back a picture can be taken from, in seconds, and how many degrees apart two pictures must be for the
# straighter one to win over the sharper one
RECENT_SECONDS = 1.2
SAME_ANGLE = 2.0
# Sharpness is measured on a copy this wide, which is quick
SHARPNESS_WIDTH = 160


def sharpness(frame):
    # How sharp a picture is: the more it varies when run through an edge filter, the sharper (a blurry picture has
    # soft edges). Used to pick between pictures that face the camera equally well
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    h, w = gray.shape[:2]
    if w > SHARPNESS_WIDTH:
        gray = cv2.resize(gray, (SHARPNESS_WIDTH, max(1, round(h * SHARPNESS_WIDTH / w))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class RecentFrames:
    # Remembers which picture from the last second or so faced the camera best, so pressing the photo button
    # doesn't depend on one picture where the head happened to be a bit off.
    # Only pictures that can still win are kept: one that is older and no better than a newer one never will be,
    # so only a handful are ever in memory, no matter how fast the camera is

    def __init__(self, seconds=RECENT_SECONDS):
        self.seconds = seconds
        self.entries = deque()  # (time, rank, frame), best (and oldest) first, the ranks getting worse

    def clear(self):
        self.entries.clear()

    @staticmethod
    def rank(turn, tilt, sharp):
        # Lower is better: first how far the head is from facing the camera (in steps of SAME_ANGLE degrees, so
        # pictures that face about equally well are told apart by sharpness)
        return (round(float(np.hypot(turn, tilt)) / SAME_ANGLE), -sharp)

    def add(self, frame, turn, tilt, now=None):
        # Remembers a picture where a face was found. frame is kept as it is (not copied)
        now = time.monotonic() if now is None else now
        rank = self.rank(turn, tilt, sharpness(frame))
        # Older pictures that this one beats will never be the best
        while self.entries and self.entries[-1][1] >= rank:
            self.entries.pop()
        self.entries.append((now, rank, frame))
        self.forget_old(now)

    def forget_old(self, now):
        while self.entries and now - self.entries[0][0] > self.seconds:
            self.entries.popleft()

    def best(self, now=None):
        # The most frontal recent picture (the sharpest of the equally frontal ones), or None if there is none
        now = time.monotonic() if now is None else now
        self.forget_old(now)
        return self.entries[0][2] if self.entries else None
