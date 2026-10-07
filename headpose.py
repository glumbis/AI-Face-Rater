import time
from collections import deque

import cv2
import numpy as np

# Which way the head is facing, and what to do about it. Everything about head direction lives in this file, and the
# rest of the program only needs head_angles(), so a better way of measuring the angles can be swapped in there.

# ---------- Limits ----------
# How many degrees the head may be turned to the side, or tilted up or down. The angles from a single picture
# jump around by several degrees even when the head doesn't move (the landmarks wobble a little), and a head
# turned 10 degrees only costs about 2 points, so a picture is only refused when the head is clearly turned away
MAX_TURN = 25
MAX_TILT = 25
# The live preview gives a gentle tip a bit earlier. It doesn't stop anything. The tip comes when the (smoothed) angle
# is over HINT_TURN / HINT_TILT, and goes away again when it's HINT_MARGIN degrees back under, so it doesn't flicker
HINT_TURN = 15
HINT_TILT = 15
HINT_MARGIN = 4

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
# Where landmarks are on an average head in 3D (nose tip at 0, y up, z towards the camera, about 5 units per mm).
# The first six are the classic head model, the rest are spread out over the face, the jaw and the brows: more points
# and a wider spread make the angles about 3 times steadier. Sideways and up-down positions follow the average
# of the two model faces (perBoy.jpg and perGirl.jpg) so a face looking straight into the camera reads as about 0 degrees.
HEAD_MODEL_POINTS = {
    30: (0.0, 0.0, 0.0),           # nose tip
    27: (0.0, 197.0, -115.0),      # top of the nose, between the eyes
    33: (0.0, -54.0, -80.0),       # under the nose
    31: (-60.0, -34.0, -90.0),     # nostril wings
    35: (60.0, -34.0, -90.0),
    36: (-225.0, 200.0, -135.0),   # outer eye corners
    45: (225.0, 200.0, -135.0),
    39: (-96.0, 187.0, -105.0),    # inner eye corners
    42: (96.0, 187.0, -105.0),
    48: (-132.0, -135.0, -125.0),  # mouth corners
    54: (132.0, -135.0, -125.0),
    51: (0.0, -116.0, -100.0),     # middle of the upper and lower lip
    57: (0.0, -204.0, -105.0),
    8: (0.0, -355.0, -65.0),       # chin
    2: (-315.0, 32.0, -330.0),     # the jaw, by the ears
    14: (315.0, 32.0, -330.0),
    4: (-270.0, -150.0, -285.0),   # the jaw, further down
    12: (270.0, -150.0, -285.0),
    17: (-300.0, 280.0, -150.0),   # outer ends of the eyebrows
    26: (300.0, 280.0, -150.0),
}
HEAD_LANDMARKS = list(HEAD_MODEL_POINTS)
HEAD_MODEL = np.array([HEAD_MODEL_POINTS[n] for n in HEAD_LANDMARKS], dtype=np.float64)


def head_angles(xList, yList, imgShape):
    # Returns (turn, tilt) in degrees from the 68 landmarks, or None if it can't be worked out.
    # turn is to the side (the sign says which side), tilt is positive when looking down and negative when looking up. This is the one place that works out the angles
    h, w = imgShape[:2]
    points = np.array([(xList[p], yList[p]) for p in HEAD_LANDMARKS], dtype=np.float64)
    # A normal webcam/phone camera: focal length about the picture width. Pretending the camera looks straight at
    # the nose tip means the answer doesn't change with where the face is in the picture (a face low in a photo
    # would otherwise look tilted down, even if the photo was just cropped differently)
    camera = np.array([[w, 0, xList[30]], [0, w, yList[30]], [0, 0, 1]], dtype=np.float64)
    # Start from a head facing the camera a bit away. HEAD_MODEL has y up and the camera y down, so that's the
    # head turned half a round around the x axis. Without this guess it sometimes lands on an upside-down answer
    rotation = np.array([[np.pi], [0.0], [0.0]])
    position = np.array([[0.0], [0.0], [3000.0]])
    ok, rotation, _ = cv2.solvePnP(HEAD_MODEL, points, camera, None, rotation, position,
                                   useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    # Which way the front of the face points, in the camera's directions (x right, y down, z away from the camera)
    forward = cv2.Rodrigues(rotation)[0] @ np.array([0.0, 0.0, 1.0])
    turn = np.degrees(np.arctan2(forward[0], -forward[2]))
    tilt = np.degrees(np.arctan2(forward[1], -forward[2]))
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


def facing_problem(xList, yList, imgShape):
    # Returns None if the face is facing the camera well enough to be rated, otherwise "side", "down" or "up"
    angles = head_angles(xList, yList, imgShape)
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
