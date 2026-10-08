# Rating a face against an ideal: a set of measured proportions (eye spacing, nose width, jaw shape, ...), each with a
# target value and a tolerance for each gender. The targets and tolerances are in idealdata.py, made by
# tools/make_ideal_face.py: each target is the average of 102 real faces (the Face Research Lab London Set), moved
# part of the way towards the model faces (perBoy*.jpg, perGirl*.jpg), and the tolerance is how much real faces differ.
#
# Why not compare the whole face with the model faces point by point (as before)? MediaPipe's landmarks all move a
# little with the expression, and comparing 162 of them at once adds all of that up: a smile changed the score as much
# as being a different person did. Single proportions between bony points (the eye corners, the nose, the chin, the
# jaw line) barely change with a smile, and what is left is corrected using MediaPipe's own expression measurements
# (blendshapes), see EXPRESSIONS.
#
# Everything here works on all 478 MediaPipe landmarks (x, y in pixels and depth z on the same scale as x), so the
# numbers below are MediaPipe landmark numbers.
import time
from collections import deque

import numpy as np

import idealdata

# ---------- Turning the face to look straight at the camera ----------
# Points that don't move with the expression: the bridge of the nose, the eye corners, the forehead and between the
# brows. The face is moved, turned (in 3D, using the depth) and resized so these lie as close as possible to the
# same points of an average face (idealdata.REFERENCE), so a head turned or tilted a little is measured as if it
# looked straight into the camera
ANCHORS = [6, 168, 197, 195, 133, 362, 33, 263, 10, 151, 9, 8]

# The face outline from the side of the face by the ear (234 on the picture's left, 454 on the right) down along the
# jaw to the chin (152)
JAW_LEFT = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152]
JAW_RIGHT = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152]


def turned(points, reference=None):
    # The face (478 x 3) moved, turned in 3D and resized so its ANCHORS lie as close as possible to those of the
    # reference face (Umeyama), so it looks straight into the camera, in the size of the reference (eye width 1)
    reference = idealdata.REFERENCE if reference is None else reference
    points = np.asarray(points, dtype=np.float64)
    p = points[ANCHORS]
    q = np.asarray(reference, dtype=np.float64)
    pMid, qMid = p.mean(axis=0), q.mean(axis=0)
    u, s, vt = np.linalg.svd((q - qMid).T @ (p - pMid))
    # Flipping the sign stops it from mirroring the face instead of turning it
    d = np.array([1.0, 1.0, np.sign(np.linalg.det(u) * np.linalg.det(vt))])
    rotation = u @ np.diag(d) @ vt
    scale = (s * d).sum() / ((p - pMid) ** 2).sum()
    return scale * (points - pMid) @ rotation.T + qMid


def frontal(points, reference=None):
    # The face as seen straight from the front: x and y of turned(), 478 x 2
    return turned(points, reference)[:, :2]


# ---------- The measurements ----------
def _dist(f, a, b):
    return float(np.linalg.norm(f[a] - f[b]))


def _jaw_sharpness(f, chain):
    # How much of the jaw line's bend is in one corner: the largest turn between two neighbouring pieces of the line,
    # as a share of all its turning from the ear down to the chin. A rounded jaw bends a little everywhere (low), a
    # sharp jaw bends mostly at its corner (high)
    steps = np.diff(f[chain], axis=0)
    angles = np.unwrap(np.arctan2(steps[:, 1], steps[:, 0]))
    turns = np.abs(np.diff(angles))
    return float(turns.max() / max(turns.sum(), 1e-9))


def _brow_arch(f, inner, peak, outer):
    # How far the highest point of a brow is above the straight line from its inner to its outer end
    a, b, c = f[inner], f[outer], f[peak]
    t = np.clip((c - a) @ (b - a) / ((b - a) @ (b - a)), 0, 1)
    return float(np.linalg.norm(c - (a + t * (b - a))))


def measure(points, reference=None):
    # The measurements of the face (478 x 3 landmarks) as {name: value}, in the order of FEATURES. Left and right are
    # averaged, so a mirrored picture measures the same
    f = frontal(points, reference)
    faceWidth = _dist(f, 234, 454)  # cheek to cheek, by the ears
    faceHeight = _dist(f, 10, 152)  # top of the forehead to the chin
    eyeWidth = (_dist(f, 33, 133) + _dist(f, 263, 362)) / 2  # the width of one eye, corner to corner
    eyeSpan = _dist(f, 33, 263)  # outer corner to outer corner
    innerCorners = _dist(f, 133, 362)
    lips = _dist(f, 0, 13) + _dist(f, 14, 17)  # the upper lip and the lower lip, top to bottom
    return {
        # Eyes
        "eyeSpacing": _dist(f, 468, 473) / faceWidth,  # pupils apart, as a share of the face width (about 0.46)
        "innerEyeGap": innerCorners / eyeWidth,  # the gap between the eyes, in eye widths
        "canthalTilt": ((f[133, 1] - f[33, 1]) / _dist(f, 33, 133) + (f[362, 1] - f[263, 1]) / _dist(f, 263, 362)) / 2,
        "eyeSize": eyeWidth / faceWidth,
        # Brows
        "browHeight": ((f[159, 1] - f[105, 1]) + (f[386, 1] - f[334, 1])) / 2 / eyeSpan,
        "browArch": (_brow_arch(f, 107, 105, 70) + _brow_arch(f, 336, 334, 300)) / 2 / eyeSpan,
        "browTilt": ((f[107, 1] - f[70, 1]) + (f[336, 1] - f[300, 1])) / 2 / eyeSpan,
        # Nose
        "noseWidth": _dist(f, 129, 358) / innerCorners,  # the nostril wings, against the gap between the eyes
        "noseLength": _dist(f, 168, 2) / faceHeight,
        # Lips
        "lipFullness": lips / _dist(f, 2, 152),  # both lips, as a share of nose-to-chin
        "upperLip": _dist(f, 0, 13) / _dist(f, 2, 0),  # the upper lip against the groove above it
        # Jaw and face shape
        "jawSharpness": (_jaw_sharpness(f, JAW_LEFT) + _jaw_sharpness(f, JAW_RIGHT)) / 2,
        "faceLength": faceHeight / faceWidth,  # a round face is short and wide
        "jawWidth": _dist(f, 58, 288) / faceWidth,
        "chinWidth": _dist(f, 148, 377) / faceWidth,
        "chinHeight": _dist(f, 17, 152) / _dist(f, 2, 152),
        "thirds": _dist(f, 9, 2) / _dist(f, 2, 152),  # middle third (brows to nose) against the lower third
    }


# How each measurement counts: (name, region, weight, which way). The region is the part of the face its score
# counts for. The weight is how much it counts. Which way: 0 = both too much and too little lower the score, +1 = more
# than the target is fine too (a sharper jaw, bigger eyes, a more upward tilt), -1 = less is fine
FEATURES = [
    ("eyeSpacing", "eyes", 1.0, 0),
    ("innerEyeGap", "eyes", 0.7, 0),
    ("canthalTilt", "eyes", 0.7, +1),
    ("eyeSize", "eyes", 0.7, +1),
    ("browHeight", "brows", 1.0, 0),
    ("browArch", "brows", 0.7, 0),
    ("browTilt", "brows", 0.5, 0),
    ("noseWidth", "nose", 1.0, 0),
    ("noseLength", "nose", 0.7, 0),
    ("lipFullness", "outerLips", 1.0, 0),
    ("upperLip", "outerLips", 0.6, 0),
    ("jawSharpness", "jaw", 1.2, +1),
    ("faceLength", "jaw", 1.2, 0),
    ("jawWidth", "jaw", 0.8, 0),
    ("chinWidth", "jaw", 0.6, 0),
    ("chinHeight", "jaw", 0.5, 0),
    ("thirds", "jaw", 0.5, 0),
]
FEATURE_NAMES = [name for name, _, _, _ in FEATURES]
REGIONS = ("jaw", "brows", "nose", "eyes", "outerLips")

# A measurement further than this many tolerances from its target counts as this many, so one odd landmark can't
# pull the whole score down
MAX_DEVIATION = 3.5


# ---------- The expression ----------
# MediaPipe also says how much the face is smiling, squinting and so on (its "blendshapes", 0 to 1). These are the
# ones that change the measurements, each the average of the left and right side
EXPRESSIONS = {
    "smile": ("mouthSmileLeft", "mouthSmileRight"),
    "squint": ("eyeSquintLeft", "eyeSquintRight"),
    "mouthOpen": ("jawOpen",),
    "upperLipUp": ("mouthUpperUpLeft", "mouthUpperUpRight"),
}
# A picture is refused when the face smiles more than this or the mouth is more open than this (a big smile changes
# the face too much to correct). Most neutral faces are under 0.2, a broad smile is about 0.65
MAX_SMILE = 0.5
MAX_MOUTH_OPEN = 0.25
# The live preview says to relax the face from these, and stops saying it when the face is HINT_EXPRESSION_MARGIN
# under them again. It follows the middle of the last EXPRESSION_SECONDS, so one odd picture doesn't make it flicker
HINT_SMILE = 0.35
HINT_MOUTH_OPEN = 0.15
HINT_EXPRESSION_MARGIN = 0.05
EXPRESSION_SECONDS = 0.5

EXPRESSION_MESSAGES = {
    "smile": "Relax your face, no smile.",
    "mouthOpen": "Close your mouth.",
}


def expression(blendshapes):
    # The EXPRESSIONS as an array, from blendshapes ({name: score}). None when there are none: then nothing is
    # corrected
    if not blendshapes:
        return None
    return np.array([np.mean([blendshapes.get(name, 0.0) for name in names]) for names in EXPRESSIONS.values()])


def _problem_for(smile, mouthOpen, maxSmile, maxMouthOpen):
    if mouthOpen > maxMouthOpen:
        return "mouthOpen"
    if smile > maxSmile:
        return "smile"
    return None


def expression_problem(blendshapes, hint=False):
    # "smile" or "mouthOpen" when the face isn't neutral enough to rate (or, with hint=True, enough to give the live
    # tip), else None
    found = expression(blendshapes)
    if found is None:
        return None
    smile, _, mouthOpen, _ = found
    if hint:
        return _problem_for(smile, mouthOpen, HINT_SMILE, HINT_MOUTH_OPEN)
    return _problem_for(smile, mouthOpen, MAX_SMILE, MAX_MOUTH_OPEN)


class ExpressionTracker:
    # The live tip about the expression ("smile", "mouthOpen" or None), from the middle (median) of the last
    # EXPRESSION_SECONDS, with a margin so the tip doesn't flicker on and off (like headpose.PoseTracker)

    def __init__(self, seconds=EXPRESSION_SECONDS):
        self.seconds = seconds
        self.history = deque()  # (time, smile, mouthOpen)
        self.problem = None

    def reset(self):
        self.history.clear()
        self.problem = None

    def update(self, blendshapes, now=None):
        now = time.monotonic() if now is None else now
        found = expression(blendshapes)
        if found is not None:
            self.history.append((now, float(found[0]), float(found[2])))
        while self.history and now - self.history[0][0] > self.seconds:
            self.history.popleft()
        if not self.history:
            self.problem = None
            return None
        smile = float(np.median([h[1] for h in self.history]))
        mouthOpen = float(np.median([h[2] for h in self.history]))
        margin = HINT_EXPRESSION_MARGIN
        if self.problem is not None:
            stay = _problem_for(smile, mouthOpen, HINT_SMILE - margin, HINT_MOUTH_OPEN - margin)
            if stay == self.problem:
                return stay
        self.problem = _problem_for(smile, mouthOpen, HINT_SMILE, HINT_MOUTH_OPEN)
        return self.problem


def neutral_measures(points, blendshapes=None):
    # The measurements as they would be with a neutral face: each one moves with the expression by about the same
    # amount in every face (idealdata.EXPRESSION_SLOPES, learnt from the same people smiling and not smiling), so
    # that is taken off
    values = measure(points)
    found = expression(blendshapes)
    if found is None:
        return values
    change = found - np.array(idealdata.NEUTRAL_EXPRESSION)
    return {name: value - float(change @ np.array(idealdata.EXPRESSION_SLOPES[name])) for name, value in values.items()}


# ---------- Against the ideal ----------
def deviations(values, gender):
    # How far each measurement is from the target of the gender, in tolerances (0 = on target), with the
    # one-sided ones (FEATURES) at 0 when they are past the target the good way
    ideal = idealdata.IDEALS[gender]
    found = {}
    for name, _, _, way in FEATURES:
        target, tolerance = ideal[name]
        z = (values[name] - target) / tolerance
        if way * z > 0:
            z = 0.0
        found[name] = float(np.clip(z, -MAX_DEVIATION, MAX_DEVIATION))
    return found


def _combined(z, names):
    weights = {name: weight for name, _, weight, _ in FEATURES}
    return float(np.sqrt(sum(weights[n] * z[n] ** 2 for n in names) / sum(weights[n] for n in names)))


def ideal_errors(values, gender):
    # (error, region errors, deviations): the weighted RMS of the deviations over all measurements, and for each
    # region over its own. 0 is the ideal face, a typical face is about 1
    z = deviations(values, gender)
    regions = {region: _combined(z, [n for n, r, _, _ in FEATURES if r == region]) for region in REGIONS}
    return _combined(z, FEATURE_NAMES), regions, z
