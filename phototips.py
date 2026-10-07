import cv2
import numpy as np

from headpose import HINT_TILT, HINT_TURN
from landmarkdetect import HINT_FACE_SIZE, MIN_FACE_SIZE_FILE

# Rule-based tips about the photo and the setup (never about the face or body). Everything here is a pure function of
# a picture and the face landmarks, so it is easy to test. The thresholds are the constants below.

# How many tips a result shows at most
MAX_TIPS = 2
# Pictures bigger than this (on the long side) are shrunk before they are measured
MEASURE_MAX_SIZE = 960

# Head direction in degrees (turn to the side, tilt up or down). The same limits as the live preview's tip (a picture
# with the head turned more than MAX_TURN / MAX_TILT is refused and never gets here). The rating turns the head back
# before comparing, but the further it is turned, the more the landmarks of the far side are guesses
TURN_TIP = HINT_TURN
TILT_TIP = HINT_TILT
# Sharpness: variance of the Laplacian of the face, on a copy of the face this wide. Lower than this is blurry
SHARPNESS_WIDTH = 128
BLURRY_BELOW = 25.0
# Average brightness of the face, 0 (black) to 255 (white)
TOO_DARK_BELOW = 70.0
TOO_BRIGHT_ABOVE = 215.0
# How much brighter one half of the face is than the other, as a share of the average brightness
UNEVEN_LIGHT_ABOVE = 0.22
# Too close: the size of the face (from the top of the forehead to the chin) as a share of the picture's width or height,
# whichever is bigger. (The face outline of the landmarks starts higher than the old landmarks did, at the top of the
# forehead and not at the brows, which made the box about 20% taller, so this limit is 20% bigger than it was)
TOO_CLOSE_ABOVE = 0.74
# Too far: the face height as a share of the picture's shorter side, the same measure as the limits for rating in
# landmarkdetect.py (face_size). A webcam picture smaller than MIN_FACE_SIZE is refused, and the live preview says
# "Move a little closer." under HINT_FACE_SIZE, so the tip comes at the same size. A picked photo may be smaller
# (MIN_FACE_SIZE_FILE), so its tip comes a bit above the lowest size that is rated
TOO_FAR_BELOW = HINT_FACE_SIZE
TOO_FAR_BELOW_FILE = 0.22
assert TOO_FAR_BELOW_FILE > MIN_FACE_SIZE_FILE

TIPS = {
    "turn": "Your head is turned a little. Look straight at the lens.",
    "tilt": "Your head is tilted a little. Look straight at the lens.",
    "blurry": "Blurry. Hold still for a second.",
    "dark": "The picture is a bit dark. Add light in front of you.",
    "bright": "Very bright. Step away from direct light.",
    "uneven": "One side of your face is brighter. Face a window or lamp.",
    "close": "Step back a little. Very close photos distort faces.",
    "far": "Your face is small in the picture. Move a bit closer.",
    "small": "Your face is small in the photo. A closer photo rates better.",
}
GREAT = "Great setup: straight, sharp and evenly lit."


def face_box(xList, yList):
    # (left, top, right, bottom) around the landmarks
    return min(xList), min(yList), max(xList), max(yList)


def to_gray(picture):
    if picture.ndim == 2:
        return picture
    return cv2.cvtColor(picture, cv2.COLOR_BGRA2GRAY if picture.shape[2] == 4 else cv2.COLOR_BGR2GRAY)


def shrink(picture):
    # A copy no bigger than MEASURE_MAX_SIZE on the long side, which is plenty for measuring and quicker
    h, w = picture.shape[:2]
    if max(h, w) <= MEASURE_MAX_SIZE:
        return picture
    factor = MEASURE_MAX_SIZE / max(h, w)
    return cv2.resize(picture, (round(w * factor), round(h * factor)), interpolation=cv2.INTER_AREA)


def crop(gray, box):
    # The part of the picture inside the box (never empty, and never outside the picture)
    h, w = gray.shape[:2]
    left, top, right, bottom = box
    left, top = min(max(0, int(left)), w - 1), min(max(0, int(top)), h - 1)
    right, bottom = min(w, max(int(right), left + 1)), min(h, max(int(bottom), top + 1))
    return gray[top:bottom, left:right]


def brightness(gray, box):
    # Average brightness of the face, 0 to 255
    return float(crop(gray, box).mean())


def light_imbalance(gray, box):
    # How different the left and right halves of the face are in brightness, as a share of the average:
    # 0 is the same, 0.5 is for example one half 1.25 times and the other 0.75 times the average
    face = crop(gray, box).astype(np.float64)
    half = face.shape[1] // 2
    if half < 1:
        return 0.0
    left, right = face[:, :half].mean(), face[:, face.shape[1] - half:].mean()
    return float(abs(left - right) / max((left + right) / 2, 1.0))


def sharpness(gray, box):
    # Variance of the Laplacian of the face, which is high for sharp edges and low for a blurry picture. The face is
    # shrunk to a fixed width first, so the number doesn't depend on how many pixels the camera has
    face = crop(gray, box)
    h, w = face.shape[:2]
    if w != SHARPNESS_WIDTH:
        face = cv2.resize(face, (SHARPNESS_WIDTH, max(1, round(h * SHARPNESS_WIDTH / w))),
                          interpolation=cv2.INTER_AREA if w > SHARPNESS_WIDTH else cv2.INTER_LINEAR)
    return float(cv2.Laplacian(face, cv2.CV_64F).var())


def face_size(box, shape):
    # How much of the picture the face fills: its width or height as a share of the picture's, the bigger of the two
    h, w = shape[:2]
    left, top, right, bottom = box
    return float(max((right - left) / w, (bottom - top) / h))


def face_share(box, shape):
    # The face height as a share of the picture's shorter side (the measure of landmarkdetect.face_size, which is the
    # height of the landmarks' box as the box goes from the top of the forehead to the chin)
    return float((box[3] - box[1]) / min(shape[:2]))


def measure(picture, xList, yList, angles=None, source="camera"):
    # Everything the tips look at. angles is (turn, tilt) in degrees if known. source is "camera" or "file" (a picked
    # photo, where a smaller face is fine)
    gray = to_gray(picture)
    box = face_box(xList, yList)
    return {"turn": None if angles is None else angles[0], "tilt": None if angles is None else angles[1],
            "sharpness": sharpness(gray, box), "brightness": brightness(gray, box),
            "imbalance": light_imbalance(gray, box), "size": face_size(box, gray.shape),
            "share": face_share(box, gray.shape), "source": source}


def pick_tips(m, maxTips=MAX_TIPS):
    # Which tips (keys of TIPS) to show for the measurements of measure(), the most important first and at most maxTips.
    # An empty list means nothing is wrong
    found = []
    if m.get("turn") is not None and abs(m["turn"]) > TURN_TIP:
        found.append("turn")
    if m.get("tilt") is not None and abs(m["tilt"]) > TILT_TIP:
        found.append("tilt")
    if m["sharpness"] < BLURRY_BELOW:
        found.append("blurry")
    if m["brightness"] < TOO_DARK_BELOW:
        found.append("dark")
    elif m["brightness"] > TOO_BRIGHT_ABOVE:
        found.append("bright")
    if m["imbalance"] > UNEVEN_LIGHT_ABOVE:
        found.append("uneven")
    if m["size"] > TOO_CLOSE_ABOVE:
        found.append("close")
    elif m.get("source") == "file" and m.get("share", 1.0) < TOO_FAR_BELOW_FILE:
        found.append("small")
    elif m.get("source") != "file" and m.get("share", 1.0) < TOO_FAR_BELOW:
        found.append("far")
    return found[:maxTips]


def tip_lines(m, maxTips=MAX_TIPS):
    # The texts to show: the tips, or one positive line when there is nothing to fix
    keys = pick_tips(m, maxTips)
    return [TIPS[key] for key in keys] if keys else [GREAT]
