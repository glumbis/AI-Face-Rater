import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
import cv2
import numpy as np
import time
import os
import sys
from collections import namedtuple

from facelayout import (BOY_PERFECT_X, BOY_PERFECT_Y, EYE_CORNERS, FACE_WIDTH_POINTS, GIRL_PERFECT_X, GIRL_PERFECT_Y,
                        LEFT_CHEEK_POINTS, MIRROR_PAIRS, NOSE_BRIDGE_POINTS, POINT_WEIGHTS as _POINT_WEIGHTS,
                        RIGHT_CHEEK_POINTS, SUBSET)
from headpose import MAX_TILT, MAX_TURN, REFUSED_MESSAGES, facing_problem, head_angles  # noqa: F401

# ---------- Put the path to the picture you want to rate here ----------
# (only used when you run this file from the terminal, face_rater_app.py has buttons for it)
IMAGE_TO_RATE = "your_picture.jpg"

# Look for files next to this script, so it works no matter which folder you run it from
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LANDMARKER_PATH = os.path.join(SCRIPT_DIR, "face_landmarker.task")
LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

# Bigger pictures are shrunk to this many pixels on the longest side before rating, so big photos don't take ages
MAX_PICTURE_SIZE = 1280
# The most faces looked for in one picture (only the biggest is rated)
MAX_FACES = 4


class FaceError(Exception):
    # The picture can't be rated, the message says why
    pass


def read_image(path):
    # cv2.imread can't open paths with letters like æ, ø and å on Windows, so read the bytes ourselves
    if not os.path.isfile(path):
        raise FaceError(f"Could not find the picture '{path}'.")
    try:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except OSError as e:
        # For example a OneDrive file that isn't downloaded while offline, or a file another program has locked
        raise FaceError(f"Could not read '{path}': {e.strerror or e}")
    except cv2.error:
        # Empty or broken files can make OpenCV fail instead of just returning None
        img = None
    if img is None:
        raise FaceError(f"Could not open '{path}' as a picture.")
    return img


def to_bgr(img):
    # Everything here expects normal colour pictures (3 channels, blue-green-red), so convert gray and see-through ones
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


# What finding a face gives: its landmarks as lists of pixel coordinates (the 162 landmarks of facelayout.py, in the
# picture's own pixels) and which way the head is facing, (turn, tilt) in degrees, or None if that couldn't be worked out
Detection = namedtuple("Detection", ["xList", "yList", "angles"])

# The model file is ~4 MB and takes a moment to load, so it's only read once, when first needed. The still-picture
# landmarker ("image") and the one for the live camera ("video", which follows the face from one picture to the next,
# so it is steadier and quicker) are separate, each is made when first used
modelData = None
landmarkers = {}
lastVideoTime = 0


def load_landmarker(live=False):
    # Returns the face landmarker for still pictures, or for the live camera (live=True). Raises FileNotFoundError if the
    # model file is missing and RuntimeError if it is damaged
    global modelData
    key = "video" if live else "image"
    if key in landmarkers:
        return landmarkers[key]
    if modelData is None:
        if not os.path.isfile(LANDMARKER_PATH):
            raise FileNotFoundError("Missing face_landmarker.task. It should be next to the scripts. Download it from "
                                    f"{LANDMARKER_URL} and put it there.")
        # The bytes are given to MediaPipe instead of the path, so folders with long names or letters like æ, ø
        # and å can't get in the way
        try:
            with open(LANDMARKER_PATH, "rb") as f:
                modelData = f.read()
        except OSError as e:
            raise RuntimeError(f"Could not read face_landmarker.task: {e.strerror or e}")
    try:
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=modelData),
            running_mode=vision.RunningMode.VIDEO if live else vision.RunningMode.IMAGE,
            num_faces=MAX_FACES, output_face_blendshapes=False, output_facial_transformation_matrixes=True)
        landmarkers[key] = vision.FaceLandmarker.create_from_options(options)
    except Exception:
        # MediaPipe's own message ("Unable to open zip archive" and so on) doesn't say what to do
        modelData = None
        raise RuntimeError("face_landmarker.task is damaged or only partly downloaded. "
                           f"Download it again from {LANDMARKER_URL} and put it next to the scripts.")
    return landmarkers[key]


def detect_face(img, detectScale=1.0, live=False):
    # Finds the biggest face in img. Returns a Detection (landmarks and head direction), or None if there is no face.
    # detectScale < 1 looks for the face in a shrunk copy, which is faster. The landmarks are still in the pixels of img.
    # live=True is for the live camera: the pictures must come one after the other from the same camera
    global lastVideoTime
    small = to_bgr(img)
    h, w = small.shape[:2]
    if detectScale != 1.0:
        small = cv2.resize(small, None, fx=detectScale, fy=detectScale, interpolation=cv2.INTER_AREA)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    landmarker = load_landmarker(live)
    if live:
        # The live landmarker needs a time for each picture, and every one later than the one before
        lastVideoTime = max(lastVideoTime + 1, int(time.monotonic() * 1000))
        result = landmarker.detect_for_video(image, lastVideoTime)
    else:
        result = landmarker.detect(image)
    if not result.face_landmarks:
        return None

    # (x, y) of the chosen landmarks in the picture's pixels. MediaPipe's numbers are shares of the picture's width
    # and height, with 0 and 1 at the outer edges of the picture, so the middle of the first pixel is at -0.5
    faces = [np.array([(p.x * w - 0.5, p.y * h - 0.5) for p in face])[SUBSET] for face in result.face_landmarks]
    # Only rate the biggest face, otherwise the landmark lists get longer than the model face's
    which = max(range(len(faces)), key=lambda n: np.ptp(faces[n][:, 0]) * np.ptp(faces[n][:, 1]))
    matrices = result.facial_transformation_matrixes
    angles = head_angles(matrices[which]) if which < len(matrices) else None
    return Detection(faces[which][:, 0].tolist(), faces[which][:, 1].tolist(), angles)


def landmark_detect(img, detectScale=1.0, live=False):
    # Returns the 162 landmarks of the biggest face as xList, yList, or None if there is no face.
    # (detect_face gives the head direction too)
    found = detect_face(img, detectScale, live)
    return None if found is None else (found.xList, found.yList)


# Colours of what is drawn on the rated picture (blue, green, red order, like all OpenCV colours)
MINT = (128, 222, 74)  # landmark dots, #4ADE80
SOFT_WHITE = (240, 240, 240)  # outline of the cheek squares
SOFT_RED = (113, 113, 248)  # uneven skin, #F87171


def draw_landmarks(img, xList, yList):
    # Dot size follows the picture size, so the dots are visible on big photos and not huge on small ones.
    # There are 162 of them, so they are smaller than the 68 dots of the old layout were
    radius = max(1, round(max(img.shape[:2]) / 400))
    # Soft mint dots: drawn smooth on a copy, then mixed 75% into the picture so the face still shows through
    dots = img.copy()
    for x, y in zip(xList, yList):
        cv2.circle(img=dots, center=(int(x), int(y)), radius=radius, color=MINT, thickness=-1, lineType=cv2.LINE_AA)
    cv2.addWeighted(dots, 0.75, img, 0.25, 0, dst=img)


# Head direction (head_angles, facing_problem and the limits) is in headpose.py


# ---------- Skin clarity (cheeks) ----------
# How different (in CIELAB Delta E) a spot's tint must be from the skin around it to count as an inconsistency.
# Brightness differences (shadows) are left out. Around 2.3 is the smallest difference a person can see, so
# tiny variations below this are ignored, and skin in deep shadow turning a little warmer stays below 7.
SKIN_THRESHOLD = 7.0
# When this share of the cheek squares is inconsistent, skin clarity is 0
SKIN_WORST_FRACTION = 0.15
# How much uneven skin lowers the score. It's added to the shape error (see SCORE_K), from 0 at clarity 1 to
# this at clarity 0, so very uneven skin lowers the score as much as a 0.015 bigger shape error would (a quarter off the score)
SKIN_PENALTY = 0.015
# A spot darker than this share of the brightness around it, and not more than HAIR_MAX_REDNESS (Lab a*) redder,
# counts as hair (stubble, beard) and is left out, so beards don't count as uneven skin.
# The downside is that dark brown spots (like moles) are left out too, only red ones count
HAIR_DARKER = 0.8
HAIR_MAX_REDNESS = 2.0
# A full beard makes the whole cheek dark, so it isn't darker than its surroundings. It is much darker than the
# skin on the nose bridge though (which beards don't cover), so a spot darker than this share of the nose
# bridge counts as hair too. Cheeks are often about half as bright as the nose bridge, and deep side shadow
# can reach this too, but then those spots are just left out
BEARD_DARKER = 0.3
# If more than this share of a cheek square is hair (a full beard), that cheek isn't judged at all
HAIR_MAX_COVER = 0.6
# Cheek square side, as a share of the face width (from one side of the face outline to the other)
CHEEK_SIZE = 0.2
# Cheek squares are resized to this many pixels per side, so the check behaves the same for any photo size
CHEEK_SAMPLE = 64
# A cheek square with less than this share inside the picture isn't judged
CHEEK_MIN_VISIBLE = 0.5
# Skin with less colour than this (average Lab a* and b*, skin is usually 10 to 30) is a black-and-white photo
MIN_SKIN_COLOUR = 2.0

# The landmarks around each cheek (LEFT_CHEEK_POINTS and RIGHT_CHEEK_POINTS in facelayout.py), the middle of them
# is the middle of the square


def cheek_square(xList, yList, points):
    left, right = FACE_WIDTH_POINTS
    side = CHEEK_SIZE * dist(xList[left], yList[left], xList[right], yList[right])
    cX = np.mean([xList[p] for p in points])
    cY = np.mean([yList[p] for p in points])
    return int(cX - side / 2), int(cY - side / 2), int(cX + side / 2), int(cY + side / 2)


def to_linear(bgr):
    # Linear light, where a shadow multiplies all three colour channels by the same number
    srgb = bgr.astype(np.float32) / 255
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4).astype(np.float32)


# How bright each colour channel looks to a person (B, G, R), to turn a colour into one brightness
LUMINANCE = np.array([0.0722, 0.7152, 0.2126], np.float32)


def nose_skin_brightness(img, xList, yList):
    # The brightness of the skin on the nose bridge (between the two NOSE_BRIDGE_POINTS), or None if it's outside the
    # picture. The median, so a shiny spot or the bridge of a pair of glasses doesn't change it much
    half = max(1, round(0.1 * dist(xList[EYE_CORNERS[0]], yList[EYE_CORNERS[0]], xList[EYE_CORNERS[1]], yList[EYE_CORNERS[1]])))
    top, bottom = NOSE_BRIDGE_POINTS
    cX, cY = round((xList[top] + xList[bottom]) / 2), round((yList[top] + yList[bottom]) / 2)
    h, w = img.shape[:2]
    sample = img[max(cY - half, 0):min(cY + half, h), max(cX - half, 0):min(cX + half, w)]
    if sample.size == 0:
        return None
    return float(np.median(to_linear(sample) @ LUMINANCE))


def cheek_inconsistencies(img, square, noseBrightness=None):
    # Returns the share of the square's skin that is inconsistent, and a mask of where (in the square's own size).
    # Returns None, None if the square is too small, mostly outside the picture, grayscale or mostly covered by beard
    x1, y1, x2, y2 = square
    # The square's real size, also when part of it is outside the picture
    side = x2 - x1
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
    if side < 4 or x2 - x1 < 4 or y2 - y1 < 4:
        return None, None
    # A thin strip at the edge of the picture is too little of the cheek to judge it
    if (x2 - x1) * (y2 - y1) < CHEEK_MIN_VISIBLE * side * side:
        return None, None

    # Look at a bit more than the square, so the skin around the square's edges is known too
    margin = side // 4
    px1, py1, px2, py2 = max(x1 - margin, 0), max(y1 - margin, 0), min(x2 + margin, w), min(y2 + margin, h)
    scale = CHEEK_SAMPLE / side
    size = (round((px2 - px1) * scale), round((py2 - py1) * scale))
    linear = to_linear(cv2.resize(img[py1:py2, px1:px2], size, interpolation=cv2.INTER_AREA))

    # A light blur removes camera noise, a heavy blur gives the skin colour around each spot.
    # Comparing to the surroundings instead of one average colour means light changing across the cheek doesn't count.
    detail = cv2.GaussianBlur(linear, (0, 0), 1)
    surroundings = cv2.GaussianBlur(linear, (0, 0), CHEEK_SAMPLE / 8)
    detailLum = detail @ LUMINANCE
    surroundingsLum = surroundings @ LUMINANCE
    surroundingsLab = cv2.cvtColor(surroundings, cv2.COLOR_LBGR2Lab)

    # Only the tint is compared, so a black-and-white photo would always look perfectly clear. Skin always has
    # some colour (a* and b* in Lab), so if there's almost none, the photo has no colour to judge
    if np.abs(surroundingsLab[..., 1:]).mean() < MIN_SKIN_COLOUR:
        return None, None

    # Stubble and beard hairs are much darker than the skin around them, but not redder (a* in Lab is how red
    # a colour is). Redness and pimples are redder, so they still count. Hair isn't uneven skin, so leave it out
    redness = cv2.cvtColor(detail, cv2.COLOR_LBGR2Lab)[..., 1] - surroundingsLab[..., 1]
    hair = (detailLum < HAIR_DARKER * surroundingsLum) & (redness < HAIR_MAX_REDNESS)
    if noseBrightness is not None:
        # Compared to the beard around it, a lighter hair can look redder, so here only the darkness counts
        hair |= detailLum < BEARD_DARKER * noseBrightness

    # Give each spot the same brightness as its surroundings, so a shadow (same skin, just darker) looks
    # identical to the skin around it and only a different tint (redness, spots) counts
    relit = detail * (surroundingsLum / np.maximum(detailLum, 1e-4))[..., None]

    # Lab, so the distance between two colours is Delta E (how different they look to a person)
    deltaE = np.linalg.norm(cv2.cvtColor(relit, cv2.COLOR_LBGR2Lab) - surroundingsLab, axis=2)
    # In very dark shadow the colour is mostly camera noise, so don't judge it
    deltaE[detailLum < 0.01] = 0

    # Only count the square itself, not the extra margin
    deltaE = cv2.resize(deltaE, (px2 - px1, py2 - py1))[y1 - py1:y2 - py1, x1 - px1:x2 - px1]
    hair = cv2.resize(hair.astype(np.float32), (px2 - px1, py2 - py1))[y1 - py1:y2 - py1, x1 - px1:x2 - px1] > 0.5
    skin = ~hair
    # A cheek that is mostly beard can't be judged
    if skin.mean() < 1 - HAIR_MAX_COVER:
        return None, None
    mask = (deltaE > SKIN_THRESHOLD) & skin
    # The share of the skin (not the whole square) that is uneven, so a beard doesn't make the cheek look clearer either
    return mask.sum() / skin.sum(), mask.astype(np.uint8)


def skin_clarity(img, drawOn, xList, yList):
    # Returns 0 (very uneven cheeks) to 1 (clear cheeks), and draws the squares and inconsistent spots on drawOn
    fractions = []
    noseBrightness = nose_skin_brightness(img, xList, yList)
    for points in (LEFT_CHEEK_POINTS, RIGHT_CHEEK_POINTS):
        square = cheek_square(xList, yList, points)
        fraction, mask = cheek_inconsistencies(img, square, noseBrightness)
        if fraction is None:
            continue
        fractions.append(fraction)

        x1, y1 = max(square[0], 0), max(square[1], 0)
        region = drawOn[y1:y1 + mask.shape[0], x1:x1 + mask.shape[1]]
        # Uneven spots in a soft red, mixed 60% into the skin so the skin still shows through
        spots = mask == 1
        region[spots] = (region[spots] * 0.4 + np.array(SOFT_RED) * 0.6).astype(np.uint8)
        # A faint dark outline behind the white one, so the square also shows on pale skin
        shadow = drawOn.copy()
        cv2.rectangle(shadow, square[:2], square[2:], color=(30, 30, 30), thickness=4, lineType=cv2.LINE_AA)
        cv2.addWeighted(shadow, 0.35, drawOn, 0.65, 0, dst=drawOn)
        cv2.rectangle(drawOn, square[:2], square[2:], color=SOFT_WHITE, thickness=2, lineType=cv2.LINE_AA)

    if not fractions:
        return None
    return 1 - min(np.mean(fractions) / SKIN_WORST_FRACTION, 1)


# ---------- Comparing to the model face ----------
# The error (shape error plus the skin and symmetry penalties) that lowers the score to 10 * e^-1 (about 3.7).
# Tuned so rating perGirl.jpg as a boy and perBoy.jpg as a girl (two quite different faces) gives about 5.
# Even the same face in another photo has an error of about 0.003 to 0.008 (the landmarks move a little), which
# costs about half a point to a point
SCORE_K = 0.054

# How much each landmark counts, both when lining the face up with the model face and when measuring the difference
# (the reasons are in facelayout.py, where the weights of the regions are)
POINT_WEIGHTS = np.array(_POINT_WEIGHTS)

# MIRROR_PAIRS (facelayout.py) has, for each landmark, the landmark in the same place on the other side of the face
# (points on the middle line, like the nose tip and chin, are their own partner)
# How lopsided a face must be (difference between the two sides, as a share of the eye width) for symmetry 0.
# The two model faces measure about 0.010, and the landmarks wobble less than the old ones did (they were 0.013 to 0.022
# and the limit was 0.08), so the limit is lower, which keeps the symmetry percentages about as they were
SYMMETRY_WORST = 0.05
# Like SKIN_PENALTY: added to the shape error, from 0 at symmetry 1 to this at symmetry 0 (the same share off
# the score as before, about an eighth)
SYMMETRY_PENALTY = 0.0075

# The model faces you can compare with. "average" is between the boy and the girl face (see average_face)
MODEL_FACES = ("boy", "girl", "average")


def dist(x1, y1, x2, y2):
    return(np.sqrt((x1-x2)**2+(y1-y2)**2))

def align(points, target, weights=POINT_WEIGHTS):
    # Moves, turns and resizes points (162 x 2) so they lie as close as possible to target (weighted Umeyama).
    # Only those three changes are allowed, so the face keeps its own shape and only the differences in shape are left.
    w = weights / weights.sum()
    pointsMid = w @ points
    targetMid = w @ target
    p = points - pointsMid
    t = target - targetMid
    # The best rotation comes from the SVD of how the two point sets vary together
    u, s, vt = np.linalg.svd((t * w[:, None]).T @ p)
    # Flipping the sign stops it from mirroring the face instead of turning it
    d = np.array([1.0, np.sign(np.linalg.det(u) * np.linalg.det(vt))])
    rotation = u @ np.diag(d) @ vt
    scale = (s * d).sum() / (w @ (p ** 2).sum(axis=1))
    return scale * p @ rotation.T + targetMid

def weighted_rms(a, b, weights=POINT_WEIGHTS):
    return np.sqrt(weights @ ((a - b) ** 2).sum(axis=1) / weights.sum())

def eye_width(points):
    # Distance between the outer eye corners, used as the face's size so errors don't depend on how big the face is
    return np.linalg.norm(points[EYE_CORNERS[1]] - points[EYE_CORNERS[0]])


def average_face():
    # The face halfway between the boy and the girl model face: the girl face is moved, turned and resized onto the
    # boy face (the same alignment as for rating), then the two are averaged landmark by landmark
    boy = np.column_stack([BOY_PERFECT_X, BOY_PERFECT_Y]).astype(np.float64)
    girl = np.column_stack([GIRL_PERFECT_X, GIRL_PERFECT_Y]).astype(np.float64)
    return (boy + align(girl, boy)) / 2


AVERAGE_PERFECT_X, AVERAGE_PERFECT_Y = (list(column) for column in average_face().T)


def getPerfs(gender):
    # The model face's landmarks as xList, yList. gender is "boy", "girl" or "average"
    if gender.lower() == "boy":
        return BOY_PERFECT_X, BOY_PERFECT_Y
    elif gender.lower() == "girl":
        return GIRL_PERFECT_X, GIRL_PERFECT_Y
    elif gender.lower() == "average":
        return AVERAGE_PERFECT_X, AVERAGE_PERFECT_Y
    raise ValueError("gender must be 'boy', 'girl' or 'average'")

def shape_error(xList, yList, gender):
    # How different the face's shape is from the model face, as a share of the model face's eye width (0 = identical)
    perfectX, perfectY = getPerfs(gender)
    perfect = np.column_stack([perfectX, perfectY]).astype(np.float64)
    selfie = np.column_stack([xList, yList]).astype(np.float64)
    # A face looks just as good in a mirror, and a mirrored photo (like from a selfie camera) shouldn't change
    # the score, so compare the face both ways round and keep the closest
    mirrored = selfie[MIRROR_PAIRS] * [-1.0, 1.0]
    err = min(weighted_rms(align(selfie, perfect), perfect), weighted_rms(align(mirrored, perfect), perfect))
    return float(err / eye_width(perfect))

def score_from_error(err):
    # Turns a shape error into a score from 0 to 10, an exact match gives 10
    return 10 * np.exp(-err / SCORE_K)

def symmetry(xList, yList):
    # Returns 0 (very lopsided) to 1 (perfectly symmetric).
    # Mirror the face, then line the mirrored face up onto the original. Lining up takes care of where the
    # middle of the face is and how the head is tilted, so only real differences between the two sides are left.
    points = np.column_stack([xList, yList]).astype(np.float64)
    mirrored = points[MIRROR_PAIRS] * [-1.0, 1.0]
    asymmetry = weighted_rms(align(mirrored, points), points) / eye_width(points)
    return float(1 - min(asymmetry / SYMMETRY_WORST, 1))


def rate_face(img, gender):
    # Rates the face in img against the boy, girl or average model face (gender is "boy", "girl" or "average").
    # Returns a dict with score (0 to 10), shapeError, clarity (None if the cheeks couldn't be judged), skinPenalty,
    # skinFactor, symmetry, symmetryPenalty, symmetryFactor and picture (a copy with the landmarks and cheek
    # squares drawn on it). Raises FaceError if it can't be rated.
    img = to_bgr(img)
    h, w = img.shape[:2]
    if max(h, w) > MAX_PICTURE_SIZE:
        shrink = MAX_PICTURE_SIZE / max(h, w)
        img = cv2.resize(img, (round(w * shrink), round(h * shrink)), interpolation=cv2.INTER_AREA)

    found = detect_face(img)
    if found is None:
        raise FaceError("No face found. Face the camera with a straight face.")
    selfieX, selfieY = found.xList, found.yList

    problem = facing_problem(found.angles)
    if problem is not None:
        raise FaceError(REFUSED_MESSAGES[problem])

    # Draw on a copy, and check the skin colour on the clean picture
    picture = img.copy()
    draw_landmarks(picture, selfieX, selfieY)

    clarity = skin_clarity(img, picture, selfieX, selfieY)

    shapeError = shape_error(selfieX, selfieY, gender)

    # Uneven skin and a lopsided face count as a little extra shape error. There's no penalty when the cheeks
    # couldn't be judged, so a hidden cheek or a beard doesn't lower the score
    skinPenalty = 0.0 if clarity is None else SKIN_PENALTY * (1 - clarity)
    faceSymmetry = symmetry(selfieX, selfieY)
    symmetryPenalty = SYMMETRY_PENALTY * (1 - faceSymmetry)
    score = score_from_error(shapeError + skinPenalty + symmetryPenalty)

    # The same penalties as how many times smaller they made the score, to show it
    skinFactor = float(np.exp(-skinPenalty / SCORE_K))
    symmetryFactor = float(np.exp(-symmetryPenalty / SCORE_K))

    # Odd landmarks (for example all on top of each other) could give "not a number", count that as 0
    score = 0.0 if np.isnan(score) else float(np.clip(score, 0, 10))

    return {"score": score, "shapeError": shapeError, "clarity": clarity, "skinPenalty": skinPenalty,
            "skinFactor": skinFactor, "symmetry": faceSymmetry, "symmetryPenalty": symmetryPenalty,
            "symmetryFactor": symmetryFactor, "picture": picture}


def main():
    # The old terminal version. For the window version, run face_rater_app.py
    imagePath = sys.argv[1] if len(sys.argv) > 1 else IMAGE_TO_RATE

    print("Take you foto with you face facing the camera, and dont make any grimaces.")

    try:
        load_landmarker()
        selfiePic = read_image(imagePath)
    except (FileNotFoundError, RuntimeError, FaceError) as e:
        sys.exit(str(e))

    gender = input("Compare with the boy, girl or average model face? (write boy, girl or average) Answer here --> ")
    if gender.lower() not in MODEL_FACES:
        print("Real beauty comes from the mind, therefor you are a fucking 0. Can't even write boy or girl... smh")
        time.sleep(7)
        sys.exit()

    try:
        result = rate_face(selfiePic, gender)
    except FaceError as e:
        print(e)
        time.sleep(7)
        sys.exit()

    score = result["score"]
    clarity = result["clarity"]

    print()
    if clarity is None:
        print("Could not judge the cheeks (hidden, covered by a beard or a black-and-white photo), "
              "so skin clarity is not counted.")
    else:
        print(f"Skin clarity on the cheeks: {round(clarity * 100)}% (score x{round(result['skinFactor'], 2)})")

    print(f"Symmetry: {round(result['symmetry'] * 100)}% (score x{round(result['symmetryFactor'], 2)})")

    print(f"Your beauty score is {round(score,1)} / 10!")
    # Count up to the score in steps of 0.1
    for i in range(round(score * 10) + 1):
        print(f"{i / 10:.1f}")
        time.sleep(0.03)

    # Show the picture with the landmarks drawn on it, scaled down to fit the screen
    picture = result["picture"]
    h, w = picture.shape[:2]
    shrink = min(1.0, 900 / max(h, w))
    cv2.imshow("Face", cv2.resize(picture, (int(w * shrink), int(h * shrink))))
    print("Press any key in the picture window to close it.")
    cv2.waitKey(delay=0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
