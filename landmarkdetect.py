import dlib
import cv2
import numpy as np
import time
import os
import sys

# ---------- Put the path to the picture you want to rate here ----------
# (only used when you run this file from the terminal, face_rater_app.py has buttons for it)
IMAGE_TO_RATE = "your_picture.jpg"

# Look for files next to this script, so it works no matter which folder you run it from
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTOR_PATH = os.path.join(SCRIPT_DIR, "shape_predictor_68_face_landmarks.dat")

# Bigger pictures are shrunk to this many pixels on the longest side before rating, so big photos don't take ages
MAX_PICTURE_SIZE = 1280


class FaceError(Exception):
    # The picture can't be rated, the message says why
    pass


def read_image(path):
    # cv2.imread can't open paths with letters like æ, ø and å on Windows, so read the bytes ourselves
    if not os.path.isfile(path):
        raise FaceError(f"Could not find the picture '{path}'.")
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FaceError(f"Could not open '{path}' as a picture.")
    return img


#punkter for aligning
i1 = 27
i2 = 8

detector = dlib.get_frontal_face_detector()

predictor = None


def load_predictor():
    # The model file is ~100 MB and takes a moment to load, so it's only loaded once, when first needed
    global predictor
    if predictor is None:
        if not os.path.isfile(PREDICTOR_PATH):
            raise FileNotFoundError("Missing shape_predictor_68_face_landmarks.dat. Download it from "
                                    "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2, "
                                    "unzip it, and put it next to the scripts.")
        predictor = dlib.shape_predictor(PREDICTOR_PATH)
    return predictor


def landmark_detect(img, detectScale=1.0):
    # Returns the 68 landmarks of the biggest face as xList, yList, or None if there is no face.
    # detectScale < 1 looks for the face in a shrunk copy, which is faster (used for the live camera)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    small = gray if detectScale == 1.0 else cv2.resize(gray, None, fx=detectScale, fy=detectScale)
    faces = detector(small)

    if len(faces) == 0:
        return None

    # Only rate the biggest face, otherwise the landmark lists get longer than the model face's
    face = max(faces, key=lambda f: f.width() * f.height())
    if detectScale != 1.0:
        face = dlib.rectangle(int(face.left() / detectScale), int(face.top() / detectScale),
                              int(face.right() / detectScale), int(face.bottom() / detectScale))

    landmarks = load_predictor()(image=gray, box=face)

    xList = [landmarks.part(n).x for n in range(0, 68)]
    yList = [landmarks.part(n).y for n in range(0, 68)]
    return xList, yList


def draw_landmarks(img, xList, yList):
    # Dot size follows the picture size, so the dots are visible on big photos and not huge on small ones
    radius = max(2, round(max(img.shape[:2]) / 250))
    for x, y in zip(xList, yList):
        cv2.circle(img=img, center=(int(x), int(y)), radius=radius, color=(0, 255, 0), thickness=-1)


# ---------- Head direction ----------
# How many degrees the head may be turned to the side, or tilted up or down, and still count as facing the camera
MAX_TURN = 20
MAX_TILT = 20

# Where some landmarks are on an average head in 3D (nose tip at 0, y up, z towards the camera),
# used to work out which way the head in the picture is facing
HEAD_MODEL = np.array([
    (0.0, 0.0, 0.0),           # nose tip (30)
    (0.0, -330.0, -65.0),      # chin (8)
    (-225.0, 170.0, -135.0),   # outer corner of one eye (36)
    (225.0, 170.0, -135.0),    # outer corner of the other eye (45)
    (-150.0, -150.0, -125.0),  # mouth corner (48)
    (150.0, -150.0, -125.0),   # other mouth corner (54)
])
HEAD_MODEL_POINTS = [30, 8, 36, 45, 48, 54]


def head_angles(xList, yList, imgShape):
    # Returns (turn, tilt) in degrees. turn is to the side, tilt is positive when looking down and negative when looking up
    h, w = imgShape[:2]
    points = np.array([(xList[p], yList[p]) for p in HEAD_MODEL_POINTS], dtype=np.float64)
    # A normal webcam/phone camera: focal length about the picture width, middle of the picture straight ahead
    camera = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rotation, _ = cv2.solvePnP(HEAD_MODEL, points, camera, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    # Which way the front of the face points, in the camera's directions (x right, y down, z away from the camera)
    forward = cv2.Rodrigues(rotation)[0] @ np.array([0.0, 0.0, 1.0])
    turn = np.degrees(np.arctan2(forward[0], -forward[2]))
    tilt = np.degrees(np.arctan2(forward[1], -forward[2]))
    return turn, tilt


def facing_problem(xList, yList, imgShape):
    # Returns None if the face is facing the camera, otherwise which way it's facing instead
    angles = head_angles(xList, yList, imgShape)
    if angles is None:
        return None
    turn, tilt = angles
    if abs(turn) > MAX_TURN:
        return "to the side"
    if tilt > MAX_TILT:
        return "down"
    if tilt < -MAX_TILT:
        return "up"
    return None


# ---------- Skin clarity (cheeks) ----------
# How different (in CIELAB Delta E) a spot's tint must be from the skin around it to count as an inconsistency.
# Brightness differences (shadows) are left out. Around 2.3 is the smallest difference a person can see, so
# tiny variations below this are ignored, and skin in deep shadow turning a little warmer stays below 7.
SKIN_THRESHOLD = 7.0
# When this share of the cheek squares is inconsistent, skin clarity is 0
SKIN_WORST_FRACTION = 0.15
# How much skin clarity can change the score: 0.2 means from -20% (clarity 0) to +20% (clarity 1)
SKIN_WEIGHT = 0.2
# Cheek square side, as a share of the face width (jaw point 0 to 16)
CHEEK_SIZE = 0.2
# Cheek squares are resized to this many pixels per side, so the check behaves the same for any photo size
CHEEK_SAMPLE = 64

# The landmarks around each cheek, the middle of them is the middle of the square
LEFT_CHEEK_POINTS = [1, 3, 31, 41]
RIGHT_CHEEK_POINTS = [15, 13, 35, 46]


def cheek_square(xList, yList, points):
    side = CHEEK_SIZE * dist(xList[0], yList[0], xList[16], yList[16])
    cX = np.mean([xList[p] for p in points])
    cY = np.mean([yList[p] for p in points])
    return int(cX - side / 2), int(cY - side / 2), int(cX + side / 2), int(cY + side / 2)


def cheek_inconsistencies(img, square):
    # Returns the share of the square that is inconsistent, and a mask of where (in the square's own size)
    x1, y1, x2, y2 = square
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None, None

    # Look at a bit more than the square, so the skin around the square's edges is known too
    margin = (x2 - x1) // 4
    px1, py1, px2, py2 = max(x1 - margin, 0), max(y1 - margin, 0), min(x2 + margin, w), min(y2 + margin, h)
    scale = CHEEK_SAMPLE / (x2 - x1)
    size = (round((px2 - px1) * scale), round((py2 - py1) * scale))
    patch = cv2.resize(img[py1:py2, px1:px2], size, interpolation=cv2.INTER_AREA)
    # Linear light, where a shadow multiplies all three colour channels by the same number
    srgb = patch.astype(np.float32) / 255
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4).astype(np.float32)

    # A light blur removes camera noise, a heavy blur gives the skin colour around each spot.
    # Comparing to the surroundings instead of one average colour means light changing across the cheek doesn't count.
    detail = cv2.GaussianBlur(linear, (0, 0), 1)
    surroundings = cv2.GaussianBlur(linear, (0, 0), CHEEK_SAMPLE / 8)

    # Give each spot the same brightness as its surroundings, so a shadow (same skin, just darker) looks
    # identical to the skin around it and only a different tint (redness, spots) counts
    luminance = np.array([0.0722, 0.7152, 0.2126], np.float32)  # B, G, R
    detailLum = detail @ luminance
    relit = detail * ((surroundings @ luminance) / np.maximum(detailLum, 1e-4))[..., None]

    # Lab, so the distance between two colours is Delta E (how different they look to a person)
    deltaE = np.linalg.norm(cv2.cvtColor(relit, cv2.COLOR_LBGR2Lab) - cv2.cvtColor(surroundings, cv2.COLOR_LBGR2Lab), axis=2)
    # In very dark shadow the colour is mostly camera noise, so don't judge it
    deltaE[detailLum < 0.01] = 0

    # Only count the square itself, not the extra margin
    deltaE = cv2.resize(deltaE, (px2 - px1, py2 - py1))[y1 - py1:y2 - py1, x1 - px1:x2 - px1]
    mask = deltaE > SKIN_THRESHOLD
    return mask.mean(), mask.astype(np.uint8)


def skin_clarity(img, drawOn, xList, yList):
    # Returns 0 (very uneven cheeks) to 1 (clear cheeks), and draws the squares and inconsistent spots on drawOn
    fractions = []
    for points in (LEFT_CHEEK_POINTS, RIGHT_CHEEK_POINTS):
        square = cheek_square(xList, yList, points)
        fraction, mask = cheek_inconsistencies(img, square)
        if fraction is None:
            continue
        fractions.append(fraction)

        x1, y1 = max(square[0], 0), max(square[1], 0)
        region = drawOn[y1:y1 + mask.shape[0], x1:x1 + mask.shape[1]]
        region[mask == 1] = (0, 0, 255)
        cv2.rectangle(drawOn, square[:2], square[2:], color=(255, 0, 0), thickness=2)

    if not fractions:
        return None
    return 1 - min(np.mean(fractions) / SKIN_WORST_FRACTION, 1)


# ---------- Comparing to the model face ----------
BOY_PERFECT_X = [44, 49, 57, 62, 73, 97, 128, 162, 200, 239, 268, 293, 315, 324, 328, 332, 334, 61, 77, 103, 129, 154, 210, 238, 263, 289, 308, 186, 187, 187, 188, 164, 177, 191, 205, 218, 95, 112, 132, 149, 131, 111, 225, 240, 259, 276, 261, 242, 138, 157, 177, 194, 211, 230, 250, 231, 213, 195, 178, 157, 146, 178, 195, 212, 241, 211, 194, 177]
BOY_PERFECT_Y = [248, 291, 334, 375, 414, 448, 476, 498, 502, 493, 467, 438, 404, 366, 325, 284, 242, 243, 226, 223, 227, 234, 231, 224, 220, 222, 236, 265, 292, 318, 346, 362, 365, 368, 363, 359, 269, 262, 262, 271, 275, 275, 269, 260, 260, 265, 271, 271, 407, 399, 395, 398, 394, 394, 400, 416, 425, 428, 428, 422, 408, 407, 407, 404, 402, 405, 408, 408]
GIRL_PERFECT_X = [123, 124, 130, 139, 154, 179, 207, 241, 280, 318, 351, 379, 402, 418, 428, 434, 437, 141, 167, 197, 224, 251, 317, 343, 370, 399, 424, 280, 280, 280, 279, 254, 266, 279, 292, 304, 171, 191, 216, 233, 210, 186, 328, 347, 372, 391, 374, 350, 217, 239, 261, 277, 293, 317, 339, 319, 296, 279, 260, 238, 229, 261, 278, 294, 327, 295, 278, 261]
GIRL_PERFECT_Y = [264, 307, 349, 391, 429, 462, 492, 517, 524, 518, 496, 467, 436, 399, 358, 315, 273, 216, 205, 208, 219, 234, 236, 223, 215, 214, 226, 271, 303, 335, 366, 381, 386, 392, 386, 380, 263, 251, 254, 274, 279, 276, 277, 259, 257, 269, 282, 283, 428, 418, 412, 416, 411, 417, 428, 453, 464, 466, 464, 453, 432, 434, 435, 433, 431, 435, 437, 435]


def getPerfs(gender):
    if gender.lower() == "boy":
        return BOY_PERFECT_X, BOY_PERFECT_Y
    elif gender.lower() == "girl":
        return GIRL_PERFECT_X, GIRL_PERFECT_Y
    raise ValueError("gender must be 'boy' or 'girl'")

def dist(x1, y1, x2, y2):
    return(np.sqrt((x1-x2)**2+(y1-y2)**2))

def movePoint(pointX, pointY, x, y):
    newX = pointX - x
    newY = pointY - y
    return newX, newY

def scalePoint(pointX, pointY, centerX, centerY, scalar):
    deltaX = pointX-centerX
    deltaY = pointY-centerY
    newX = centerX+scalar*deltaX
    newY = centerY+scalar*deltaY

    return newX, newY

def rotatePoint(pointX, pointY, centerX, centerY, angle):
    dX = pointX - centerX
    dY = pointY - centerY
    L = np.sqrt(dX**2+dY**2)

    oA = np.arctan2(dY, dX)

    nA = oA + angle
    newX = centerX + np.cos(nA)*L
    newY = centerY + np.sin(nA)*L
    return newX, newY

def findFactors(selfieX, selfieY, perfectX, perfectY, pointIndex1 = i1, pointIndex2 = i2):


    x = selfieX[pointIndex1]-perfectX[pointIndex1]
    y = selfieY[pointIndex1]-perfectY[pointIndex1]

    scalar = dist(perfectX[pointIndex1], perfectY[pointIndex1], perfectX[pointIndex2], perfectY[pointIndex2])/dist(selfieX[pointIndex1], selfieY[pointIndex1], selfieX[pointIndex2], selfieY[pointIndex2])

    sdX = selfieX[pointIndex1] - selfieX[pointIndex2]
    sdY = selfieY[pointIndex1] - selfieY[pointIndex2]
    selfieRotation = np.arctan2(sdY, sdX)


    pdX = perfectX[pointIndex1] - perfectX[pointIndex2]
    pdY = perfectY[pointIndex1] - perfectY[pointIndex2]
    perfectRotation = np.arctan2(pdY, pdX)


    angle = selfieRotation-perfectRotation

    return  x, y, scalar, angle


def changePoints(selfieX, selfieY, perfectX, perfectY):
    x,y,scalar,angle = findFactors(selfieX, selfieY, perfectX, perfectY)

    for i in range(len(selfieX)):
        selfieX[i], selfieY[i] = movePoint(selfieX[i], selfieY[i], x, y)

    cX, cY = selfieX[i1], selfieY[i1]
    for i in range(len(selfieX)):
        selfieX[i], selfieY[i] = scalePoint(selfieX[i], selfieY[i], cX, cY, scalar)

    for i in range(len(selfieX)):
       selfieX[i], selfieY[i] = rotatePoint(selfieX[i], selfieY[i], cX, cY, -angle)

def giveScore(selfieX, selfieY, perfectX, perfectY):
    score = 0
    for i in range(len(selfieX)):
        score += dist(selfieX[i], selfieY[i], perfectX[i], perfectY[i])**2

    # A distance of 0 (rating one of the model faces itself) would divide by zero
    score = (1/max(np.sqrt(score), 1e-3))*100000
    return score


def rate_face(img, gender):
    # Rates the face in img against the boy or girl model face.
    # Returns a dict with score, clarity (None if the cheeks couldn't be seen), skinFactor and picture
    # (a copy with the landmarks and cheek squares drawn on it). Raises FaceError if it can't be rated.
    h, w = img.shape[:2]
    if max(h, w) > MAX_PICTURE_SIZE:
        shrink = MAX_PICTURE_SIZE / max(h, w)
        img = cv2.resize(img, (round(w * shrink), round(h * shrink)), interpolation=cv2.INTER_AREA)

    found = landmark_detect(img)
    if found is None:
        raise FaceError("No face found. Face the camera with a straight face.")
    selfieX, selfieY = found

    problem = facing_problem(selfieX, selfieY, img.shape)
    if problem is not None:
        raise FaceError(f"Face the camera! You are facing {problem}.")

    # Draw on a copy, and check the skin colour on the clean picture
    picture = img.copy()
    draw_landmarks(picture, selfieX, selfieY)

    # Before changePoints, which moves the landmarks away from where they are in the picture
    clarity = skin_clarity(img, picture, selfieX, selfieY)

    perfectX, perfectY = getPerfs(gender)
    changePoints(selfieX, selfieY, perfectX, perfectY)
    score = giveScore(selfieX, selfieY, perfectX, perfectY)

    skinFactor = 1.0
    if clarity is not None:
        # Clarity 0.5 leaves the score as it is, clearer skin raises it and uneven skin lowers it
        skinFactor = 1 + SKIN_WEIGHT * (2 * clarity - 1)
        score *= skinFactor

    return {"score": score, "clarity": clarity, "skinFactor": skinFactor, "picture": picture}


def main():
    # The old terminal version. For the window version, run face_rater_app.py
    imagePath = sys.argv[1] if len(sys.argv) > 1 else IMAGE_TO_RATE

    print("Take you foto with you face facing the camera, and dont make any grimaces.")

    try:
        load_predictor()
        selfiePic = read_image(imagePath)
    except (FileNotFoundError, FaceError) as e:
        sys.exit(str(e))

    gender = input("Are you a boy or a girl? (write boy or girl) Answer here --> ")
    if gender.lower() not in ("boy", "girl"):
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
        print("Could not see the cheeks, so skin clarity is not counted.")
    else:
        print(f"Skin clarity on the cheeks: {round(clarity * 100)}% (score x{round(result['skinFactor'], 2)})")

    print(f"Your beauty score is {round(score,1)}!")
    # Count up to the score, but don't count forever on a huge score
    for i in range(min(round(score), 2000)):
        print(i)
        time.sleep(0.01)

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
