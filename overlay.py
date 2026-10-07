# What is drawn on the face, on the rated picture and in the live preview: thin smooth contour lines (face outline,
# brows, eyes, nose, lips) with a few accent dots, and on the result picture soft corner brackets around the cheeks
# and a faint red tint where the skin is uneven. The style is written down in web/OVERLAY_STYLE.md, so the website
# can match it.
#
# Everything is drawn 2 to 4 times bigger than needed and shrunk again, which makes the lines smooth, and only in a
# crop around the face, so it is fast enough for the live preview. "Left" and "right" are as seen in the picture.
import cv2
import numpy as np

from facelayout import FACE_WIDTH_POINTS, SUBSET

# ---------- Colours (blue, green, red order, like all OpenCV colours) ----------
LINE_COLOUR = (255, 255, 255)  # the contour lines: white
LINE_TURNED = (140, 205, 255)  # the lines in the live preview when the head is turned: a soft amber, #FFCD8C
HALO_COLOUR = (20, 20, 20)  # the dark glow behind the lines, so they also show on pale skin
ACCENT = (255, 205, 96)  # the accent dots, #60CDFF (the app's accent colour in dark mode)
ACCENT_TURNED = (40, 160, 245)  # the accent dots when the head is turned, #F5A028
SOFT_RED = (113, 113, 248)  # uneven skin, #F87171
# Kept for the legend in the window, which shows the same colours
ACCENT_HEX = "#60cdff"
RED_HEX = "#f87171"

# ---------- Sizes, as shares of the face width (the distance between the two sides of the outline) ----------
LINE_WIDTH = 0.004  # at least MIN_LINE pixels
MIN_LINE = 1.2
HALO_WIDTH = 3.4  # times the line width
DOT_RADIUS = 1.35  # of the accent dots, times the line width. A white ring (+0.7) and a dark halo (+1.6) go around it
BRACKET_ARM = 0.07  # the length of the straight part of a cheek bracket, as a share of the face width
BRACKET_CURVE = 0.03  # the radius of its rounded corner

# ---------- How strong things are (0 to 1) ----------
LINE_ALPHA = 0.78
HALO_ALPHA = 0.34
ACCENT_ALPHA = 0.95
SKIN_ALPHA = 0.5  # the strongest red tint of uneven skin

# ---------- The contours, as MediaPipe landmark numbers. All of them are in facelayout's 162 landmarks ----------
# Each is (landmarks in order, closed, strength of the line 0 to 1). A landmark number can also be a pair, which is
# the middle between the two landmarks (used for the brows, which are drawn as one line along the middle)
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
        148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
EYE_RIGHT = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
EYE_LEFT = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
BROW_RIGHT = [(276, 300), (283, 293), (282, 334), (295, 296), (285, 336)]
BROW_LEFT = [(46, 70), (53, 63), (52, 105), (65, 66), (55, 107)]
LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
NOSE_BRIDGE = [168, 6, 197, 195, 5, 4]
NOSE_BASE = [48, 64, 98, 97, 2, 326, 327, 294, 278]
CONTOURS = (
    (OVAL, True, 0.55, 0.9),
    (BROW_RIGHT, False, 1.0, 1.5), (BROW_LEFT, False, 1.0, 1.5),
    (EYE_RIGHT, True, 1.0, 1.0), (EYE_LEFT, True, 1.0, 1.0),
    (LIPS_OUTER, True, 1.0, 1.0), (LIPS_INNER, True, 0.5, 1.0),
    (NOSE_BRIDGE, False, 0.7, 1.0), (NOSE_BASE, False, 0.7, 1.0),
)  # (landmarks, closed, strength of the line, thickness compared to the normal line). The brows are one line along
# the middle, so they are a little thicker than the outlines
# The accent dots: the outer corners of the eyes, the tip of the nose, the corners of the mouth and the chin
KEY_POINTS = [33, 263, 4, 61, 291, 152]

_POSITION = {mediapipe: n for n, mediapipe in enumerate(SUBSET)}


def _resolve(contour):
    # The places in the 162 lists of the two landmarks (the same one twice, unless it is a pair) of each point
    return [np.array([_POSITION[p if isinstance(p, int) else p[k]] for p in contour]) for k in (0, 1)]


_CONTOURS = [(_resolve(points), closed, strength, thick) for points, closed, strength, thick in CONTOURS]
_KEY_POINTS = [_POSITION[m] for m in KEY_POINTS]


def face_width(xList, yList):
    left, right = FACE_WIDTH_POINTS
    return float(np.hypot(xList[left] - xList[right], yList[left] - yList[right]))


def _smooth(points, closed, rounds):
    # Chaikin's corner cutting: every round replaces each corner with two points nearer to the middle of its sides,
    # so the contour turns into a smooth curve through (almost) the same points. Open lines keep their end points
    for _ in range(rounds):
        a, b = (points, np.roll(points, -1, axis=0)) if closed else (points[:-1], points[1:])
        out = np.empty((2 * len(a), 2))
        out[0::2], out[1::2] = 0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b
        points = out if closed else np.vstack([points[:1], out, points[-1:]])
    return points


SHIFT = 4  # OpenCV draws at 1/16 of a (big) pixel with this, so the lines don't jump from pixel to pixel


class _Canvas:
    # Three masks for one crop of the picture: the dark halo behind everything (soft, so it is drawn at the normal
    # size), the white lines and the rings round the dots (drawn big), and the accent dots (small, normal size).
    # finish() shrinks the big one (every pixel becomes the average of the big pixels it covers, which is what makes
    # the lines smooth) and mixes the colours into the picture

    def __init__(self, img, box, ss, unit):
        self.img = img
        self.x0, self.y0, self.x1, self.y1 = box
        self.ss = ss
        w, h = self.x1 - self.x0, self.y1 - self.y0
        self.size = (w, h)
        self.unit = unit
        self.halo = np.zeros((-(-h // 2), -(-w // 2)), np.uint8)  # half the size: it is soft anyway
        self.lines = np.zeros((h * ss, w * ss), np.uint8)
        self.dots = np.zeros((h, w), np.uint8)

    def lines_halo(self, paths, width):
        # The soft dark halo behind all the lines, in one go (the same strength everywhere, it's only a glow)
        shifted = [np.rint((p - [self.x0 + 0.5, self.y0 + 0.5]) * (1 << SHIFT) / 2 - 8).astype(np.int32) for p in paths]
        cv2.polylines(self.halo, shifted, False, 200, max(1, round(width / 2)), cv2.LINE_AA, SHIFT)

    def line(self, pts, width, value, closed=False):
        # A white line (width is in the picture's pixels) with round ends and corners
        fixed = np.rint((pts - [self.x0 + 0.5, self.y0 + 0.5]) * (self.ss << SHIFT) - 0.5 * (1 << SHIFT)).astype(np.int32)
        cv2.polylines(self.lines, [fixed], closed, value, max(1, round(width * self.ss)), cv2.LINE_AA, SHIFT)

    def points(self, pts, ss):
        # In the pixels of a mask, as 1/16 pixels. Landmark numbers have the middle of the first pixel at 0
        pts = (np.asarray(pts, np.float64) - [self.x0 + 0.5, self.y0 + 0.5]) * ss - 0.5
        return np.round(pts * (1 << SHIFT)).astype(np.int32)

    def dot(self, x, y, radius):
        # An accent dot with a white ring and a dark halo
        for mask, ss, r in ((self.halo, 0.5, radius + 1.6 * self.unit), (self.lines, self.ss, radius + 0.7 * self.unit),
                            (self.dots, 1, radius)):
            c = self.points([(x, y)], ss)[0]
            cv2.circle(mask, (int(c[0]), int(c[1])), round(r * ss * (1 << SHIFT)), 255, -1, cv2.LINE_AA, SHIFT)

    def finish(self, lineColour, accent, scale=1.0):
        # Shrinks the big mask and mixes the colours into the picture: halo first, the lines on top of it, the
        # accent dots on top of those. All in 8-bit pictures, which OpenCV does very quickly
        w, h = self.size
        # The halo is blurred a little, so it is a soft glow and not a second outline
        halo = cv2.resize(cv2.GaussianBlur(self.halo, (0, 0), self.unit * 0.4), (w, h), interpolation=cv2.INTER_LINEAR)
        lines = self.lines if self.ss == 1 else cv2.resize(self.lines, (w, h), interpolation=cv2.INTER_AREA)
        region = self.img[self.y0:self.y1, self.x0:self.x1]
        picture = np.ascontiguousarray(region)
        for mask, alpha, colour in ((halo, HALO_ALPHA, HALO_COLOUR), (lines, LINE_ALPHA, lineColour),
                                    (self.dots, ACCENT_ALPHA, accent)):
            strength = cv2.merge([cv2.convertScaleAbs(mask, alpha=alpha * scale)] * 3)
            picture = cv2.add(cv2.multiply(picture, cv2.bitwise_not(strength), scale=1 / 255),
                              cv2.multiply(strength, (*colour, 0), scale=1 / 255))
        region[:] = picture


def _box(img, xs, ys, margin):
    h, w = img.shape[:2]
    x0, x1 = int(max(np.floor(min(xs) - margin), 0)), int(min(np.ceil(max(xs) + margin) + 1, w))
    y0, y1 = int(max(np.floor(min(ys) - margin), 0)), int(min(np.ceil(max(ys) + margin) + 1, h))
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _supersampling(box, preview):
    # 4 times bigger for the result picture, 2 for the live preview (which has to be fast), less if the crop is huge
    w, h = box[2] - box[0], box[3] - box[1]
    ss = 2 if preview else 4
    while ss > 1 and w * h * ss * ss > 3e7:
        ss -= 1
    return ss


def draw_face(img, xList, yList, turned=False, preview=False):
    # Draws the contour lines and accent dots on img (a BGR picture), in place. turned=True is the amber version
    # the live preview uses when the head is turned (a hint, not an error)
    width = face_width(xList, yList)
    line = max(MIN_LINE, LINE_WIDTH * width)
    box = _box(img, xList, yList, 4 * line + 3)
    if box is None or width < 8:
        return
    canvas = _Canvas(img, box, _supersampling(box, preview), line)

    xs, ys = np.asarray(xList), np.asarray(yList)
    paths = []
    for (first, second), closed, strength, thick in _CONTOURS:
        pts = np.column_stack([(xs[first] + xs[second]) / 2, (ys[first] + ys[second]) / 2])
        pts = _smooth(pts, closed, 1 if preview else 2)
        canvas.line(pts, line * thick, round(255 * strength), closed)
        paths.append(np.vstack([pts, pts[:1]]) if closed else pts)

    canvas.lines_halo(paths, line * HALO_WIDTH)
    for n in _KEY_POINTS:
        x, y = xList[n], yList[n]
        canvas.dot(x, y, DOT_RADIUS * line)

    # The dots' halo and ring are drawn after the lines, so a dot on a line cuts it cleanly. The accent mask is
    # separate, so it ends up on top of its own white ring
    canvas.finish(LINE_TURNED if turned else LINE_COLOUR, ACCENT_TURNED if turned else ACCENT)


def draw_cheek(img, square, mask, faceWidth):
    # The uneven skin of one cheek as a soft red tint, and soft corner brackets where the cheek was looked at.
    # square is (x1, y1, x2, y2) in the picture's pixels (it can stick out of the picture), mask is 1 where the skin
    # is uneven, in the size of the part of the square that is inside the picture
    x1, y1, x2, y2 = square
    h, w = img.shape[:2]
    side = x2 - x1
    if mask is not None and mask.any():
        sx1, sy1 = max(x1, 0), max(y1, 0)
        mh, mw = mask.shape
        region = img[sy1:sy1 + mh, sx1:sx1 + mw]
        soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), max(1.0, side / 20))
        # The tint fades out towards the edge of the square, so the square doesn't show as a hard edge
        edge = max(2.0, side * 0.12)
        fadeX = np.clip(np.minimum(np.arange(mw) + sx1 - x1 + 1, x2 - sx1 - np.arange(mw)) / edge, 0, 1)
        fadeY = np.clip(np.minimum(np.arange(mh) + sy1 - y1 + 1, y2 - sy1 - np.arange(mh)) / edge, 0, 1)
        alpha = np.clip(soft * 1.6, 0, 1) * np.outer(fadeY, fadeX) * SKIN_ALPHA
        tinted = region * (1 - alpha[..., None]) + np.array(SOFT_RED, np.float32) * alpha[..., None]
        region[:] = np.clip(tinted + 0.5, 0, 255).astype(np.uint8)

    line = max(MIN_LINE, LINE_WIDTH * faceWidth)
    box = _box(img, (x1, x2), (y1, y2), 4 * line + 3)
    if box is None:
        return
    canvas = _Canvas(img, box, _supersampling(box, False), line)
    arm, curve = BRACKET_ARM * faceWidth, BRACKET_CURVE * faceWidth
    arm = max(arm, curve + 2)
    paths = []
    for cx, cy, sx, sy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x2, y2, -1, -1), (x1, y2, 1, -1)):
        # Along the top (or bottom) edge, round the corner (a quarter circle), then along the side
        ccx, ccy = cx + sx * curve, cy + sy * curve
        pts = [(cx + sx * arm, cy)]
        pts += [(ccx - sx * curve * np.sin(t), ccy - sy * curve * np.cos(t)) for t in np.linspace(0, np.pi / 2, 9)]
        pts.append((cx, cy + sy * arm))
        pts = np.array(pts)
        paths.append(pts)
        canvas.line(pts, line * 1.1, 255)
    canvas.lines_halo(paths, line * 1.1 * HALO_WIDTH)
    canvas.finish(LINE_COLOUR, ACCENT, scale=0.7)
