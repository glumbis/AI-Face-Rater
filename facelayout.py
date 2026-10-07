# Which MediaPipe face landmarks the program uses, and the fixed data that goes with them (mirror pairs, weights).
# The Face Landmarker finds 478 points. 162 of them are enough to describe the shape of the face, so only
# those are used for rating and drawing. "Left" and "right" are as seen in the picture (the viewer's left).
# The landmarks of the model faces are in modelfaces.py (made by tools/make_model_faces.py).
# All the lists below are in the order of SUBSET, so number n in a list of landmarks is SUBSET[n] in MediaPipe.

# ---------- The 162 landmarks, by region ----------
# The outline of the face: top of the forehead (10), chin (152), and the two sides from the temples down to the jaw
JAW = [10, 21, 54, 58, 67, 93, 103, 109, 127, 132, 136, 148, 149, 150, 152, 162, 172, 176, 234,
       251, 284, 288, 297, 323, 332, 338, 356, 361, 365, 377, 378, 379, 389, 397, 400, 454]
# Eyebrows: the right one, then the left one
BROWS = [276, 282, 283, 285, 293, 295, 296, 300, 334, 336,
         46, 52, 53, 55, 63, 65, 66, 70, 105, 107]
# Nose: tip (4), bridge (6, 168, 195, 197) and underside (1, 2) along the middle, then the nostrils and sides of the nose
NOSE = [1, 2, 4, 5, 6, 19, 45, 48, 64, 94, 97, 98, 115, 168, 195, 197, 220, 275, 278, 294, 326, 327, 344, 440]
# Eyes: the outline of the right eye, the left eye, then the irises (the centre of the pupils is 468 and 473)
EYES = [249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466,
        7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246,
        474, 475, 476, 477, 469, 470, 471, 472, 468, 473]
# The outside edge of the lips, starting at the left corner (61), over the upper lip (0 is the middle) and back
# under the lower lip (17 is the middle)
OUTER_LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
# The inside edge of the lips (where the mouth opens), starting at the left corner (78)
INNER_LIPS = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

REGIONS = {"jaw": JAW, "brows": BROWS, "nose": NOSE, "eyes": EYES, "outerLips": OUTER_LIPS, "innerLips": INNER_LIPS}
SUBSET = JAW + BROWS + NOSE + EYES + OUTER_LIPS + INNER_LIPS
POINT_COUNT = len(SUBSET)  # 162


def position(mediapipeIndex):
    # Where a MediaPipe landmark number is in the 162-landmark lists
    return SUBSET.index(mediapipeIndex)


# ---------- Which landmark is where on the face ----------
# The landmarks around each cheek, the middle of them is the middle of the cheek square
LEFT_CHEEK_POINTS = [position(i) for i in (234, 58, 98, 144)]    # by the ear, the jaw, the nostril and the lower eyelid
RIGHT_CHEEK_POINTS = [position(i) for i in (454, 288, 327, 373)]
# The two sides of the face outline at the temples (the widest part), used as the face width
FACE_WIDTH_POINTS = (position(127), position(356))
# The outer corners of the eyes, the distance between them is the face's size
EYE_CORNERS = (position(33), position(263))
# Two points on the bridge of the nose, where beards don't grow
NOSE_BRIDGE_POINTS = (position(197), position(195))

# ---------- Left and right ----------
# The pairs of landmarks in the same place on each side of the face. The landmarks on the middle line
# (forehead, chin, nose, the middle of the lips) are their own partner and are not in the list
MIRROR_PAIRS_MEDIAPIPE = [
    (21, 251), (54, 284), (58, 288), (67, 297), (93, 323), (103, 332), (109, 338), (127, 356), (132, 361),  # face outline
    (136, 365), (148, 377), (149, 378), (150, 379), (162, 389), (172, 397), (176, 400), (234, 454),
    (46, 276), (52, 282), (53, 283), (55, 285), (63, 293), (65, 295), (66, 296), (70, 300), (105, 334),  # brows
    (107, 336),
    (45, 275), (48, 278), (64, 294), (97, 326), (98, 327), (115, 344), (220, 440),  # nose
    (7, 249), (33, 263), (133, 362), (144, 373), (145, 374), (153, 380), (154, 381), (155, 382), (157, 384),  # eyes
    (158, 385), (159, 386), (160, 387), (161, 388), (163, 390), (173, 398), (246, 466),
    (468, 473), (469, 476), (470, 475), (471, 474), (472, 477),  # irises
    (37, 267), (39, 269), (40, 270), (61, 291), (84, 314), (91, 321), (146, 375), (181, 405), (185, 409),  # outer lips
    (78, 308), (80, 310), (81, 311), (82, 312), (87, 317), (88, 318), (95, 324), (178, 402), (191, 415),  # inner lips
]
_partner = {}
for _a, _b in MIRROR_PAIRS_MEDIAPIPE:
    _partner[_a], _partner[_b] = _b, _a
# For each landmark (in the 162 lists), the number of the landmark in the same place on the other side of the face
MIRROR_PAIRS = [position(_partner.get(i, i)) for i in SUBSET]

# ---------- How much each landmark counts ----------
# How much each region counts, both when lining the face up with the model face and when measuring the difference.
# The jaw line is noisy and moves with hair, beard and head angle, so it counts little. The nose and eyes are
# found very reliably and are the core of the face's shape, so they count the most. Brows move a bit with
# expression, so they count a medium amount. The outer lips move a lot when smiling (a small smile would cost
# about 2 points at 0.6), so they count little. The inner lips mostly show if the mouth is open or smiling,
# which isn't the face's shape, so they don't count.
# These are the totals for the whole region, spread evenly over its landmarks. They are the same as with the
# old 68-landmark layout (jaw 17 landmarks at 0.3, brows 10 at 0.6, nose 9 and eyes 12 at 1.0, outer lips
# 12 at 0.3), so the regions count in the same proportions as before.
REGION_WEIGHTS = {"jaw": 17 * 0.3, "brows": 10 * 0.6, "nose": 9 * 1.0, "eyes": 12 * 1.0,
                  "outerLips": 12 * 0.3, "innerLips": 0.0}
POINT_WEIGHTS = [REGION_WEIGHTS[name] / len(points) for name, points in REGIONS.items() for _ in points]

# The regions that count, as positions in the 162 lists (the inner lips don't count, so they get no score of their own)
REGION_POINTS = {name: [position(i) for i in points] for name, points in REGIONS.items() if REGION_WEIGHTS[name] > 0}
