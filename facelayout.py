# Which MediaPipe face landmarks the program uses, and the fixed data that goes with them (model faces, mirror pairs,
# weights). The Face Landmarker finds 478 points. 162 of them are enough to describe the shape of the face, so only
# those are used for rating and drawing. "Left" and "right" are as seen in the picture (the viewer's left).
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

# ---------- The model faces ----------
# The landmarks of perBoy.jpg and perGirl.jpg (in the pixels of those pictures), from this same detector
BOY_PERFECT_X = [
    182.4, 53.1, 62.0, 68.5, 109.0, 54.8, 79.6, 142.4, 49.1, 60.3, 96.2, 169.9, 131.8, 115.4, 196.4, 49.7, 81.0,
    149.2, 51.5, 318.9, 307.4, 319.5, 256.4, 327.7, 287.4, 222.4, 327.8, 325.2, 294.1, 223.0, 260.2, 275.9, 324.7,
    308.4, 243.3, 328.4, 298.6, 268.8, 286.7, 212.2, 293.4, 244.2, 246.5, 305.4, 273.0, 216.0, 72.3, 100.1, 83.0,
    158.5, 75.4, 125.0, 121.6, 65.0, 94.7, 152.7, 190.5, 191.3, 189.9, 189.0, 187.4, 190.9, 177.5, 150.3, 149.9,
    191.1, 174.1, 154.4, 157.5, 186.6, 188.5, 187.8, 166.8, 202.3, 231.9, 232.8, 208.5, 228.4, 223.6, 213.5, 272.4,
    276.8, 221.9, 260.7, 250.8, 240.2, 230.8, 224.8, 233.0, 243.9, 254.8, 264.2, 270.5, 267.5, 225.4, 274.2, 102.1,
    97.2, 152.9, 113.8, 124.1, 134.4, 144.0, 150.4, 141.0, 129.7, 119.1, 109.3, 103.4, 107.0, 149.2, 99.9, 263.7,
    250.6, 238.1, 251.0, 137.4, 123.9, 111.6, 125.2, 124.6, 251.0, 133.2, 139.9, 148.2, 159.2, 175.1, 192.2, 209.2,
    225.4, 237.2, 246.2, 253.4, 246.8, 237.6, 225.0, 209.9, 193.1, 176.7, 161.5, 148.9, 140.0, 138.2, 147.2, 155.2,
    165.9, 178.5, 192.5, 206.9, 219.8, 230.6, 239.0, 248.8, 239.3, 230.8, 219.8, 206.7, 192.4, 178.5, 166.0, 155.2,
    147.3
]
BOY_PERFECT_Y = [
    160.5, 218.5, 197.4, 396.1, 168.9, 329.5, 180.5, 162.8, 271.2, 361.7, 444.5, 497.9, 477.6, 463.8, 499.2, 241.9,
    423.6, 489.8, 299.6, 208.1, 187.9, 385.3, 163.5, 318.6, 172.6, 159.7, 259.8, 351.1, 435.6, 495.6, 472.1, 456.6,
    230.6, 413.6, 485.7, 288.4, 240.0, 230.9, 233.5, 245.2, 224.0, 234.3, 222.9, 232.6, 219.7, 228.2, 246.9, 235.5,
    239.2, 246.9, 230.1, 237.6, 226.3, 240.1, 224.4, 230.2, 357.9, 367.3, 344.9, 326.2, 279.1, 363.8, 344.9, 350.0,
    354.5, 366.0, 366.3, 361.3, 347.0, 262.6, 310.1, 295.1, 345.5, 344.3, 347.2, 351.6, 365.1, 358.8, 344.8, 343.9,
    265.8, 263.0, 268.6, 269.1, 270.1, 269.6, 268.9, 268.7, 260.6, 257.7, 257.3, 258.5, 260.5, 267.6, 265.1, 262.0,
    270.9, 268.5, 270.4, 273.9, 274.1, 272.8, 271.4, 270.9, 263.1, 260.7, 260.9, 262.8, 265.2, 272.6, 267.3, 267.0,
    262.5, 252.0, 263.1, 274.0, 265.7, 255.4, 267.0, 277.7, 266.4, 263.0, 404.1, 401.5, 398.7, 395.3, 392.3, 395.3,
    391.3, 393.2, 395.4, 397.4, 399.0, 404.9, 411.7, 419.3, 424.8, 427.0, 426.5, 422.8, 416.0, 410.0, 404.3, 404.6,
    404.8, 404.9, 405.8, 406.4, 405.1, 403.2, 401.9, 401.0, 399.8, 401.1, 402.0, 403.2, 405.2, 406.5, 406.1, 405.9,
    405.4, 405.1
]

GIRL_PERFECT_X = [
    284.9, 132.0, 145.5, 139.0, 200.5, 123.4, 167.0, 238.7, 120.6, 129.5, 169.8, 249.6, 210.2, 192.1, 276.3, 124.4,
    152.7, 228.7, 121.0, 434.3, 422.7, 415.5, 369.6, 434.1, 402.5, 331.4, 440.5, 426.6, 383.3, 302.6, 342.4, 360.7,
    439.7, 400.9, 323.6, 438.2, 412.4, 380.0, 400.0, 314.8, 406.6, 352.6, 355.3, 419.6, 385.0, 320.0, 152.1, 184.9,
    165.0, 248.5, 159.1, 211.9, 210.1, 145.3, 180.8, 244.5, 278.1, 278.4, 278.3, 278.6, 280.1, 278.2, 265.7, 240.6,
    240.6, 278.2, 262.7, 244.4, 246.2, 280.7, 279.1, 279.4, 255.2, 290.9, 316.2, 316.2, 293.8, 312.0, 310.4, 301.4,
    383.7, 389.0, 322.0, 369.1, 357.1, 344.6, 333.3, 325.6, 336.2, 349.3, 362.6, 374.1, 381.7, 377.6, 326.6, 385.9,
    178.6, 173.6, 239.4, 192.2, 204.4, 216.5, 228.1, 236.1, 225.6, 212.6, 199.8, 187.8, 180.7, 184.1, 235.2, 176.6,
    374.7, 359.5, 343.8, 358.7, 218.6, 204.4, 188.0, 202.3, 203.3, 359.4, 217.2, 224.1, 232.7, 244.5, 261.9, 278.8,
    295.8, 312.4, 324.1, 332.2, 338.7, 332.8, 324.2, 311.7, 295.7, 277.5, 259.2, 243.6, 231.5, 223.4, 223.0, 232.9,
    241.1, 251.9, 264.2, 278.3, 292.3, 304.6, 315.0, 322.9, 333.2, 323.1, 314.8, 304.5, 291.9, 278.2, 264.1, 251.8,
    241.1, 233.0
]
GIRL_PERFECT_Y = [
    140.3, 190.4, 168.2, 394.1, 142.9, 316.5, 152.3, 139.4, 250.0, 353.6, 454.4, 517.5, 491.8, 476.7, 521.9, 217.0,
    428.3, 506.1, 282.2, 201.0, 177.9, 403.5, 149.1, 327.9, 160.6, 142.7, 261.3, 364.2, 462.2, 519.7, 497.5, 483.3,
    227.9, 437.0, 510.3, 293.6, 235.0, 226.4, 227.1, 248.7, 215.0, 234.0, 219.7, 226.0, 212.3, 229.3, 225.7, 219.1,
    218.4, 245.9, 206.1, 228.3, 214.1, 216.3, 204.7, 226.1, 375.7, 386.3, 361.8, 341.4, 289.1, 382.0, 361.0, 365.8,
    371.4, 384.8, 384.5, 378.9, 362.3, 267.0, 323.9, 307.6, 361.0, 362.1, 368.0, 373.3, 385.0, 380.5, 364.4, 362.4,
    273.7, 268.6, 277.3, 278.8, 279.2, 278.0, 277.0, 277.3, 265.1, 259.3, 257.9, 259.7, 263.2, 276.9, 272.5, 266.2,
    265.0, 259.6, 273.7, 271.8, 273.2, 272.8, 272.6, 273.7, 260.0, 252.8, 250.4, 251.4, 254.3, 268.8, 268.7, 257.1,
    267.4, 253.4, 266.8, 281.2, 261.5, 246.8, 259.5, 274.8, 260.5, 267.2, 425.0, 422.1, 418.8, 414.8, 410.6, 414.7,
    411.0, 415.3, 419.5, 423.0, 425.9, 434.5, 444.8, 455.4, 462.3, 464.2, 461.6, 454.2, 443.3, 433.8, 426.2, 428.2,
    429.7, 430.8, 432.3, 433.6, 432.9, 431.4, 430.0, 428.7, 427.0, 428.8, 430.1, 431.3, 433.0, 433.6, 432.1, 431.0,
    429.5, 428.3
]

