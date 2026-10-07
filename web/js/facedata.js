// GENERATED FILE. Do not edit by hand.
// Made by tools/export_web_data.py from facelayout.py, modelfaces.py, landmarkdetect.py, headpose.py and phototips.py.
// Whenever a limit or layout changes in the Python files, re-run (from the repo root):
//     py -3.14 tools/export_web_data.py && py -3.14 tools/make_golden.py
// and the website is in sync again. All numbers here come from the Python modules.
// Names: CONST.<NAME> as in Python. When headpose.py, landmarkdetect.py and phototips.py define the same name with
// different values (today SHARPNESS_WIDTH: 160 in headpose, 128 in phototips), CONST has MODULE_NAME for each
// (HEADPOSE_SHARPNESS_WIDTH, PHOTOTIPS_SHARPNESS_WIDTH) and the plain name is headpose's.

// MediaPipe landmark numbers used (162 of the 478). Everything below is in the order of SUBSET.
export const SUBSET = [10, 21, 54, 58, 67, 93, 103, 109, 127, 132, 136, 148, 149, 150, 152, 162, 172, 176, 234, 251, 284, 288, 297, 323, 332, 338, 356, 361, 365, 377, 378, 379, 389, 397, 400, 454, 276, 282, 283, 285, 293, 295, 296, 300, 334, 336, 46, 52, 53, 55, 63, 65, 66, 70, 105, 107, 1, 2, 4, 5, 6, 19, 45, 48, 64, 94, 97, 98, 115, 168, 195, 197, 220, 275, 278, 294, 326, 327, 344, 440, 249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466, 7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246, 474, 475, 476, 477, 469, 470, 471, 472, 468, 473, 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95];

// SUBSET split by region, as MediaPipe landmark numbers.
export const REGIONS = {
  "jaw": [10, 21, 54, 58, 67, 93, 103, 109, 127, 132, 136, 148, 149, 150, 152, 162, 172, 176, 234, 251, 284, 288, 297, 323, 332, 338, 356, 361, 365, 377, 378, 379, 389, 397, 400, 454],
  "brows": [276, 282, 283, 285, 293, 295, 296, 300, 334, 336, 46, 52, 53, 55, 63, 65, 66, 70, 105, 107],
  "nose": [1, 2, 4, 5, 6, 19, 45, 48, 64, 94, 97, 98, 115, 168, 195, 197, 220, 275, 278, 294, 326, 327, 344, 440],
  "eyes": [249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466, 7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246, 474, 475, 476, 477, 469, 470, 471, 472, 468, 473],
  "outerLips": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146],
  "innerLips": [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95],
};

// How much each of the 162 points counts in alignment and error.
export const WEIGHTS = [0.14166666666666666, 0.14166666666666666, 0.14166666666666666, 0.14166666666666666, 0.14166666666666666, 0.14166666666666666, 0.14166666666666666, 0.14166666666666666, 0.14166666666666666, 0.14166666666666666, 0.14166666666666666, 0.14166666666666666, 0.14166666666666666, 0.14166666666666666, 0.14166666666666666, 0.14166666666666666, 0.14166666666666666, 0.14166666666666666, 0.14166666666666666, 0.14166666666666666, 0.14166666666666666, 0.14166666666666666, 0.14166666666666666, 0.14166666666666666, 0.14166666666666666, 0.14166666666666666, 0.14166666666666666, 0.14166666666666666, 0.14166666666666666, 0.14166666666666666, 0.14166666666666666, 0.14166666666666666, 0.14166666666666666, 0.14166666666666666, 0.14166666666666666, 0.14166666666666666, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

// For each of the 162 points, the position of its left/right partner (itself on the middle line).
export const MIRROR = [0, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 14, 32, 33, 34, 35, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 56, 57, 58, 59, 60, 61, 73, 74, 75, 65, 76, 77, 78, 69, 70, 71, 79, 62, 63, 64, 66, 67, 68, 72, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 118, 117, 116, 119, 114, 113, 112, 115, 121, 120, 132, 131, 130, 129, 128, 127, 126, 125, 124, 123, 122, 141, 140, 139, 138, 137, 136, 135, 134, 133, 152, 151, 150, 149, 148, 147, 146, 145, 144, 143, 142, 161, 160, 159, 158, 157, 156, 155, 154, 153];

// Positions (in the 162 list) around the left cheek, as seen in the picture.
export const CHEEK_LEFT = [18, 3, 67, 99];

// Positions (in the 162 list) around the right cheek.
export const CHEEK_RIGHT = [35, 21, 77, 83];

// Positions of the outer eye corners (their distance is the face size).
export const EYE_OUTER = [97, 81];

// Positions at the temples (the face width).
export const FACE_WIDTH = [8, 26];

// Positions on the bridge of the nose.
export const NOSE_BRIDGE = [71, 70];

// The model faces of each gender (from modelfaces.py): a list of faces, 162 [x, y] each.
export const MODEL_FACES = {
  "boy": [[[182.4, 160.5], [53.1, 218.5], [62.0, 197.4], [68.5, 396.1], [109.0, 168.9], [54.8, 329.5], [79.6, 180.5], [142.4, 162.8], [49.1, 271.2], [60.3, 361.7], [96.2, 444.5], [169.9, 497.9], [131.8, 477.6], [115.4, 463.8], [196.4, 499.2], [49.7, 241.9], [81.0, 423.6], [149.2, 489.8], [51.5, 299.6], [318.9, 208.1], [307.4, 187.9], [319.5, 385.3], [256.4, 163.5], [327.7, 318.6], [287.4, 172.6], [222.4, 159.7], [327.8, 259.8], [325.2, 351.1], [294.1, 435.6], [223.0, 495.6], [260.2, 472.1], [275.9, 456.6], [324.7, 230.6], [308.4, 413.6], [243.3, 485.7], [328.4, 288.4], [298.6, 240.0], [268.8, 230.9], [286.7, 233.5], [212.2, 245.2], [293.4, 224.0], [244.2, 234.3], [246.5, 222.9], [305.4, 232.6], [273.0, 219.7], [216.0, 228.2], [72.3, 246.9], [100.1, 235.5], [83.0, 239.2], [158.5, 246.9], [75.4, 230.1], [125.0, 237.6], [121.6, 226.3], [65.0, 240.1], [94.7, 224.4], [152.7, 230.2], [190.5, 357.9], [191.3, 367.3], [189.9, 344.9], [189.0, 326.2], [187.4, 279.1], [190.9, 363.8], [177.5, 344.9], [150.3, 350.0], [149.9, 354.5], [191.1, 366.0], [174.1, 366.3], [154.4, 361.3], [157.5, 347.0], [186.6, 262.6], [188.5, 310.1], [187.8, 295.1], [166.8, 345.5], [202.3, 344.3], [231.9, 347.2], [232.8, 351.6], [208.5, 365.1], [228.4, 358.8], [223.6, 344.8], [213.5, 343.9], [272.4, 265.8], [276.8, 263.0], [221.9, 268.6], [260.7, 269.1], [250.8, 270.1], [240.2, 269.6], [230.8, 268.9], [224.8, 268.7], [233.0, 260.6], [243.9, 257.7], [254.8, 257.3], [264.2, 258.5], [270.5, 260.5], [267.5, 267.6], [225.4, 265.1], [274.2, 262.0], [102.1, 270.9], [97.2, 268.5], [152.9, 270.4], [113.8, 273.9], [124.1, 274.1], [134.4, 272.8], [144.0, 271.4], [150.4, 270.9], [141.0, 263.1], [129.7, 260.7], [119.1, 260.9], [109.3, 262.8], [103.4, 265.2], [107.0, 272.6], [149.2, 267.3], [99.9, 267.0], [263.7, 262.5], [250.6, 252.0], [238.1, 263.1], [251.0, 274.0], [137.4, 265.7], [123.9, 255.4], [111.6, 267.0], [125.2, 277.7], [124.6, 266.4], [251.0, 263.0], [133.2, 404.1], [139.9, 401.5], [148.2, 398.7], [159.2, 395.3], [175.1, 392.3], [192.2, 395.3], [209.2, 391.3], [225.4, 393.2], [237.2, 395.4], [246.2, 397.4], [253.4, 399.0], [246.8, 404.9], [237.6, 411.7], [225.0, 419.3], [209.9, 424.8], [193.1, 427.0], [176.7, 426.5], [161.5, 422.8], [148.9, 416.0], [140.0, 410.0], [138.2, 404.3], [147.2, 404.6], [155.2, 404.8], [165.9, 404.9], [178.5, 405.8], [192.5, 406.4], [206.9, 405.1], [219.8, 403.2], [230.6, 401.9], [239.0, 401.0], [248.8, 399.8], [239.3, 401.1], [230.8, 402.0], [219.8, 403.2], [206.7, 405.2], [192.4, 406.5], [178.5, 406.1], [166.0, 405.9], [155.2, 405.4], [147.3, 405.1]]],
  "girl": [[[284.9, 140.3], [132.0, 190.4], [145.5, 168.2], [139.0, 394.1], [200.5, 142.9], [123.4, 316.5], [167.0, 152.3], [238.7, 139.4], [120.6, 250.0], [129.5, 353.6], [169.8, 454.4], [249.6, 517.5], [210.2, 491.8], [192.1, 476.7], [276.3, 521.9], [124.4, 217.0], [152.7, 428.3], [228.7, 506.1], [121.0, 282.2], [434.3, 201.0], [422.7, 177.9], [415.5, 403.5], [369.6, 149.1], [434.1, 327.9], [402.5, 160.6], [331.4, 142.7], [440.5, 261.3], [426.6, 364.2], [383.3, 462.2], [302.6, 519.7], [342.4, 497.5], [360.7, 483.3], [439.7, 227.9], [400.9, 437.0], [323.6, 510.3], [438.2, 293.6], [412.4, 235.0], [380.0, 226.4], [400.0, 227.1], [314.8, 248.7], [406.6, 215.0], [352.6, 234.0], [355.3, 219.7], [419.6, 226.0], [385.0, 212.3], [320.0, 229.3], [152.1, 225.7], [184.9, 219.1], [165.0, 218.4], [248.5, 245.9], [159.1, 206.1], [211.9, 228.3], [210.1, 214.1], [145.3, 216.3], [180.8, 204.7], [244.5, 226.1], [278.1, 375.7], [278.4, 386.3], [278.3, 361.8], [278.6, 341.4], [280.1, 289.1], [278.2, 382.0], [265.7, 361.0], [240.6, 365.8], [240.6, 371.4], [278.2, 384.8], [262.7, 384.5], [244.4, 378.9], [246.2, 362.3], [280.7, 267.0], [279.1, 323.9], [279.4, 307.6], [255.2, 361.0], [290.9, 362.1], [316.2, 368.0], [316.2, 373.3], [293.8, 385.0], [312.0, 380.5], [310.4, 364.4], [301.4, 362.4], [383.7, 273.7], [389.0, 268.6], [322.0, 277.3], [369.1, 278.8], [357.1, 279.2], [344.6, 278.0], [333.3, 277.0], [325.6, 277.3], [336.2, 265.1], [349.3, 259.3], [362.6, 257.9], [374.1, 259.7], [381.7, 263.2], [377.6, 276.9], [326.6, 272.5], [385.9, 266.2], [178.6, 265.0], [173.6, 259.6], [239.4, 273.7], [192.2, 271.8], [204.4, 273.2], [216.5, 272.8], [228.1, 272.6], [236.1, 273.7], [225.6, 260.0], [212.6, 252.8], [199.8, 250.4], [187.8, 251.4], [180.7, 254.3], [184.1, 268.8], [235.2, 268.7], [176.6, 257.1], [374.7, 267.4], [359.5, 253.4], [343.8, 266.8], [358.7, 281.2], [218.6, 261.5], [204.4, 246.8], [188.0, 259.5], [202.3, 274.8], [203.3, 260.5], [359.4, 267.2], [217.2, 425.0], [224.1, 422.1], [232.7, 418.8], [244.5, 414.8], [261.9, 410.6], [278.8, 414.7], [295.8, 411.0], [312.4, 415.3], [324.1, 419.5], [332.2, 423.0], [338.7, 425.9], [332.8, 434.5], [324.2, 444.8], [311.7, 455.4], [295.7, 462.3], [277.5, 464.2], [259.2, 461.6], [243.6, 454.2], [231.5, 443.3], [223.4, 433.8], [223.0, 426.2], [232.9, 428.2], [241.1, 429.7], [251.9, 430.8], [264.2, 432.3], [278.3, 433.6], [292.3, 432.9], [304.6, 431.4], [315.0, 430.0], [322.9, 428.7], [333.2, 427.0], [323.1, 428.8], [314.8, 430.1], [304.5, 431.3], [291.9, 433.0], [278.2, 433.6], [264.1, 432.1], [251.8, 431.0], [241.1, 429.5], [233.0, 428.3]]],
};

// Their names ("Boy 1", "Boy 2", ...), in the same order.
export const MODEL_FACE_NAMES = {
  "boy": ["Boy 1"],
  "girl": ["Girl 1"],
};

// Every upper-case constant of landmarkdetect.py, headpose.py and phototips.py (flat). A name that several modules define with different values (SHARPNESS_WIDTH) is there as HEADPOSE_SHARPNESS_WIDTH and PHOTOTIPS_SHARPNESS_WIDTH, and the plain name is headpose's (see CONST_* below too).
export const CONST = {
  "MAX_PICTURE_SIZE": 1280,
  "MAX_FACES": 4,
  "SOFT_RED": [113, 113, 248],
  "FACE_HEIGHT_POINTS": [0, 14],
  "MIN_FACE_SIZE": 0.3,
  "HINT_FACE_SIZE": 0.35,
  "HINT_SIZE_MARGIN": 0.02,
  "MIN_FACE_SIZE_FILE": 0.15,
  "SIZE_MESSAGES": {"near": "Move a little closer.", "far": "Move closer to the camera."},
  "FILE_TOO_SMALL_MESSAGE": "The face is too small in this photo. Use a photo where the face fills more of the picture.",
  "SKIN_THRESHOLD": 7.0,
  "SKIN_WORST_FRACTION": 0.15,
  "SKIN_PENALTY": 0.0025,
  "HAIR_DARKER": 0.8,
  "HAIR_MAX_REDNESS": 2.0,
  "BEARD_DARKER": 0.3,
  "HAIR_MAX_COVER": 0.6,
  "CHEEK_SIZE": 0.2,
  "CHEEK_SAMPLE": 64,
  "CHEEK_MIN_VISIBLE": 0.5,
  "MIN_SKIN_COLOUR": 2.0,
  "SCORE_MID": 0.027,
  "SCORE_POWER": 6.0,
  "TILT_SEARCH": 40,
  "TURN_SEARCH": 20,
  "POSE_STEP": 2.0,
  "POSE_FINE_STEP": 0.25,
  "SYMMETRY_WORST": 0.05,
  "SYMMETRY_PENALTY": 0.0012,
  "MAX_TURN": 12,
  "MAX_TILT": 25,
  "HINT_TURN": 8,
  "HINT_TILT": 18,
  "HINT_MARGIN": 2,
  "HINT_MESSAGES": {"side": "Turn a little towards the camera.", "down": "Lift your chin a little.", "up": "Lower your chin a little."},
  "REFUSED_MESSAGES": {"side": "Head turned too far. Look at the camera.", "down": "Head tilted too far down. Lift your chin.", "up": "Head tilted too far up. Lower your chin."},
  "TILT_OFFSET": 14.0,
  "SMOOTH_FRAMES": 9,
  "SMOOTH_SECONDS": 0.6,
  "RECENT_SECONDS": 1.2,
  "SAME_ANGLE": 2.0,
  "HEADPOSE_SHARPNESS_WIDTH": 160,
  "PHOTOTIPS_SHARPNESS_WIDTH": 128,
  "SHARPNESS_WIDTH": 160,
  "MAX_TIPS": 2,
  "MEASURE_MAX_SIZE": 960,
  "TURN_TIP": 8,
  "TILT_TIP": 18,
  "BLURRY_BELOW": 25.0,
  "TOO_DARK_BELOW": 70.0,
  "TOO_BRIGHT_ABOVE": 215.0,
  "UNEVEN_LIGHT_ABOVE": 0.22,
  "TOO_CLOSE_ABOVE": 0.74,
  "TOO_FAR_BELOW": 0.35,
  "TOO_FAR_BELOW_FILE": 0.22,
  "TIPS": {"turn": "Your head is turned a little. Look straight at the lens.", "tilt": "Your head is tilted a little. Look straight at the lens.", "blurry": "Blurry. Hold still for a second.", "dark": "The picture is a bit dark. Add light in front of you.", "bright": "Very bright. Step away from direct light.", "uneven": "One side of your face is brighter. Face a window or lamp.", "close": "Step back a little. Very close photos distort faces.", "far": "Your face is small in the picture. Move a bit closer.", "small": "Your face is small in the photo. A closer photo rates better."},
  "GREAT": "Great setup: straight, sharp and evenly lit.",
};

// The same constants per module (CONST_HEADPOSE, CONST_PHOTOTIPS): use these for names more than one module defines.
export const CONST_LANDMARKDETECT = {
  "MAX_PICTURE_SIZE": 1280,
  "MAX_FACES": 4,
  "SOFT_RED": [113, 113, 248],
  "FACE_HEIGHT_POINTS": [0, 14],
  "MIN_FACE_SIZE": 0.3,
  "HINT_FACE_SIZE": 0.35,
  "HINT_SIZE_MARGIN": 0.02,
  "MIN_FACE_SIZE_FILE": 0.15,
  "SIZE_MESSAGES": {"near": "Move a little closer.", "far": "Move closer to the camera."},
  "FILE_TOO_SMALL_MESSAGE": "The face is too small in this photo. Use a photo where the face fills more of the picture.",
  "SKIN_THRESHOLD": 7.0,
  "SKIN_WORST_FRACTION": 0.15,
  "SKIN_PENALTY": 0.0025,
  "HAIR_DARKER": 0.8,
  "HAIR_MAX_REDNESS": 2.0,
  "BEARD_DARKER": 0.3,
  "HAIR_MAX_COVER": 0.6,
  "CHEEK_SIZE": 0.2,
  "CHEEK_SAMPLE": 64,
  "CHEEK_MIN_VISIBLE": 0.5,
  "MIN_SKIN_COLOUR": 2.0,
  "SCORE_MID": 0.027,
  "SCORE_POWER": 6.0,
  "TILT_SEARCH": 40,
  "TURN_SEARCH": 20,
  "POSE_STEP": 2.0,
  "POSE_FINE_STEP": 0.25,
  "SYMMETRY_WORST": 0.05,
  "SYMMETRY_PENALTY": 0.0012,
};

export const CONST_HEADPOSE = {
  "MAX_TURN": 12,
  "MAX_TILT": 25,
  "HINT_TURN": 8,
  "HINT_TILT": 18,
  "HINT_MARGIN": 2,
  "HINT_MESSAGES": {"side": "Turn a little towards the camera.", "down": "Lift your chin a little.", "up": "Lower your chin a little."},
  "REFUSED_MESSAGES": {"side": "Head turned too far. Look at the camera.", "down": "Head tilted too far down. Lift your chin.", "up": "Head tilted too far up. Lower your chin."},
  "TILT_OFFSET": 14.0,
  "SMOOTH_FRAMES": 9,
  "SMOOTH_SECONDS": 0.6,
  "RECENT_SECONDS": 1.2,
  "SAME_ANGLE": 2.0,
  "SHARPNESS_WIDTH": 160,
};

export const CONST_PHOTOTIPS = {
  "HINT_FACE_SIZE": 0.35,
  "MIN_FACE_SIZE_FILE": 0.15,
  "MAX_TIPS": 2,
  "MEASURE_MAX_SIZE": 960,
  "TURN_TIP": 8,
  "TILT_TIP": 18,
  "SHARPNESS_WIDTH": 128,
  "BLURRY_BELOW": 25.0,
  "TOO_DARK_BELOW": 70.0,
  "TOO_BRIGHT_ABOVE": 215.0,
  "UNEVEN_LIGHT_ABOVE": 0.22,
  "TOO_CLOSE_ABOVE": 0.74,
  "TOO_FAR_BELOW": 0.35,
  "TOO_FAR_BELOW_FILE": 0.22,
  "TIPS": {"turn": "Your head is turned a little. Look straight at the lens.", "tilt": "Your head is tilted a little. Look straight at the lens.", "blurry": "Blurry. Hold still for a second.", "dark": "The picture is a bit dark. Add light in front of you.", "bright": "Very bright. Step away from direct light.", "uneven": "One side of your face is brighter. Face a window or lamp.", "close": "Step back a little. Very close photos distort faces.", "far": "Your face is small in the picture. Move a bit closer.", "small": "Your face is small in the photo. A closer photo rates better."},
  "GREAT": "Great setup: straight, sharp and evenly lit.",
};

// Hash of the Python sources this file was made from (golden.json carries the same one).
export const SOURCE_HASH = "3b3347d94981";
