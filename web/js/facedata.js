// GENERATED FILE. Do not edit by hand.
// Made by tools/export_web_data.py from facelayout.py, idealface.py, idealdata.py, landmarkdetect.py, headpose.py and
// phototips.py.
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

// The genders you can be rated as (each has its own ideal).
export const GENDERS = ["boy", "girl"];

// The ideal face (idealface.py and idealdata.py): ANCHORS, JAW_LEFT/JAW_RIGHT, FEATURES [name, region, weight, way], REGIONS, the expression limits and messages, REFERENCE, NEUTRAL_EXPRESSION, EXPRESSION_SLOPES and IDEALS {gender: {feature: [target, tolerance]}}. All MediaPipe landmark numbers (of the 478).
export const IDEAL = {
  "ANCHORS": [6, 168, 197, 195, 133, 362, 33, 263, 10, 151, 9, 8],
  "JAW_LEFT": [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152],
  "JAW_RIGHT": [454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152],
  "FEATURES": [["eyeSpacing", "eyes", 1.0, 0], ["innerEyeGap", "eyes", 0.7, 0], ["canthalTilt", "eyes", 0.7, 1], ["eyeSize", "eyes", 0.7, 1], ["browHeight", "brows", 1.0, 0], ["browArch", "brows", 0.7, 0], ["browTilt", "brows", 0.5, 0], ["noseWidth", "nose", 1.0, 0], ["noseLength", "nose", 0.7, 0], ["lipFullness", "outerLips", 1.0, 0], ["upperLip", "outerLips", 0.6, 0], ["jawSharpness", "jaw", 1.2, 1], ["faceLength", "jaw", 1.2, 0], ["jawWidth", "jaw", 0.8, 0], ["chinWidth", "jaw", 0.6, 0], ["chinHeight", "jaw", 0.5, 0], ["thirds", "jaw", 0.5, 0]],
  "FEATURE_NAMES": ["eyeSpacing", "innerEyeGap", "canthalTilt", "eyeSize", "browHeight", "browArch", "browTilt", "noseWidth", "noseLength", "lipFullness", "upperLip", "jawSharpness", "faceLength", "jawWidth", "chinWidth", "chinHeight", "thirds"],
  "REGIONS": ["jaw", "brows", "nose", "eyes", "outerLips"],
  "MAX_DEVIATION": 3.5,
  "EXPRESSIONS": {"smile": ["mouthSmileLeft", "mouthSmileRight"], "squint": ["eyeSquintLeft", "eyeSquintRight"], "mouthOpen": ["jawOpen"], "upperLipUp": ["mouthUpperUpLeft", "mouthUpperUpRight"]},
  "MAX_SMILE": 0.5,
  "MAX_MOUTH_OPEN": 0.25,
  "HINT_SMILE": 0.35,
  "HINT_MOUTH_OPEN": 0.15,
  "HINT_EXPRESSION_MARGIN": 0.05,
  "EXPRESSION_SECONDS": 0.5,
  "EXPRESSION_MESSAGES": {"smile": "Relax your face, no smile.", "mouthOpen": "Close your mouth."},
  "REFERENCE": [[0, 0.13348, -0.09707], [0, 0.045873, -0.049273], [0, 0.21245, -0.1533], [0, 0.28488, -0.21528], [-0.18699, 0.10168, 0.14537], [0.18699, 0.10168, 0.14537], [-0.5, 0.078626, 0.20654], [0.5, 0.078626, 0.20654], [0, -0.51002, -0.0074717], [0, -0.33503, -0.047953], [0, -0.14538, -0.075781], [0, -0.046867, -0.057688]],
  "NEUTRAL_EXPRESSION": [0.044514, 0.26957, 0.0043033, 0.0038199],
  "EXPRESSION_SLOPES": {"eyeSpacing": [-0.00017012, 0.0030793, 0.00035126, 0.002525], "innerEyeGap": [0.039787, 0.079257, 0.004952, 0.01644], "canthalTilt": [-0.0042294, 0.0014787, 0.0073478, 0.0066419], "eyeSize": [-0.00299, -0.008103, -0.0011581, -0.00074935], "browHeight": [0.0031904, -0.010074, -0.0026192, 0.0001563], "browArch": [0.001499, -0.0027616, -0.0017083, -0.0046246], "browTilt": [-0.011137, -0.011359, 0.0075932, 0.014059], "noseWidth": [0.10926, 0.052991, 0.0082191, 0.041057], "noseLength": [-0.0079023, -0.0041498, -0.0074914, -0.00062306], "lipFullness": [-0.03111, -0.0050907, -0.029953, -0.038045], "upperLip": [-0.023631, -0.025647, -0.082539, 0.055332], "jawSharpness": [-0.0010099, -0.0014258, 0.018035, 0.02235], "faceLength": [0.01871, -0.0022699, 0.020253, 0.0095954], "jawWidth": [0.0066061, 0.0018765, -0.0021877, 0.00026241], "chinWidth": [-0.0036888, 0.0029219, 0.0012986, 0.0036143], "chinHeight": [0.011497, -0.0080329, -0.048083, -0.066765], "thirds": [-0.042624, -0.035324, -0.044793, -0.033668]},
  "MODEL_PICTURES": {"boy": ["perBoy.jpg", "perBoy2.jpg", "perBoy3.jpg", "perBoy4.jpg", "perBoy5.jpg"], "girl": ["perGirl.jpg", "perGirl2.jpg", "perGirl3.jpg", "perGirl4.jpg", "perGirl5.jpg"]},
  "IDEALS": {"boy": {"eyeSpacing": [0.459, 0.024477], "innerEyeGap": [1.1743, 0.084041], "canthalTilt": [0.060659, 0.027277], "eyeSize": [0.20198, 0.0098623], "browHeight": [0.19892, 0.025116], "browArch": [0.072877, 0.0060405], "browTilt": [-0.06083, 0.016065], "noseWidth": [1.2364, 0.066776], "noseLength": [0.2964, 0.010724], "lipFullness": [0.25986, 0.060838], "upperLip": [0.48454, 0.167], "jawSharpness": [0.2, 0.019461], "faceLength": [1.2296, 0.044667], "jawWidth": [0.89082, 0.017424], "chinWidth": [0.1895, 0.0078718], "chinHeight": [0.52136, 0.05283], "thirds": [0.96733, 0.055191]}, "girl": {"eyeSpacing": [0.47297, 0.020524], "innerEyeGap": [1.1969, 0.084896], "canthalTilt": [0.1125, 0.021847], "eyeSize": [0.2081, 0.0094495], "browHeight": [0.2262, 0.027897], "browArch": [0.082199, 0.0065287], "browTilt": [-0.046473, 0.014346], "noseWidth": [1.1179, 0.067794], "noseLength": [0.30625, 0.0074793], "lipFullness": [0.32593, 0.045899], "upperLip": [0.65855, 0.15125], "jawSharpness": [0.20753, 0.015793], "faceLength": [1.2007, 0.045343], "jawWidth": [0.8594, 0.015777], "chinWidth": [0.17136, 0.0059597], "chinHeight": [0.46434, 0.039352], "thirds": [1.0905, 0.055359]}},
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
  "SKIN_PENALTY": 0.1,
  "HAIR_DARKER": 0.8,
  "HAIR_MAX_REDNESS": 2.0,
  "BEARD_DARKER": 0.3,
  "HAIR_MAX_COVER": 0.6,
  "CHEEK_SIZE": 0.2,
  "CHEEK_SAMPLE": 64,
  "CHEEK_MIN_VISIBLE": 0.5,
  "MIN_SKIN_COLOUR": 2.0,
  "SCORE_MID": 1.15,
  "SCORE_POWER": 3.0,
  "SYMMETRY_WORST": 0.05,
  "SYMMETRY_PENALTY": 0.05,
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
  "SKIN_PENALTY": 0.1,
  "HAIR_DARKER": 0.8,
  "HAIR_MAX_REDNESS": 2.0,
  "BEARD_DARKER": 0.3,
  "HAIR_MAX_COVER": 0.6,
  "CHEEK_SIZE": 0.2,
  "CHEEK_SAMPLE": 64,
  "CHEEK_MIN_VISIBLE": 0.5,
  "MIN_SKIN_COLOUR": 2.0,
  "SCORE_MID": 1.15,
  "SCORE_POWER": 3.0,
  "SYMMETRY_WORST": 0.05,
  "SYMMETRY_PENALTY": 0.05,
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
export const SOURCE_HASH = "a35bdbc00a30";
