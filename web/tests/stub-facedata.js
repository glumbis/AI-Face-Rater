// Stand-in for js/facedata.js, used only by tests/features.html (through its import map), so the tips tests work
// before the generated file exists and always run with the same numbers. The values are phototips.py's (TURN_TIP = HINT_TURN, TOO_FAR_BELOW = HINT_FACE_SIZE, ...).
export const SUBSET = [10, 152, 234, 454, 1, 33, 263];
export const CONST = {
  MAX_TIPS: 2, MEASURE_MAX_SIZE: 960, TURN_TIP: 8, TILT_TIP: 10, SHARPNESS_WIDTH: 128, BLURRY_BELOW: 25.0,
  TOO_DARK_BELOW: 70.0, TOO_BRIGHT_ABOVE: 215.0, UNEVEN_LIGHT_ABOVE: 0.22, TOO_CLOSE_ABOVE: 0.74, TOO_FAR_BELOW: 0.35, TOO_FAR_BELOW_FILE: 0.22,
};
