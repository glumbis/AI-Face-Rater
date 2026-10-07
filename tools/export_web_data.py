"""Writes web/js/facedata.js from the Python modules, so the website uses exactly the same data and constants.

Run it from the repo root whenever facelayout.py, landmarkdetect.py, headpose.py or phototips.py change:

    py -3.14 tools/export_web_data.py

Nothing in the output is typed by hand: the layout (landmark subset, weights, mirror pairs, cheek/eye/nose points),
the boy / girl / average model faces and every upper-case constant of landmarkdetect, headpose and phototips are
read from the imported modules.
"""
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import facelayout  # noqa: E402
import headpose  # noqa: E402
import landmarkdetect  # noqa: E402
import phototips  # noqa: E402

OUT = os.path.join(ROOT, "web", "js", "facedata.js")
SOURCES = ("facelayout.py", "landmarkdetect.py", "headpose.py", "phototips.py")

# Module-level names that are not constants for the website (file paths and the like)
SKIP = {"SCRIPT_DIR", "LANDMARKER_PATH", "LANDMARKER_URL", "IMAGE_TO_RATE",
        "AVERAGE_PERFECT_X", "AVERAGE_PERFECT_Y", "MODEL_FACES"}  # the model faces are exported as MODEL_FACES
# Where each module's constants go, in this order (a name used by several modules gets the value of the last one in
# the flat CONST; the per-module objects below always have each module's own value)
# For a name several modules define differently, the plain name in CONST is the first of these
PREFERRED = ("headpose", "landmarkdetect", "phototips")
MODULES = (("landmarkdetect", landmarkdetect), ("headpose", headpose), ("phototips", phototips))


def jsonable(value):
    # Plain Python data for json.dumps, or raises TypeError for anything that is not a plain constant
    if isinstance(value, (bool, str)) or value is None:
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    raise TypeError(type(value))


def constants(module, others):
    # The upper-case constants the module defines itself (not the ones it imports from another module of the program)
    found = {}
    for name, value in vars(module).items():
        if not name.isupper() or name.startswith("_") or name in SKIP:
            continue
        if any(getattr(other, name, None) is value for other in others if other is not module):
            continue
        try:
            found[name] = jsonable(value)
        except TypeError:
            continue  # numpy arrays (POINT_WEIGHTS) and so on are exported under their layout names
    return found


def points(xs, ys):
    return [[float(x), float(y)] for x, y in zip(xs, ys)]


def digest():
    h = hashlib.sha1()
    for name in SOURCES:
        with open(os.path.join(ROOT, name), "rb") as f:
            h.update(f.read().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]


def dump(value):
    return json.dumps(value, separators=(", ", ": "), ensure_ascii=False)


def main():
    others = [facelayout, headpose]
    per_module = {name: constants(mod, others) for name, mod in MODULES}
    # The flat CONST has every name once. A name that more than one module defines with different values (like
    # SHARPNESS_WIDTH) is there as MODULE_NAME for each of them (HEADPOSE_SHARPNESS_WIDTH, PHOTOTIPS_SHARPNESS_WIDTH),
    # and the plain name is the value of the first module in PREFERRED
    values = {}
    for name, consts in per_module.items():
        for key, value in consts.items():
            values.setdefault(key, {})[name] = value
    flat = {}
    for key, by_module in values.items():
        if len({json.dumps(v, sort_keys=True) for v in by_module.values()}) == 1:
            flat[key] = next(iter(by_module.values()))
            continue
        for name, value in by_module.items():
            flat[f"{name.upper()}_{key}"] = value
        winner = next(name for name in PREFERRED if name in by_module)
        flat[key] = by_module[winner]
        print(f"note: {key} differs between {', '.join(by_module)}: CONST.{key} is the one from {winner}, the others are "
              + ", ".join(f"CONST.{name.upper()}_{key}" for name in by_module))

    exports = {
        "SUBSET": facelayout.SUBSET,
        "REGIONS": facelayout.REGIONS,
        "WEIGHTS": [float(w) for w in facelayout.POINT_WEIGHTS],
        "MIRROR": facelayout.MIRROR_PAIRS,
        "CHEEK_LEFT": facelayout.LEFT_CHEEK_POINTS,
        "CHEEK_RIGHT": facelayout.RIGHT_CHEEK_POINTS,
        "EYE_OUTER": list(facelayout.EYE_CORNERS),
        "FACE_WIDTH": list(facelayout.FACE_WIDTH_POINTS),
        "NOSE_BRIDGE": list(facelayout.NOSE_BRIDGE_POINTS),
        "MODEL_FACES": {name: points(*landmarkdetect.getPerfs(name)) for name in landmarkdetect.MODEL_FACES},
        "CONST": flat,
        "CONST_LANDMARKDETECT": per_module["landmarkdetect"],
        "CONST_HEADPOSE": per_module["headpose"],
        "CONST_PHOTOTIPS": per_module["phototips"],
        "SOURCE_HASH": digest(),
    }
    comments = {
        "SUBSET": "MediaPipe landmark numbers used (162 of the 478). Everything below is in the order of SUBSET.",
        "REGIONS": "SUBSET split by region, as MediaPipe landmark numbers.",
        "WEIGHTS": "How much each of the 162 points counts in alignment and error.",
        "MIRROR": "For each of the 162 points, the position of its left/right partner (itself on the middle line).",
        "CHEEK_LEFT": "Positions (in the 162 list) around the left cheek, as seen in the picture.",
        "CHEEK_RIGHT": "Positions (in the 162 list) around the right cheek.",
        "EYE_OUTER": "Positions of the outer eye corners (their distance is the face size).",
        "FACE_WIDTH": "Positions at the temples (the face width).",
        "NOSE_BRIDGE": "Positions on the bridge of the nose.",
        "MODEL_FACES": "Model faces boy / girl / average: 162 [x, y] each.",
        "CONST": "Every upper-case constant of landmarkdetect.py, headpose.py and phototips.py (flat). A name that several "
                 "modules define with different values (SHARPNESS_WIDTH) is there as HEADPOSE_SHARPNESS_WIDTH and "
                 "PHOTOTIPS_SHARPNESS_WIDTH, and the plain name is headpose's (see CONST_* below too).",
        "CONST_LANDMARKDETECT": "The same constants per module (CONST_HEADPOSE, CONST_PHOTOTIPS): use these for names more than one module defines.",
        "SOURCE_HASH": "Hash of the Python sources this file was made from (golden.json carries the same one).",
    }
    object_exports = ("MODEL_FACES", "REGIONS", "CONST", "CONST_LANDMARKDETECT", "CONST_HEADPOSE", "CONST_PHOTOTIPS")
    lines = [
        "// GENERATED FILE. Do not edit by hand.",
        "// Made by tools/export_web_data.py from facelayout.py, landmarkdetect.py, headpose.py and phototips.py.",
        "// Whenever a limit or layout changes in the Python files, re-run (from the repo root):",
        "//     py -3.14 tools/export_web_data.py && py -3.14 tools/make_golden.py",
        "// and the website is in sync again. All numbers here come from the Python modules.",
        "// Names: CONST.<NAME> as in Python. When headpose.py, landmarkdetect.py and phototips.py define the same name with",
        "// different values (today SHARPNESS_WIDTH: 160 in headpose, 128 in phototips), CONST has MODULE_NAME for each",
        "// (HEADPOSE_SHARPNESS_WIDTH, PHOTOTIPS_SHARPNESS_WIDTH) and the plain name is headpose's.",
        "",
    ]
    for name, value in exports.items():
        if name in comments:
            lines.append("// " + comments[name])
        if name in object_exports:
            body = ",\n".join(f"  {json.dumps(k)}: {dump(v)}" for k, v in value.items())
            lines.append(f"export const {name} = {{\n{body},\n}};")
        else:
            lines.append(f"export const {name} = {dump(value)};")
        lines.append("")
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    print(f"wrote {os.path.relpath(OUT, ROOT)} ({os.path.getsize(OUT)} bytes, source hash {exports['SOURCE_HASH']})")


if __name__ == "__main__":
    main()
