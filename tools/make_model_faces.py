"""Writes modelfaces.py: the landmarks of every model face picture, so the app doesn't have to find them each time.

The model faces are perBoy.jpg, perBoy2.jpg, perBoy3.jpg, ... and perGirl.jpg, perGirl2.jpg, ... next to the scripts
(perBoy.jpg is "Boy 1"). To add one, put the picture there with the next number and run from the repo root:

    py -3.14 tools/make_model_faces.py

and then tools/export_web_data.py and tools/make_golden.py, so the website gets it too. A good model face picture is
sharp and evenly lit, with the face looking straight into the camera and a neutral expression.
"""
import glob
import os
import re
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import landmarkdetect as ld  # noqa: E402

OUT = os.path.join(ROOT, "modelfaces.py")
# The start of the picture names, and of the model face names, for each gender
PICTURES = {"boy": ("perBoy", "Boy"), "girl": ("perGirl", "Girl")}


def pictures(start):
    # (number, file name) of every picture called start.jpg (number 1) or start<number>.jpg, in number order
    found = {}
    for path in glob.glob(os.path.join(ROOT, start + "*.jpg")):
        match = re.fullmatch(re.escape(start) + r"(\d*)\.jpg", os.path.basename(path))
        if not match:
            continue
        number = int(match.group(1) or 1)
        if number in found:
            sys.exit(f"{found[number]} and {os.path.basename(path)} are both model face number {number}. Rename one.")
        found[number] = os.path.basename(path)
    return sorted(found.items())


def numbers(values):
    # The numbers as indented lines of Python, 0.1 pixel is plenty
    return textwrap.fill(", ".join(f"{round(v, 1)}" for v in values), 112, initial_indent=" " * 12,
                         subsequent_indent=" " * 12)


def main():
    ld.load_landmarker()
    lines = [
        "# GENERATED FILE, made by tools/make_model_faces.py. Do not edit by hand: to add a model face, put the picture",
        "# next to the scripts (perBoy2.jpg, perGirl3.jpg, ...) and run the tool again (see the README).",
        "# For each model face: its name, its picture and its 162 landmarks (in the order of facelayout.SUBSET), in the",
        "# pixels of the picture after shrinking it like a rated picture (landmarkdetect.MAX_PICTURE_SIZE).",
        "",
        "FACES = {",
    ]
    for gender, (start, label) in PICTURES.items():
        found = pictures(start)
        if not found:
            sys.exit(f"There is no {start}.jpg. Every gender needs at least one model face.")
        lines.append(f'    "{gender}": [')
        for number, name in found:
            picture = ld.limit_size(ld.read_image(os.path.join(ROOT, name)))
            face = ld.detect_face(picture)
            if face is None:
                sys.exit(f"No face found in {name}.")
            turn, tilt = face.angles or (0.0, 0.0)
            print(f"{name}: {label} {number}, head turned {turn:.1f} and tilted {tilt:.1f} degrees")
            if ld.facing_problem(face.angles) is not None or abs(turn) > 5:
                print(f"  note: {name} doesn't look straight into the camera, a straight-on photo makes a better model face")
            lines += [f'        {{"name": "{label} {number}", "file": "{name}",',
                      '         "x": [', numbers(face.xList), "         ],",
                      '         "y": [', numbers(face.yList), "         ]},"]
        lines.append("    ],")
    lines += ["}", ""]
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    print(f"wrote {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
