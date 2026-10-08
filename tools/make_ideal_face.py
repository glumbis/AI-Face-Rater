"""Writes idealdata.py: the ideal face the app rates against (see idealface.py).

It needs the front photos of the Face Research Lab London Set (DeBruine & Jones 2017, CC BY 4.0,
https://doi.org/10.6084/m9.figshare.5047666): 102 adults photographed with a neutral face and smiling. They are
downloaded once (about 90 MB) into .cache/london-set next to the scripts. From them it works out:

- the average face, to turn every face to look straight into the camera (REFERENCE)
- how much each measurement moves with a smile, a squint and so on, from the same people neutral and smiling
  (EXPRESSION_SLOPES and NEUTRAL_EXPRESSION)
- for each gender, how big each measurement is on average and how much it differs between people (the tolerance)

The target of each measurement is that average, moved MODEL_PULL of the way towards the model faces (perBoy*.jpg and
perGirl*.jpg next to the scripts), but at most MAX_PULL tolerances. To add a model face, put the picture next to the
scripts with the next free number (perBoy6.jpg) and run, from the repo root:

    py -3.14 tools/make_ideal_face.py

and then tools/export_web_data.py and tools/make_golden.py, so the website gets it too. A good model face picture is
sharp and evenly lit, with the face looking straight into the camera and a neutral expression.
"""
import csv
import glob
import os
import re
import sys
import urllib.request
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import idealface  # noqa: E402
import landmarkdetect as ld  # noqa: E402

OUT = os.path.join(ROOT, "idealdata.py")
CACHE = os.path.join(ROOT, ".cache", "london-set")
DOWNLOADS = {
    "neutral_front.zip": "https://ndownloader.figshare.com/files/8541961",
    "smiling_front.zip": "https://ndownloader.figshare.com/files/8541964",
    "london_faces_info.csv": "https://ndownloader.figshare.com/files/27397184",
    "london_faces_ratings.csv": "https://ndownloader.figshare.com/files/8542045",
}
# The start of the model face picture names for each gender (perBoy.jpg, perBoy2.jpg, ...)
PICTURES = {"boy": "perBoy", "girl": "perGirl"}
# How far each target is moved from the average towards the model faces (1 = all the way), and the most it can be
# moved, in tolerances. Not all the way, because a few model faces are a small sample and some of their measurements
# are just how that person looks, not what makes a face look good
MODEL_PULL = 0.75
MAX_PULL = 1.5
# How strongly the expression slopes are held back towards 0 (ridge regression), so a few odd smiles can't make them big
SLOPE_RIDGE = 0.01


def download():
    os.makedirs(CACHE, exist_ok=True)
    for name, url in DOWNLOADS.items():
        path = os.path.join(CACHE, name)
        if not os.path.isfile(path):
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(url, path + ".part")
            os.replace(path + ".part", path)
        if name.endswith(".zip") and not os.path.isdir(os.path.join(CACHE, name[:-4])):
            with zipfile.ZipFile(path) as z:
                z.extractall(CACHE)


def detect(path):
    found = ld.detect_face(ld.limit_size(ld.read_image(path)))
    if found is None:
        sys.exit(f"No face found in {path}.")
    return found


def expression_of(found):
    return idealface.expression(found.blendshapes)


def london_faces():
    # {person: {"gender", "neutral": Detection, "smiling": Detection}}
    with open(os.path.join(CACHE, "london_faces_info.csv"), newline="") as f:
        genders = {row["face_id"]: ("boy" if row["face_gender"] == "male" else "girl") for row in csv.DictReader(f)}
    people = {}
    for kind in ("neutral", "smiling"):
        for path in sorted(glob.glob(os.path.join(CACHE, f"{kind}_front", "*.jpg"))):
            person = os.path.basename(path).split("_")[0]
            people.setdefault(person, {"gender": genders[person]})[kind] = detect(path)
    return {p: v for p, v in people.items() if "neutral" in v and "smiling" in v}


def model_faces(gender):
    found = []
    for path in glob.glob(os.path.join(ROOT, PICTURES[gender] + "*.jpg")):
        match = re.fullmatch(re.escape(PICTURES[gender]) + r"(\d*)\.jpg", os.path.basename(path))
        if match:
            found.append((int(match.group(1) or 1), path))
    if not found:
        sys.exit(f"There is no {PICTURES[gender]}.jpg. Every gender needs at least one model face.")
    return [path for _, path in sorted(found)]


def average_anchors(faces):
    # The ANCHORS of the average face: every face lined up (in 3D) onto the average, a few times over, centred and
    # resized so the outer eye corners are 1 apart
    # The average is made exactly symmetric (averaged with its mirror image), so a mirrored face is turned the
    # mirrored way and measures exactly the same
    partner = [idealface.ANCHORS.index(p) for p in (6, 168, 197, 195, 362, 133, 263, 33, 10, 151, 9, 8)]
    reference = np.asarray(faces[0].points)[idealface.ANCHORS]
    outer = [idealface.ANCHORS.index(33), idealface.ANCHORS.index(263)]
    for _ in range(4):
        reference = np.mean([idealface.turned(f.points, reference)[idealface.ANCHORS] for f in faces], axis=0)
        reference -= reference.mean(axis=0)
        reference = (reference + reference[partner] * [-1.0, 1.0, 1.0]) / 2
        reference /= np.linalg.norm(reference[outer[0]] - reference[outer[1]])
    return reference


def vector(values):
    return np.array([values[name] for name in idealface.FEATURE_NAMES])


def numbers(values, digits=5):
    return "[" + ", ".join(f"{float(v):.{digits}g}" for v in values) + "]"


def main():
    download()
    ld.load_landmarker()
    people = london_faces()
    print(f"{len(people)} people with a neutral and a smiling photo")
    reference = average_anchors([p["neutral"] for p in people.values()])

    neutral = np.array([vector(idealface.measure(p["neutral"].points, reference)) for p in people.values()])
    smiling = np.array([vector(idealface.measure(p["smiling"].points, reference)) for p in people.values()])
    exprNeutral = np.array([expression_of(p["neutral"]) for p in people.values()])
    exprSmiling = np.array([expression_of(p["smiling"]) for p in people.values()])

    # How much each measurement moves per unit of each expression: the change from neutral to smiling against the
    # change in expression, over all the people (a ridge regression)
    x, y = exprSmiling - exprNeutral, smiling - neutral
    slopes = np.linalg.solve(x.T @ x + SLOPE_RIDGE * len(x) * np.eye(x.shape[1]), x.T @ y)  # expressions x features
    baseline = exprNeutral.mean(axis=0)

    def corrected(values, expr):
        return values - (expr - baseline) @ slopes

    neutralFixed = np.array([corrected(v, e) for v, e in zip(neutral, exprNeutral)])
    genders = np.array([p["gender"] for p in people.values()])
    ideals, report, pictures = {}, [], {}
    for gender in ("boy", "girl"):
        population = neutralFixed[genders == gender]
        mean, spread = population.mean(axis=0), population.std(axis=0)
        models = []
        pictures[gender] = [os.path.basename(path) for path in model_faces(gender)]
        for path in model_faces(gender):
            found = detect(path)
            models.append(corrected(vector(idealface.measure(found.points, reference)), expression_of(found)))
        pull = np.clip(MODEL_PULL * ((np.array(models) - mean) / spread).mean(axis=0), -MAX_PULL, MAX_PULL)
        target = mean + pull * spread
        ideals[gender] = {name: (float(target[i]), float(spread[i])) for i, name in enumerate(idealface.FEATURE_NAMES)}
        report.append((gender, len(population), len(models), pull))

    lines = [
        "# GENERATED FILE, made by tools/make_ideal_face.py. Do not edit by hand: run the tool again (see the README).",
        "# The ideal face the app rates against (see idealface.py), worked out from the front photos of the Face Research",
        "# Lab London Set (DeBruine & Jones 2017, CC BY 4.0, https://doi.org/10.6084/m9.figshare.5047666) and the model",
        "# face pictures.",
        "",
        "# The idealface.ANCHORS of the average face (x, y, depth), outer eye corners 1 apart",
        "REFERENCE = [",
        *[f"    {numbers(row)}," for row in reference],
        "]",
        "",
        "# The expression (idealface.EXPRESSIONS: smile, squint, mouthOpen, upperLipUp) of a typical neutral face",
        f"NEUTRAL_EXPRESSION = {numbers(baseline)}",
        "",
        "# How much each measurement changes per unit (0 to 1) of each expression, in the order of NEUTRAL_EXPRESSION",
        "EXPRESSION_SLOPES = {",
        *[f'    "{name}": {numbers(slopes[:, i])},' for i, name in enumerate(idealface.FEATURE_NAMES)],
        "}",
        "",
        "# The model face pictures the targets were moved towards",
        f"MODEL_PICTURES = {pictures!r}",
        "",
        "# For each gender and measurement: (target, tolerance). The target is the average of real faces moved towards the",
        "# model faces, the tolerance is how much real faces differ (one standard deviation)",
        "IDEALS = {",
    ]
    for gender, values in ideals.items():
        lines.append(f'    "{gender}": {{')
        lines += [f'        "{name}": ({target:.5g}, {tolerance:.5g}),' for name, (target, tolerance) in values.items()]
        lines.append("    },")
    lines += ["}", ""]
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    print(f"wrote {os.path.relpath(OUT, ROOT)}")
    for gender, people_count, model_count, pull in report:
        print(f"{gender}: {people_count} real faces, {model_count} model faces. Target moved from the average by (tolerances):")
        print("   " + ", ".join(f"{name} {p:+.1f}" for name, p in zip(idealface.FEATURE_NAMES, pull)))


if __name__ == "__main__":
    main()
