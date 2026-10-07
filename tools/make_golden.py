"""Writes web/tests/golden.json: what the Python pipeline gives for a set of pictures, so the website's JS can be checked
against it (web/tests/index.html runs scoring.js and headpose.js on every case).

Run from the repo root whenever the Python scoring changes (and after tools/export_web_data.py):

    py -3.14 tools/make_golden.py

Each case is a (downscaled) picture, stored as a PNG, with the 478 landmarks and the head matrix that MediaPipe gave for
it, and everything the Python code works out from them. The pictures are rated in Python at that same size, so the JS
rates exactly the same pixels. The camera is never used.
"""
import base64
import json
import os
import sys

import cv2
import mediapipe as mp
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import export_web_data  # noqa: E402  (for the source hash, so the test can tell if facedata.js is out of date)
import facelayout  # noqa: E402
import headpose  # noqa: E402
import landmarkdetect as ld  # noqa: E402

OUT = os.path.join(ROOT, "web", "tests", "golden.json")
CASE_SIZE = 280  # long side of the stored pictures
REFERENCES = ld.MODEL_FACES
SOURCES = ("camera", "file")


def b64(data):
    return base64.b64encode(data).decode("ascii")


def png(img):
    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    assert ok
    return buf.tobytes()


def detect(img):
    # The same as landmarkdetect.detect_face, but it also keeps all 478 points (x, y and depth) and the matrix
    h, w = img.shape[:2]
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(ld.to_bgr(img), cv2.COLOR_BGR2RGB))
    result = ld.load_landmarker(False).detect(image)
    if not result.face_landmarks:
        return None
    faces = [np.array([(p.x * w - 0.5, p.y * h - 0.5, p.z * w) for p in face]) for face in result.face_landmarks]
    which = max(range(len(faces)), key=lambda n: np.ptp(faces[n][SUBSET, 0]) * np.ptp(faces[n][SUBSET, 1]))
    return np.round(faces[which], 3), np.round(np.array(result.facial_transformation_matrixes[which]), 6)


SUBSET = facelayout.SUBSET


def evaluate(img, points, matrix):
    # Everything the Python code works out for this picture and these landmarks
    xs, ys, zs = points[SUBSET, 0].tolist(), points[SUBSET, 1].tolist(), points[SUBSET, 2].tolist()
    angles = headpose.head_angles(matrix)
    problem = headpose.facing_problem(angles)
    size = ld.face_size(xs, ys, img.shape)
    # What rate_face refuses with, for a webcam picture and for a picked photo (the size is checked first)
    refusal = {}
    for source in SOURCES:
        if source == "camera" and size < ld.MIN_FACE_SIZE:
            refusal[source] = ld.SIZE_MESSAGES["far"]
        elif source == "file" and size < ld.MIN_FACE_SIZE_FILE:
            refusal[source] = ld.FILE_TOO_SMALL_MESSAGE
        elif problem is not None:
            refusal[source] = headpose.REFUSED_MESSAGES[problem]
        else:
            refusal[source] = None

    # The cheeks, like skin_clarity() does them
    nose = ld.nose_skin_brightness(img, xs, ys)
    cheeks, fractions = [], []
    for cheekPoints in (facelayout.LEFT_CHEEK_POINTS, facelayout.RIGHT_CHEEK_POINTS):
        square = ld.cheek_square(xs, ys, cheekPoints)
        fraction, mask = ld.cheek_inconsistencies(img, square, nose)
        if fraction is None:
            continue
        fractions.append(float(fraction))
        cheeks.append({"x1": square[0], "y1": square[1], "x2": square[2], "y2": square[3],
                       "w": int(mask.shape[1]), "h": int(mask.shape[0]), "fraction": float(fraction),
                       "mask": b64(np.packbits(mask.flatten()).tobytes())})
    clarity = ld.skin_clarity(img, img.copy(), xs, ys)
    clarity = None if clarity is None else float(clarity)
    assert (clarity is None) == (not fractions)

    sym = ld.symmetry(xs, ys)
    skinPenalty = 0.0 if clarity is None else ld.SKIN_PENALTY * (1 - clarity)
    symmetryPenalty = ld.SYMMETRY_PENALTY * (1 - sym)
    refs = {}
    for name in REFERENCES:
        errors = ld.model_errors(xs, ys, name, zs)
        err, modelFace = ld.closest_model_face(xs, ys, name, zs)
        score = float(np.clip(ld.score_from_error(err + skinPenalty + symmetryPenalty), 0, 10))
        refs[name] = {"shapeError": err, "score": score, "modelFace": modelFace, "modelFaceCount": len(errors),
                      "errors": [e for e, _ in errors]}
        for source in SOURCES:
            # Check against the real thing, which does the whole detection again
            if refusal[source] is None:
                full = ld.rate_face(img, name, source)
                assert abs(full["score"] - score) < 1e-3 and abs(full["shapeError"] - err) < 1e-4, (name, full["score"], score)
                assert full["modelFace"] == modelFace and full["modelFaceCount"] == len(errors)
            else:
                try:
                    ld.rate_face(img, name, source)
                except ld.FaceError as e:
                    assert str(e) == refusal[source], (str(e), refusal[source])
                else:
                    raise AssertionError("should have been refused")
    return {
        "angles": None if angles is None else {"turn": angles[0], "tilt": angles[1]},
        "problem": problem,
        "size": size, "refusal": refusal,
        "clarity": clarity, "cheeks": cheeks,
        "skinPenalty": skinPenalty, "symmetry": float(sym), "symmetryPenalty": float(symmetryPenalty),
        "refs": refs, "sharpness": headpose.sharpness(img),
    }


# ---------- The variants ----------

def shrink(img, size):
    h, w = img.shape[:2]
    f = size / max(h, w)
    return cv2.resize(img, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)


def rotated(img, degrees):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def cheek_squares(img):
    points, _ = detect(img)
    xs, ys = points[SUBSET, 0].tolist(), points[SUBSET, 1].tolist()
    return [ld.cheek_square(xs, ys, p) for p in (facelayout.LEFT_CHEEK_POINTS, facelayout.RIGHT_CHEEK_POINTS)]


def blotchy(img, count=7, radius=(0.06, 0.12)):
    # Red blotches on both cheeks, so there is uneven skin to find
    out = img.copy()
    rng = np.random.default_rng(1)
    for x1, y1, x2, y2 in cheek_squares(img):
        side = x2 - x1
        layer = out.copy()
        for _ in range(count):
            c = (int(rng.uniform(x1 + 0.15 * side, x2 - 0.15 * side)), int(rng.uniform(y1 + 0.15 * side, y2 - 0.15 * side)))
            cv2.circle(layer, c, max(2, int(side * rng.uniform(*radius))), (70, 70, 215), -1, cv2.LINE_AA)
        out = cv2.addWeighted(layer, 0.7, out, 0.3, 0)
    return out


def bearded(img):
    # The first (left) cheek in dark stubble speckles, the second one a solid dark beard
    out = img.copy()
    rng = np.random.default_rng(2)
    (a, b) = cheek_squares(img)
    x1, y1, x2, y2 = a
    region = out[y1:y2, x1:x2]
    speckle = rng.random(region.shape[:2]) < 0.4
    region[speckle] = (region[speckle] * 0.25).astype(np.uint8)
    x1, y1, x2, y2 = b
    out[y1:y2, x1:x2] = (out[y1:y2, x1:x2] * 0.12).astype(np.uint8)
    return out


def shadowed(img):
    # Light from the left: a smooth dark gradient over the face, which must not count as uneven skin
    h, w = img.shape[:2]
    ramp = np.linspace(1.0, 0.35, w, dtype=np.float32)[None, :, None]
    return np.clip(img.astype(np.float32) * ramp, 0, 255).astype(np.uint8)


def cut_left(img, share):
    # The picture cut so that the left cheek square is only partly inside (the share of its side that is cut off)
    x1, y1, x2, y2 = cheek_squares(img)[0]
    start = int(x1 + share * (x2 - x1))
    return img[:, max(start, 0):].copy()


def with_face_size(img, target):
    # The picture shrunk and put in the middle of a plain 320 x 320 square, so the face is `target` of the square's side
    points, _ = detect(img)
    ys = points[SUBSET, 1].tolist()
    f = target * 320 / abs(ys[facelayout.position(152)] - ys[facelayout.position(10)])
    small = cv2.resize(img, (round(img.shape[1] * f), round(img.shape[0] * f)), interpolation=cv2.INTER_AREA)
    canvas = np.full((320, 320, 3), (112, 120, 128), np.uint8)
    h, w = small.shape[:2]
    y0, x0 = max(0, (320 - h) // 2), max(0, (320 - w) // 2)
    cropped = small[max(0, (h - 320) // 2):, max(0, (w - 320) // 2):][:320, :320]
    canvas[y0:y0 + cropped.shape[0], x0:x0 + cropped.shape[1]] = cropped
    return canvas


def make_variants(name, base):
    variants = {"base": base, "rot": rotated(base, 18), "small": shrink(base, 200), "flip": cv2.flip(base, 1),
                "gray": cv2.cvtColor(cv2.cvtColor(base, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR),
                "edge": cut_left(base, 0.3), "edge-gone": cut_left(base, 0.7),
                "blotchy": blotchy(base), "mild": blotchy(base, 2, (0.04, 0.07)), "beard": bearded(base), "shadow": shadowed(base),
                "far": with_face_size(base, 0.22), "tiny": with_face_size(base, 0.11)}
    return variants


# ---------- The other tests: angles, trackers, recent frames, image operations ----------

def head_cases(real_matrices):
    cases = []

    def case(label, matrix):
        angles = headpose.head_angles(matrix)
        problem = headpose.facing_problem(angles)
        rows = np.asarray(matrix, dtype=float).tolist()
        # JSON has no NaN: null stands for it
        rows = [[None if v != v else v for v in row] for row in rows]
        cases.append({"label": label, "matrix": rows,
                      "angles": None if angles is None else {"turn": angles[0], "tilt": angles[1]},
                      "problem": problem, "message": None if problem is None else headpose.REFUSED_MESSAGES[problem]})

    for label, matrix in real_matrices:
        case(label, matrix)

    def rot(axis, degrees, size=1.0, shift=(1.5, -2.0, -40.0)):
        a = np.radians(degrees)
        c, s = np.cos(a), np.sin(a)
        r = {"y": [[c, 0, s], [0, 1, 0], [-s, 0, c]], "x": [[1, 0, 0], [0, c, -s], [0, s, c]]}[axis]
        m = np.eye(4)
        m[:3, :3] = np.array(r) * size
        m[:3, 3] = shift
        return m

    for deg in (-40, -26, -10, 0, 12, 24, 26, 33, 60):
        case(f"turn {deg}", rot("y", deg, 1.3))
    for deg in (-45, -30, -12, -10, 0, 14, 30, 38, 40, 55):
        case(f"tilt {deg}", rot("x", deg, 0.8))
    case("3x3 identity", np.eye(3))
    case("zero", np.zeros((4, 4)))
    case("nan", np.full((4, 4), np.nan))
    return cases


def tracker_cases():
    cases = []
    for seed, drift in ((11, 1.0), (12, 0.6)):
        rng = np.random.default_rng(seed)
        tracker = headpose.HeadTracker()
        steps, t = [], 100.0
        turn = tilt = 0.0
        for i in range(160):
            t += float(rng.choice([0.033, 0.033, 0.05, 0.4]))
            turn += rng.normal(0, 1.2) * drift + (0.7 * drift if (i // 40) % 2 == 0 else -0.7 * drift)
            tilt += rng.normal(0, 1.0) * drift + (0.5 * drift if (i // 55) % 2 == 0 else -0.6 * drift)
            turn, tilt = float(np.clip(turn, -30, 30)), float(np.clip(tilt, -30, 30))
            if i == 70:
                turn += 25  # one wild picture
            problem = tracker.update(round(turn, 3), round(tilt, 3), now=round(t, 4))
            steps.append({"t": round(t, 4), "turn": round(turn, 3), "tilt": round(tilt, 3), "problem": problem})
        cases.append(steps)
    return cases


def recent_cases():
    cases = []
    sharp_of = {}
    original = headpose.sharpness
    headpose.sharpness = lambda frame: sharp_of[frame]
    try:
        for seed in (21, 22):
            rng = np.random.default_rng(seed)
            recent = headpose.RecentFrames()
            steps, t = [], 50.0
            for i in range(120):
                t = round(t + float(rng.choice([0.03, 0.04, 0.2, 0.5])), 4)
                turn = round(float(rng.normal(0, 8)), 3)
                tilt = round(float(rng.normal(0, 6)), 3)
                # Some pictures face exactly as well as the one before (so only sharpness decides)
                if i % 7 == 3 and steps:
                    turn, tilt = steps[-1]["turn"], steps[-1]["tilt"]
                sharp = round(float(rng.uniform(5, 300)), 3)
                sharp_of[i] = sharp
                recent.add(i, turn, tilt, now=t)
                step = {"t": t, "turn": turn, "tilt": tilt, "sharp": sharp, "id": i}
                if i % 5 == 4:
                    step["bestAt"] = round(t + float(rng.choice([0.0, 0.3, 1.0])), 4)
                    best = recent.best(now=step["bestAt"])
                    step["best"] = best
                steps.append(step)
            cases.append(steps)
    finally:
        headpose.sharpness = original
    return cases


def flist(a):
    # float32 numbers as short decimals that read back as exactly the same float32
    return [float(np.format_float_scientific(v, unique=True)) for v in np.asarray(a, dtype=np.float32).flatten()]


def ops_cases(base):
    rng = np.random.default_rng(5)
    ops = {}
    # INTER_AREA on bytes: up (small cheek squares), down, and the 160 px wide sharpness copy
    crops = []
    for (w, h, dw, dh) in ((37, 29, 64, 52), (64, 56, 40, 33), (80, 70, 64, 56), (31, 31, 64, 64), (90, 60, 45, 30),
                           (30, 25, 30, 25), (48, 36, 16, 12)):
        y0, x0 = rng.integers(0, base.shape[0] - h), rng.integers(0, base.shape[1] - w)
        patch = np.ascontiguousarray(base[y0:y0 + h, x0:x0 + w]) if h <= base.shape[0] and w <= base.shape[1] else None
        if patch is None:
            patch = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        out = cv2.resize(patch, (dw, dh), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        crops.append({"w": w, "h": h, "dw": dw, "dh": dh, "bgr": patch.flatten().tolist(), "out": out.flatten().tolist(),
                      "gray": gray.flatten().tolist(), "lap": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                      "grayOut": cv2.resize(gray, (dw, dh), interpolation=cv2.INTER_AREA).flatten().tolist()})
    ops["area"] = crops
    # Gaussian blur, float32 3 channels, sigma 1 and 8, also on pictures smaller than the kernel
    blur = []
    for (w, h) in ((48, 40), (30, 24), (20, 17)):
        src = np.round(rng.random((h, w, 3)) ** 2, 5).astype(np.float32)
        for sigma in (1.0, 8.0):
            blur.append({"w": w, "h": h, "sigma": sigma, "src": flist(src),
                         "out": flist(cv2.GaussianBlur(src, (0, 0), sigma))})
    ops["blur"] = blur
    # Linear BGR to Lab, including values over 1 and in the dark
    lin = np.round(rng.random((300, 1, 3)), 5).astype(np.float32)
    lin[:50] *= 0.002
    lin[50:60] *= 1.4
    lin = lin.astype(np.float32)
    ops["lab"] = {"src": flist(lin), "out": flist(cv2.cvtColor(lin, cv2.COLOR_LBGR2Lab))}
    # Bilinear resizes of float32 maps
    linear = []
    for (w, h, dw, dh) in ((64, 64, 96, 80), (64, 56, 30, 37), (50, 50, 50, 50), (20, 25, 41, 31), (64, 64, 7, 9)):
        src = np.round(rng.random((h, w)) * 20, 4).astype(np.float32)
        linear.append({"w": w, "h": h, "dw": dw, "dh": dh, "src": flist(src),
                       "out": flist(cv2.resize(src, (dw, dh)))})
    ops["linear"] = linear
    return ops


def main():
    ld.load_landmarker(False)
    cases, real_matrices = [], []
    for source in ("perBoy", "perGirl"):
        original = ld.read_image(os.path.join(ROOT, source + ".jpg"))
        base = shrink(original, CASE_SIZE)
        for variant, img in make_variants(source, base).items():
            if source == "perGirl" and variant in ("small", "flip", "shadow", "beard", "tiny"):
                continue  # keeps golden.json small
            found = detect(img)
            if found is None:
                print(f"skipped {source}/{variant}: no face found")
                continue
            points, matrix = found
            gray = variant == "gray"
            case = {"name": f"{source}-{variant}", "width": img.shape[1], "height": img.shape[0],
                    "png": b64(png(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if gray else img)),
                    "points": points.tolist(), "matrix": matrix.tolist(), "expected": evaluate(img, points, matrix)}
            cases.append(case)
            real_matrices.append((case["name"], matrix))
            e = case["expected"]
            print(f"{case['name']:18} {img.shape[1]}x{img.shape[0]}  clarity {e['clarity']}  problem {e['problem']}  "
                  f"size {e['size']:.2f}  refusal {e['refusal']}  scores {[round(r['score'], 2) for r in e['refs'].values()]}  "
                  f"uneven {[round(c['fraction'], 3) for c in e['cheeks']]}  png {len(case['png']) // 1024} KB")

    golden = {"sourceHash": export_web_data.digest(), "cases": cases, "headCases": head_cases(real_matrices),
              "sizeCases": [{"size": s, "current": c, "problem": ld.size_problem(s, c)}
                            for s in (0.1, 0.29, 0.3, 0.31, 0.319, 0.321, 0.34, 0.35, 0.36, 0.369, 0.371, 0.6)
                            for c in (None, "near", "far")],
              "trackers": tracker_cases(),"recents": recent_cases(),
              "ops": ops_cases(shrink(ld.read_image(os.path.join(ROOT, "perGirl.jpg")), 400))}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(golden, f, separators=(",", ":"))
    print(f"wrote {os.path.relpath(OUT, ROOT)} ({os.path.getsize(OUT) / 1024:.0f} KB, {len(cases)} cases)")


if __name__ == "__main__":
    main()
