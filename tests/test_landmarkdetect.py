import os
import re
import shutil

import cv2
import numpy as np
import pytest

# These tests check things that should stay true however the score is tuned (for example "the boy model face scores
# higher as a boy" or "turning the picture doesn't change the face"), not exact score numbers.

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_FILE = os.path.join(REPO_DIR, "face_landmarker.task")
BOY_PICTURE = os.path.join(REPO_DIR, "perBoy.jpg")
GIRL_PICTURE = os.path.join(REPO_DIR, "perGirl.jpg")

if not os.path.isfile(MODEL_FILE):
    pytest.skip("face_landmarker.task is missing. It is part of the repo, put it next to the scripts to run "
                "these tests.", allow_module_level=True)

import facelayout  # noqa: E402  (imported after the check, so a missing model skips instead of failing)
import idealdata  # noqa: E402
import idealface  # noqa: E402
import landmarkdetect  # noqa: E402
from landmarkdetect import FaceError, detect_face, landmark_detect, rate_face, read_image  # noqa: E402


def model(found):
    # The 162 landmarks of a detection as float arrays x, y
    return np.array(found.xList, dtype=np.float64), np.array(found.yList, dtype=np.float64)


@pytest.fixture(scope="session", autouse=True)
def landmarker():
    # Loading the model takes a moment. landmarkdetect keeps it once it's loaded, so loading it here once means
    # the rest of the tests are quick
    return landmarkdetect.load_landmarker()


@pytest.fixture(scope="session")
def boy_img():
    return read_image(BOY_PICTURE)


@pytest.fixture(scope="session")
def girl_img():
    return read_image(GIRL_PICTURE)


@pytest.fixture(scope="session")
def boy_found(boy_img):
    # What detect_face finds in perBoy.jpg (shrunk like a rated picture)
    return detect_face(landmarkdetect.limit_size(boy_img))


@pytest.fixture(scope="session")
def girl_found(girl_img):
    return detect_face(landmarkdetect.limit_size(girl_img))


def rotate(img, degrees):
    # Turn the picture in the image plane, with a bigger canvas so no part of the face is cut off
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    newW, newH = int(h * sin + w * cos), int(h * cos + w * sin)
    matrix[0, 2] += newW / 2 - w / 2
    matrix[1, 2] += newH / 2 - h / 2
    return cv2.warpAffine(img, matrix, (newW, newH), borderMode=cv2.BORDER_REPLICATE)


# ---------- read_image ----------

def test_read_image_missing_path_raises_face_error(tmp_path):
    with pytest.raises(FaceError):
        read_image(str(tmp_path / "does_not_exist.jpg"))


def test_read_image_with_norwegian_letters_in_path(tmp_path):
    # cv2.imread fails on paths like this on Windows, read_image must not
    path = tmp_path / "bilde_æøå.jpg"
    shutil.copy(BOY_PICTURE, path)
    img = read_image(str(path))
    assert img.ndim == 3 and img.shape[2] == 3


def test_read_image_not_a_picture_raises_face_error(tmp_path):
    path = tmp_path / "not_a_picture.jpg"
    path.write_text("hello")
    with pytest.raises(FaceError):
        read_image(str(path))


# ---------- The model file ----------

@pytest.fixture
def other_model_file(monkeypatch):
    # Pretend the model file is somewhere else, with nothing loaded yet. The loaded landmarkers are put back after
    monkeypatch.setattr(landmarkdetect, "landmarkers", {})
    monkeypatch.setattr(landmarkdetect, "modelData", None)
    return lambda path: monkeypatch.setattr(landmarkdetect, "LANDMARKER_PATH", str(path))


def test_missing_model_file_says_what_to_do(tmp_path, other_model_file):
    other_model_file(tmp_path / "face_landmarker.task")
    with pytest.raises(FileNotFoundError, match="face_landmarker.task"):
        landmarkdetect.load_landmarker()


@pytest.mark.parametrize("content", [b"", b"this is not a model", os.urandom(5000)], ids=["empty", "text", "random"])
@pytest.mark.parametrize("live", [False, True], ids=["image", "video"])
def test_damaged_model_file_says_what_to_do(tmp_path, other_model_file, content, live):
    path = tmp_path / "face_landmarker.task"
    path.write_bytes(content)
    other_model_file(path)
    with pytest.raises(RuntimeError, match="damaged"):
        landmarkdetect.load_landmarker(live)
    # and it doesn't remember the damaged file: with the right file there it works
    other_model_file(MODEL_FILE)
    assert landmarkdetect.load_landmarker(live) is not None


def test_model_loads_from_a_folder_with_norwegian_letters(tmp_path, other_model_file):
    path = tmp_path / "mappe_æøå" / "face_landmarker.task"
    path.parent.mkdir()
    shutil.copy(MODEL_FILE, path)
    other_model_file(path)
    assert landmarkdetect.load_landmarker() is not None


# ---------- The 162-landmark layout ----------

def test_landmark_subset_is_162_distinct_mediapipe_landmarks():
    subset = facelayout.SUBSET
    assert len(subset) == len(set(subset)) == facelayout.POINT_COUNT == 162
    assert all(0 <= n < 478 for n in subset)
    assert sum(len(points) for points in facelayout.REGIONS.values()) == 162
    assert len(facelayout.POINT_WEIGHTS) == 162 and len(landmarkdetect.POINT_WEIGHTS) == 162


def test_region_weights_keep_the_proportions_of_the_old_layout():
    weights = landmarkdetect.POINT_WEIGHTS
    start = 0
    totals = {}
    for name, points in facelayout.REGIONS.items():
        totals[name] = weights[start:start + len(points)].sum()
        start += len(points)
    assert totals["jaw"] == pytest.approx(17 * 0.3)
    assert totals["brows"] == pytest.approx(10 * 0.6)
    assert totals["nose"] == pytest.approx(9) and totals["eyes"] == pytest.approx(12)
    assert totals["outerLips"] == pytest.approx(12 * 0.3)
    # The inner lips (the mouth opening) don't count at all
    assert totals["innerLips"] == 0 and (weights[:-20] > 0).all()


def test_mirror_pairs_swap_left_and_right():
    pairs = landmarkdetect.MIRROR_PAIRS
    assert sorted(pairs) == list(range(162))
    # Mirroring twice gives the same face back
    assert [pairs[p] for p in pairs] == list(range(162))
    # A landmark's partner is in the same region and has the same weight, and only the middle line is its own partner
    weights = landmarkdetect.POINT_WEIGHTS
    assert all(weights[i] == weights[pairs[i]] for i in range(162))
    middle = [facelayout.SUBSET[i] for i in range(162) if pairs[i] == i]
    assert sorted(middle) == [0, 1, 2, 4, 5, 6, 10, 13, 14, 17, 19, 94, 152, 168, 195, 197]
    start = 0
    for points in facelayout.REGIONS.values():
        assert all(start <= pairs[i] < start + len(points) for i in range(start, start + len(points)))
        start += len(points)


def test_mirror_pairs_are_on_opposite_sides_of_the_model_faces(boy_found, girl_found):
    for found in (boy_found, girl_found):
        x, y = model(found)
        middle = (x[facelayout.position(168)] + x[facelayout.position(152)]) / 2
        for i, partner in enumerate(landmarkdetect.MIRROR_PAIRS):
            if partner != i:
                assert (x[i] - middle) * (x[partner] - middle) < 0
                assert abs(y[i] - y[partner]) < 0.1 * np.ptp(y)


def test_cheeks_and_face_width_landmarks_are_where_they_should_be(boy_found):
    x, y = model(boy_found)
    left, right = facelayout.LEFT_CHEEK_POINTS, facelayout.RIGHT_CHEEK_POINTS
    assert len(left) == len(right) == 4
    assert x[left].mean() < x[facelayout.position(4)] < x[right].mean()
    assert all(landmarkdetect.MIRROR_PAIRS[a] == b for a, b in zip(left, right))
    # The cheeks are below the eyes and above the mouth
    eyes, mouth = y[list(facelayout.EYE_CORNERS)].mean(), y[facelayout.position(0)]
    assert eyes < y[left].mean() < mouth and eyes < y[right].mean() < mouth
    a, b = facelayout.FACE_WIDTH_POINTS
    assert x[a] < x[facelayout.position(4)] < x[b]
    assert landmarkdetect.MIRROR_PAIRS[a] == b
    top, bottom = facelayout.NOSE_BRIDGE_POINTS
    assert y[top] < y[bottom] < y[facelayout.position(4)]


# ---------- landmark_detect and detect_face ----------

def test_landmark_detect_finds_162_points(boy_img):
    found = landmark_detect(boy_img)
    assert found is not None
    xList, yList = found
    assert len(xList) == 162 and len(yList) == 162


def test_landmark_detect_blank_picture_finds_nothing():
    assert landmark_detect(np.zeros((300, 300, 3), np.uint8)) is None
    assert detect_face(np.zeros((300, 300, 3), np.uint8)) is None


def test_detect_face_gives_landmarks_and_head_direction(boy_img):
    found = detect_face(boy_img)
    assert len(found.xList) == len(found.yList) == len(found.zList) == 162
    # The nose tip is nearer the camera than the outer eye corners (depth grows away from the camera)
    assert found.zList[facelayout.position(4)] < found.zList[facelayout.position(33)]
    turn, tilt = found.angles
    # (a frontal face must sit well inside the preview tip's limits, HINT_TURN and HINT_TILT)
    assert abs(turn) < 3 and abs(tilt) < 7


def test_landmarks_are_in_the_pixels_of_the_picture_whatever_the_size_searched(boy_img):
    # Looking in a shrunk copy gives the landmarks in the full picture's pixels too
    full = detect_face(boy_img)
    modelX, modelY = model(full)
    small = detect_face(boy_img, detectScale=0.5)
    assert np.abs(np.array(small.xList) - modelX).max() < 3 and np.abs(np.array(small.yList) - modelY).max() < 3


def test_landmarks_follow_the_picture_when_it_is_moved(boy_img):
    # Padding the picture on the left and top moves every landmark by exactly that much
    padded = cv2.copyMakeBorder(boy_img, 100, 0, 150, 0, cv2.BORDER_REPLICATE)
    before, after = detect_face(boy_img), detect_face(padded)
    assert np.abs(np.array(after.xList) - np.array(before.xList) - 150).mean() < 1.5
    assert np.abs(np.array(after.yList) - np.array(before.yList) - 100).mean() < 1.5


def test_the_biggest_face_is_the_one_found(boy_img, girl_img):
    # The boy is shrunk to be much smaller than the girl, the two are put next to each other
    small_boy = cv2.resize(boy_img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    canvas = np.full((1000, 1100, 3), 128, np.uint8)
    canvas[20:20 + girl_img.shape[0], 20:20 + girl_img.shape[1]] = girl_img
    canvas[20:20 + small_boy.shape[0], 700:700 + small_boy.shape[1]] = small_boy
    found = detect_face(canvas)
    assert np.mean(found.xList) < 600
    # and the other way round
    canvas[:] = 128
    big_boy = cv2.resize(boy_img, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_CUBIC)
    small_girl = cv2.resize(girl_img, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
    canvas[10:10 + big_boy.shape[0], 10:10 + big_boy.shape[1]] = big_boy
    canvas[550:550 + small_girl.shape[0], 700:700 + small_girl.shape[1]] = small_girl
    assert np.mean(detect_face(canvas).xList) < 600


def test_live_mode_follows_a_series_of_pictures(boy_img, girl_img):
    # The live landmarker is a separate one that needs ever-increasing times. Many pictures in a quick row must work,
    # also when the face changes, and give about the same as looking at one picture
    scale = 320 / boy_img.shape[1]
    still = detect_face(boy_img, detectScale=scale)
    for _ in range(10):
        live = detect_face(boy_img, detectScale=scale, live=True)
    assert np.abs(np.array(live.xList) - still.xList).max() < 3 and np.abs(np.array(live.yList) - still.yList).max() < 3
    assert np.allclose(live.angles, still.angles, atol=2)
    assert detect_face(np.zeros((480, 640, 3), np.uint8), live=True) is None
    assert detect_face(girl_img, live=True) is not None
    assert landmarkdetect.load_landmarker(True) is not landmarkdetect.load_landmarker(False)


def test_gray_and_see_through_pictures_work(boy_img):
    gray = cv2.cvtColor(boy_img, cv2.COLOR_BGR2GRAY)
    assert detect_face(gray) is not None
    assert detect_face(cv2.cvtColor(boy_img, cv2.COLOR_BGR2BGRA)) is not None


def test_a_small_picture_is_found_too(boy_img):
    small = cv2.resize(boy_img, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
    assert min(small.shape[:2]) < 200
    assert detect_face(small) is not None


def test_model_face_is_facing_the_camera(boy_img, girl_img):
    for img in (boy_img, girl_img):
        assert landmarkdetect.facing_problem(detect_face(img).angles) is None


# ---------- The ideal ----------

@pytest.mark.parametrize("gender", ["dog", "average", ""])
def test_an_unknown_gender_raises_value_error(boy_img, gender):
    with pytest.raises(ValueError):
        rate_face(boy_img, gender)


def test_the_ideal_has_every_measurement_for_every_gender():
    for gender in landmarkdetect.MODEL_FACES:
        ideal = idealdata.IDEALS[gender]
        assert list(ideal) == idealface.FEATURE_NAMES
        assert all(tolerance > 0 for _, tolerance in ideal.values())
    assert list(idealdata.EXPRESSION_SLOPES) == idealface.FEATURE_NAMES
    assert all(len(slopes) == len(idealface.EXPRESSIONS) for slopes in idealdata.EXPRESSION_SLOPES.values())
    assert np.shape(idealdata.REFERENCE) == (len(idealface.ANCHORS), 3)
    assert {region for _, region, _, _ in idealface.FEATURES} == set(idealface.REGIONS)


def test_every_model_face_picture_is_in_the_ideal():
    # Adding a picture (perBoy6.jpg, ...) needs tools/make_ideal_face.py to be run again
    for gender, start in (("boy", "perBoy"), ("girl", "perGirl")):
        pictures = sorted(name for name in os.listdir(REPO_DIR) if re.fullmatch(start + r"\d*\.jpg", name))
        assert sorted(idealdata.MODEL_PICTURES[gender]) == pictures


def turned_in_3d(points, tilt, turn, roll=0.0, scale=1.0, shift=(0.0, 0.0, 0.0)):
    # The landmarks (n x 3) of the head tilted, turned and rolled in 3D by that many degrees, resized and moved
    a, b, c = np.radians(tilt), np.radians(turn), np.radians(roll)
    tilting = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    turning = np.array([[np.cos(b), 0, -np.sin(b)], [0, 1, 0], [np.sin(b), 0, np.cos(b)]])
    rolling = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    points = np.asarray(points, dtype=np.float64)
    middle = points.mean(axis=0)
    return scale * (points - middle) @ (rolling @ tilting @ turning).T + middle + shift


@pytest.mark.parametrize("tilt, turn, roll", [(20, 0, 0), (-20, 0, 0), (0, 10, 0), (15, -8, 5), (0, 0, 30)])
def test_the_measurements_dont_change_when_the_head_is_turned(boy_found, tilt, turn, roll):
    # The face is turned back in 3D (with the landmarks' depth) before it is measured, so a head tilted, turned or
    # rolled, nearer or further away, measures the same
    before = idealface.measure(boy_found.points)
    after = idealface.measure(turned_in_3d(boy_found.points, tilt, turn, roll, scale=1.3, shift=(40, -25, 7)))
    assert after == pytest.approx(before, abs=1e-9)


def test_the_measurements_are_the_same_for_a_mirrored_face(boy_found):
    # The landmarks of the mirror image (x the other way round, every landmark swapped with its partner) measure the
    # same, since each measurement takes both sides
    points = np.asarray(boy_found.points)
    mirrored = points * [-1.0, 1.0, 1.0]
    partner = np.arange(len(points))
    for a, b in facelayout.MIRROR_PAIRS_MEDIAPIPE + [(129, 358)]:
        partner[a], partner[b] = b, a
    assert idealface.measure(mirrored[partner]) == pytest.approx(idealface.measure(points), abs=1e-9)


def test_the_ideal_face_scores_ten():
    # A face measuring exactly the targets is the ideal face
    for gender in landmarkdetect.MODEL_FACES:
        targets = {name: target for name, (target, _) in idealdata.IDEALS[gender].items()}
        error, regions, deviations = idealface.ideal_errors(targets, gender)
        assert error == pytest.approx(0) and all(e == pytest.approx(0) for e in regions.values())
        assert all(z == 0 for z in deviations.values())
        assert landmarkdetect.score_from_error(error) == pytest.approx(10)


def test_each_region_only_counts_its_own_measurements():
    for gender in landmarkdetect.MODEL_FACES:
        targets = {name: target for name, (target, _) in idealdata.IDEALS[gender].items()}
        for name, region, _, way in idealface.FEATURES:
            changed = dict(targets)
            # three tolerances the way that counts (less than the target for the one-sided ones)
            changed[name] -= 3 * idealdata.IDEALS[gender][name][1] * (1 if way >= 0 else -1)
            _, regions, deviations = idealface.ideal_errors(changed, gender)
            assert deviations[name] == pytest.approx(-3 if way >= 0 else 3)
            assert regions[region] > 0.5
            assert all(e == pytest.approx(0) for r, e in regions.items() if r != region)


def test_past_the_target_is_fine_for_the_one_sided_measurements():
    # A sharper jaw, bigger eyes or a more upward eye tilt than the target doesn't lower the score
    for gender in landmarkdetect.MODEL_FACES:
        targets = {name: target for name, (target, _) in idealdata.IDEALS[gender].items()}
        for name, _, _, way in idealface.FEATURES:
            changed = dict(targets)
            changed[name] += 2 * idealdata.IDEALS[gender][name][1] * (way or 1)
            error = idealface.ideal_errors(changed, gender)[0]
            assert (error == pytest.approx(0)) == (way != 0)


def test_one_odd_measurement_cant_take_the_whole_score(boy_found):
    values = idealface.measure(boy_found.points)
    values["noseWidth"] = 100.0
    assert idealface.deviations(values, "boy")["noseWidth"] == idealface.MAX_DEVIATION


def rounded_jaw(points, rounds=3):
    # The jaw line made rounder: each point of the outline between the ear and the chin is moved part of the way to
    # the middle of its neighbours a few times, which smooths away the corner (the ends stay where they are)
    points = np.array(points, dtype=np.float64)
    for chain in (idealface.JAW_LEFT, idealface.JAW_RIGHT):
        for _ in range(rounds):
            inner = points[chain[1:-1]]
            points[chain[1:-1]] = 0.4 * inner + 0.3 * (points[chain[:-2]] + points[chain[2:]])
    return points


def test_a_sharp_jaw_scores_higher_than_a_rounded_one(boy_found, girl_found):
    for found, gender in ((boy_found, "boy"), (girl_found, "girl")):
        sharp = idealface.measure(found.points)
        rounded = idealface.measure(rounded_jaw(found.points))
        assert rounded["jawSharpness"] < sharp["jawSharpness"] - 0.03
        assert (landmarkdetect.ideal_error(rounded_jaw(found.points), gender)[1]["jaw"]
                > landmarkdetect.ideal_error(found.points, gender)[1]["jaw"] + 0.3)


def test_a_rounder_face_scores_lower(boy_found):
    # The same face 12% wider: a round face is short and wide
    points = np.asarray(boy_found.points)
    wide = points * [1.12, 1.0, 1.12]
    assert idealface.measure(wide)["faceLength"] < idealface.measure(points)["faceLength"] * 0.92
    assert landmarkdetect.ideal_error(wide, "boy")[1]["jaw"] > landmarkdetect.ideal_error(points, "boy")[1]["jaw"] + 0.5
    assert landmarkdetect.ideal_error(wide, "boy")[0] > landmarkdetect.ideal_error(points, "boy")[0] + 0.3


# ---------- The expression ----------

def blendshapes(smile=0.0, squint=0.3, mouthOpen=0.0, upperLipUp=0.0):
    return {"mouthSmileLeft": smile, "mouthSmileRight": smile, "eyeSquintLeft": squint, "eyeSquintRight": squint,
            "jawOpen": mouthOpen, "mouthUpperUpLeft": upperLipUp, "mouthUpperUpRight": upperLipUp}


def test_detect_face_gives_all_landmarks_and_the_expression(boy_img):
    found = detect_face(boy_img)
    assert np.shape(found.points) == (478, 3)
    assert np.allclose(np.asarray(found.points)[facelayout.SUBSET, 0], found.xList)
    assert {"mouthSmileLeft", "jawOpen", "eyeSquintRight"} <= set(found.blendshapes)
    assert all(0 <= value <= 1 for value in found.blendshapes.values())


def test_the_expression_is_taken_off_the_measurements(boy_found):
    points = boy_found.points
    plain = idealface.measure(points)
    # A typical neutral face is not corrected at all, and without blendshapes nothing is
    neutral = dict(zip(idealface.EXPRESSIONS, idealdata.NEUTRAL_EXPRESSION))
    assert idealface.neutral_measures(points, blendshapes(**neutral)) == pytest.approx(plain)
    assert idealface.neutral_measures(points, None) == plain
    # A smile is taken off with the slopes the tool worked out
    smiling = idealface.neutral_measures(points, blendshapes(**dict(neutral, smile=neutral["smile"] + 0.4)))
    for name in idealface.FEATURE_NAMES:
        assert smiling[name] == pytest.approx(plain[name] - 0.4 * idealdata.EXPRESSION_SLOPES[name][0])


def with_blendshapes(monkeypatch, shapes):
    # detect_face as it is, but with made-up blendshapes
    real = detect_face
    monkeypatch.setattr(landmarkdetect, "detect_face", lambda img, *a, **k: real(img, *a, **k)._replace(blendshapes=shapes))


def test_a_big_smile_or_an_open_mouth_is_refused(boy_img, monkeypatch):
    for shapes, message in ((blendshapes(smile=0.8), "no smile"), (blendshapes(mouthOpen=0.5), "Close your mouth")):
        with_blendshapes(monkeypatch, shapes)
        for source in ("camera", "file"):
            with pytest.raises(FaceError, match=message):
                rate_face(boy_img, "boy", source)
    # a slight smile is still rated
    with_blendshapes(monkeypatch, blendshapes(smile=0.3))
    assert 0 <= rate_face(boy_img, "boy")["score"] <= 10


def test_the_expression_correction_is_small_for_a_slight_smile(boy_img, monkeypatch):
    # Here only the blendshapes change, so the score moves by just the correction a slight smile gets: it must stay
    # small, or a smile that the landmarks barely show would still change the score
    scores = []
    for smile in (0.0, 0.2, 0.4):
        with_blendshapes(monkeypatch, blendshapes(smile=smile))
        scores.append(rate_face(boy_img, "boy")["score"])
    assert max(scores) - min(scores) < 1.0


def test_expression_problem_limits():
    problem = idealface.expression_problem
    assert problem(None) is None and problem({}) is None
    assert problem(blendshapes()) is None
    assert problem(blendshapes(smile=0.4)) is None and problem(blendshapes(smile=0.4), hint=True) == "smile"
    assert problem(blendshapes(smile=0.6)) == "smile"
    assert problem(blendshapes(mouthOpen=0.2)) is None and problem(blendshapes(mouthOpen=0.2), hint=True) == "mouthOpen"
    assert problem(blendshapes(smile=0.9, mouthOpen=0.9)) == "mouthOpen"
    assert idealface.HINT_SMILE < idealface.MAX_SMILE and idealface.HINT_MOUTH_OPEN < idealface.MAX_MOUTH_OPEN


def test_the_expression_tip_is_steady():
    tracker = idealface.ExpressionTracker()
    t = 100.0
    for _ in range(5):
        t += 0.05
        assert tracker.update(blendshapes(smile=0.1), t) is None
    # one odd picture doesn't bring the tip
    t += 0.05
    assert tracker.update(blendshapes(smile=0.9), t) is None
    for _ in range(12):
        t += 0.05
        tracker.update(blendshapes(smile=0.5), t)
    assert tracker.problem == "smile"
    # just under the limit it stays (the margin), clearly under it goes
    for _ in range(12):
        t += 0.05
        tracker.update(blendshapes(smile=idealface.HINT_SMILE - 0.02), t)
    assert tracker.problem == "smile"
    for _ in range(12):
        t += 0.05
        tracker.update(blendshapes(smile=0.1), t)
    assert tracker.problem is None
    tracker.update(blendshapes(smile=0.9), t + 0.05)
    tracker.reset()
    assert tracker.problem is None and tracker.update(None, t + 0.1) is None


# ---------- rate_face ----------

def test_rate_face_result_has_the_expected_keys(boy_img):
    result = rate_face(boy_img, "boy")
    assert {"score", "clarity", "symmetry", "picture", "regions", "deviations", "shapeError"} <= set(result)
    assert list(result["deviations"]) == idealface.FEATURE_NAMES
    # The factors say how many times smaller each penalty made the score
    assert 0 < result["skinFactor"] <= 1 and 0 < result["symmetryFactor"] <= 1
    picture = result["picture"]
    assert isinstance(picture, np.ndarray)
    assert picture.ndim == 3 and picture.shape[2] == 3 and picture.dtype == np.uint8


# ---------- the score of each region ----------

def test_the_regions_are_the_ones_shown(boy_img):
    regions = rate_face(boy_img, "boy")["regions"]
    assert list(regions) == list(idealface.REGIONS) == ["jaw", "brows", "nose", "eyes", "outerLips"]
    assert all(0 <= score <= 10 for score in regions.values())


def test_region_scores_use_the_same_formula_as_the_total():
    scores = landmarkdetect.region_scores({"jaw": 0.0, "eyes": landmarkdetect.SCORE_MID, "nose": 3.0})
    assert scores["jaw"] == pytest.approx(10) and scores["eyes"] == pytest.approx(5) and scores["nose"] < 1


def test_the_result_picture_has_the_landmarks_drawn_on_it(boy_img):
    picture = rate_face(boy_img, "boy")["picture"]
    changed = (picture != boy_img).any(axis=2)
    assert changed.sum() > 500
    # Dots on the face, nothing in the corners
    ys, xs = np.nonzero(changed)
    assert xs.min() > 20 and ys.min() > 20


def test_the_model_faces_score_high(boy_img, girl_img):
    # The ideal is moved towards the model faces, so they score well above an average face (5), though not 10: the
    # ideal is a mix of real faces and all the model faces
    assert rate_face(boy_img, "boy")["score"] > 8.5
    assert rate_face(girl_img, "girl")["score"] > 7


def test_boy_face_scores_higher_as_boy(boy_img):
    assert rate_face(boy_img, "boy")["score"] > rate_face(boy_img, "girl")["score"]


def test_girl_face_scores_higher_as_girl(girl_img):
    assert rate_face(girl_img, "girl")["score"] > rate_face(girl_img, "boy")["score"]


@pytest.mark.parametrize("image", [
    np.zeros((400, 400, 3), np.uint8),
    np.full((400, 400, 3), 255, np.uint8),
    np.random.default_rng(0).integers(0, 256, (400, 400, 3), dtype=np.uint8),
], ids=["black", "white", "noise"])
def test_rate_face_without_a_face_raises_face_error(image):
    with pytest.raises(FaceError):
        rate_face(image, "boy")


def test_score_symmetry_and_clarity_are_in_range(boy_img, girl_img):
    for img in (boy_img, girl_img):
        for gender in landmarkdetect.MODEL_FACES:
            result = rate_face(img, gender)
            assert 0 <= result["score"] <= 10
            assert 0 <= result["symmetry"] <= 1
            assert result["clarity"] is None or 0 <= result["clarity"] <= 1


def test_a_head_turned_far_is_refused(boy_img, monkeypatch):
    # (a picture of a really turned head is needed to see this for real, so the angles are made up here)
    monkeypatch.setattr(landmarkdetect, "head_angles", lambda matrix: (40.0, 0.0))
    with pytest.raises(FaceError, match="turned"):
        rate_face(boy_img, "boy")
    monkeypatch.setattr(landmarkdetect, "head_angles", lambda matrix: (0.0, -40.0))
    with pytest.raises(FaceError, match="up"):
        rate_face(boy_img, "boy")


def test_a_picture_without_head_angles_is_still_rated(boy_img, monkeypatch):
    monkeypatch.setattr(landmarkdetect, "head_angles", lambda matrix: None)
    assert rate_face(boy_img, "boy")["score"] > 8.5


# The scores come from the ideal error (0 = the ideal face, a typical face about 1, see SCORE_MID). MediaPipe's
# landmarks wobble a little from picture to picture, which moves the error by about 0.1, while two different faces
# are about 1 apart, so a change that is only landmark noise must stay well under that
SAME_FACE_TOLERANCE = 0.3


def ideal_error_of(img, gender):
    found = detect_face(img)
    return landmarkdetect.ideal_error(found.points, gender, found.blendshapes)[0]


@pytest.mark.parametrize("picture, gender", [
    (BOY_PICTURE, "boy"),
    (BOY_PICTURE, "girl"),
    (GIRL_PICTURE, "boy"),
    (GIRL_PICTURE, "girl"),
], ids=["perBoy-as-boy", "perBoy-as-girl", "perGirl-as-boy", "perGirl-as-girl"])
@pytest.mark.parametrize("degrees", [10, -10])
def test_small_rotation_barely_changes_the_ideal_error(picture, gender, degrees):
    # A head tilted a little to one side is the same face. This compares the error and not the score,
    # so a score that is capped at 10 can't hide a change.
    img = read_image(picture)
    assert abs(ideal_error_of(rotate(img, degrees), gender) - ideal_error_of(img, gender)) < SAME_FACE_TOLERANCE


@pytest.mark.parametrize("scale", [0.5, 2.0])
def test_picture_size_barely_changes_the_ideal_error(boy_img, scale):
    resized = cv2.resize(boy_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    assert abs(ideal_error_of(resized, "girl") - ideal_error_of(boy_img, "girl")) < SAME_FACE_TOLERANCE


@pytest.mark.parametrize("picture, gender", [(BOY_PICTURE, "boy"), (BOY_PICTURE, "girl"), (GIRL_PICTURE, "boy"),
                                             (GIRL_PICTURE, "girl")],
                         ids=["perBoy-as-boy", "perBoy-as-girl", "perGirl-as-boy", "perGirl-as-girl"])
def test_mirrored_picture_barely_changes_the_ideal_error(picture, gender):
    # A mirrored selfie (like the camera preview) is the same face, so it should get about the same error
    img = read_image(picture)
    assert abs(ideal_error_of(cv2.flip(img, 1), gender) - ideal_error_of(img, gender)) < SAME_FACE_TOLERANCE


def test_the_same_face_after_small_changes_still_scores_about_the_same(boy_img, girl_img):
    # Rotated, smaller, mirrored, darker, blurred and squeezed into a jpg: still the same face
    def jpeg(img):
        return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 40])[1], 1)

    changes = [lambda i: rotate(i, 5), lambda i: cv2.resize(i, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA),
               lambda i: cv2.flip(i, 1), lambda i: np.clip(i * 0.8, 0, 255).astype(np.uint8),
               lambda i: cv2.GaussianBlur(i, (0, 0), 1.5), jpeg]
    for img, gender in ((boy_img, "boy"), (girl_img, "girl")):
        original = rate_face(img, gender)["score"]
        scores = np.array([rate_face(change(img), gender)["score"] for change in changes])
        assert np.abs(scores - original).max() < 1.5 and np.abs(scores - original).mean() < 0.7


def test_score_from_error_goes_down_as_the_error_goes_up():
    scores = [landmarkdetect.score_from_error(err) for err in (0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2, 3.5)]
    assert all(a > b for a, b in zip(scores, scores[1:]))
    assert all(0 <= s <= 10 for s in scores)
    assert landmarkdetect.score_from_error(0) == pytest.approx(10)


def test_the_score_spreads_ordinary_faces_out():
    mid = landmarkdetect.SCORE_MID
    score = landmarkdetect.score_from_error
    assert score(mid) == pytest.approx(5)
    # Ordinary faces are about 0.7 to 1.6 from the ideal, which spreads over most of the scale
    assert score(0.7) > 7.5 and score(1.6) < 3
    # while very uneven skin takes about a quarter off a middling score
    assert 3.5 < score(mid + landmarkdetect.SKIN_PENALTY) < 4.5


@pytest.mark.parametrize("tilt, turn", [(20, 0), (-20, 0), (0, 10), (15, -8)])
def test_a_tilted_or_turned_head_is_turned_back_before_measuring(boy_found, tilt, turn):
    # Seen flat, 20 degrees of tilt changes the face more than going to a different face, but with the landmarks'
    # depth the face is turned back first, so the error is the same
    moved = turned_in_3d(boy_found.points, tilt, turn)
    assert landmarkdetect.ideal_error(moved, "boy")[0] == pytest.approx(
        landmarkdetect.ideal_error(boy_found.points, "boy")[0], abs=1e-9)


# ---------- alignment and symmetry ----------

def test_align_undoes_a_move_turn_and_resize():
    rng = np.random.default_rng(1)
    points = rng.uniform(0, 300, (162, 2))
    angle = np.radians(25)
    turn = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    moved = 1.7 * points @ turn.T + [40, -15]
    assert np.allclose(landmarkdetect.align(moved, points), points)


def test_symmetry_ignores_a_mirrored_picture(boy_img):
    xList, yList = landmark_detect(boy_img)
    # The same landmarks mirrored left to right are exactly as symmetric
    mirroredX = [-x for x in xList]
    assert landmarkdetect.symmetry(mirroredX, yList) == pytest.approx(landmarkdetect.symmetry(xList, yList), abs=1e-6)


def test_a_lopsided_face_is_less_symmetric(boy_found):
    x, y = model(boy_found)
    eye = landmarkdetect.eye_width(np.column_stack([x, y]))
    normal = landmarkdetect.symmetry(x, y)
    # One eye (and iris) moved down by 12% of the eye width, so the two sides are different
    lopsided = y.copy()
    for part in facelayout.EYES[:16] + facelayout.EYES[-10:-5]:
        lopsided[facelayout.position(part)] += 0.12 * eye
    assert landmarkdetect.symmetry(x, lopsided) < normal - 0.15
    # and a face that is made exactly symmetric is perfect
    points = np.column_stack([x, y])
    mirrored = points[landmarkdetect.MIRROR_PAIRS] * [-1.0, 1.0]
    symmetric = (points + landmarkdetect.align(mirrored, points)) / 2
    assert landmarkdetect.symmetry(symmetric[:, 0], symmetric[:, 1]) > 0.97


def test_huge_picture_still_rates(girl_img):
    h, w = girl_img.shape[:2]
    scale = 3000 / max(h, w)
    huge = cv2.resize(girl_img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_CUBIC)
    result = rate_face(huge, "girl")
    assert np.isfinite(result["score"])


# ---------- Distance to the camera ----------

def on_frame(img, share, frame=(720, 1280)):
    # The picture shrunk so the face is `share` of the frame height, on a gray frame the size of a webcam picture
    h, w = frame
    found = detect_face(img)
    top, chin = landmarkdetect.FACE_HEIGHT_POINTS
    scale = share * min(h, w) / (found.yList[chin] - found.yList[top])
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w, 3), 120, np.uint8)
    sh, sw = small.shape[:2]
    y0, x0 = (h - sh) // 2, (w - sw) // 2
    canvas[max(y0, 0):y0 + sh, max(x0, 0):x0 + sw] = small[max(-y0, 0):max(-y0, 0) + h, max(-x0, 0):max(-x0, 0) + w]
    return canvas


def test_face_size_is_the_face_height_share_of_the_shorter_side(boy_img):
    found = detect_face(boy_img)
    h, w = boy_img.shape[:2]
    size = landmarkdetect.face_size(found.xList, found.yList, boy_img.shape)
    assert size == pytest.approx(0.529 * h / min(h, w), abs=0.02)
    # the same face in a bigger picture is a smaller share
    assert landmarkdetect.face_size(found.xList, found.yList, (h * 2, w * 2, 3)) == pytest.approx(size / 2)


def test_the_distance_limits_are_ordered_as_intended():
    assert landmarkdetect.MIN_FACE_SIZE_FILE < landmarkdetect.MIN_FACE_SIZE < landmarkdetect.HINT_FACE_SIZE


def test_a_face_that_is_too_far_from_the_camera_is_refused(boy_img):
    for share in (0.2, 0.26):
        with pytest.raises(FaceError, match="Move closer to the camera."):
            rate_face(on_frame(boy_img, share), "boy", "camera")


def test_a_face_close_enough_to_the_camera_is_rated(boy_img, girl_img):
    for img, gender in ((boy_img, "boy"), (girl_img, "girl")):
        assert rate_face(on_frame(img, 0.34), gender, "camera")["score"] > 5
    assert rate_face(boy_img, "boy", "camera")["score"] > 8.5


def test_a_picked_photo_may_have_a_smaller_face_but_not_a_tiny_one(boy_img):
    # A face a fifth of the picture's height is a normal portrait photo, but too far away for the webcam
    assert rate_face(on_frame(boy_img, 0.2), "boy", "file")["score"] > 5
    with pytest.raises(FaceError, match="too small"):
        rate_face(on_frame(boy_img, 0.1, frame=(1280, 1280)), "boy", "file")


def test_distance_tip_has_hysteresis():
    size_problem = landmarkdetect.size_problem
    assert size_problem(0.25) == "far" and size_problem(0.32) == "near" and size_problem(0.4) is None
    # Once showing, a tip stays until the face is a margin past the limit
    assert size_problem(0.31, "far") == "far" and size_problem(0.33, "far") == "near"
    assert size_problem(0.36, "near") == "near" and size_problem(0.38, "near") is None
    assert size_problem(0.36) is None
