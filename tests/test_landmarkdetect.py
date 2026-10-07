import os
import shutil

import cv2
import numpy as np
import pytest

# These tests check things that should stay true however the score is tuned (for example "the boy face looks
# more like the boy model face" or "turning the picture doesn't change the face"), not exact score numbers.

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_FILE = os.path.join(REPO_DIR, "face_landmarker.task")
BOY_PICTURE = os.path.join(REPO_DIR, "perBoy.jpg")
GIRL_PICTURE = os.path.join(REPO_DIR, "perGirl.jpg")

if not os.path.isfile(MODEL_FILE):
    pytest.skip("face_landmarker.task is missing. It is part of the repo, put it next to the scripts to run "
                "these tests.", allow_module_level=True)

import facelayout  # noqa: E402  (imported after the check, so a missing model skips instead of failing)
import landmarkdetect  # noqa: E402
from landmarkdetect import FaceError, detect_face, getPerfs, landmark_detect, rate_face, read_image  # noqa: E402


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


def test_mirror_pairs_are_on_opposite_sides_of_the_model_faces():
    for gender in ("boy", "girl", "average"):
        x, y = (np.array(a) for a in getPerfs(gender))
        middle = (x[facelayout.position(168)] + x[facelayout.position(152)]) / 2
        for i, partner in enumerate(landmarkdetect.MIRROR_PAIRS):
            if partner != i:
                assert (x[i] - middle) * (x[partner] - middle) < 0
                assert abs(y[i] - y[partner]) < 0.1 * np.ptp(y)


def test_cheeks_and_face_width_landmarks_are_where_they_should_be():
    x, y = (np.array(a) for a in getPerfs("boy"))
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
    assert len(found.xList) == len(found.yList) == 162
    turn, tilt = found.angles
    assert abs(turn) < 5 and abs(tilt) < 10


def test_landmarks_are_in_the_pixels_of_the_picture_whatever_the_size_searched(boy_img):
    # The model faces were made from perBoy.jpg with this same detector, so its landmarks are the model face's.
    # Looking in a shrunk copy gives the landmarks in the full picture's pixels too
    modelX, modelY = getPerfs("boy")
    full = detect_face(boy_img)
    assert np.abs(np.array(full.xList) - modelX).max() < 0.2 and np.abs(np.array(full.yList) - modelY).max() < 0.2
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


# ---------- getPerfs and the average face ----------

def test_get_perfs_unknown_gender_raises_value_error():
    with pytest.raises(ValueError):
        getPerfs("dog")


def test_get_perfs_gives_a_landmark_list_for_every_model_face():
    for gender in landmarkdetect.MODEL_FACES:
        x, y = getPerfs(gender)
        assert len(x) == len(y) == 162
    assert getPerfs("Average") == getPerfs("average")


def test_the_average_face_is_halfway_between_the_boy_and_the_girl_face():
    boy, girl = (np.column_stack(getPerfs(g)) for g in ("boy", "girl"))
    average = np.column_stack(getPerfs("average"))
    # It is in the same place and size as the boy face (the girl face was lined up onto it)
    assert np.abs(average.mean(axis=0) - boy.mean(axis=0)).max() < 15
    assert landmarkdetect.eye_width(average) == pytest.approx(landmarkdetect.eye_width(boy), rel=0.05)
    # Equally far from both, and each of the two is half as far from it as from each other
    toBoy = landmarkdetect.shape_error(*getPerfs("average"), "boy")
    toGirl = landmarkdetect.shape_error(*getPerfs("average"), "girl")
    apart = landmarkdetect.shape_error(*getPerfs("girl"), "boy")
    assert toBoy == pytest.approx(toGirl, rel=0.1)
    assert toBoy == pytest.approx(apart / 2, rel=0.1)
    assert landmarkdetect.shape_error(*getPerfs("average"), "average") == pytest.approx(0, abs=1e-9)


def test_average_face_is_as_symmetric_as_the_model_faces():
    for gender in ("boy", "girl", "average"):
        assert landmarkdetect.symmetry(*getPerfs(gender)) > 0.6


def test_a_model_face_looks_like_itself_and_the_average_is_in_between(boy_img, girl_img):
    for img, own, other in ((boy_img, "boy", "girl"), (girl_img, "girl", "boy")):
        ownScore = rate_face(img, own)["score"]
        averageScore = rate_face(img, "average")["score"]
        otherScore = rate_face(img, other)["score"]
        assert ownScore > averageScore > otherScore


# ---------- rate_face ----------

def test_rate_face_result_has_the_expected_keys(boy_img):
    result = rate_face(boy_img, "boy")
    assert {"score", "clarity", "symmetry", "picture"} <= set(result)
    picture = result["picture"]
    assert isinstance(picture, np.ndarray)
    assert picture.ndim == 3 and picture.shape[2] == 3 and picture.dtype == np.uint8


def test_the_result_picture_has_the_landmarks_drawn_on_it(boy_img):
    picture = rate_face(boy_img, "boy")["picture"]
    changed = (picture != boy_img).any(axis=2)
    assert changed.sum() > 500
    # Dots on the face, nothing in the corners
    ys, xs = np.nonzero(changed)
    assert xs.min() > 20 and ys.min() > 20


def test_a_model_face_scores_itself_near_the_top(boy_img, girl_img):
    # Not exactly 10: a bit of lopsidedness and uneven skin always counts for something
    assert rate_face(boy_img, "boy")["score"] > 9.3
    assert rate_face(girl_img, "girl")["score"] > 9.3
    assert rate_face(boy_img, "boy")["shapeError"] < 0.001


def test_two_quite_different_faces_score_about_five(boy_img, girl_img):
    # This is what SCORE_K is tuned for
    assert 4.5 < rate_face(boy_img, "girl")["score"] < 5.5
    assert 4.5 < rate_face(girl_img, "boy")["score"] < 5.5


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
    assert rate_face(boy_img, "boy")["score"] > 9


# The scores come from the shape error (0 = identical). Two different real faces (perBoy against the girl model
# face and the other way round) are about 0.035 apart, and the same face in a slightly different picture 0.003 to
# 0.008 (this detector's landmarks wobble about half as much as dlib's did), so a change that is only landmark
# noise must stay well below the difference between two faces
SAME_FACE_TOLERANCE = 0.015


def shape_error_of(img, gender):
    xList, yList = landmark_detect(img)
    return landmarkdetect.shape_error(xList, yList, gender)


@pytest.mark.parametrize("picture, gender", [
    (BOY_PICTURE, "boy"),
    (BOY_PICTURE, "girl"),
    (GIRL_PICTURE, "boy"),
    (GIRL_PICTURE, "average"),
], ids=["perBoy-as-boy", "perBoy-as-girl", "perGirl-as-boy", "perGirl-as-average"])
@pytest.mark.parametrize("degrees", [10, -10])
def test_small_rotation_barely_changes_the_shape_error(picture, gender, degrees):
    # A head tilted a little to one side is the same face. This compares the shape error and not the score,
    # so a score that is capped at 10 can't hide a change.
    img = read_image(picture)
    assert abs(shape_error_of(rotate(img, degrees), gender) - shape_error_of(img, gender)) < SAME_FACE_TOLERANCE


@pytest.mark.parametrize("scale", [0.5, 2.0])
def test_picture_size_barely_changes_the_shape_error(boy_img, scale):
    resized = cv2.resize(boy_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    assert abs(shape_error_of(resized, "girl") - shape_error_of(boy_img, "girl")) < SAME_FACE_TOLERANCE


@pytest.mark.parametrize("picture, gender", [(BOY_PICTURE, "girl"), (GIRL_PICTURE, "boy")],
                         ids=["perBoy-as-girl", "perGirl-as-boy"])
def test_mirrored_picture_barely_changes_the_shape_error(picture, gender):
    # A mirrored selfie (like the camera preview) is the same face, so it should get the same shape error
    img = read_image(picture)
    assert abs(shape_error_of(cv2.flip(img, 1), gender) - shape_error_of(img, gender)) < SAME_FACE_TOLERANCE


def test_the_same_face_after_small_changes_still_scores_high(boy_img, girl_img):
    # Rotated, smaller, mirrored, darker, blurred and squeezed into a jpg: still the same face
    def jpeg(img):
        return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 40])[1], 1)

    changes = [lambda i: rotate(i, 5), lambda i: cv2.resize(i, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA),
               lambda i: cv2.flip(i, 1), lambda i: np.clip(i * 0.8, 0, 255).astype(np.uint8),
               lambda i: cv2.GaussianBlur(i, (0, 0), 1.5), jpeg]
    for img, gender in ((boy_img, "boy"), (girl_img, "girl")):
        scores = [rate_face(change(img), gender)["score"] for change in changes]
        assert min(scores) > 8 and np.mean(scores) > 8.8


def test_shape_error_of_the_model_face_itself_is_zero():
    for gender in landmarkdetect.MODEL_FACES:
        perfectX, perfectY = getPerfs(gender)
        assert landmarkdetect.shape_error(perfectX, perfectY, gender) == pytest.approx(0, abs=1e-9)
    assert landmarkdetect.score_from_error(0) == pytest.approx(10)


def test_the_boy_and_girl_model_faces_are_different_by_about_the_tuned_amount():
    # SCORE_K is tuned so that this difference is worth about 5 points
    err = landmarkdetect.shape_error(*getPerfs("girl"), "boy")
    assert landmarkdetect.score_from_error(err) == pytest.approx(5.1, abs=0.4)


def test_score_from_error_goes_down_as_the_error_goes_up():
    scores = [landmarkdetect.score_from_error(err) for err in (0, 0.02, 0.05, 0.1, 0.5)]
    assert scores == sorted(scores, reverse=True)
    assert all(0 <= s <= 10 for s in scores)


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


def test_a_lopsided_face_is_less_symmetric():
    x, y = (np.array(a, dtype=np.float64) for a in getPerfs("average"))
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
