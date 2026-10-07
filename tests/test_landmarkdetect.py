import os
import shutil

import cv2
import numpy as np
import pytest

# These tests check things that should stay true however the score is tuned (for example "the boy face looks
# more like the boy model face" or "turning the picture doesn't change the face"), not exact score numbers.

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_FILE = os.path.join(REPO_DIR, "shape_predictor_68_face_landmarks.dat")
BOY_PICTURE = os.path.join(REPO_DIR, "perBoy.jpg")
GIRL_PICTURE = os.path.join(REPO_DIR, "perGirl.jpg")

if not os.path.isfile(MODEL_FILE):
    pytest.skip("shape_predictor_68_face_landmarks.dat is missing. Download it (see README) and put it "
                "next to the scripts to run these tests.", allow_module_level=True)

import landmarkdetect  # noqa: E402  (imported after the check, so a missing model skips instead of failing)
from landmarkdetect import FaceError, facing_problem, getPerfs, landmark_detect, rate_face, read_image  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def predictor():
    # Loading the ~100 MB model takes a second or two. landmarkdetect keeps it once it's loaded,
    # so loading it here once means the rest of the tests are quick
    return landmarkdetect.load_predictor()


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


# ---------- landmark_detect and facing_problem ----------

def test_landmark_detect_finds_68_points(boy_img):
    found = landmark_detect(boy_img)
    assert found is not None
    xList, yList = found
    assert len(xList) == 68 and len(yList) == 68


def test_landmark_detect_blank_picture_finds_nothing():
    assert landmark_detect(np.zeros((300, 300, 3), np.uint8)) is None


def test_model_face_is_facing_the_camera(boy_img):
    xList, yList = landmark_detect(boy_img)
    assert facing_problem(xList, yList, boy_img.shape) is None


# ---------- getPerfs ----------

def test_get_perfs_unknown_gender_raises_value_error():
    with pytest.raises(ValueError):
        getPerfs("dog")


# ---------- rate_face ----------

def test_rate_face_result_has_the_expected_keys(boy_img):
    result = rate_face(boy_img, "boy")
    assert {"score", "clarity", "symmetry", "picture"} <= set(result)
    picture = result["picture"]
    assert isinstance(picture, np.ndarray)
    assert picture.ndim == 3 and picture.shape[2] == 3 and picture.dtype == np.uint8


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
        for gender in ("boy", "girl"):
            result = rate_face(img, gender)
            assert 0 <= result["score"] <= 10
            assert 0 <= result["symmetry"] <= 1
            assert result["clarity"] is None or 0 <= result["clarity"] <= 1


# Two different real faces (perBoy against the girl model face and the other way round) are about 0.045 apart in
# shape error, so a change that is only landmark noise must stay well below that
SAME_FACE_TOLERANCE = 0.02


def shape_error_of(img, gender):
    xList, yList = landmark_detect(img)
    return landmarkdetect.shape_error(xList, yList, gender)


@pytest.mark.parametrize("picture, gender", [
    (BOY_PICTURE, "boy"),
    (BOY_PICTURE, "girl"),
    (GIRL_PICTURE, "boy"),
], ids=["perBoy-as-boy", "perBoy-as-girl", "perGirl-as-boy"])
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


def test_shape_error_of_the_model_face_itself_is_zero():
    perfectX, perfectY = getPerfs("boy")
    assert landmarkdetect.shape_error(perfectX, perfectY, "boy") == pytest.approx(0, abs=1e-9)
    assert landmarkdetect.score_from_error(0) == pytest.approx(10)


def test_score_from_error_goes_down_as_the_error_goes_up():
    scores = [landmarkdetect.score_from_error(err) for err in (0, 0.02, 0.05, 0.1, 0.5)]
    assert scores == sorted(scores, reverse=True)
    assert all(0 <= s <= 10 for s in scores)


# ---------- alignment and symmetry ----------

def test_mirror_pairs_swap_left_and_right():
    pairs = landmarkdetect.MIRROR_PAIRS
    assert sorted(pairs) == list(range(68))
    # Mirroring twice gives the same face back
    assert [pairs[p] for p in pairs] == list(range(68))


def test_align_undoes_a_move_turn_and_resize():
    rng = np.random.default_rng(1)
    points = rng.uniform(0, 300, (68, 2))
    angle = np.radians(25)
    turn = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    moved = 1.7 * points @ turn.T + [40, -15]
    assert np.allclose(landmarkdetect.align(moved, points), points)


def test_symmetry_ignores_a_mirrored_picture(boy_img):
    xList, yList = landmark_detect(boy_img)
    # The same landmarks mirrored left to right are exactly as symmetric
    mirroredX = [-x for x in xList]
    assert landmarkdetect.symmetry(mirroredX, yList) == pytest.approx(landmarkdetect.symmetry(xList, yList), abs=1e-6)


def test_huge_picture_still_rates(girl_img):
    h, w = girl_img.shape[:2]
    scale = 3000 / max(h, w)
    huge = cv2.resize(girl_img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_CUBIC)
    result = rate_face(huge, "girl")
    assert np.isfinite(result["score"])
