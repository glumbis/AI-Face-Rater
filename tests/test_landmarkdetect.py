import os
import shutil

import cv2
import numpy as np
import pytest

# These tests only use the public functions of landmarkdetect.py and only check things that should stay
# true however the score is worked out (for example "the boy face looks more like the boy model face"),
# not exact score numbers.

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
    assert {"score", "clarity", "skinFactor", "picture"} <= set(result)
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


@pytest.mark.parametrize("picture, gender", [
    pytest.param(BOY_PICTURE, "boy", marks=pytest.mark.xfail(
        strict=False,
        reason="The old 1/distance score is huge when a model face is rated against itself (distance about 0), "
               "so any small change makes it jump. Should pass once the score is 0-10.")),
    (BOY_PICTURE, "girl"),
    (GIRL_PICTURE, "boy"),
], ids=["perBoy-as-boy", "perBoy-as-girl", "perGirl-as-boy"])
@pytest.mark.parametrize("degrees", [10, -10])
def test_small_rotation_barely_changes_the_score(picture, gender, degrees):
    # A head tilted a little to one side is the same face, so the score should stay almost the same
    img = read_image(picture)
    straight = rate_face(img, gender)["score"]
    rotated = rate_face(rotate(img, degrees), gender)["score"]
    assert abs(rotated - straight) / straight < 0.15


def test_huge_picture_still_rates(girl_img):
    h, w = girl_img.shape[:2]
    scale = 3000 / max(h, w)
    huge = cv2.resize(girl_img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_CUBIC)
    result = rate_face(huge, "girl")
    assert np.isfinite(result["score"])
