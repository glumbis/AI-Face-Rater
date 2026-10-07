from PIL import Image
import pytest

import dataset_rater
from dataset_rater import (count_rated, fit_size, list_images, load_ratings, next_unrated, save_ratings,
                           undo_last)

# Tests for the file handling in dataset_rater.py. No window is opened.


@pytest.fixture
def folder(tmp_path):
    # A folder with four tiny pictures and a file that isn't a picture
    pictures = tmp_path / "faces"
    pictures.mkdir()
    for name in ["c.png", "a.jpg", "b.PNG", "d.jpeg"]:
        Image.new("RGB", (4, 3), "white").save(pictures / name, format="PNG")
    (pictures / "notes.txt").write_text("not a picture")
    return pictures


def test_list_images_sorted_and_only_pictures(folder):
    assert list_images(str(folder)) == ["a.jpg", "b.PNG", "c.png", "d.jpeg"]


def test_list_images_missing_folder(tmp_path):
    assert list_images(str(tmp_path / "nope")) == []


def test_save_writes_header_and_load_reads_it_back(tmp_path):
    path = str(tmp_path / "ratings.csv")
    save_ratings(path, [["a.jpg", 7], ["b.png", 10]])
    with open(path, encoding="utf-8") as f:
        assert f.read().splitlines() == ["filename,rating", "a.jpg,7", "b.png,10"]
    assert load_ratings(path) == [["a.jpg", "7"], ["b.png", "10"]]


def test_old_headerless_file_still_loads(tmp_path):
    path = tmp_path / "ratings.csv"
    path.write_text("a.jpg,7\n\nb.png,3\n")
    assert load_ratings(str(path)) == [["a.jpg", "7"], ["b.png", "3"]]


def test_missing_file_loads_as_empty(tmp_path):
    assert load_ratings(str(tmp_path / "ratings.csv")) == []


def test_resume_at_first_unrated_picture_not_after_last_line():
    images = ["a.jpg", "b.png", "c.png", "d.jpeg"]
    # b was skipped last time, and the last line is d, so "the picture after the last line" would be the end
    rows = [["a.jpg", "5"], ["c.png", "6"], ["d.jpeg", "7"]]
    assert next_unrated(images, rows) == 1
    assert next_unrated(images, rows, start=2) == 4
    assert count_rated(images, rows) == 3


def test_rows_from_another_folder_are_ignored():
    images = ["a.jpg", "b.png"]
    rows = [["other.jpg", "5"]]
    assert next_unrated(images, rows) == 0
    assert count_rated(images, rows) == 0


def test_undo_removes_last_row_and_goes_back_to_its_picture(tmp_path):
    path = str(tmp_path / "ratings.csv")
    images = ["a.jpg", "b.png", "c.png"]
    rows = [["a.jpg", 5], ["b.png", 8]]
    save_ratings(path, rows)

    assert undo_last(images, rows) == 1
    save_ratings(path, rows)
    assert load_ratings(path) == [["a.jpg", "5"]]
    assert next_unrated(images, rows) == 1


def test_undo_with_nothing_to_undo():
    assert undo_last(["a.jpg"], []) is None


def test_undo_does_not_remove_a_picture_from_another_folder():
    rows = [["other.jpg", "5"]]
    assert undo_last(["a.jpg"], rows) is None
    assert rows == [["other.jpg", "5"]]


def test_old_file_gets_header_after_next_rating(tmp_path):
    path = tmp_path / "ratings.csv"
    path.write_text("a.jpg,7\n")
    rows = load_ratings(str(path))
    rows.append(["b.png", 4])
    save_ratings(str(path), rows)
    assert path.read_text().splitlines() == ["filename,rating", "a.jpg,7", "b.png,4"]


@pytest.mark.parametrize("size, expected", [
    ((800, 400), (400, 200)),  # wide, shrunk
    ((400, 800), (200, 400)),  # tall, shrunk
    ((100, 50), (400, 200)),   # small, enlarged
    ((400, 400), (400, 400)),
])
def test_fit_size_keeps_the_shape(size, expected):
    assert fit_size(*size, 400) == expected


def test_last_folder_is_remembered(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_rater, "SETTINGS_FILE", str(tmp_path / "settings.json"))
    assert dataset_rater.load_last_folder() is None
    dataset_rater.save_last_folder("C:/faces")
    assert dataset_rater.load_last_folder() == "C:/faces"


def test_broken_settings_file_is_ignored(tmp_path, monkeypatch):
    settings = tmp_path / "settings.json"
    settings.write_text("{not json")
    monkeypatch.setattr(dataset_rater, "SETTINGS_FILE", str(settings))
    assert dataset_rater.load_last_folder() is None


def test_folder_from_command_line_wins(folder, tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_rater, "SETTINGS_FILE", str(tmp_path / "settings.json"))
    dataset_rater.save_last_folder(str(tmp_path))
    monkeypatch.setattr("sys.argv", ["dataset_rater.py", str(folder)])
    assert dataset_rater.choose_folder() == str(folder)


def test_last_folder_used_without_command_line(folder, tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_rater, "SETTINGS_FILE", str(tmp_path / "settings.json"))
    dataset_rater.save_last_folder(str(folder))
    monkeypatch.setattr("sys.argv", ["dataset_rater.py"])
    assert dataset_rater.choose_folder() == str(folder)
