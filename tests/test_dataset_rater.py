import os
import tkinter as tk

from PIL import Image
import pytest

import dataset_rater
from dataset_rater import (RatingsFileError, count_rated, fit_size, list_images, load_ratings, next_unrated,
                           save_ratings, undo_last)

# Tests for the file handling in dataset_rater.py. The few window tests use a hidden window
# and never wait for input.


@pytest.fixture
def folder(tmp_path):
    # A folder with four tiny pictures and a file that isn't a picture
    pictures = tmp_path / "faces"
    pictures.mkdir()
    for name in ["c.png", "a.jpg", "b.PNG", "d.jpeg"]:
        Image.new("RGB", (4, 3), "white").save(pictures / name, format="PNG")
    (pictures / "notes.txt").write_text("not a picture")
    return pictures


@pytest.fixture
def settings(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    monkeypatch.setattr(dataset_rater, "SETTINGS_FILE", str(path))
    return path


# ---------- Finding the pictures ----------

def test_list_images_sorted_and_only_pictures(folder):
    assert list_images(str(folder)) == ["a.jpg", "b.PNG", "c.png", "d.jpeg"]


def test_list_images_missing_folder(tmp_path):
    assert list_images(str(tmp_path / "nope")) == []


@pytest.mark.parametrize("size, expected", [
    ((800, 400), (400, 200)),  # wide, shrunk
    ((400, 800), (200, 400)),  # tall, shrunk
    ((100, 50), (400, 200)),   # small, enlarged
    ((400, 400), (400, 400)),
])
def test_fit_size_keeps_the_shape(size, expected):
    assert fit_size(*size, 400) == expected


# ---------- Reading ratings.csv ----------

def test_save_writes_header_and_load_reads_it_back(tmp_path):
    path = str(tmp_path / "ratings.csv")
    save_ratings(path, [["a.jpg", 7], ["b.png", 10]])
    with open(path, encoding="utf-8") as f:
        assert f.read().splitlines() == ["filename,rating", "a.jpg,7", "b.png,10"]
    assert load_ratings(path) == ([["a.jpg", "7"], ["b.png", "10"]], ",")


def test_old_headerless_file_still_loads(tmp_path):
    path = tmp_path / "ratings.csv"
    path.write_text("a.jpg,7\n\nb.png,3\n")
    assert load_ratings(str(path)) == ([["a.jpg", "7"], ["b.png", "3"]], ",")


def test_missing_file_loads_as_empty(tmp_path):
    assert load_ratings(str(tmp_path / "ratings.csv")) == ([], ",")


def test_semicolon_file_from_excel(tmp_path):
    # Excel with Norwegian settings uses ; between the columns
    path = tmp_path / "ratings.csv"
    path.write_bytes(b"filename;rating\r\na.jpg;7\r\nb.png;3\r\n")
    rows, delimiter = load_ratings(str(path))
    assert rows == [["a.jpg", "7"], ["b.png", "3"]]
    assert delimiter == ";"

    # Saving keeps the ; so Excel still opens it in two columns
    save_ratings(str(path), rows + [["c.png", 5]], delimiter)
    assert path.read_text(encoding="utf-8").splitlines() == ["filename;rating", "a.jpg;7", "b.png;3", "c.png;5"]


def test_old_row_with_comma_in_the_filename(tmp_path):
    # The old tools didn't put quotes around names, so the rating is the last cell
    path = tmp_path / "ratings.csv"
    path.write_text("a.jpg,7\nsmith, john.jpg,8\n")
    rows, delimiter = load_ratings(str(path))
    assert rows == [["a.jpg", "7"], ["smith, john.jpg", "8"]]

    # Written back with quotes, so it reads back the same
    save_ratings(str(path), rows, delimiter)
    assert load_ratings(str(path))[0] == rows


def test_old_cp1252_file_with_norwegian_letters(tmp_path):
    path = tmp_path / "ratings.csv"
    path.write_bytes("bilde_æøå.jpg,7\n".encode("cp1252"))
    rows, delimiter = load_ratings(str(path))
    assert rows == [["bilde_æøå.jpg", "7"]]

    # Saved as utf-8 from now on
    save_ratings(str(path), rows, delimiter)
    assert "bilde_æøå.jpg,7" in path.read_bytes().decode("utf-8")


@pytest.mark.parametrize("text", ["filename,rating\na.jpg,7\n", "a.jpg,7\n"], ids=["with header", "without header"])
def test_utf8_bom_from_excel(tmp_path, text):
    path = tmp_path / "ratings.csv"
    path.write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))
    assert load_ratings(str(path))[0] == [["a.jpg", "7"]]


@pytest.mark.parametrize("bad_line", ["just_a_name.jpg", "a.jpg,seven", "a.jpg;7"])
def test_line_that_cant_be_read_stops_instead_of_being_dropped(tmp_path, bad_line):
    path = tmp_path / "ratings.csv"
    text = f"filename,rating\na.jpg,7\n{bad_line}\nb.png,3\n"
    path.write_text(text)
    with pytest.raises(RatingsFileError, match="Line 3"):
        load_ratings(str(path))
    assert path.read_text() == text


# ---------- Saving ----------

def test_failing_save_leaves_the_file_as_it_was(tmp_path, monkeypatch):
    path = tmp_path / "ratings.csv"
    save_ratings(str(path), [["a.jpg", 7]])
    before = path.read_bytes()
    rows = [["a.jpg", 7], ["b.png", 3]]

    tries = []

    def locked(src, dst):
        tries.append(src)
        raise PermissionError("locked by Excel")

    monkeypatch.setattr(dataset_rater, "SAVE_WAIT", 0)
    monkeypatch.setattr(dataset_rater.os, "replace", locked)
    with pytest.raises(OSError):
        save_ratings(str(path), rows)

    assert len(tries) == dataset_rater.SAVE_TRIES
    assert path.read_bytes() == before
    assert rows == [["a.jpg", 7], ["b.png", 3]]
    assert not os.path.exists(str(path) + ".tmp")


def test_save_works_when_the_lock_goes_away(tmp_path, monkeypatch):
    path = tmp_path / "ratings.csv"
    realReplace = os.replace
    tries = []

    def locked_once(src, dst):
        tries.append(src)
        if len(tries) == 1:
            raise PermissionError("locked for a moment")
        realReplace(src, dst)

    monkeypatch.setattr(dataset_rater, "SAVE_WAIT", 0)
    monkeypatch.setattr(dataset_rater.os, "replace", locked_once)
    save_ratings(str(path), [["a.jpg", 7]])
    assert load_ratings(str(path))[0] == [["a.jpg", "7"]]


def test_old_file_gets_header_after_next_rating(tmp_path):
    path = tmp_path / "ratings.csv"
    path.write_text("a.jpg,7\n")
    rows, delimiter = load_ratings(str(path))
    save_ratings(str(path), rows + [["b.png", 4]], delimiter)
    assert path.read_text().splitlines() == ["filename,rating", "a.jpg,7", "b.png,4"]


# ---------- Resume and undo ----------

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


def test_undo_removes_last_row_and_goes_back_to_its_picture():
    images = ["a.jpg", "b.png", "c.png"]
    rows = [["a.jpg", 5], ["b.png", 8]]
    newRows, index = undo_last(images, rows)
    assert index == 1
    assert newRows == [["a.jpg", 5]]
    assert rows == [["a.jpg", 5], ["b.png", 8]]  # The old list isn't changed, in case saving fails
    assert next_unrated(images, newRows) == 1


def test_undo_with_nothing_to_undo():
    assert undo_last(["a.jpg"], []) == ([], None)


def test_undo_does_not_remove_a_picture_from_another_folder():
    rows = [["other.jpg", "5"]]
    assert undo_last(["a.jpg"], rows) == (rows, None)


# ---------- Remembering and picking the folder ----------

def test_last_folder_is_remembered(settings):
    assert dataset_rater.load_last_folder() is None
    dataset_rater.save_last_folder("C:/faces")
    assert dataset_rater.load_last_folder() == "C:/faces"


@pytest.mark.parametrize("content", ["{not json", '{"last_folder": ["x"]}', '{"last_folder": 5}', '["x"]'])
def test_broken_settings_file_is_ignored(settings, content):
    settings.write_text(content)
    assert dataset_rater.load_last_folder() is None


def test_folder_from_command_line_wins(folder, tmp_path, settings, monkeypatch):
    dataset_rater.save_last_folder(str(tmp_path))
    monkeypatch.setattr("sys.argv", ["dataset_rater.py", str(folder)])
    assert dataset_rater.choose_folder() == str(folder)


def test_last_folder_used_without_command_line(folder, settings, monkeypatch):
    dataset_rater.save_last_folder(str(folder))
    monkeypatch.setattr("sys.argv", ["dataset_rater.py"])
    assert dataset_rater.choose_folder() == str(folder)


def test_folder_without_pictures_opens_the_picker(folder, tmp_path, settings, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    dataset_rater.save_last_folder(str(empty))
    monkeypatch.setattr("sys.argv", ["dataset_rater.py"])
    asked = []
    monkeypatch.setattr(dataset_rater.filedialog, "askdirectory", lambda title: asked.append(title) or str(folder))
    assert dataset_rater.choose_folder() == str(folder)
    assert len(asked) == 1 and "No .jpg or .png pictures" in asked[0]


def test_cancel_in_the_picker_gives_none(tmp_path, settings, monkeypatch):
    monkeypatch.setattr("sys.argv", ["dataset_rater.py", str(tmp_path / "nope")])
    monkeypatch.setattr(dataset_rater.filedialog, "askdirectory", lambda title: "")
    assert dataset_rater.choose_folder() is None


# ---------- The window (hidden, driven from code) ----------

@pytest.fixture(scope="module")
def tk_root():
    # Only one Tk for all the tests: making a new one for each test sometimes fails on Windows
    try:
        root = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"tkinter can't open a window here: {e}")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def app(tk_root, folder, tmp_path, monkeypatch):
    # Each test gets its own hidden window inside the one Tk
    window = tk.Toplevel(tk_root)
    window.withdraw()
    # A message box would wait for a click, so record the messages instead
    errors = []
    monkeypatch.setattr(dataset_rater.messagebox, "showerror", lambda title, text: errors.append(text))
    monkeypatch.setattr(dataset_rater, "RATINGS_FILE", str(tmp_path / "ratings.csv"))
    monkeypatch.setattr(dataset_rater, "SAVE_WAIT", 0)
    rater = dataset_rater.DatasetRater(window, str(folder), [], ",")
    rater.errors = errors
    yield rater
    window.destroy()


def test_locked_file_keeps_window_and_file_in_step(app, monkeypatch):
    app.rate(7)
    before = open(dataset_rater.RATINGS_FILE, "rb").read()

    def locked(src, dst):
        raise PermissionError("locked by Excel")

    monkeypatch.setattr(dataset_rater.os, "replace", locked)
    app.rate(8)
    app.undo()
    assert app.rows == [["a.jpg", 7]]
    assert app.index == 1
    assert open(dataset_rater.RATINGS_FILE, "rb").read() == before
    assert len(app.errors) == 2 and "locked" in app.errors[0]

    # Once the file is free again, nothing was saved twice
    monkeypatch.undo()
    app.rate(8)
    assert load_ratings(dataset_rater.RATINGS_FILE)[0] == [["a.jpg", "7"], ["b.PNG", "8"]]


def test_broken_picture_cant_be_rated_but_can_be_skipped(app, folder):
    (folder / "a.jpg").write_text("not really a picture")
    app.show_picture()
    assert app.broken
    app.rate(10)
    assert app.rows == [] and not os.path.exists(dataset_rater.RATINGS_FILE)
    app.skip()
    assert app.index == 1 and not app.broken
    app.rate(6)
    assert app.rows == [["b.PNG", 6]]
