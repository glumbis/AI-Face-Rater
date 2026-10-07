import os
import sys
from datetime import date, datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import history  # noqa: E402


def add(folder, model, score, daysAgo=0, clarity=0.8, source="file"):
    when = datetime(2026, 5, 20, 12, 0) - timedelta(days=daysAgo)
    assert history.record(model, score, clarity, 0.9, source, when=when, folder=str(folder))


def test_save_and_load_roundtrip(tmp_path):
    add(tmp_path, "boy", 7.25, clarity=None, source="camera")
    add(tmp_path, "girl", 5.0)
    entries = history.load(str(tmp_path))
    assert [e.model for e in entries] == ["boy", "girl"]
    assert entries[0].score == 7.25 and entries[0].clarity is None and entries[0].source == "camera"
    assert entries[1].clarity == 0.8 and entries[1].symmetry == 0.9
    # Only numbers and short words are saved
    text = open(history.history_path(str(tmp_path)), encoding="utf-8").read()
    assert text.splitlines()[0] == "time,model,score,clarity,symmetry,source"


def test_missing_folder_is_created_and_missing_file_is_empty(tmp_path):
    folder = tmp_path / "new" / "folder"
    assert history.load(str(folder)) == []
    add(folder, "boy", 6.0)
    assert len(history.load(str(folder))) == 1


def test_bad_rows_are_skipped(tmp_path):
    add(tmp_path, "boy", 6.0)
    with open(history.history_path(str(tmp_path)), "a", encoding="utf-8") as f:
        f.write("not a date,boy,abc,,0.9,file\n2026-05-20T12:00:00,cat,5,,0.9,file\n")
    assert len(history.load(str(tmp_path))) == 1


def test_write_error_is_quiet(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    # A folder can't be made inside a file
    assert history.record("boy", 6.0, 0.8, 0.9, "file", folder=str(blocker / "sub")) is False


def test_clear(tmp_path):
    add(tmp_path, "boy", 6.0)
    assert history.clear(str(tmp_path)) is True
    assert history.load(str(tmp_path)) == []
    assert history.clear(str(tmp_path)) is True  # nothing to delete is fine


def test_compare_best_average_and_trend(tmp_path):
    add(tmp_path, "boy", 6.0, daysAgo=1)
    add(tmp_path, "boy", 8.0)
    add(tmp_path, "girl", 2.0)  # another model face doesn't count
    before = history.load(str(tmp_path))
    up = history.compare(before, "boy", 8.5)
    assert up["trend"] == "up" and up["count"] == 3 and up["best"] == 8.5
    assert abs(up["average"] - 22.5 / 3) < 1e-9
    # The arrow compares with the previous photo (8.0), not the average (7.0)
    assert abs(up["diff"] - 0.5) < 1e-9
    down = history.compare(before, "boy", 7.0)
    assert down["trend"] == "down" and abs(down["diff"] + 1.0) < 1e-9
    assert history.compare(before, "boy", 8.04)["trend"] is None  # shown as 8.0, like the previous photo
    assert history.compare(before, "boy", 7.96)["trend"] is None
    assert history.compare([], "boy", 7.0) is None
    assert history.compare(before, "girl", 3.0)["count"] == 2


def test_arrow_uses_the_shown_one_decimal_scores(tmp_path):
    add(tmp_path, "boy", 7.04)
    before = history.load(str(tmp_path))
    # 7.04 -> 7.16 is shown as 7.0 -> 7.2: the arrow says 0.2, not 0.1
    assert history.compare(before, "boy", 7.16)["diff"] == 0.2
    # 7.04 -> 7.13 is shown as 7.0 -> 7.1: an arrow, although the scores are less than 0.1 apart
    assert history.compare(before, "boy", 7.13)["trend"] == "up"
    # 7.04 -> 6.96 is shown as 7.0 -> 7.0: no arrow
    assert history.compare(before, "boy", 6.96)["trend"] is None
    # 7.0 -> 7.1 is up, although 7.1 - 7.0 is a hair under 0.1 in floating point
    assert history.compare([history.Entry(datetime(2026, 5, 20), "boy", 7.0, None, 0.9, "file")], "boy", 7.1)["trend"] == "up"


def test_arrow_compares_with_the_previous_photo_of_the_same_model_face(tmp_path):
    add(tmp_path, "boy", 9.0, daysAgo=2)
    add(tmp_path, "girl", 3.0, daysAgo=1)  # a later photo, but for another model face
    add(tmp_path, "boy", 5.0)
    before = history.load(str(tmp_path))
    compare = history.compare(before, "boy", 5.6)
    assert compare["trend"] == "up" and abs(compare["diff"] - 0.6) < 1e-9
    assert history.compare(before, "girl", 3.0)["trend"] is None  # no change from 3.0
    assert history.compare(before, "girl", 2.0)["trend"] == "down"


def test_streak(tmp_path):
    today = date(2026, 5, 20)
    assert history.streak([], today) == 0
    for daysAgo in (0, 1, 2, 4):
        add(tmp_path, "boy", 6.0, daysAgo=daysAgo)
    entries = history.load(str(tmp_path))
    assert history.streak(entries, today) == 3
    # Nothing rated yet today: yesterday still counts, a longer gap doesn't
    assert history.streak(entries, today + timedelta(days=1)) == 3
    assert history.streak(entries, today + timedelta(days=2)) == 0


def test_each_model_face_has_its_own_history_including_the_average_face(tmp_path):
    add(tmp_path, "boy", 6.0)
    add(tmp_path, "average", 7.0)
    add(tmp_path, "average", 8.0, daysAgo=1)
    entries = history.load(str(tmp_path))
    assert sorted(e.model for e in entries) == ["average", "average", "boy"]
    compare = history.compare(entries, "average", 9.0)
    assert compare["count"] == 3 and compare["best"] == 9.0 and compare["average"] == 8.0
    assert history.compare(entries, "girl", 5.0) is None
