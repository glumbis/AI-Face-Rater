import csv
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta

# Every rating is saved on this computer as a row of numbers (never a picture), so the app can show how a new
# score compares with your earlier ones. Everything here fails quietly: a missing or locked file never breaks the app.

HISTORY_FILE = "history.csv"
COLUMNS = ["time", "model", "score", "clarity", "symmetry", "source"]
MODELS = ("boy", "girl")
# The up/down arrow only shows when the score is at least this far from your earlier average
TREND_MIN_DIFF = 0.1


@dataclass
class Entry:
    when: datetime
    model: str  # "boy" or "girl"
    score: float
    clarity: float | None
    symmetry: float
    source: str  # "camera" or "file"


def data_folder():
    # %APPDATA%\AI Face Rater on Windows, a folder in the home folder anywhere else
    base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "AI Face Rater")


def history_path(folder=None):
    return os.path.join(folder or data_folder(), HISTORY_FILE)


def record(model, score, clarity, symmetry, source, when=None, folder=None):
    # Adds one rating to the file. Returns False (and does nothing else) if it can't be written
    when = when or datetime.now()
    path = history_path(folder)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        isNew = not os.path.isfile(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if isNew:
                writer.writerow(COLUMNS)
            writer.writerow([when.isoformat(timespec="seconds"), model, f"{score:.3f}",
                             "" if clarity is None else f"{clarity:.3f}", f"{symmetry:.3f}", source])
        return True
    except OSError:
        return False


def load(folder=None):
    # All saved ratings, oldest first. Rows that can't be read are skipped
    entries = []
    try:
        with open(history_path(folder), newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    model = row["model"]
                    if model not in MODELS:
                        continue
                    entries.append(Entry(datetime.fromisoformat(row["time"]), model, float(row["score"]),
                                         float(row["clarity"]) if row.get("clarity") else None,
                                         float(row["symmetry"]), row.get("source") or ""))
                except (KeyError, TypeError, ValueError):
                    continue
    except (OSError, UnicodeDecodeError, csv.Error):
        pass
    return entries


def clear(folder=None):
    # Deletes the history file. Returns True if there is no history left
    try:
        os.remove(history_path(folder))
    except FileNotFoundError:
        pass
    except OSError:
        return False
    return True


def compare(entries, model, score):
    # How a new score looks next to the earlier ones for the same model face. entries are the ratings from before
    # this one. Returns None when there is nothing to compare with, otherwise a dict with best and average (both
    # including the new score), count (including it), trend ("up", "down" or None) and diff (the new score minus
    # the earlier average)
    earlier = [e.score for e in entries if e.model == model]
    if not earlier:
        return None
    diff = score - sum(earlier) / len(earlier)
    trend = "up" if diff >= TREND_MIN_DIFF else "down" if diff <= -TREND_MIN_DIFF else None
    allScores = earlier + [score]
    return {"best": max(allScores), "average": sum(allScores) / len(allScores), "count": len(allScores),
            "trend": trend, "diff": diff}


def streak(entries, today=None):
    # How many days in a row (ending today, or yesterday if nothing is rated yet today) have at least one rating
    today = today or date.today()
    days = {e.when.date() for e in entries}
    day = today if today in days else today - timedelta(days=1)
    count = 0
    while day in days:
        count += 1
        day -= timedelta(days=1)
    return count
