# AI Face Rater

Old, unfinished experiments with face landmark detection (MediaPipe Face Landmarker + OpenCV) and a tool for rating a face dataset.

- `face_rater_app.py` is the app: double-click `Face Rater.bat` (or run `python face_rater_app.py`).
  - The camera picture is on the left, with a small status line under it that tells you if your face is OK. If your head is turned or tilted a little (over 8 / 10 degrees) or your face is a bit small in the picture it gives a short tip ("Turn a little towards the camera.", "Move a little closer.") but never stops you. The tip follows the middle of the last few pictures and only goes when you're clearly back, so it doesn't flicker.
  - On the right you pick the model face (Boy, Girl or Average) and see your score out of 10, with skin clarity and symmetry as percentages.
  - **Take photo** (or Space) rates the picture from the last second or so where your head faced the camera best (and was sharpest), so one bad moment doesn't ruin it, and **Pick a photo…** (or Ctrl+O) rates a file. After a result the main button becomes **Back to camera** (or Esc). Changing the model face rates the same picture again.
  - **Photo tips** (`phototips.py`): under a result you get at most two short tips about the photo and setup (head turned or tilted, blurry, too dark or too bright, one side lit more than the other, too close or too far), or a "Great setup" line when nothing is wrong. The thresholds are named constants at the top of that file.
  - **History:** every rating is saved on this computer as numbers only (time, model face, score, skin clarity, symmetry, camera or file, never a picture) in `%APPDATA%\AI Face Rater\history.csv`. A result shows your best and average for that model face, an arrow for above or below your average, and a day streak from 2 days in a row. The **i** button opens a small About window with a privacy note and **Clear history** (with a confirmation).
  - The window follows the Windows light/dark setting and fits screens from 1366×768 up to high-DPI screens.
- `landmarkdetect.py` does the rating, and `headpose.py` works out which way the head is facing (from the head position MediaPipe gives with the landmarks). `facelayout.py` lists the 162 landmarks that are used, which ones are each other's left/right partner, how much each counts, and the landmarks of the model faces. `face_landmarker.task` is MediaPipe's face model (3.8 MB, Apache-2.0 licence) and must stay next to the scripts. You can also run it in the terminal: set `IMAGE_TO_RATE` at the top of the file to the picture you want to rate, or pass it on the command line: `python landmarkdetect.py picture.jpg`.
- `perBoy.jpg` and `perGirl.jpg` are the model faces the "perfect" landmark coordinates come from. The third model face, "Average", is the two of them put on top of each other (the girl face is moved, turned and resized to fit the boy face, then the landmarks are averaged), so it is a face somewhere between them.
- `dataset_rater.py` (optional extra) rates a folder of face pictures from 1 to 10 into `ratings.csv` next to the script. Run `python dataset_rater.py path\to\folder`. Without a folder it uses the folder from last time, or asks you to pick one.
  - Keys **1-9** rate right away, **0** = 10. Or drag the slider and press **Submit** (or Enter).
  - **Backspace** or **Left arrow** undoes the last rating and goes back to that picture.
  - **S** or **Right arrow** skips a picture without rating it. Skipped pictures come back the next time you start.
  - It always starts at the first picture in the folder that isn't rated yet, and shows how many are done ("12 / 340").
  - `ratings.csv` has a `filename,rating` header and can be edited in Excel (comma- or semicolon-separated). If it has a line the tool can't read, it tells you which line and doesn't start, so no rating is lost.

## How the score works

The 162 face landmarks (picked from the 478 that MediaPipe finds) are lined up with a model face (boy, girl or average) by moving, turning and resizing them to fit as closely as possible. The face is also compared the other way round (mirrored), so a mirrored selfie gets the same score. Each region counts differently: eyes and nose the most, brows a medium amount, the jaw line and outer lips a little, and the inner lips not at all, since they mostly change with expression. The remaining difference, measured relative to the distance between the eyes, is turned into a score from 0 to 10.

Uneven skin on the cheeks and a lopsided face (found by comparing the face with its own mirror image) add a small penalty. Only the tint of the skin is compared, so shadows don't count, and stubble, beards and black-and-white photos are not counted as uneven skin. The head must roughly face the camera: a picture is refused when the head is turned more than `MAX_TURN` (12) degrees to the side or tilted more than `MAX_TILT` (15) degrees up or down (both in `headpose.py`), and the live tip comes at 8 and 10 degrees (`HINT_TURN`, `HINT_TILT`). A head turned 10 degrees or more loses some points, because the face looks lopsided. The face must also be close enough: a camera picture is refused when the face (forehead to chin) is under `MIN_FACE_SIZE` (30%) of the picture's height (about 60 cm from a laptop webcam), with a live tip under `HINT_FACE_SIZE` (35%); a picked photo only needs `MIN_FACE_SIZE_FILE` (15%) of its shorter side, so a group photo with tiny faces is refused (all in `landmarkdetect.py`).

On the result picture, the landmarks are mint dots, the checked cheek squares are white outlines and uneven spots are soft red.

The score is just a fun comparison with one model face, not a real measure of beauty.

## Setup

Needs Python 3.10 or newer with tkinter (included in the normal Windows installer).

```
pip install -r requirements.txt
```

That is all: the face model (`face_landmarker.task`) is in the repo. With several Pythons installed, `Face Rater.bat` automatically starts the app with the first Python that has the packages (newest first). In VS Code, pick that Python with **Python: Select Interpreter**.

## Tests

```
pip install -r requirements-dev.txt
pytest
```

The tests for `landmarkdetect.py` need `face_landmarker.task` next to the scripts (it is in the repo). Without it they are skipped, and the `dataset_rater.py` tests still run.
