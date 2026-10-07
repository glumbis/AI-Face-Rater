# AI Face Rater

Rates a face by comparing its landmarks (MediaPipe Face Landmarker) with a few model faces.

**Try it in the browser, no install: https://glumbis.github.io/AI-Face-Rater/**. Everything runs on your device and no photo is uploaded. The website is in `web/` (see `web/FEATURES.md`) and deploys to GitHub Pages on every push to main.

- `face_rater_app.py` is the app: double-click `Face Rater.bat` (or run `python face_rater_app.py`).
  - The camera picture is on the left, with a small status line under it that tells you if your face is OK. If your head is turned a little (over 8 degrees) or tilted quite far up or down (over 18 degrees), or your face is a bit small in the picture it gives a short tip ("Turn a little towards the camera.", "Move a little closer.") but never stops you. The tip follows the middle of the last few pictures and only goes when you're clearly back, so it doesn't flicker.
  - On the right you pick the model face (Boy or Girl) and see your score out of 10, with skin clarity and symmetry as percentages and, under them, a score for each part of the face ("Eyes 7.9 · Nose 3.7 · ...", best first). When there are several model faces of that gender, your face gets the score of the one it is most like, and the status line says which ("Closest to Boy 3").
  - **Take photo** (or Space) rates the picture from the last second or so where your head faced the camera best (and was sharpest), so one bad moment doesn't ruin it, and **Pick a photo…** (or Ctrl+O) rates a file. After a result the main button becomes **Back to camera** (or Esc). Changing the model face rates the same picture again.
  - **Photo tips** (`phototips.py`): under a result you get at most two short tips about the photo and setup (head turned or tilted, blurry, too dark or too bright, one side lit more than the other, too close or too far), or a "Great setup" line when nothing is wrong. The thresholds are named constants at the top of that file.
  - **History:** every rating is saved on this computer as numbers only (time, model face, score, skin clarity, symmetry, camera or file, never a picture) in `%APPDATA%\AI Face Rater\history.csv`. A result shows your best and average for that model face, an arrow with how much higher or lower the score is than your previous photo for that model face (no arrow on the first photo), and a day streak from 2 days in a row. The **i** button opens a small About window with a privacy note and **Clear history** (with a confirmation).
  - The window follows the Windows light/dark setting and fits screens from 1366×768 up to high-DPI screens.
- `landmarkdetect.py` does the rating, and `headpose.py` works out which way the head is facing (from the head position MediaPipe gives with the landmarks). `facelayout.py` lists the 162 landmarks that are used, which ones are each other's left/right partner and how much each counts, and `modelfaces.py` has the landmarks of the model faces (made by a tool, see below). `face_landmarker.task` is MediaPipe's face model (3.8 MB, Apache-2.0 licence) and must stay next to the scripts. You can also run it in the terminal: set `IMAGE_TO_RATE` at the top of the file to the picture you want to rate, or pass it on the command line: `python landmarkdetect.py picture.jpg`.
- `perBoy.jpg`, `perBoy2.jpg`, `perBoy3.jpg`, ... and `perGirl.jpg`, `perGirl2.jpg`, ... are the model faces the "perfect" landmark coordinates come from (`perBoy.jpg` is "Boy 1", `perBoy2.jpg` "Boy 2" and so on). There can be any number of each, but at least `perBoy.jpg` and `perGirl.jpg`. To add one:
  1. Put the picture next to the scripts with the next free number, for example `perBoy2.jpg`. A sharp, evenly lit photo of a face looking straight into the camera with a neutral expression works best.
  2. Run `python tools/make_model_faces.py`. It finds the landmarks in every model face picture and writes them to `modelfaces.py` (it says if a face isn't looking straight into the camera).
  3. Run `python tools/export_web_data.py` and `python tools/make_golden.py`, so the website gets the new face too, and commit the picture with the changed files.
- `dataset_rater.py` (optional extra) rates a folder of face pictures from 1 to 10 into `ratings.csv` next to the script. Run `python dataset_rater.py path\to\folder`. Without a folder it uses the folder from last time, or asks you to pick one.
  - Keys **1-9** rate right away, **0** = 10. Or drag the slider and press **Submit** (or Enter).
  - **Backspace** or **Left arrow** undoes the last rating and goes back to that picture.
  - **S** or **Right arrow** skips a picture without rating it. Skipped pictures come back the next time you start.
  - It always starts at the first picture in the folder that isn't rated yet, and shows how many are done ("12 / 340").
  - `ratings.csv` has a `filename,rating` header and can be edited in Excel (comma- or semicolon-separated). If it has a line the tool can't read, it tells you which line and doesn't start, so no rating is lost.

## Leaderboard setup (website, optional)

The website can show a leaderboard and a community average, kept in a free Supabase database. Only a nickname, a score and the region scores are stored, only for people who opt in, and never a photo. It stays hidden until you set it up:

1. Create a Supabase project in an EU region.
2. In the **SQL Editor** run `supabase/setup.sql` (safe to run again).
3. Paste the Project URL and the anon / publishable key (it is public by design) into `SUPABASE_URL` and `SUPABASE_KEY` in `web/js/config.js`.

A weekly workflow (`.github/workflows/keepalive.yml`) keeps the free project from pausing. More in `web/FEATURES.md`.

## How the score works

The 162 face landmarks (picked from the 478 that MediaPipe finds) are lined up with a model face by moving, turning and resizing them to fit as closely as possible. The face is also compared the other way round (mirrored), so a mirrored selfie gets the same score. Each region counts differently: eyes and nose the most, brows a medium amount, the jaw line and outer lips a little, and the inner lips not at all, since they mostly change with expression. The face is compared with every model face of the chosen gender, and the closest one counts.

A head that is tilted or turned a little changes the shape in the picture as much as a different face would (5 degrees of tilt used to cost about 2 points). MediaPipe also says how deep each landmark is, so before comparing, the face is turned back in 3D: every tilt (up to 40 degrees) and turn (up to 20) is tried and the best fit is kept (`TILT_SEARCH` and the constants under it in `landmarkdetect.py`).

The remaining difference, measured relative to the distance between the eyes, is turned into a score from 0 to 10: `score = 10 / (1 + (difference / SCORE_MID) ** SCORE_POWER)`. A difference of `SCORE_MID` (0.027) gives 5, a fifth less 7.9 and a fifth more 2.5, so ordinary faces (which are all roughly as far from a model face) spread out over the scale instead of all getting about 5, while a model face rates itself 10. It was tuned on made-up faces: the model faces with the eyes, nose, mouth, brows and jaw changed by as much as real faces differ. With five model faces per gender, nine in ten of them score between about 3 and 9; with only one per gender the scores are lower (half of them under 4), because a close model face is found less often.

**By region:** the result also shows a score from 0 to 10 for the brows, eyes, nose, lips and jaw (`region_scores` in `landmarkdetect.py`). It uses the same lined-up face as the total score, so there is no separate alignment: each region's own difference to the closest model face (the same weighted RMS, relative to the distance between the eyes) goes through the same score formula, so the numbers are comparable to the main score and show which parts of the face match best and least. A region can score above or below the total, because the total is a mix of all of them. The inner lips don't count, so they have no score.

Uneven skin on the cheeks and a lopsided face (found by comparing the face with its own mirror image) add a small penalty (very uneven skin takes about a quarter off a score of 5). Only the tint of the skin is compared, so shadows don't count, and stubble, beards and black-and-white photos are not counted as uneven skin. The head must roughly face the camera: a picture is refused when the head is turned more than `MAX_TURN` (12) degrees to the side or tilted more than `MAX_TILT` (25) degrees up or down (both in `headpose.py`), and the live tip comes at 8 and 18 degrees (`HINT_TURN`, `HINT_TILT`). The face must also be close enough: a camera picture is refused when the face (forehead to chin) is under `MIN_FACE_SIZE` (30%) of the picture's height (about 60 cm from a laptop webcam), with a live tip under `HINT_FACE_SIZE` (35%); a picked photo only needs `MIN_FACE_SIZE_FILE` (15%) of its shorter side, so a group photo with tiny faces is refused (all in `landmarkdetect.py`).

On the result picture, thin white lines follow the face outline, brows, eyes, nose and lips (with a few blue accent dots), soft corner brackets mark the checked cheeks and uneven spots are tinted soft red.

The score is just a fun comparison with a few model faces, not a real measure of beauty.

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
