# AI Face Rater

Rates a face by measuring its proportions (with the landmarks of MediaPipe Face Landmarker) and comparing them with an ideal: the average of real faces, moved part of the way towards a few model faces.

**Try it in the browser, no install: https://glumbis.github.io/AI-Face-Rater/**. Everything runs on your device and no photo is uploaded. The website is in `web/` (see `web/FEATURES.md`) and deploys to GitHub Pages on every push to main.

- `face_rater_app.py` is the app: double-click `Face Rater.bat` (or run `python face_rater_app.py`).
  - The camera picture is on the left, with a small status line under it that tells you if your face is OK. If your head is turned a little (over 8 degrees) or tilted quite far up or down (over 18 degrees), your face is a bit small in the picture, or you smile or open your mouth, it gives a short tip ("Turn a little towards the camera.", "Move a little closer.", "Relax your face, no smile.") but never stops you. The tip follows the middle of the last few pictures and only goes when you're clearly back, so it doesn't flicker.
  - On the right you pick the ideal to rate against (Boy or Girl) and see your score out of 10, with skin clarity and symmetry as percentages and, under them, a score for each part of the face ("Eyes 7.9 · Jaw 6.8 · ...", best first).
  - **Take photo** (or Space) rates the picture from the last second or so where your head faced the camera best (and was sharpest, leaving out moments with a big smile), so one bad moment doesn't ruin it, and **Pick a photo…** (or Ctrl+O) rates a file. After a result the main button becomes **Back to camera** (or Esc). Changing between Boy and Girl rates the same picture again.
  - **Photo tips** (`phototips.py`): under a result you get at most two short tips about the photo and setup (head turned or tilted, blurry, too dark or too bright, one side lit more than the other, too close or too far), or a "Great setup" line when nothing is wrong. The thresholds are named constants at the top of that file.
  - **History:** every rating is saved on this computer as numbers only (time, Boy or Girl, score, skin clarity, symmetry, camera or file, never a picture) in `%APPDATA%\AI Face Rater\history.csv`. A result shows your best and average for Boy or Girl, an arrow with how much higher or lower the score is than your previous photo for it (no arrow on the first photo), and a day streak from 2 days in a row. The **i** button opens a small About window with a privacy note and **Clear history** (with a confirmation).
  - The window follows the Windows light/dark setting and fits screens from 1366×768 up to high-DPI screens.
- `landmarkdetect.py` does the rating, `idealface.py` measures the face and compares it with the ideal, and `headpose.py` works out which way the head is facing (from the head position MediaPipe gives with the landmarks). `idealdata.py` has the ideal (made by a tool, see below). `facelayout.py` lists the 162 landmarks that are drawn and used for the symmetry and skin checks, and which ones are each other's left/right partner. `face_landmarker.task` is MediaPipe's face model (3.8 MB, Apache-2.0 licence) and must stay next to the scripts. You can also run it in the terminal: set `IMAGE_TO_RATE` at the top of the file to the picture you want to rate, or pass it on the command line: `python landmarkdetect.py picture.jpg`.
- `perBoy.jpg`, `perBoy2.jpg`, ... and `perGirl.jpg`, `perGirl2.jpg`, ... are the model faces: the ideal is moved towards them (see below). There can be any number of each, but at least `perBoy.jpg` and `perGirl.jpg`. To add one:
  1. Put the picture next to the scripts with the next free number, for example `perBoy6.jpg`. A sharp, evenly lit photo of a face looking straight into the camera with a neutral expression works best.
  2. Run `python tools/make_ideal_face.py`. It works out the ideal again and writes `idealdata.py`. The first time it downloads the front photos of the [Face Research Lab London Set](https://doi.org/10.6084/m9.figshare.5047666) (about 90 MB, CC BY 4.0) into `.cache/london-set`, which is not committed.
  3. Run `python tools/export_web_data.py` and `python tools/make_golden.py`, so the website gets the new ideal too, and commit the picture with the changed files.
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

The face is measured: 17 proportions, each between landmarks that sit on bone or barely move with the expression (`FEATURES` in `idealface.py`). Before measuring, the face is turned in 3D (MediaPipe also says how deep each landmark is) so it looks straight into the camera, using points on the nose bridge, the eye corners and the forehead. So a head that is tilted, turned or nearer the camera measures the same.

| Part | Measurements |
| --- | --- |
| Eyes | how far apart the pupils are against the face width, the gap between the eyes in eye widths, the tilt of the eyes (outer corners higher is fine), the eye size |
| Brows | how high above the eyes, how arched, how tilted |
| Nose | its width against the gap between the eyes, its length against the face |
| Lips | how full both lips are, the upper lip against the groove above it |
| Jaw | how sharp the jaw corner is (a sharper corner than the target is fine), the face length against its width (a round face is short and wide), the jaw and chin width, the chin height, the middle third of the face against the lower third |

Each measurement has a target and a tolerance for each gender (`idealdata.py`, made by `tools/make_ideal_face.py`). The tolerance is how much 102 real people differ (the front photos of the Face Research Lab London Set). The target is their average, moved three quarters of the way towards the average of the model faces (but at most 1.5 tolerances). Real faces near the average look good to most people. The model faces show which way to move from it: for boys, a longer face and a sharper jaw; for girls, a narrower jaw and chin, eyes tilted up and fuller lips. A sharp jaw counts for both. How far each measurement is from its target, in tolerances, is combined (a weighted RMS, the jaw and the face length count the most) into the ideal error. The error becomes a score from 0 to 10: `score = 10 / (1 + (error / SCORE_MID) ** SCORE_POWER)`. An error of `SCORE_MID` (1.15) gives 5. Of the 102 real faces, half score over 5 and eight in ten between 3 and 7.7. The model faces score between 6 and 9.5, and a face that measures exactly the targets gets 10.

**The expression:** MediaPipe also measures the expression (its blendshapes: how much the face smiles, squints, opens the mouth). The same 102 people were also photographed smiling, which shows how much each measurement moves with a smile or a squint. That much is taken off before comparing, so a slight smile moves the score by about 0.2 on average (it used to be over 0.5, and a broad smile over 2). A big smile (`MAX_SMILE`) or an open mouth (`MAX_MOUTH_OPEN`) changes the face too much to correct, so that picture is refused with a tip, and the live preview says "Relax your face, no smile." a bit before that.

**By region:** the result also shows a score from 0 to 10 for the brows, eyes, nose, lips and jaw (`region_scores` in `landmarkdetect.py`): the same formula on only the measurements of that part, so a region can score above or below the total.

**Checked on real faces:** on the London Set, the score goes up with how attractive about 2,500 people rated those faces (rank correlation 0.27 for men, 0.40 for women), down with a rounder face (-0.56 for men), and the jaw score up with a sharper jaw (0.5). Taking the same photo again (slightly smaller and turned 4 degrees) moves the score by 0.25 on average.

Uneven skin on the cheeks and a lopsided face (found by comparing the face with its own mirror image) add a little to the error (very uneven skin takes about a quarter off a score of 5). Only the tint of the skin is compared, so shadows don't count, and stubble, beards and black-and-white photos are not counted as uneven skin. The head must roughly face the camera: a picture is refused when the head is turned more than `MAX_TURN` (12) degrees to the side or tilted more than `MAX_TILT` (25) degrees up or down (both in `headpose.py`), and the live tip comes at 8 and 18 degrees (`HINT_TURN`, `HINT_TILT`). The face must also be close enough: a camera picture is refused when the face (forehead to chin) is under `MIN_FACE_SIZE` (30%) of the picture's height (about 60 cm from a laptop webcam), with a live tip under `HINT_FACE_SIZE` (35%); a picked photo only needs `MIN_FACE_SIZE_FILE` (15%) of its shorter side, so a group photo with tiny faces is refused (all in `landmarkdetect.py`).

On the result picture, thin white lines follow the face outline, brows, eyes, nose and lips (with a few blue accent dots), soft corner brackets mark the checked cheeks and uneven spots are tinted soft red.

The score is just a fun comparison with an average and a few model faces, not a real measure of beauty.

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
