# AI Face Rater

Old, unfinished experiments with face landmark detection (dlib + OpenCV) and a tool for rating a face dataset.

- `face_rater_app.py` is the app: double-click `Face Rater.bat` (or run `python face_rater_app.py`).
  - The camera picture is on the left, with a small status line under it that tells you if your face is OK or you should turn to the camera.
  - On the right you pick the model face (Boy or Girl) and see your score out of 10, with skin clarity and symmetry as percentages.
  - **Take photo** (or Space) rates the live picture, and **Pick a photo…** (or Ctrl+O) rates a file. After a result the main button becomes **Back to camera** (or Esc). Changing Boy/Girl rates the same picture again.
  - The window follows the Windows light/dark setting and fits screens from 1366×768 up to high-DPI screens.
- `landmarkdetect.py` does the rating. You can also run it in the terminal: set `IMAGE_TO_RATE` at the top of the file to the picture you want to rate, or pass it on the command line: `python landmarkdetect.py picture.jpg`.
- `perBoy.jpg` and `perGirl.jpg` are the model faces the "perfect" landmark coordinates come from.
- `dataset_rater.py` (optional extra) rates a folder of face pictures from 1 to 10 into `ratings.csv` next to the script. Run `python dataset_rater.py path\to\folder`. Without a folder it uses the folder from last time, or asks you to pick one.
  - Keys **1-9** rate right away, **0** = 10. Or drag the slider and press **Submit** (or Enter).
  - **Backspace** or **Left arrow** undoes the last rating and goes back to that picture.
  - **S** or **Right arrow** skips a picture without rating it. Skipped pictures come back the next time you start.
  - It always starts at the first picture in the folder that isn't rated yet, and shows how many are done ("12 / 340").
  - `ratings.csv` has a `filename,rating` header and can be edited in Excel (comma- or semicolon-separated). If it has a line the tool can't read, it tells you which line and doesn't start, so no rating is lost.

## How the score works

The 68 face landmarks are lined up with a model face (boy or girl) by moving, turning and resizing them to fit as closely as possible. The face is also compared the other way round (mirrored), so a mirrored selfie gets the same score. Each region counts differently: eyes and nose the most, brows a medium amount, the jaw line and outer lips a little, and the inner lips not at all, since they mostly change with expression. The remaining difference, measured relative to the distance between the eyes, is turned into a score from 0 to 10.

Uneven skin on the cheeks and a lopsided face (found by comparing the face with its own mirror image) add a small penalty. Only the tint of the skin is compared, so shadows don't count, and stubble, beards and black-and-white photos are not counted as uneven skin. The head must face the camera: turned at most `MAX_TURN` (12) degrees to the side and tilted at most `MAX_TILT` (20) degrees up or down.

On the result picture, the landmarks are mint dots, the checked cheek squares are white outlines and uneven spots are soft red.

The score is just a fun comparison with one model face, not a real measure of beauty.

## Setup

Needs Python 3.10 or newer with tkinter (included in the normal Windows installer).

```
pip install -r requirements.txt
```

On Windows, `pip install dlib` may need CMake and the Visual Studio C++ build tools unless a prebuilt wheel exists for your Python version.

The dlib model `shape_predictor_68_face_landmarks.dat` isn't in this repo because it's too large. Download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2, unzip it, and put it next to the scripts.

## Tests

```
pip install -r requirements-dev.txt
pytest
```

The tests for `landmarkdetect.py` need `shape_predictor_68_face_landmarks.dat` next to the scripts. Without it they are skipped, and the `dataset_rater.py` tests still run.
