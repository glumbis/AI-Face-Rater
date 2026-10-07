# AI Face Rater

Old, unfinished experiments with face landmark detection (dlib + OpenCV) and a tool for rating a face dataset.

- `face_rater_app.py` is the app: double-click `Face Rater.bat` (or run `python face_rater_app.py`). It shows your PC's default camera live and says "Face the camera" when you're facing to the side, up or down. Press **Take photo** (or Space) to rate yourself, or **Pick a photo...** to rate a picture from your PC. Choose whether to compare with the boy or girl model face; changing it rates the same picture again.
- `landmarkdetect.py` does the rating. You can also run it in the terminal: set `IMAGE_TO_RATE` at the top of the file to the picture you want to rate, or pass it on the command line: `python landmarkdetect.py picture.jpg`.
  It refuses pictures where the head is turned more than `MAX_TURN` degrees to the side or tilted more than `MAX_TILT` degrees up or down.
  It also checks skin clarity in a square on each cheek: spots whose colour differs from the skin around them by more than `SKIN_THRESHOLD` count as inconsistencies. Only the tint is compared, not the brightness, so shadows on the face don't count. Clear cheeks raise the score by up to 20% and uneven cheeks lower it by up to 20%. The cheek squares are drawn in blue and the inconsistent spots in red on the result picture.
- `perBoy.jpg` and `perGirl.jpg` are the model faces the "perfect" landmark coordinates come from.
- `datasetfacerater.py` (slider) and `datasetmaker.py` (keys 1-9, 0 = 10) are tools for rating a folder of face images into `ratings.csv`. Set `IMAGE_FOLDER` at the top of the file, or pick a folder when asked.

## Setup

Needs Python 3.10 or newer with tkinter (included in the normal Windows installer).

```
pip install -r requirements.txt
```

On Windows, `pip install dlib` may need CMake and the Visual Studio C++ build tools unless a prebuilt wheel exists for your Python version.

The dlib model `shape_predictor_68_face_landmarks.dat` isn't in this repo because it's too large. Download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2, unzip it, and put it next to the scripts.
