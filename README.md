# AI Face Rater

Old, unfinished experiments with face landmark detection (dlib + OpenCV) and a tool for rating a face dataset.

- `landmarkdetect.py` rates a face by comparing its landmarks to a model face. Set `IMAGE_TO_RATE` at the top of the file to the picture you want to rate, or pass it on the command line: `python landmarkdetect.py picture.jpg`.
- `perBoy.jpg` and `perGirl.jpg` are the model faces the "perfect" landmark coordinates come from.
- `datasetfacerater.py` (slider) and `datasetmaker.py` (keys 1-9, 0 = 10) are tools for rating a folder of face images into `ratings.csv`. Set `IMAGE_FOLDER` at the top of the file, or pick a folder when asked.

## Setup

Needs Python 3.10 or newer with tkinter (included in the normal Windows installer).

```
pip install -r requirements.txt
```

On Windows, `pip install dlib` may need CMake and the Visual Studio C++ build tools unless a prebuilt wheel exists for your Python version.

The dlib model `shape_predictor_68_face_landmarks.dat` isn't in this repo because it's too large. Download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2, unzip it, and put it next to the scripts.
