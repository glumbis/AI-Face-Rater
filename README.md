# AI Face Rater

Old, unfinished experiments with face landmark detection (dlib + OpenCV) and a tool for rating a face dataset.

- `landmarkdetect.py` rates a face by comparing its landmarks to a model face. Set `IMAGE_TO_RATE` at the top of the file to the picture you want to rate.
- `perBoy.jpg` and `perGirl.jpg` are the model faces the "perfect" landmark coordinates come from.
- `datasetfacerater.py` / `datasetmaker.py` are tools for rating a folder of face images into `ratings.csv`.

The dlib model `shape_predictor_68_face_landmarks.dat` isn't in this repo because it's too large. Download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2, unzip it, and put it next to the scripts.
