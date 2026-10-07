# AI Face Rater website: features and build plan

A static website that does the same as the desktop app, entirely in the browser. MediaPipe Face Landmarker runs on the visitor's device and no photo is ever uploaded. It is hosted on GitHub Pages and needs no install or account. It is in English only.

## Features (v1)

1. **Live camera**
   - The preview is mirrored, with the landmark overlay drawn in the same clean style as the desktop app.
   - A status pill and hint show the face state: "Move closer", "Turn a little towards the camera", "No face found".
   - The status is smoothed (median over recent frames, with hysteresis) so it doesn't flicker.
2. **Take photo** (button or Space). It rates the most frontal, sharpest frame from the last ~1.2 s (`RecentFrames`), not just the current frame.
3. **Rate a photo from a file.** Use the file picker, drag & drop onto the page, or paste an image (Ctrl+V).
4. **Reference toggle: Boy / Girl.** Changing it re-rates the current picture. Each gender can have several model faces (`perBoy.jpg`, `perBoy2.jpg`, ...); the face gets the score of the closest one, and with more than one the status says which ("Closest to Boy 3").
5. **Result view**
   - A big score shown as "7.4" with "/ 10", tinted by band.
   - Skin clarity and symmetry as percentages with thin bars.
   - **By region:** a score from 0 to 10 for the brows, eyes, nose, lips and jaw, best first ("Eyes 7.9, Nose 3.6, ..."), with a thin bar each; the best one is tinted blue and the weakest amber. It shows which parts of the face are closest to the model face and which are furthest. The scores are comparable to the main score.
   - The result picture with the overlay drawn on it, and a one-row legend.
   - A **Back to camera** button (or Esc).
6. **Photo tips.** At most two tips, about the photo and setup only (angle, blur, light, distance). There are the same rules as `phototips.py`.
7. **Local history** (localStorage, numbers only)
   - "Your best · Average (n photos)".
   - A ▲/▼ arrow with the difference "since last photo" (your previous photo for the same reference, none on the first). It compares the scores as shown, with one decimal, like `history.compare`.
   - The day streak.
   - A tiny sparkline of the last 12 scores for that model face (from 3 ratings on), with a text summary for screen readers.
   - A small "New personal best!" badge when the score, as shown, beats your earlier best for that model face (never on the first photo).
   - Clear history.
8. **About ("i") panel** with a short "How it works" paragraph (landmarks compared with the model faces, a fun comparison and not a real measure of beauty) and "Photos are processed on this device and never uploaded.", plus Clear history. In camera mode, while the camera is on, a caption under the score says what to do and that nothing leaves the device.
9. **Share card.** A PNG with the score, the stats and the reference, and **no photo**. Use the Web Share API, falling back to a download.
10. **Look and feel**
    - Follows the system light/dark setting.
    - Modern and clean, matching the desktop design: one accent colour, rounded cards, Segoe UI Variable / system-ui.
    - Responsive: a desktop two-column layout and a phone portrait layout. On a phone the button bar sticks to the bottom of the screen and a new result scrolls the score into view. Usable from the keyboard, with visible focus rings; the animated score is not a live region, the final score is announced once, and the result picture's label includes the score.
11. **Robust states**
    - Loading the model (with progress).
    - Camera permission denied or no camera (the file picker still works).
    - No face, or face too far or turned.
    - Unsupported browser.
12. **Installable PWA that works offline after the first visit.** The service worker caches the app, the model and the MediaPipe files.

Later (not v1): daily photo-quality challenges, and friend groups.

## Tech
- **No build step and no Node:** plain HTML, CSS and ES modules under `web/`.
- **MediaPipe:** `@mediapipe/tasks-vision`, loaded from jsDelivr pinned to an exact version (both the JS bundle and the WASM files). The model `web/face_landmarker.task` is a copy of the repo root file and is served by the site.
- **CPU for rating, GPU only for the live preview:** the still-photo landmarker (`createLandmarker('IMAGE')`) always uses the CPU delegate. MediaPipe's GPU delegate moves the landmarks by about 0.9 % of the eye distance on average (up to 1.7 %), which lowered perBoy-as-boy from 9.67 (Python) to 8.8 with the score of that time. On the CPU the browser's landmarks equal Python's to 1e-5 eye distances (the version, 0.10.21 vs 1.1.0, and the input type img/canvas/ImageData/bitmap make no difference), so the scores and the model faces from `facedata.js` agree with the desktop app. `web/tests/landmarks.html` is a debug page to compare delegates and versions.
- **Local testing:** `py -3.14 -m http.server <port> -d web`. The camera works on localhost; GitHub Pages serves HTTPS.
- **Deploy:** `.github/workflows/pages.yml` uploads `web/` to GitHub Pages. The owner enables Pages (Source: GitHub Actions) once.
- **The Python app is the reference.** `tools/export_web_data.py` writes `web/js/facedata.js` (layout, model faces from `modelfaces.py` and every scoring, tips and head-pose constant) straight from the Python modules. `tools/make_golden.py` writes `web/tests/golden.json` (landmarks and matrix in, Python results out). Re-run both whenever the Python constants change, so the site never drifts.

## Module contract (each module has one owner)

| File | Owner | Exports |
|---|---|---|
| `js/facedata.js` | generated by `tools/export_web_data.py` | `SUBSET` (indices into the 478 points), `WEIGHTS`, `MIRROR`, `REGION_POINTS` (the regions that count, as positions in the 162 list), `CHEEK_LEFT`, `CHEEK_RIGHT`, `EYE_OUTER`, `FACE_WIDTH`, `NOSE_BRIDGE`, `MODEL_FACES` {boy, girl} (each a list of model faces, 162 [x,y] each), `MODEL_FACE_NAMES` {boy, girl} ("Boy 1", ... in the same order), `CONST` {every numeric constant from landmarkdetect/headpose/phototips} |
| `js/scoring.js` | scoring agent | `rateFace(imageData, points478px, matrix, reference, source)` → `{score, clarity, symmetry, shapeError, modelFace, modelFaceCount, skinPenalty, symmetryPenalty, regions, cheeks:[{x1,y1,x2,y2,w,h,mask:Uint8Array}]}` (`modelFace` is the name of the closest model face, like "Boy 3"; `regions` is `{jaw, brows, nose, eyes, outerLips}` → 0 to 10, the score of each region of the face against that same model face, from the same lined-up face as the total, so they are comparable to `score`; the inner lips have no weight and no score), or throws `FaceError(message)`. Also `shapeError(points, reference)`, `modelErrors(points, reference)` → `[{error, name}]`, `regionScores(aligned, model)`, `symmetry(points)`, `align(points, target, weights)` |
| `js/headpose.js` | scoring agent | `headAngles(matrix)` → `{turn, tilt}`; `facingProblem(angles)` → `null \| "side" \| "up" \| "down"`; `class HeadTracker` (`add(angles)`, `.hint` → `null` or kind); `class RecentFrames` (`add(frameCanvas, angles, sharpness)`, `best()`, `reset()`); `sharpness(imageData)` |
| `js/tips.js` | features agent | `photoTips(imageData, points478px, angles)` → `string[]` (at most 2, or the single "great setup" line) |
| `js/history.js` | features agent | `addRating({reference, score, clarity, symmetry, source})`, `stats(reference)` → `{best, average, count, trend, diff, streak}` (`trend` is `"up"`, `"down"` or `null` and `diff` is the score minus the previous photo's, both from the scores rounded to one decimal as shown), `series(reference, n = 12)` → the latest `n` scores for that model face, oldest first (`[]` if none), `clearHistory()` |
| `js/share.js` | features agent | `makeShareCard({score, clarity, symmetry, reference, stats, theme})` → `Promise<Blob>`, `shareOrDownload(blob)` |
| `sw.js`, `manifest.webmanifest`, icons | features agent | offline caching and install |
| `index.html`, `css/app.css`, `js/app.js`, `js/landmarker.js`, `js/overlay.js` | UI agent | the page, MediaPipe setup (`createLandmarker(mode)`, `detect(source, ts)` → `{points478px: [[x, y, depth] * 478], matrix}` for the largest face, or `null`), camera, states, drawing, wiring of all modules |
| `tools/`, `tests/` (`web/tests/index.html` runs `golden.json` against `scoring.js` and `headpose.js` in the browser) | scoring agent | |

Points are always pixel coordinates in the analysed image: `x = x_norm * width - 0.5`, the same as Python, with the depth `z_norm * width` as a third number (the rating uses it to turn the head back before comparing; everything else only reads x and y). Until the real modules land, the UI agent uses small stubs with the same exports.
