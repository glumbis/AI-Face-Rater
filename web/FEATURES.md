# AI Face Rater website: features and build plan

A static website that does the same as the desktop app, entirely in the browser. MediaPipe Face Landmarker runs on the visitor's device and no photo is ever uploaded. It is hosted on GitHub Pages and needs no install or account. It is in English only.

## Features (v1)

1. **Live camera**
   - The preview is mirrored, with the landmark overlay drawn in the same clean style as the desktop app.
   - A status pill and hint show the face state: "Move closer", "Turn a little towards the camera", "Relax your face, no smile.", "Close your mouth.", "No face found".
   - The status is smoothed (median over recent frames, with hysteresis) so it doesn't flicker. The expression tip uses MediaPipe's blendshapes the same way (`ExpressionTracker`), and a frame with a big smile or an open mouth is not kept for Take photo.
2. **Take photo** (button or Space). It rates the most frontal, sharpest frame from the last ~1.2 s (`RecentFrames`), not just the current frame. Frames that are too small or have a big smile or an open mouth can't be rated, so they are not kept.
3. **Rate a photo from a file.** Use the file picker, drag & drop onto the page, or paste an image (Ctrl+V).
4. **Ideal toggle: Boy / Girl.** Changing it re-rates the current picture (the status says "Rating…" meanwhile, then "Rated against the boy ideal"). Each gender has its own ideal: a target and a tolerance for 17 measured proportions (eye spacing, nose width, jaw shape, ...), made from 102 real faces moved part of the way towards the model faces. There is no comparison with single model faces any more.
5. **Result view**
   - A big score shown as "7.4" with "/ 10", tinted by band. The scale is the desktop app's (see "How the score works" in the README): the ideal face would get 10, half of ordinary faces score over about 5.
   - Skin clarity and symmetry as percentages with thin bars.
   - **By region:** a score from 0 to 10 for the brows, eyes, nose, lips and jaw, best first ("Eyes 7.9, Nose 3.6, ..."), with a thin bar each; the best one is tinted blue and the weakest amber. It shows which parts of the face are closest to the ideal and which are furthest. Each region only counts its own measurements, and the scores are comparable to the main score.
   - The result picture with the overlay drawn on it, and a one-row legend.
   - A **Back to camera** button (or Esc).
6. **Photo tips.** At most two tips, about the photo and setup only (angle, blur, light, distance). There are the same rules as `phototips.py`.
7. **Local history** (localStorage, numbers only)
   - "Your best · Average (n photos)".
   - A ▲/▼ arrow with the difference "since last photo" (your previous photo for the same reference, none on the first). It compares the scores as shown, with one decimal, like `history.compare`.
   - The day streak.
   - A tiny sparkline of the last 12 scores for that ideal (from 3 ratings on), with a text summary for screen readers.
   - A small "New personal best!" badge when the score, as shown, beats your earlier best for that ideal (never on the first photo).
   - Clear history.
8. **About ("i") panel** with a short "How it works" paragraph (proportions measured after turning the head back to face the camera, a smile corrected for, compared with an ideal face, a fun comparison and not a real measure of beauty) and "Photos are processed on this device and never uploaded.", plus Clear history. In camera mode, while the camera is on, a caption under the score says what to do and that nothing leaves the device.
9. **Share card.** A PNG with the score, the stats and the reference, and **no photo**. Use the Web Share API, falling back to a download.
10. **Look and feel**
    - Follows the system light/dark setting.
    - Modern and clean, matching the desktop design: one accent colour, rounded cards, Segoe UI Variable / system-ui.
    - Responsive: a desktop two-column layout and a phone portrait layout. On a phone the button bar sticks to the bottom of the screen and a new result scrolls the score into view. Usable from the keyboard, with visible focus rings; the animated score is not a live region, the final score is announced once, and the result picture's label includes the score.
11. **Robust states**
    - Loading the model (with progress).
    - Camera permission denied or no camera (the file picker still works).
    - No face, or face too far, turned, smiling too much or with the mouth open.
    - Unsupported browser.
12. **Installable PWA that works offline after the first visit.** The service worker caches the app, the model and the MediaPipe files.

13. **Optional leaderboard and community average** (website only, hidden until set up). With a Supabase URL and key in `js/config.js`:
    - Under a result: "Better than 63% of 120 players · Average 5.8" (from `community_stats`; hidden when offline or with no players) and a small **Add to leaderboard** button. It opens a dialog with a nickname (2 to 20 characters; use a nickname, not your real name; a small blocklist) and an **unticked** consent box (13 or older, publish nickname and score, include the score and region scores in the average; no photo; delete any time under About). Submit stays disabled until it is ticked and the nickname is valid. Submitting again replaces your entry for that ideal.
    - A **Leaderboard** button (bar-chart icon) next to the "i" button shows the top 20 for Boy or Girl, starting on the selected ideal.
    - **About** has the privacy line and **Delete my leaderboard entries** (with a confirmation).
    - Only the nickname, the score (1 decimal) and the five region scores are sent, never a photo or landmarks. Each browser has a random 64-character token in localStorage; the database stores only its SHA-256, which is how "your" entries are found for deleting. Calls are plain `fetch` to `<url>/rest/v1/rpc/<function>` (`js/leaderboard.js`), never cached by the service worker.
    - The weekly `.github/workflows/keepalive.yml` calls `top_scores` so a free Supabase project doesn't pause for being idle. It skips when `config.js` has no URL and key.

### Leaderboard setup

1. Create a Supabase project (free) in an EU region.
2. Open **SQL Editor**, paste `supabase/setup.sql` and run it (running it again is safe).
3. Under **Project Settings > API** copy the Project URL and the anon / publishable key (it is meant to be public) into `SUPABASE_URL` and `SUPABASE_KEY` in `web/js/config.js`, and push. Until both are filled in, none of the leaderboard shows.

Later (not v1): daily photo-quality challenges, and friend groups.

## Tech
- **No build step and no Node:** plain HTML, CSS and ES modules under `web/`.
- **MediaPipe:** `@mediapipe/tasks-vision`, loaded from jsDelivr pinned to an exact version (both the JS bundle and the WASM files). The model `web/face_landmarker.task` is a copy of the repo root file and is served by the site.
- **CPU for rating, GPU only for the live preview:** the still-photo landmarker (`createLandmarker('IMAGE')`) always uses the CPU delegate. MediaPipe's GPU delegate moves the landmarks by about 0.9 % of the eye distance on average (up to 1.7 %), which lowered perBoy-as-boy from 9.67 (Python) to 8.8 with the score of that time. On the CPU the browser's landmarks equal Python's to 1e-5 eye distances (the version, 0.10.21 vs 1.1.0, and the input type img/canvas/ImageData/bitmap make no difference), so the scores and the ideal face from `facedata.js` agree with the desktop app. Both landmarkers also output the face blendshapes (the expression), which the rating uses to take a smile off the measurements. `web/tests/landmarks.html` is a debug page to compare delegates and versions.
- **Local testing:** `py -3.14 -m http.server <port> -d web`. The camera works on localhost; GitHub Pages serves HTTPS.
- **Deploy:** `.github/workflows/pages.yml` uploads `web/` to GitHub Pages. The owner enables Pages (Source: GitHub Actions) once.
- **The Python app is the reference.** `tools/export_web_data.py` writes `web/js/facedata.js` (layout, the ideal face from `idealface.py` and `idealdata.py`, and every scoring, tips and head-pose constant) straight from the Python modules. `tools/make_golden.py` writes `web/tests/golden.json` (landmarks, matrix and blendshapes in, Python results out). Re-run both whenever the Python constants change (and `tools/make_ideal_face.py` first when a model face picture is added), so the site never drifts.

## Module contract (each module has one owner)

| File | Owner | Exports |
|---|---|---|
| `js/facedata.js` | generated by `tools/export_web_data.py` | `SUBSET` (indices into the 478 points), `WEIGHTS`, `MIRROR`, `CHEEK_LEFT`, `CHEEK_RIGHT`, `EYE_OUTER`, `FACE_WIDTH`, `NOSE_BRIDGE` (these are for the symmetry and the cheeks), `GENDERS` (`["boy", "girl"]`), `IDEAL` (every upper-case name of `idealface.py` and `idealdata.py`: `ANCHORS`, `JAW_LEFT`, `JAW_RIGHT`, `FEATURES` as `[name, region, weight, way]`, `REGIONS`, `MAX_DEVIATION`, the expression limits and `EXPRESSION_MESSAGES`, `REFERENCE`, `NEUTRAL_EXPRESSION`, `EXPRESSION_SLOPES`, `IDEALS` as `{gender: {feature: [target, tolerance]}}`), `CONST` {every numeric constant from landmarkdetect/headpose/phototips} |
| `js/scoring.js` | scoring agent | `rateFace(imageData, points478px, matrix, reference, source, blendshapes)` → `{score, shapeError, clarity, symmetry, skinPenalty, symmetryPenalty, skinFactor, symmetryFactor, regions, deviations, cheeks:[{x1,y1,x2,y2,w,h,mask:Uint8Array}]}` (`shapeError` is the ideal error, 0 = the ideal face; `regions` is `{jaw, brows, nose, eyes, outerLips}` → 0 to 10, the score of each region from its own measurements only, so they are comparable to `score`; the inner lips have no weight and no score; `deviations` is how far each of the 17 measurements is from its target in tolerances; `blendshapes` is `{categoryName: score}` or `null`, a big smile or an open mouth is refused and a smaller expression is taken off the measurements), or throws `FaceError(message)`. The port of `idealface.py`: `turned(points)` (the face turned to look at the camera, a 3D Umeyama alignment of the 12 anchors found with Horn's quaternion method and a Jacobi eigen-solver), `measure(points)`, `neutralMeasures(points, blendshapes)`, `deviations(values, gender)`, `idealErrors(values, gender)` → `{error, regions, deviations}`, `idealError(points, gender, blendshapes)`, `regionScores(regionErrors)`, `expression(blendshapes)`, `expressionProblem(blendshapes, hint)`, `class ExpressionTracker` (`add(blendshapes)`, `.hint`, `reset()`). Also `symmetry(points)`, `align(points, target, weights)` |
| `js/headpose.js` | scoring agent | `headAngles(matrix)` → `{turn, tilt}`; `facingProblem(angles)` → `null \| "side" \| "up" \| "down"`; `class HeadTracker` (`add(angles)`, `.hint` → `null` or kind); `class RecentFrames` (`add(frameCanvas, angles, sharpness)`, `best()`, `reset()`); `sharpness(imageData)` |
| `js/tips.js` | features agent | `photoTips(imageData, points478px, angles, source)` → `string[]` (at most 2, or the single "great setup" line) |
| `js/history.js` | features agent | `addRating({reference, score, clarity, symmetry, source})`, `stats(reference)` → `{best, average, count, trend, diff, streak}` (`trend` is `"up"`, `"down"` or `null` and `diff` is the score minus the previous photo's, both from the scores rounded to one decimal as shown), `series(reference, n = 12)` → the latest `n` scores for that ideal, oldest first (`[]` if none), `clearHistory()` |
| `js/share.js` | features agent | `makeShareCard({score, clarity, symmetry, reference, stats, theme})` → `Promise<Blob>`, `shareOrDownload(blob)` |
| `js/leaderboard.js` | features agent | `enabled`, `nameProblem(name)` → `null \| string`, `submitScore({name, score, reference, regions})`, `topScores(reference)` → `[{name, score}]`, `communityStats(reference, score)` → `{players, average, below} \| null`, `deleteMyEntries()` → count; settings `SUPABASE_URL`, `SUPABASE_KEY`, `LEADERBOARD_MIN_AGE`, `CONSENT_VERSION` in `js/config.js` |
| `sw.js`, `manifest.webmanifest`, icons | features agent | offline caching and install |
| `index.html`, `css/app.css`, `js/app.js`, `js/landmarker.js`, `js/overlay.js` | UI agent | the page, MediaPipe setup (`createLandmarker(mode)`, `detect(source, ts)` → `{points478px: [[x, y, depth] * 478], matrix, blendshapes}` for the largest face, or `null`), camera, states, drawing, wiring of all modules |
| `tools/`, `tests/` (`web/tests/index.html` runs `golden.json` against `scoring.js` and `headpose.js` in the browser: the ratings of the pictures also with other blendshapes, the turned face and the measurements, the expression helpers and trackers, and the behaviours the Python tests check) | scoring agent | |

Points are always pixel coordinates in the analysed image: `x = x_norm * width - 0.5`, the same as Python, with the depth `z_norm * width` as a third number (the rating uses it to turn the head back before measuring; the symmetry and the skin only read x and y).
