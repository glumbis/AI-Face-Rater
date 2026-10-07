// The rating: a port of landmarkdetect.py (rate_face, shape_error, symmetry, skin_clarity). Every number comes from
// facedata.js, which tools/export_web_data.py writes from the Python modules.
//
// rateFace(imageData, points478px, matrix, reference, source = "camera") -> { score, clarity, symmetry, shapeError,
//   modelFace, modelFaceCount, skinPenalty, symmetryPenalty, skinFactor, symmetryFactor, regions,
//   cheeks: [{ x1, y1, x2, y2, w, h, square, fraction, mask }] }
// The face gets the score of the model face of that gender it is most like: modelFace is its name ("Boy 3") and
// modelFaceCount how many that gender has. regions is a score from 0 to 10 for each of jaw, brows, nose, eyes and
// outerLips against that same model face ({ jaw: 4.1, ... }), see regionScores().
// or throws FaceError(message) (error.name === "FaceError"). source is "camera" or "file": a face that is too small is
// refused (camera: CONST.MIN_FACE_SIZE, file: CONST.MIN_FACE_SIZE_FILE) before the head angles are checked.
// clarity is null when the cheeks could not be judged (then there is no skin penalty).
// A cheek's x1..y2 is the part of its square inside the picture, w * h pixels, where its mask (Uint8Array, 1 = uneven
// skin) belongs; cheek.square is the whole square as in Python (it may reach outside the picture).
// Also exported: faceSize(points, width, height) and sizeProblem(size, current) for the live "move closer" hint.
// The picture should be no bigger than CONST.MAX_PICTURE_SIZE on the long side (shrink it before looking for the face,
// as the Python app does).
import {
  CHEEK_LEFT, CHEEK_RIGHT, CONST, EYE_OUTER, FACE_WIDTH, MIRROR, MODEL_FACE_NAMES, MODEL_FACES, NOSE_BRIDGE,
  REGION_POINTS, SUBSET, WEIGHTS,
} from './facedata.js';
import { facingProblem, headAngles } from './headpose.js';
import {
  LINEAR_LUT, gaussianBlurF32, lbgrToLab, pyRound, resizeAreaU8, resizeLinearF32,
} from './imageops.js';

export class FaceError extends Error {
  constructor(message) {
    super(message);
    this.name = 'FaceError';
  }
}

// ---------- Points ----------

// The 162 landmarks used for rating, as [x, y] pairs (or [x, y, depth] when the depth is there), from the 478 points
// of MediaPipe. The 478 can be [x, y] or [x, y, depth] arrays, {x, y} objects or a flat array x0, y0, x1, y1, ...
export function subsetPoints(points478) {
  const get = (i) => {
    if (typeof points478[0] === 'number') return [points478[2 * i], points478[2 * i + 1]];
    const p = points478[i];
    if (Array.isArray(p)) return p.length > 2 ? [p[0], p[1], p[2]] : [p[0], p[1]];
    return [p.x, p.y];
  };
  return SUBSET.map(get);
}

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

// ---------- Shape: weighted Umeyama alignment ----------

// Moves, turns and resizes points (n x [x, y]) onto target as closely as possible (weights as in WEIGHTS)
export function align(points, target, weights = WEIGHTS) {
  const n = points.length;
  let wsum = 0;
  for (let i = 0; i < n; i++) wsum += weights[i];
  const w = weights.map((v) => v / wsum);
  let pmx = 0, pmy = 0, tmx = 0, tmy = 0;
  for (let i = 0; i < n; i++) {
    pmx += w[i] * points[i][0]; pmy += w[i] * points[i][1];
    tmx += w[i] * target[i][0]; tmy += w[i] * target[i][1];
  }
  // M = (t * w)^T p: how the two point sets vary together (2 x 2), and the spread of p
  let m00 = 0, m01 = 0, m10 = 0, m11 = 0, spread = 0;
  for (let i = 0; i < n; i++) {
    const px = points[i][0] - pmx, py = points[i][1] - pmy;
    const tx = target[i][0] - tmx, ty = target[i][1] - tmy;
    m00 += tx * w[i] * px; m01 += tx * w[i] * py;
    m10 += ty * w[i] * px; m11 += ty * w[i] * py;
    spread += w[i] * (px * px + py * py);
  }
  // The best proper rotation (never a mirroring) is the one that maximises trace(R^T M), which has a closed form in 2D
  const c = m00 + m11, s = m10 - m01;
  const best = Math.hypot(c, s);
  const cos = best === 0 ? 1 : c / best, sin = best === 0 ? 0 : s / best;
  const scale = best / spread;
  return points.map((p) => {
    const x = p[0] - pmx, y = p[1] - pmy;
    return [scale * (cos * x - sin * y) + tmx, scale * (sin * x + cos * y) + tmy];
  });
}

export function weightedRms(a, b, weights = WEIGHTS) {
  let sum = 0, wsum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += weights[i] * ((a[i][0] - b[i][0]) ** 2 + (a[i][1] - b[i][1]) ** 2);
    wsum += weights[i];
  }
  return Math.sqrt(sum / wsum);
}

const eyeWidth = (points) => dist(points[EYE_OUTER[0]], points[EYE_OUTER[1]]);
// Mirrored left to right: the partner landmarks with x the other way round (a depth stays as it is)
const mirrorOf = (points) => MIRROR.map((j) => [-points[j][0], ...points[j].slice(1)]);
const RADIANS = Math.PI / 180;

// The face ([x, y, depth] points) as the camera would see it with the head tilted `tilt` degrees further down and
// turned `turn` degrees further to the right, as [x, y] points (posed() in landmarkdetect.py)
function posed(points, tilt, turn) {
  const a = tilt * RADIANS, b = turn * RADIANS;
  return points.map((p) => {
    const z = p[0] * Math.sin(b) + p[2] * Math.cos(b);
    return [p[0] * Math.cos(b) - p[2] * Math.sin(b), p[1] * Math.cos(a) - z * Math.sin(a)];
  });
}

// middle, and the numbers every step from it up to reach away on both sides
function around(middle, step, reach) {
  const n = Math.round(reach / step);
  const out = [];
  for (let i = -n; i <= n; i++) out.push(middle + i * step);
  return out;
}

// The first value with the smallest error(value), like Python's min(values, key=error)
function best(values, error) {
  let found = values[0], lowest = Infinity;
  for (const v of values) {
    const e = error(v);
    if (e < lowest) { lowest = e; found = v; }
  }
  return found;
}

// The difference between the face (162 points, with depth or without) and one model face (162 [x, y]) after lining
// them up, with the head turned back the way that fits best (CONST.TILT_SEARCH), as a share of the model's eye width.
// Returns { error, aligned } with the face as it was lined up (162 [x, y]), for regionScores()
function faceFit(points, model) {
  if (points[0].length < 3) {
    const aligned = align(points, model);
    return { error: weightedRms(aligned, model) / eyeWidth(model), aligned };
  }
  const error = (tilt, turn) => weightedRms(align(posed(points, tilt, turn), model), model);
  const { TILT_SEARCH, TURN_SEARCH, POSE_STEP, POSE_FINE_STEP } = CONST;
  let tilt = best(around(0, POSE_STEP, TILT_SEARCH), (t) => error(t, 0));
  let turn = best(around(0, POSE_STEP, TURN_SEARCH), (u) => error(tilt, u));
  tilt = best(around(tilt, POSE_FINE_STEP, POSE_STEP), (t) => error(t, turn));
  turn = best(around(turn, POSE_FINE_STEP, POSE_STEP), (u) => error(tilt, u));
  return { error: error(tilt, turn) / eyeWidth(model), aligned: align(posed(points, tilt, turn), model) };
}

// How different the face shape is from each model face of the gender: [{ error, name, aligned, model }], the error as a
// share of the model's eye width. points: 162 points (or all 478, which are reduced here), [x, y] or [x, y, depth]
// (without the depth the head isn't turned back). reference: "boy" or "girl", or one model face as a list of 162 [x, y]
// pairs. aligned is the face lined up with that model face
function modelFits(points, reference) {
  const pts = points.length === 162 ? points : subsetPoints(points);
  const gender = typeof reference === 'string' ? reference.toLowerCase() : null;
  const models = gender === null ? [reference] : MODEL_FACES[gender];
  if (!models || !models.length) throw new Error("reference must be 'boy' or 'girl'");
  const names = gender === null ? ['Model face'] : MODEL_FACE_NAMES[gender];
  const mirrored = mirrorOf(pts);
  // A mirrored photo should score the same, so keep the closer of the face and its mirror image
  return models.map((model, i) => {
    const a = faceFit(pts, model), b = faceFit(mirrored, model);
    const fit = b.error < a.error ? b : a;
    return { error: fit.error, name: names[i], aligned: fit.aligned, model };
  });
}

// modelFits without the lined-up face: [{ error, name }]
export const modelErrors = (points, reference) => modelFits(points, reference).map(({ error, name }) => ({ error, name }));

// The model face of the gender that the face is most like: { error, name } (the first one if two are as close)
export function closestModelFace(points, reference) {
  const { error, name } = modelFits(points, reference).reduce((a, b) => (b.error < a.error ? b : a));
  return { error, name };
}

// The shape error against the closest model face of the gender
export const shapeError = (points, reference) => closestModelFace(points, reference).error;

// A score from 0 to 10 for each region that counts (jaw, brows, nose, eyes, outerLips): the region's own weighted RMS
// error in the lined-up face (as a share of the model's eye width) through scoreFromError, so the numbers are
// comparable to the main score (region_scores in landmarkdetect.py)
export function regionScores(aligned, model) {
  const size = eyeWidth(model);
  const scores = {};
  for (const [name, idx] of Object.entries(REGION_POINTS)) {
    const error = weightedRms(idx.map((i) => aligned[i]), idx.map((i) => model[i]), idx.map((i) => WEIGHTS[i])) / size;
    scores[name] = scoreFromError(error);
  }
  return scores;
}

// 0 (very lopsided) to 1 (symmetric)
export function symmetry(points) {
  const pts = points.length === 162 ? points : subsetPoints(points);
  const asymmetry = weightedRms(align(mirrorOf(pts), pts), pts) / eyeWidth(pts);
  return 1 - Math.min(asymmetry / CONST.SYMMETRY_WORST, 1);
}

// 10 for an exact match, 5 at CONST.SCORE_MID (score_from_error in landmarkdetect.py)
export const scoreFromError = (err) => 10 / (1 + (err / CONST.SCORE_MID) ** CONST.SCORE_POWER);

// ---------- Skin clarity ----------

const LUM = [0.0722, 0.7152, 0.2126]; // how bright B, G, R look (LUMINANCE in landmarkdetect.py)
const lum = (a, i) => a[i] * LUM[0] + a[i + 1] * LUM[1] + a[i + 2] * LUM[2];

// Blue-green-red bytes of the rectangle x1..x2, y1..y2 of an RGBA picture
function cropBGR(rgba, w, x1, y1, x2, y2) {
  const cw = x2 - x1, ch = y2 - y1;
  const out = new Uint8Array(cw * ch * 3);
  for (let y = 0; y < ch; y++) {
    for (let x = 0; x < cw; x++) {
      const s = ((y1 + y) * w + x1 + x) * 4, d = (y * cw + x) * 3;
      out[d] = rgba[s + 2]; out[d + 1] = rgba[s + 1]; out[d + 2] = rgba[s];
    }
  }
  return out;
}

function toLinear(bytes) {
  const out = new Float32Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) out[i] = LINEAR_LUT[bytes[i]];
  return out;
}

function median(values) {
  const sorted = Array.from(values).sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

// Cheek square: [x1, y1, x2, y2] in whole pixels, like cheek_square() (int() cuts towards zero)
export function cheekSquare(pts, cheekPoints) {
  const side = CONST.CHEEK_SIZE * dist(pts[FACE_WIDTH[0]], pts[FACE_WIDTH[1]]);
  let cx = 0, cy = 0;
  for (const p of cheekPoints) { cx += pts[p][0]; cy += pts[p][1]; }
  cx /= cheekPoints.length; cy /= cheekPoints.length;
  return [Math.trunc(cx - side / 2), Math.trunc(cy - side / 2), Math.trunc(cx + side / 2), Math.trunc(cy + side / 2)];
}

// Brightness (linear light) of the skin on the nose bridge, or null if that is outside the picture
export function noseSkinBrightness(img, pts) {
  const { data, width: w, height: h } = img;
  const half = Math.max(1, pyRound(0.1 * dist(pts[EYE_OUTER[0]], pts[EYE_OUTER[1]])));
  const top = pts[NOSE_BRIDGE[0]], bottom = pts[NOSE_BRIDGE[1]];
  const cx = pyRound((top[0] + bottom[0]) / 2), cy = pyRound((top[1] + bottom[1]) / 2);
  const x1 = Math.max(cx - half, 0), x2 = Math.min(cx + half, w);
  const y1 = Math.max(cy - half, 0), y2 = Math.min(cy + half, h);
  if (x2 <= x1 || y2 <= y1) return null;
  const linear = toLinear(cropBGR(data, w, x1, y1, x2, y2));
  const values = new Float32Array(linear.length / 3);
  for (let i = 0; i < values.length; i++) values[i] = lum(linear, i * 3);
  return median(values);
}

// The share of the square's skin that is uneven, and where. Returns null if the square is too small, mostly outside
// the picture, grayscale, or mostly covered by beard
export function cheekInconsistencies(img, square, noseBrightness = null) {
  const { data, width: w, height: h } = img;
  let [x1, y1, x2, y2] = square;
  const side = x2 - x1;
  x1 = Math.max(x1, 0); y1 = Math.max(y1, 0); x2 = Math.min(x2, w); y2 = Math.min(y2, h);
  if (side < 4 || x2 - x1 < 4 || y2 - y1 < 4) return null;
  if ((x2 - x1) * (y2 - y1) < CONST.CHEEK_MIN_VISIBLE * side * side) return null;

  // A bit more than the square, so the skin around its edges is known too
  const margin = Math.floor(side / 4);
  const px1 = Math.max(x1 - margin, 0), py1 = Math.max(y1 - margin, 0);
  const px2 = Math.min(x2 + margin, w), py2 = Math.min(y2 + margin, h);
  const pw = px2 - px1, ph = py2 - py1;
  const scale = CONST.CHEEK_SAMPLE / side;
  const sw = pyRound(pw * scale), sh = pyRound(ph * scale);
  if (sw < 1 || sh < 1) return null;
  const small = resizeAreaU8(cropBGR(data, w, px1, py1, px2, py2), pw, ph, 3, sw, sh);
  const linear = toLinear(small);
  const n = sw * sh;

  const detail = gaussianBlurF32(linear, sw, sh, 3, 1);
  const surroundings = gaussianBlurF32(linear, sw, sh, 3, CONST.CHEEK_SAMPLE / 8);
  const detailLum = new Float32Array(n), surroundingsLum = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    detailLum[i] = lum(detail, i * 3);
    surroundingsLum[i] = lum(surroundings, i * 3);
  }
  const surroundingsLab = lbgrToLab(surroundings, n, new Float32Array(n * 3));

  // Only the tint is compared, so a black-and-white photo has nothing to judge
  let colour = 0;
  for (let i = 0; i < n; i++) colour += Math.abs(surroundingsLab[i * 3 + 1]) + Math.abs(surroundingsLab[i * 3 + 2]);
  if (colour / (n * 2) < CONST.MIN_SKIN_COLOUR) return null;

  // Hair (stubble, beard): much darker than its surroundings and not redder, or much darker than the nose bridge
  const detailLab = lbgrToLab(detail, n, new Float32Array(n * 3));
  const hair = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const redness = detailLab[i * 3 + 1] - surroundingsLab[i * 3 + 1];
    let isHair = detailLum[i] < CONST.HAIR_DARKER * surroundingsLum[i] && redness < CONST.HAIR_MAX_REDNESS;
    if (noseBrightness !== null && detailLum[i] < CONST.BEARD_DARKER * noseBrightness) isHair = true;
    hair[i] = isHair ? 1 : 0;
  }

  // Give each spot the brightness of its surroundings, so only a different tint counts, then the Lab distance (Delta E)
  const relit = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const f = surroundingsLum[i] / Math.max(detailLum[i], 1e-4);
    relit[i * 3] = detail[i * 3] * f; relit[i * 3 + 1] = detail[i * 3 + 1] * f; relit[i * 3 + 2] = detail[i * 3 + 2] * f;
  }
  const relitLab = lbgrToLab(relit, n, new Float32Array(n * 3));
  const deltaE = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const dl = relitLab[i * 3] - surroundingsLab[i * 3];
    const da = relitLab[i * 3 + 1] - surroundingsLab[i * 3 + 1];
    const db = relitLab[i * 3 + 2] - surroundingsLab[i * 3 + 2];
    deltaE[i] = detailLum[i] < 0.01 ? 0 : Math.sqrt(dl * dl + da * da + db * db);
  }

  // Back to the picture's own size, and only the square itself, not the extra margin
  const big = resizeLinearF32(deltaE, sw, sh, pw, ph);
  const bigHair = resizeLinearF32(hair, sw, sh, pw, ph);
  const cw = x2 - x1, ch = y2 - y1, ox = x1 - px1, oy = y1 - py1;
  const mask = new Uint8Array(cw * ch);
  let skinCount = 0, maskCount = 0;
  for (let y = 0; y < ch; y++) {
    for (let x = 0; x < cw; x++) {
      const k = (y + oy) * pw + x + ox;
      if (bigHair[k] > 0.5) continue;
      skinCount++;
      if (big[k] > CONST.SKIN_THRESHOLD) { mask[y * cw + x] = 1; maskCount++; }
    }
  }
  // A cheek that is mostly beard can't be judged
  if (skinCount / (cw * ch) < 1 - CONST.HAIR_MAX_COVER) return null;
  return { fraction: maskCount / skinCount, mask, w: cw, h: ch };
}

// 0 (very uneven cheeks) to 1 (clear), or null when no cheek could be judged. Also gives the cheek results
export function skinClarity(img, pts) {
  const fractions = [];
  const cheeks = [];
  const noseBrightness = noseSkinBrightness(img, pts);
  for (const points of [CHEEK_LEFT, CHEEK_RIGHT]) {
    const square = cheekSquare(pts, points);
    const found = cheekInconsistencies(img, square, noseBrightness);
    if (found === null) continue;
    fractions.push(found.fraction);
    const mx = Math.max(square[0], 0), my = Math.max(square[1], 0);
    // x1..y2 is the part of the square inside the picture (w * h, where the mask belongs); square is the whole one
    cheeks.push({
      x1: mx, y1: my, x2: mx + found.w, y2: my + found.h, w: found.w, h: found.h, square,
      fraction: found.fraction, mask: found.mask,
    });
  }
  if (!fractions.length) return { clarity: null, cheeks };
  const mean = fractions.reduce((a, b) => a + b, 0) / fractions.length;
  return { clarity: 1 - Math.min(mean / CONST.SKIN_WORST_FRACTION, 1), cheeks };
}

// ---------- The rating ----------

// ---------- How close the face is ----------

// The face height (landmarks 10 to 152) as a share of the picture's shorter side. pts: the 162 [x, y] pairs (or 478);
// Call it as faceSize(points, width, height) or faceSize(points, picture) with anything that has width and height
// (ImageData, canvas)
export function faceSize(pts, widthOrPicture, height) {
  const p = pts.length === 162 ? pts : subsetPoints(pts);
  const [top, chin] = CONST.FACE_HEIGHT_POINTS;
  const w = typeof widthOrPicture === 'number' ? widthOrPicture : widthOrPicture.width;
  const h = typeof widthOrPicture === 'number' ? height : widthOrPicture.height;
  return Math.abs(p[chin][1] - p[top][1]) / Math.min(w, h);
}

// What to tell the live preview about the distance: null (fine), "near" (a little closer) or "far" (much closer).
// current is what is showing now; it stays until the face has grown a margin past the limit (no flickering)
export function sizeProblem(size, current = null) {
  const margin = CONST.HINT_SIZE_MARGIN;
  if (size < CONST.MIN_FACE_SIZE + (current === 'far' ? margin : 0)) return 'far';
  if (size < CONST.HINT_FACE_SIZE + (current === 'near' || current === 'far' ? margin : 0)) return 'near';
  return null;
}

// Rates the face. imageData: the picture (ImageData or {data: RGBA, width, height}); points478px: the 478 landmarks in
// the picture's pixels (x = x_norm * width - 0.5), as [x, y, depth] with depth = z_norm * width (without the depth
// the head isn't turned back before comparing); matrix: MediaPipe's head matrix (see headpose.js; may be null);
// reference: "boy" or "girl"; source: "camera" (a webcam picture, the face must fill more of it) or "file"
export function rateFace(imageData, points478px, matrix, reference, source = 'camera') {
  const pts = subsetPoints(points478px);
  // Too far away is checked first, then the way the head faces
  const size = faceSize(pts, imageData);
  if (source === 'camera') {
    if (size < CONST.MIN_FACE_SIZE) throw new FaceError(CONST.SIZE_MESSAGES.far);
  } else if (size < CONST.MIN_FACE_SIZE_FILE) {
    throw new FaceError(CONST.FILE_TOO_SMALL_MESSAGE);
  }
  const problem = facingProblem(headAngles(matrix));
  if (problem !== null) throw new FaceError(CONST.REFUSED_MESSAGES[problem]);

  const { clarity, cheeks } = skinClarity(imageData, pts);
  const fits = modelFits(pts, reference);
  const closest = fits.reduce((a, b) => (b.error < a.error ? b : a));
  const err = closest.error;
  const skinPenalty = clarity === null ? 0 : CONST.SKIN_PENALTY * (1 - clarity);
  const sym = symmetry(pts);
  const symmetryPenalty = CONST.SYMMETRY_PENALTY * (1 - sym);
  let score = scoreFromError(err + skinPenalty + symmetryPenalty);
  // How many times smaller each penalty made the score
  const skinFactor = score / scoreFromError(err + symmetryPenalty);
  const symmetryFactor = score / scoreFromError(err + skinPenalty);
  score = Number.isNaN(score) ? 0 : Math.min(Math.max(score, 0), 10);
  return {
    score, shapeError: err, modelFace: closest.name, modelFaceCount: fits.length, clarity, skinPenalty, skinFactor,
    symmetry: sym, symmetryPenalty, symmetryFactor, regions: regionScores(closest.aligned, closest.model), cheeks,
  };
}
