// The rating: a port of landmarkdetect.py (rate_face, symmetry, skin_clarity) and idealface.py (the measurements, the
// expression and the ideal error). Every number comes from facedata.js, which tools/export_web_data.py writes from the
// Python modules.
//
// rateFace(imageData, points478px, matrix, reference, source = "camera", blendshapes = null) -> { score, shapeError,
//   clarity, skinPenalty, skinFactor, symmetry, symmetryPenalty, symmetryFactor, regions, deviations,
//   cheeks: [{ x1, y1, x2, y2, w, h, square, fraction, mask }] }
// or throws FaceError(message) (error.name === "FaceError"). The face is measured (eye spacing, nose width, jaw shape and
// so on, see measure()) and each measurement is compared with the ideal of the gender ("boy" or "girl"). shapeError is
// the weighted RMS of those differences (0 = the ideal face, a typical face is about 1). regions is a score from 0 to 10
// for each of jaw, brows, nose, eyes and outerLips ({ jaw: 4.1, ... }), see regionScores(). deviations is how far each
// measurement is from its target in tolerances (negative = too little).
// blendshapes is MediaPipe's expression measurements as { categoryName: score } (or null): a big smile or an open mouth
// is refused, and a smaller expression is taken off the measurements before comparing.
// source is "camera" or "file": a face that is too small is refused (camera: CONST.MIN_FACE_SIZE, file:
// CONST.MIN_FACE_SIZE_FILE) before the head angles and the expression are checked.
// clarity is null when the cheeks could not be judged (then there is no skin penalty).
// A cheek's x1..y2 is the part of its square inside the picture, w * h pixels, where its mask (Uint8Array, 1 = uneven
// skin) belongs; cheek.square is the whole square as in Python (it may reach outside the picture).
// Also exported: faceSize(points, width, height) and sizeProblem(size, current) for the live "move closer" hint, and
// ExpressionTracker and expressionProblem() for the live "relax your face" hint.
// The picture should be no bigger than CONST.MAX_PICTURE_SIZE on the long side (shrink it before looking for the face,
// as the Python app does).
import {
  CHEEK_LEFT, CHEEK_RIGHT, CONST, EYE_OUTER, FACE_WIDTH, GENDERS, IDEAL, MIRROR, NOSE_BRIDGE, SUBSET, WEIGHTS,
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

// 0 (very lopsided) to 1 (symmetric)
export function symmetry(points) {
  const pts = points.length === 162 ? points : subsetPoints(points);
  const asymmetry = weightedRms(align(mirrorOf(pts), pts), pts) / eyeWidth(pts);
  return 1 - Math.min(asymmetry / CONST.SYMMETRY_WORST, 1);
}

// 10 for an exact match, 5 at CONST.SCORE_MID (score_from_error in landmarkdetect.py)
export const scoreFromError = (err) => 10 / (1 + (err / CONST.SCORE_MID) ** CONST.SCORE_POWER);

// ---------- The ideal face ----------
// A port of idealface.py: 17 proportions measured on all 478 landmarks after turning the face to look straight at the
// camera, each compared with the target and tolerance of the gender (IDEAL.IDEALS), and combined into an error.
const { ANCHORS, JAW_LEFT, JAW_RIGHT, FEATURES, FEATURE_NAMES, MAX_DEVIATION, EXPRESSIONS } = IDEAL;

// The eigenvalues and eigenvectors of a small symmetric matrix by Jacobi rotations: { values, vectors } where
// vectors[k] is the eigenvector that belongs to values[k]
function jacobiEigen(matrix) {
  const n = matrix.length;
  const a = matrix.map((row) => row.slice());
  const v = a.map((_, i) => a.map((__, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0, all = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { all += a[i][j] ** 2; if (i !== j) off += a[i][j] ** 2; }
    if (off <= 1e-30 * all) break;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        if (a[p][q] === 0) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        // A = J^T A J (columns first, then rows), and the same turn on the eigenvectors
        for (let k = 0; k < n; k++) {
          const kp = a[k][p], kq = a[k][q];
          a[k][p] = c * kp - s * kq; a[k][q] = s * kp + c * kq;
        }
        for (let k = 0; k < n; k++) {
          const pk = a[p][k], qk = a[q][k];
          a[p][k] = c * pk - s * qk; a[q][k] = s * pk + c * qk;
        }
        for (let k = 0; k < n; k++) {
          const kp = v[k][p], kq = v[k][q];
          v[k][p] = c * kp - s * kq; v[k][q] = s * kp + c * kq;
        }
      }
    }
  }
  return { values: a.map((row, i) => row[i]), vectors: a.map((_, k) => v.map((row) => row[k])) };
}

// The 3 x 3 turn and the scale that bring the points `from` closest to the points `to` (lists of [x, y, z], both already
// moved so their middle is the origin). Horn's method: the best proper rotation is the biggest eigenvector of a 4 x 4
// matrix, as a quaternion. numpy's Umeyama in turned() (an SVD with the sign fix) finds the same turn, because the best
// one is unique
function bestTurn(from, to) {
  const S = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]; // S[a][b] = sum of from_a * to_b
  let spread = 0;
  for (let i = 0; i < from.length; i++) {
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) S[a][b] += from[i][a] * to[i][b];
    spread += from[i][0] ** 2 + from[i][1] ** 2 + from[i][2] ** 2;
  }
  const [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = S;
  const { values, vectors } = jacobiEigen([
    [xx + yy + zz, yz - zy, zx - xz, xy - yx],
    [yz - zy, xx - yy - zz, xy + yx, zx + xz],
    [zx - xz, xy + yx, -xx + yy - zz, yz + zy],
    [xy - yx, zx + xz, yz + zy, -xx - yy + zz],
  ]);
  let top = 0;
  for (let k = 1; k < 4; k++) if (values[k] > values[top]) top = k;
  const length = Math.hypot(...vectors[top]);
  const [w, x, y, z] = vectors[top].map((v) => v / length);
  const rotation = [
    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
  ];
  // The scale is how much of the target the turned points reach, over how big the points are
  let reach = 0;
  for (let i = 0; i < from.length; i++) {
    for (let a = 0; a < 3; a++) {
      reach += to[i][a] * (rotation[a][0] * from[i][0] + rotation[a][1] * from[i][1] + rotation[a][2] * from[i][2]);
    }
  }
  return { rotation, scale: reach / spread };
}

// All 478 landmarks as [x, y, depth] (a missing depth is 0). They can be [x, y, depth] arrays or {x, y, z} objects
const fullPoints = (points478) => Array.from(points478, (p) => (Array.isArray(p) ? [p[0], p[1], p[2] ?? 0] : [p.x, p.y, p.z ?? 0]));

const middleOf = (list) => [0, 1, 2].map((a) => list.reduce((sum, p) => sum + p[a], 0) / list.length);

// The face (478 x [x, y, depth]) moved, turned in 3D and resized so its ANCHORS lie as close as possible to those of the
// reference face (turned() in idealface.py), so it looks straight into the camera, in the size of the reference
export function turned(points, reference = IDEAL.REFERENCE) {
  const all = fullPoints(points);
  const p = ANCHORS.map((i) => all[i]);
  const pMid = middleOf(p), qMid = middleOf(reference);
  const { rotation: r, scale } = bestTurn(p.map((v) => v.map((x, a) => x - pMid[a])), reference.map((v) => v.map((x, a) => x - qMid[a])));
  return all.map((pt) => {
    const d = [pt[0] - pMid[0], pt[1] - pMid[1], pt[2] - pMid[2]];
    return [0, 1, 2].map((a) => scale * (r[a][0] * d[0] + r[a][1] * d[1] + r[a][2] * d[2]) + qMid[a]);
  });
}

// The face as seen straight from the front: x and y of turned(), 478 x [x, y]
export const frontal = (points, reference) => turned(points, reference).map((p) => [p[0], p[1]]);

const distance = (f, a, b) => Math.hypot(f[a][0] - f[b][0], f[a][1] - f[b][1]);

// How much of the jaw line's bend is in one corner: the largest turn between two neighbouring pieces of the line, as a
// share of all its turning (_jaw_sharpness in idealface.py)
function jawSharpness(f, chain) {
  const angles = [];
  for (let i = 1; i < chain.length; i++) {
    angles.push(Math.atan2(f[chain[i]][1] - f[chain[i - 1]][1], f[chain[i]][0] - f[chain[i - 1]][0]));
  }
  // Each turn like numpy.unwrap gives it: a jump of half a turn or more is taken the short way round instead
  const whole = 2 * Math.PI;
  const turns = [];
  for (let i = 1; i < angles.length; i++) {
    const jump = angles[i] - angles[i - 1];
    let wrapped = ((((jump + Math.PI) % whole) + whole) % whole) - Math.PI;
    if (wrapped === -Math.PI && jump > 0) wrapped = Math.PI;
    turns.push(Math.abs(Math.abs(jump) < Math.PI ? jump : wrapped));
  }
  return Math.max(...turns) / Math.max(turns.reduce((a, b) => a + b, 0), 1e-9);
}

// How far the highest point of a brow is above the straight line from its inner to its outer end
function browArch(f, inner, peak, outer) {
  const a = f[inner], b = f[outer], c = f[peak];
  const bx = b[0] - a[0], by = b[1] - a[1];
  const t = Math.min(Math.max(((c[0] - a[0]) * bx + (c[1] - a[1]) * by) / (bx * bx + by * by), 0), 1);
  return Math.hypot(c[0] - (a[0] + t * bx), c[1] - (a[1] + t * by));
}

// The measurements of the face (478 landmarks) as { name: value }, in the order of IDEAL.FEATURES (measure() in
// idealface.py; see there for what each one is)
export function measure(points, reference) {
  const f = frontal(points, reference);
  const d = (a, b) => distance(f, a, b);
  const faceWidth = d(234, 454);
  const faceHeight = d(10, 152);
  const eyeWidthAvg = (d(33, 133) + d(263, 362)) / 2;
  const eyeSpan = d(33, 263);
  const innerCorners = d(133, 362);
  const lips = d(0, 13) + d(14, 17);
  return {
    // Eyes
    eyeSpacing: d(468, 473) / faceWidth,
    innerEyeGap: innerCorners / eyeWidthAvg,
    canthalTilt: ((f[133][1] - f[33][1]) / d(33, 133) + (f[362][1] - f[263][1]) / d(263, 362)) / 2,
    eyeSize: eyeWidthAvg / faceWidth,
    // Brows
    browHeight: ((f[159][1] - f[105][1]) + (f[386][1] - f[334][1])) / 2 / eyeSpan,
    browArch: (browArch(f, 107, 105, 70) + browArch(f, 336, 334, 300)) / 2 / eyeSpan,
    browTilt: ((f[107][1] - f[70][1]) + (f[336][1] - f[300][1])) / 2 / eyeSpan,
    // Nose
    noseWidth: d(129, 358) / innerCorners,
    noseLength: d(168, 2) / faceHeight,
    // Lips
    lipFullness: lips / d(2, 152),
    upperLip: d(0, 13) / d(2, 0),
    // Jaw and face shape
    jawSharpness: (jawSharpness(f, JAW_LEFT) + jawSharpness(f, JAW_RIGHT)) / 2,
    faceLength: faceHeight / faceWidth,
    jawWidth: d(58, 288) / faceWidth,
    chinWidth: d(148, 377) / faceWidth,
    chinHeight: d(17, 152) / d(2, 152),
    thirds: d(9, 2) / d(2, 152),
  };
}

// ---------- The expression ----------
// The IDEAL.EXPRESSIONS as a list [smile, squint, mouthOpen, upperLipUp], each the average of the left and right side,
// from blendshapes ({ categoryName: score }). null when there are none: then nothing is corrected
export function expression(blendshapes) {
  if (!blendshapes || !Object.keys(blendshapes).length) return null;
  return Object.values(EXPRESSIONS).map((names) => names.reduce((sum, name) => sum + (blendshapes[name] ?? 0), 0) / names.length);
}

function expressionLimit(smile, mouthOpen, maxSmile, maxMouthOpen) {
  if (mouthOpen > maxMouthOpen) return 'mouthOpen';
  if (smile > maxSmile) return 'smile';
  return null;
}

// "smile" or "mouthOpen" when the face isn't neutral enough to rate (or, with hint = true, enough to give the live tip),
// else null (also with no blendshapes)
export function expressionProblem(blendshapes, hint = false) {
  const found = expression(blendshapes);
  if (found === null) return null;
  const [smile, , mouthOpen] = found;
  if (hint) return expressionLimit(smile, mouthOpen, IDEAL.HINT_SMILE, IDEAL.HINT_MOUTH_OPEN);
  return expressionLimit(smile, mouthOpen, IDEAL.MAX_SMILE, IDEAL.MAX_MOUTH_OPEN);
}

// The live tip about the expression ("smile", "mouthOpen" or null), from the middle (median) of the last
// EXPRESSION_SECONDS, with a margin so the tip doesn't flicker (like HeadTracker in headpose.js)
export class ExpressionTracker {
  constructor(seconds = IDEAL.EXPRESSION_SECONDS) {
    this.seconds = seconds;
    this.history = []; // [time, smile, mouthOpen]
    this.problem = null;
  }

  // null or "smile" / "mouthOpen": the tip to show (the same as .problem)
  get hint() {
    return this.problem;
  }

  reset() {
    this.history = [];
    this.problem = null;
  }

  // blendshapes: { categoryName: score } of the newest picture (or null). now is in seconds. Returns the tip
  add(blendshapes, now = performance.now() / 1000) {
    const found = expression(blendshapes);
    if (found !== null) this.history.push([now, found[0], found[2]]);
    while (this.history.length && now - this.history[0][0] > this.seconds) this.history.shift();
    if (!this.history.length) {
      this.problem = null;
      return null;
    }
    const smile = median(this.history.map((h) => h[1])), mouthOpen = median(this.history.map((h) => h[2]));
    const margin = IDEAL.HINT_EXPRESSION_MARGIN;
    if (this.problem !== null) {
      const stay = expressionLimit(smile, mouthOpen, IDEAL.HINT_SMILE - margin, IDEAL.HINT_MOUTH_OPEN - margin);
      if (stay === this.problem) return stay;
    }
    this.problem = expressionLimit(smile, mouthOpen, IDEAL.HINT_SMILE, IDEAL.HINT_MOUTH_OPEN);
    return this.problem;
  }
}

// The measurements as they would be with a neutral face: each one moves with the expression by about the same amount in
// every face (IDEAL.EXPRESSION_SLOPES), so that is taken off
export function neutralMeasures(points, blendshapes = null) {
  const values = measure(points);
  const found = expression(blendshapes);
  if (found === null) return values;
  const change = found.map((v, i) => v - IDEAL.NEUTRAL_EXPRESSION[i]);
  const out = {};
  for (const [name, value] of Object.entries(values)) {
    out[name] = value - change.reduce((sum, c, i) => sum + c * IDEAL.EXPRESSION_SLOPES[name][i], 0);
  }
  return out;
}

// ---------- Against the ideal ----------
function checkGender(gender) {
  const name = String(gender).toLowerCase();
  if (!GENDERS.includes(name)) throw new Error("gender must be 'boy' or 'girl'");
  return name;
}

// How far each measurement is from the target of the gender, in tolerances (0 = on target), with the one-sided ones at 0
// when they are past the target the good way
export function deviations(values, gender) {
  const ideal = IDEAL.IDEALS[checkGender(gender)];
  const found = {};
  for (const [name, , , way] of FEATURES) {
    const [target, tolerance] = ideal[name];
    let z = (values[name] - target) / tolerance;
    if (way * z > 0) z = 0;
    found[name] = Math.min(Math.max(z, -MAX_DEVIATION), MAX_DEVIATION);
  }
  return found;
}

function combined(z, names) {
  const weight = Object.fromEntries(FEATURES.map(([name, , w]) => [name, w]));
  return Math.sqrt(names.reduce((sum, n) => sum + weight[n] * z[n] ** 2, 0) / names.reduce((sum, n) => sum + weight[n], 0));
}

// { error, regions, deviations }: the weighted RMS of the deviations over all measurements, and for each region over its
// own. 0 is the ideal face, a typical face is about 1
export function idealErrors(values, gender) {
  const z = deviations(values, gender);
  const regions = {};
  for (const region of IDEAL.REGIONS) {
    regions[region] = combined(z, FEATURES.filter((f) => f[1] === region).map((f) => f[0]));
  }
  return { error: combined(z, FEATURE_NAMES), regions, deviations: z };
}

// idealErrors of the face (478 landmarks) with the expression taken off first
export const idealError = (points, gender, blendshapes = null) => idealErrors(neutralMeasures(points, blendshapes), gender);

// A score from 0 to 10 for each region (jaw, brows, nose, eyes, outerLips) from its own error, through the same formula
// as the total, so the numbers are comparable to the main score (region_scores in landmarkdetect.py)
export function regionScores(regionErrors) {
  return Object.fromEntries(Object.entries(regionErrors).map(([name, error]) => [name, scoreFromError(error)]));
}

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
// the picture's pixels (x = x_norm * width - 0.5), as [x, y, depth] with depth = z_norm * width; matrix: MediaPipe's
// head matrix (see headpose.js; may be null); reference: "boy" or "girl"; source: "camera" (a webcam picture, the face
// must fill more of it) or "file"; blendshapes: MediaPipe's expression measurements as { categoryName: score }, or null
// (then there is no correction and no refusal for a smile)
export function rateFace(imageData, points478px, matrix, reference, source = 'camera', blendshapes = null) {
  const gender = checkGender(reference);
  const pts = subsetPoints(points478px);
  // Too far away is checked first, then the way the head faces, then the expression
  const size = faceSize(pts, imageData);
  if (source === 'camera') {
    if (size < CONST.MIN_FACE_SIZE) throw new FaceError(CONST.SIZE_MESSAGES.far);
  } else if (size < CONST.MIN_FACE_SIZE_FILE) {
    throw new FaceError(CONST.FILE_TOO_SMALL_MESSAGE);
  }
  const problem = facingProblem(headAngles(matrix));
  if (problem !== null) throw new FaceError(CONST.REFUSED_MESSAGES[problem]);
  const unrelaxed = expressionProblem(blendshapes);
  if (unrelaxed !== null) throw new FaceError(IDEAL.EXPRESSION_MESSAGES[unrelaxed]);

  const { clarity, cheeks } = skinClarity(imageData, pts);
  const { error: err, regions: regionErrors, deviations: z } = idealError(points478px, gender, blendshapes);
  const skinPenalty = clarity === null ? 0 : CONST.SKIN_PENALTY * (1 - clarity);
  const sym = symmetry(pts);
  const symmetryPenalty = CONST.SYMMETRY_PENALTY * (1 - sym);
  let score = scoreFromError(err + skinPenalty + symmetryPenalty);
  // How many times smaller each penalty made the score
  const skinFactor = score / scoreFromError(err + symmetryPenalty);
  const symmetryFactor = score / scoreFromError(err + skinPenalty);
  score = Number.isNaN(score) ? 0 : Math.min(Math.max(score, 0), 10);
  return {
    score, shapeError: err, clarity, skinPenalty, skinFactor, symmetry: sym, symmetryPenalty, symmetryFactor,
    regions: regionScores(regionErrors), deviations: z, cheeks,
  };
}
