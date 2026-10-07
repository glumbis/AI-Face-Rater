// Which way the head is facing: a port of headpose.py. All limits come from facedata.js (made from headpose.py).
//
//   headAngles(matrix) -> { turn, tilt } in degrees, or null
//   facingProblem(angles) -> null | "side" | "up" | "down"        (angles may be null: then the picture is rated)
//   class HeadTracker      add(angles, now?) -> hint,  .hint,  .problem,  reset()
//   class RecentFrames     add(frame, angles, sharp, now?),  best() -> frame | null,  bestEntry(),  reset()
//   sharpness(imageData) -> number
//
// matrix: MediaPipe's facial transformation matrix. It can be an array of 4 rows (or 3), a { rows, columns, data }
// object as MediaPipe's JS API returns (data is column-major), or a flat array of 16 numbers (the layout is
// recognised: a row-major matrix has 0, 0, 0 where a column-major one has its translation).
// now is in seconds (default: performance.now() / 1000).
import { CONST_HEADPOSE as HP } from './facedata.js';
import { grayFromRGBA, laplacianVariance, pyRound, resizeAreaU8 } from './imageops.js';

const DEG = 180 / Math.PI;

// Rows of the 3 x 3 turning part of the matrix, or null
function rotationRows(matrix) {
  if (!matrix) return null;
  let rows = null;
  if (Array.isArray(matrix) && Array.isArray(matrix[0])) {
    rows = matrix.slice(0, 3).map((r) => Array.from(r).slice(0, 3));
  } else {
    let data = matrix.data ?? matrix;
    if (!data || typeof data.length !== 'number') return null;
    const n = matrix.rows && matrix.columns ? matrix.rows : Math.round(Math.sqrt(data.length));
    const columns = matrix.columns ?? n;
    if (n * columns !== data.length || n < 3 || columns < 3) return null;
    let columnMajor = true;
    if (!matrix.data && n === 4) {
      // flat array: row-major has the bottom row 0, 0, 0, 1, column-major has its translation there
      columnMajor = !(data[12] === 0 && data[13] === 0 && data[14] === 0);
    }
    rows = [0, 1, 2].map((i) => [0, 1, 2].map((j) => (columnMajor ? data[j * n + i] : data[i * columns + j])));
  }
  if (rows.length !== 3 || rows.some((r) => r.length !== 3)) return null;
  return rows;
}

// Turn (positive when the face points to the right in the picture) and tilt (positive when looking down), in degrees.
// The turning part of the matrix says where the front of the face points. TILT_OFFSET is taken off the tilt.
export function headAngles(matrix) {
  const r = rotationRows(matrix);
  if (!r || r.some((row) => row.some((v) => typeof v !== 'number' || !Number.isFinite(v)))) return null;
  // The matrix also resizes the head: columns of length 1 are left with only the turning
  const lengths = [0, 1, 2].map((j) => Math.hypot(r[0][j], r[1][j], r[2][j]));
  if (lengths.some((l) => l < 1e-9)) return null;
  const forward = [r[0][2] / lengths[2], r[1][2] / lengths[2], r[2][2] / lengths[2]];
  const turn = Math.atan2(forward[0], forward[2]) * DEG;
  const tilt = Math.atan2(-forward[1], forward[2]) * DEG - HP.TILT_OFFSET;
  return { turn, tilt };
}

// "side", "down" or "up" if the angles are over the limits, otherwise null
export function problemFor(turn, tilt, maxTurn = HP.MAX_TURN, maxTilt = HP.MAX_TILT) {
  if (Math.abs(turn) > maxTurn) return 'side';
  if (tilt > maxTilt) return 'down';
  if (tilt < -maxTilt) return 'up';
  return null;
}

// null if the face is facing the camera well enough to be rated, otherwise "side", "down" or "up"
export function facingProblem(angles) {
  if (!angles) return null;
  return problemFor(angles.turn, angles.tilt);
}

const nowSeconds = () => performance.now() / 1000;

function median(values) {
  const sorted = values.slice().sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

// ---------- Live preview: steady angles ----------

export class HeadTracker {
  constructor(frames = HP.SMOOTH_FRAMES, seconds = HP.SMOOTH_SECONDS) {
    this.frames = frames;
    this.seconds = seconds;
    this.history = [];
    this.problem = null;
  }

  // null or "side" / "down" / "up": the tip to show (the same as .hint)
  get hint() {
    return this.problem;
  }

  reset() {
    this.history = [];
    this.problem = null;
  }

  // angles: { turn, tilt } of the newest picture. Returns the hint
  add(angles, now = nowSeconds()) {
    if (!angles) return this.problem;
    this.history.push([now, angles.turn, angles.tilt]);
    while (this.history.length > this.frames) this.history.shift();
    while (this.history.length && now - this.history[0][0] > this.seconds) this.history.shift();
    this.problem = this.nextProblem(median(this.history.map((h) => h[1])), median(this.history.map((h) => h[2])));
    return this.problem;
  }

  // The tip comes at the hint limit and only goes at the limit minus the margin, so it doesn't flicker
  nextProblem(turn, tilt) {
    if (this.problem === null) return problemFor(turn, tilt, HP.HINT_TURN, HP.HINT_TILT);
    const stay = problemFor(turn, tilt, HP.HINT_TURN - HP.HINT_MARGIN, HP.HINT_TILT - HP.HINT_MARGIN);
    if (stay === this.problem) return stay;
    return problemFor(turn, tilt, HP.HINT_TURN, HP.HINT_TILT);
  }
}

// ---------- Taking a photo: the best of the last moments ----------

// How sharp a picture is: the variance of its Laplacian on a copy 160 px wide (higher is sharper). imageData: ImageData
export function sharpness(imageData) {
  const { data, width: w, height: h } = imageData;
  let gray = grayFromRGBA(data, w, h);
  let gw = w, gh = h;
  if (w > HP.SHARPNESS_WIDTH) {
    gh = Math.max(1, pyRound((h * HP.SHARPNESS_WIDTH) / w));
    gray = resizeAreaU8(gray, w, h, 1, HP.SHARPNESS_WIDTH, gh);
    gw = HP.SHARPNESS_WIDTH;
  }
  return laplacianVariance(gray, gw, gh);
}

// Remembers which picture of the last second or so faced the camera best (the sharpest of the equally frontal ones).
// Only pictures that can still win are kept
export class RecentFrames {
  constructor(seconds = HP.RECENT_SECONDS) {
    this.seconds = seconds;
    this.entries = []; // { time, rank: [angleSteps, -sharp], frame, angles }, best (and oldest) first
  }

  reset() {
    this.entries = [];
  }

  clear() {
    this.reset();
  }

  // Lower is better: first how far the head is from facing the camera (in steps of SAME_ANGLE degrees), then sharpness
  static rank(turn, tilt, sharp) {
    return [pyRound(Math.hypot(turn, tilt) / HP.SAME_ANGLE), -sharp];
  }

  // frame: anything (a canvas, an ImageData), kept as it is. angles: { turn, tilt }. sharp: from sharpness()
  add(frame, angles, sharp, now = nowSeconds()) {
    const rank = RecentFrames.rank(angles.turn, angles.tilt, sharp);
    while (this.entries.length) {
      const last = this.entries[this.entries.length - 1].rank;
      // last >= rank, like tuple comparison in Python
      if (last[0] > rank[0] || (last[0] === rank[0] && last[1] >= rank[1])) this.entries.pop();
      else break;
    }
    this.entries.push({ time: now, rank, frame, angles });
    this.forgetOld(now);
  }

  forgetOld(now) {
    while (this.entries.length && now - this.entries[0].time > this.seconds) this.entries.shift();
  }

  bestEntry(now = nowSeconds()) {
    this.forgetOld(now);
    return this.entries.length ? this.entries[0] : null;
  }

  // The most frontal recent picture, or null
  best(now = nowSeconds()) {
    const entry = this.bestEntry(now);
    return entry ? entry.frame : null;
  }
}
