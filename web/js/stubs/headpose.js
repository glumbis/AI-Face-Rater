// STUB with the contract's exports, ported from headpose.py. Replaced by the real js/headpose.js.
const C = { MAX_TURN: 12, MAX_TILT: 15, HINT_TURN: 8, HINT_TILT: 10, HINT_MARGIN: 4, TILT_OFFSET: 14 };

// matrix: 4x4 nested arrays (row-major), as numpy gives it in the Python app.
export function headAngles(matrix) {
  if (!matrix) return null;
  const R = [0, 1, 2].map((r) => [0, 1, 2].map((c) => Number(matrix[r][c])));
  if (R.flat().some((v) => !Number.isFinite(v))) return null;
  const len = [0, 1, 2].map((c) => Math.hypot(R[0][c], R[1][c], R[2][c]));
  if (len.some((l) => l < 1e-9)) return null;
  const fx = R[0][2] / len[2], fy = R[1][2] / len[2], fz = R[2][2] / len[2];
  const deg = 180 / Math.PI;
  return { turn: Math.atan2(fx, fz) * deg, tilt: Math.atan2(-fy, fz) * deg - C.TILT_OFFSET };
}

function problemFor(turn, tilt, maxTurn = C.MAX_TURN, maxTilt = C.MAX_TILT) {
  if (Math.abs(turn) > maxTurn) return 'side';
  if (tilt > maxTilt) return 'down';
  if (tilt < -maxTilt) return 'up';
  return null;
}

export function facingProblem(angles) {
  return angles ? problemFor(angles.turn, angles.tilt) : null;
}

const median = (a) => { const s = [...a].sort((x, y) => x - y); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };

export class HeadTracker {
  constructor(frames = 9, seconds = 0.6) { this.frames = frames; this.seconds = seconds; this.items = []; this.hint = null; }
  reset() { this.items = []; this.hint = null; }
  add(angles, now = performance.now() / 1000) {
    if (!angles) return this.hint;
    this.items.push([now, angles.turn, angles.tilt]);
    while (this.items.length > this.frames || (this.items.length && now - this.items[0][0] > this.seconds)) this.items.shift();
    const turn = median(this.items.map((i) => i[1])), tilt = median(this.items.map((i) => i[2]));
    if (this.hint === null) this.hint = problemFor(turn, tilt, C.HINT_TURN, C.HINT_TILT);
    else {
      const stay = problemFor(turn, tilt, C.HINT_TURN - C.HINT_MARGIN, C.HINT_TILT - C.HINT_MARGIN);
      this.hint = stay === this.hint ? stay : problemFor(turn, tilt, C.HINT_TURN, C.HINT_TILT);
    }
    return this.hint;
  }
}

export function sharpness(imageData) {
  // Variance of a Laplacian on the grayscale picture (the caller passes a small one)
  const { width: w, height: h, data } = imageData;
  const g = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) g[i] = 0.2126 * data[i * 4] + 0.7152 * data[i * 4 + 1] + 0.0722 * data[i * 4 + 2];
  let sum = 0, sum2 = 0, n = 0;
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const i = y * w + x, l = g[i - 1] + g[i + 1] + g[i - w] + g[i + w] - 4 * g[i];
    sum += l; sum2 += l * l; n++;
  }
  return n ? sum2 / n - (sum / n) ** 2 : 0;
}

export class RecentFrames {
  constructor(seconds = 1.2) { this.seconds = seconds; this.items = []; }
  reset() { this.items = []; }
  add(canvas, angles, sharp, now = performance.now() / 1000) {
    this.items.push({ now, canvas, angles, sharp });
    while (this.items.length && now - this.items[0].now > this.seconds) this.items.shift();
  }
  best() {
    if (!this.items.length) return null;
    const off = (i) => Math.hypot(i.angles.turn, i.angles.tilt);
    const straightest = Math.min(...this.items.map(off));
    const pool = this.items.filter((i) => off(i) <= straightest + 2);
    return pool.reduce((a, b) => (b.sharp > a.sharp ? b : a)).canvas;
  }
}
