// Draws the face overlay on a canvas 2D context, in the same style as the desktop app (overlay.py; the style is
// written down in web/OVERLAY_STYLE.md): thin smooth contour lines with a soft dark halo, six accent dots, and on a
// result picture soft corner brackets round the cheeks and a faint red tint where the skin is uneven.
//
// Everything is sized from the face width W (landmarks 127 to 356): the line width is L = max(1.2 px, 0.004 * W).
// The contour index lists are worked out once at load time, so the live preview only has to smooth and stroke.

export const STYLE = {
  line: '#ffffff',
  lineTurned: '#ffcd8c', // the lines in the live preview while there is a tip to show
  halo: '#141414',
  accent: '#60cdff',
  accentTurned: '#f5a028',
  skinTint: [248, 113, 113], // #F87171
  lineAlpha: 0.78,
  haloAlpha: 0.34,
  accentAlpha: 0.95,
  skinAlpha: 0.5, // the strongest tint of uneven skin
  minLine: 1.0, // in device pixels (the desktop uses 1.2 on a bigger picture)
  lineWidth: 0.004, // of the face width
  haloWidth: 3.4, // times the line width
  dotRadius: 1.35, // times the line width. A ring (+0.7) and a halo (+1.6) go around it
  bracketArm: 0.07, // of the face width
  bracketCurve: 0.03,
  bracketScale: 0.7, // cheek brackets are a bit fainter than the face lines
};

export const FACE_WIDTH_POINTS = [127, 356];
// The outer eye corners, the tip of the nose, the corners of the mouth and the chin
export const KEY_POINTS = [33, 263, 4, 61, 291, 152];

const OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
  148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];
const EYE_RIGHT = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249];
const EYE_LEFT = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7];
// The brows are one line along the middle of each pair of landmarks
const BROW_RIGHT = [[276, 300], [283, 293], [282, 334], [295, 296], [285, 336]];
const BROW_LEFT = [[46, 70], [53, 63], [52, 105], [65, 66], [55, 107]];
const LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146];
const LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95];
const NOSE_BRIDGE = [168, 6, 197, 195, 5, 4];
const NOSE_BASE = [48, 64, 98, 97, 2, 326, 327, 294, 278];

// [landmarks, closed, strength of the line (times its alpha), thickness compared to the normal line]
const CONTOUR_SPEC = [
  [OVAL, true, 0.55, 0.9],
  [BROW_RIGHT, false, 1, 1.5], [BROW_LEFT, false, 1, 1.5],
  [EYE_RIGHT, true, 1, 1], [EYE_LEFT, true, 1, 1],
  [LIPS_OUTER, true, 1, 1], [LIPS_INNER, true, 0.5, 1],
  [NOSE_BRIDGE, false, 0.7, 1], [NOSE_BASE, false, 0.7, 1],
];
// Each point is two landmark numbers (the same one twice, unless it is a brow pair), whose middle is used
const CONTOURS = CONTOUR_SPEC.map(([list, closed, strength, thick]) => ({
  a: Int16Array.from(list, (p) => (typeof p === 'number' ? p : p[0])),
  b: Int16Array.from(list, (p) => (typeof p === 'number' ? p : p[1])),
  closed, strength, thick,
}));

// ---------- Helpers ----------

// Chaikin's corner cutting on a flat [x0, y0, x1, y1, ...] array: every round replaces each corner with two points
// nearer to the middle of its sides. Open lines keep their end points.
function chaikin(src, closed, rounds) {
  let pts = src;
  for (let r = 0; r < rounds; r++) {
    const n = pts.length / 2;
    const sides = closed ? n : n - 1;
    const out = new Array(sides * 4 + (closed ? 0 : 4));
    let k = 0;
    if (!closed) { out[k++] = pts[0]; out[k++] = pts[1]; }
    for (let i = 0; i < sides; i++) {
      const j = (i + 1) % n;
      const ax = pts[2 * i], ay = pts[2 * i + 1], bx = pts[2 * j], by = pts[2 * j + 1];
      out[k++] = 0.75 * ax + 0.25 * bx; out[k++] = 0.75 * ay + 0.25 * by;
      out[k++] = 0.25 * ax + 0.75 * bx; out[k++] = 0.25 * ay + 0.75 * by;
    }
    if (!closed) { out[k++] = pts[2 * n - 2]; out[k++] = pts[2 * n - 1]; }
    pts = out;
  }
  return pts;
}

function pathOf(pts, closed) {
  const path = new Path2D();
  path.moveTo(pts[0], pts[1]);
  for (let i = 2; i < pts.length; i += 2) path.lineTo(pts[i], pts[i + 1]);
  if (closed) path.closePath();
  return path;
}

export function faceWidth(points) {
  const a = points[FACE_WIDTH_POINTS[0]], b = points[FACE_WIDTH_POINTS[1]];
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

// How many device pixels one canvas pixel is (the canvas is scaled by CSS), so the thinnest line can be 1.2 device px
function displayRatio(canvas, cover) {
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  if (!cw || !ch || !canvas.width || !canvas.height) return 1;
  const rx = cw / canvas.width, ry = ch / canvas.height;
  const r = (cover ? Math.max(rx, ry) : Math.min(rx, ry)) * (window.devicePixelRatio || 1);
  return r > 0 ? r : 1;
}

function lineUnit(W, ratio) {
  return Math.max(STYLE.minLine / ratio, STYLE.lineWidth * W);
}

// A soft dark glow: two strokes (a wide faint one under a narrower one) look like a blurred line and cost almost
// nothing, unlike ctx.filter, which is slow and not supported everywhere.
function strokeHalo(ctx, path, width, alpha) {
  ctx.strokeStyle = STYLE.halo;
  ctx.lineWidth = width * 1.25;
  ctx.globalAlpha = alpha * 0.2;
  ctx.stroke(path);
  ctx.lineWidth = width * 0.85;
  ctx.globalAlpha = alpha * 0.6;
  ctx.stroke(path);
}

function fillHaloCircle(ctx, x, y, r, alpha) {
  ctx.fillStyle = STYLE.halo;
  ctx.globalAlpha = alpha * 0.2;
  ctx.beginPath();
  ctx.arc(x, y, r + 0.45 * (r / 3), 0, Math.PI * 2);
  ctx.fill();
  ctx.globalAlpha = alpha * 0.6;
  ctx.beginPath();
  ctx.arc(x, y, r * 0.85, 0, Math.PI * 2);
  ctx.fill();
}

function disc(ctx, x, y, r, colour, alpha) {
  ctx.fillStyle = colour;
  ctx.globalAlpha = alpha;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fill();
}

// ---------- The face ----------

// The contour lines and accent dots of a face. points is [[x, y] * 478] in the canvas' pixels.
// turned: the amber version (preview with a tip). preview: one round of smoothing instead of two.
export function drawFace(ctx, points, { turned = false, preview = false, cover = false } = {}) {
  if (!points || !points[FACE_WIDTH_POINTS[0]] || !points[FACE_WIDTH_POINTS[1]]) return;
  const W = faceWidth(points);
  if (!(W >= 8)) return;
  const L = lineUnit(W, displayRatio(ctx.canvas, cover));
  const lineColour = turned ? STYLE.lineTurned : STYLE.line;
  const accent = turned ? STYLE.accentTurned : STYLE.accent;
  const rounds = preview ? 1 : 2;

  const paths = [];
  const halo = new Path2D();
  for (const c of CONTOURS) {
    const n = c.a.length;
    const flat = new Array(n * 2);
    for (let i = 0; i < n; i++) {
      const p = points[c.a[i]], q = points[c.b[i]];
      flat[2 * i] = (p[0] + q[0]) / 2;
      flat[2 * i + 1] = (p[1] + q[1]) / 2;
    }
    const path = pathOf(chaikin(flat, c.closed, rounds), c.closed);
    paths.push(path);
    halo.addPath(path);
  }

  ctx.save();
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  // 1. halo, the same strength on every contour
  strokeHalo(ctx, halo, STYLE.haloWidth * L, STYLE.haloAlpha);
  const r = STYLE.dotRadius * L;
  for (const i of KEY_POINTS) {
    const p = points[i];
    if (p) fillHaloCircle(ctx, p[0], p[1], r + 1.6 * L, STYLE.haloAlpha);
  }
  // 2. the lines
  ctx.strokeStyle = lineColour;
  for (let i = 0; i < CONTOURS.length; i++) {
    const c = CONTOURS[i];
    ctx.globalAlpha = STYLE.lineAlpha * c.strength;
    ctx.lineWidth = L * c.thick;
    ctx.stroke(paths[i]);
  }
  // 3. the rings and 4. the accent dots
  for (const i of KEY_POINTS) {
    const p = points[i];
    if (p) disc(ctx, p[0], p[1], r + 0.7 * L, lineColour, STYLE.lineAlpha);
  }
  for (const i of KEY_POINTS) {
    const p = points[i];
    if (p) disc(ctx, p[0], p[1], r, accent, STYLE.accentAlpha);
  }
  ctx.restore();
}

// ---------- The cheeks ----------

// Rounded corner brackets round a cheek square [x1, y1, x2, y2], with the same halo as the lines
function drawBrackets(ctx, square, W, L) {
  const [x1, y1, x2, y2] = square;
  const curve = STYLE.bracketCurve * W;
  const arm = Math.max(STYLE.bracketArm * W, curve + 2);
  const path = new Path2D();
  for (const [cx, cy, sx, sy] of [[x1, y1, 1, 1], [x2, y1, -1, 1], [x2, y2, -1, -1], [x1, y2, 1, -1]]) {
    path.moveTo(cx + sx * arm, cy);
    path.lineTo(cx + sx * curve, cy);
    path.arcTo(cx, cy, cx, cy + sy * curve, curve);
    path.lineTo(cx, cy + sy * arm);
  }
  const k = STYLE.bracketScale;
  ctx.save();
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  strokeHalo(ctx, path, STYLE.haloWidth * 1.1 * L, STYLE.haloAlpha * k);
  ctx.strokeStyle = STYLE.line;
  ctx.globalAlpha = STYLE.lineAlpha * k;
  ctx.lineWidth = 1.1 * L;
  ctx.stroke(path);
  ctx.restore();
}

// Gaussian blur of a w * h float image, in place (separable, edges mirrored like OpenCV's default)
function blur(data, w, h, sigma) {
  const radius = Math.max(1, Math.ceil(sigma * 3));
  const kernel = new Float32Array(2 * radius + 1);
  let sum = 0;
  for (let i = -radius; i <= radius; i++) { kernel[i + radius] = Math.exp(-(i * i) / (2 * sigma * sigma)); sum += kernel[i + radius]; }
  for (let i = 0; i < kernel.length; i++) kernel[i] /= sum;
  const reflect = (i, n) => {
    if (n === 1) return 0;
    while (i < 0 || i >= n) i = i < 0 ? -i : 2 * (n - 1) - i;
    return i;
  };
  const tmp = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let k = -radius; k <= radius; k++) s += kernel[k + radius] * data[y * w + reflect(x + k, w)];
      tmp[y * w + x] = s;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let k = -radius; k <= radius; k++) s += kernel[k + radius] * tmp[reflect(y + k, h) * w + x];
      data[y * w + x] = s;
    }
  }
}

// The uneven skin of one cheek as a soft red tint: blur the spot mask, boost it, fade it to nothing towards the edges
// of the square, and mix in the red with at most skinAlpha. Drawn at the cheek's own size, so no hard pixels.
function drawTint(ctx, c) {
  if (!c.mask || !(c.w > 0) || !(c.h > 0)) return;
  const { w, h } = c;
  let any = false;
  const soft = new Float32Array(w * h);
  for (let i = 0; i < soft.length; i++) if (c.mask[i]) { soft[i] = 1; any = true; }
  if (!any) return;
  const sq = c.square ?? [c.x1, c.y1, c.x2, c.y2];
  const side = sq[2] - sq[0];
  blur(soft, w, h, Math.max(1, side / 20));
  const edge = Math.max(2, side * 0.12);
  const fadeX = new Float32Array(w), fadeY = new Float32Array(h);
  for (let x = 0; x < w; x++) fadeX[x] = Math.min(1, Math.max(0, Math.min(x + c.x1 - sq[0] + 1, sq[2] - c.x1 - x) / edge));
  for (let y = 0; y < h; y++) fadeY[y] = Math.min(1, Math.max(0, Math.min(y + c.y1 - sq[1] + 1, sq[3] - c.y1 - y) / edge));

  const tint = document.createElement('canvas');
  tint.width = w;
  tint.height = h;
  const tctx = tint.getContext('2d');
  const image = tctx.createImageData(w, h);
  const [r, g, b] = STYLE.skinTint;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const a = Math.min(1, soft[i] * 1.6) * fadeX[x] * fadeY[y] * STYLE.skinAlpha;
      image.data[i * 4] = r;
      image.data[i * 4 + 1] = g;
      image.data[i * 4 + 2] = b;
      image.data[i * 4 + 3] = Math.round(a * 255);
    }
  }
  tctx.putImageData(image, 0, 0);
  ctx.save();
  ctx.globalAlpha = 1;
  ctx.drawImage(tint, c.x1, c.y1, w, h);
  ctx.restore();
}

// Each cheek is { x1, y1, x2, y2, w, h, square, mask }: the tint, then the brackets. points gives the face width.
export function drawCheeks(ctx, cheeks, points, { cover = false } = {}) {
  if (!cheeks?.length || !points) return;
  const W = faceWidth(points);
  if (!(W >= 8)) return;
  const L = lineUnit(W, displayRatio(ctx.canvas, cover));
  for (const c of cheeks) drawTint(ctx, c);
  for (const c of cheeks) drawBrackets(ctx, c.square ?? [c.x1, c.y1, c.x2, c.y2], W, L);
}

// ---------- Entry points ----------

// The live preview: clears the canvas and draws the contours, in amber while there is a tip to show.
export function drawLive(ctx, points, { tip = false } = {}) {
  const { width, height } = ctx.canvas;
  ctx.clearRect(0, 0, width, height);
  if (!points) return;
  drawFace(ctx, points, { turned: tip, preview: true, cover: true });
}

// A rated picture: the picture, then the skin tint, the cheek brackets and the face lines on top of it.
export function drawResult(ctx, picture, points, cheeks) {
  const { width, height } = ctx.canvas;
  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(picture, 0, 0, width, height);
  drawCheeks(ctx, cheeks, points);
  drawFace(ctx, points);
}

export function clear(ctx) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
}
