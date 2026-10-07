// Photo tips: a port of phototips.py. Rule-based tips about the photo and the setup (never about the face or body),
// all pure functions of a picture and the face landmarks. The thresholds are NOT written here: they come from
// facedata.js (CONST, written by tools/export_web_data.py from the Python constants), so the site follows the desktop
// app whenever those change.
import { CONST, SUBSET } from './facedata.js';

// Constant lookup. Python names are used as they are (CONST.TURN_TIP and so on). headpose.py also has a
// SHARPNESS_WIDTH (for another purpose), so the exporter may write phototips.py's one as PHOTOTIPS_SHARPNESS_WIDTH;
// a prefixed name is preferred when it exists.
function k(name) {
  for (const key of ['PHOTOTIPS_' + name, name]) {
    const v = CONST?.[key];
    if (typeof v === 'number' && Number.isFinite(v)) return v;
  }
  throw new Error('facedata CONST is missing ' + name);
}

export const TIPS = {
  turn: 'Your head is turned a little. Look straight at the lens.',
  tilt: 'Your head is tilted a little. Look straight at the lens.',
  blurry: 'Blurry. Hold still for a second.',
  dark: 'The picture is a bit dark. Add light in front of you.',
  bright: 'Very bright. Step away from direct light.',
  uneven: 'One side of your face is brighter. Face a window or lamp.',
  close: 'Step back a little. Very close photos distort faces.',
  far: 'Your face is small in the picture. Move a bit closer.',
};
export const GREAT = 'Great setup: straight, sharp and evenly lit.';

// ---- small image helpers (all on 8-bit gray: {data: Uint8ClampedArray, w, h}) ----

// Python's round(): halves go to the even number
function roundHalfEven(x) {
  const r = Math.round(x);
  return (Math.abs(x % 1) === 0.5 && r % 2 !== 0) ? r - 1 : r;
}

// Gray like OpenCV's BGR2GRAY / RGBA2GRAY (fixed point, so the same numbers)
export function toGray(imageData) {
  const { data, width: w, height: h } = imageData;
  const out = new Uint8ClampedArray(w * h);
  for (let i = 0, j = 0; i < out.length; i++, j += 4) {
    out[i] = (data[j] * 4899 + data[j + 1] * 9617 + data[j + 2] * 1868 + 8192) >> 14;
  }
  return { data: out, w, h };
}

// Smaller copy by area averaging, like cv2.INTER_AREA
function resizeArea(g, nw, nh) {
  const { data, w, h } = g;
  const weights = (n, m) => {
    const scale = n / m, list = [];
    for (let o = 0; o < m; o++) {
      const start = o * scale, end = (o + 1) * scale, items = [];
      for (let i = Math.floor(start); i < Math.min(n, Math.ceil(end)); i++) {
        const wt = Math.min(end, i + 1) - Math.max(start, i);
        if (wt > 1e-9) items.push([i, wt / scale]);
      }
      list.push(items);
    }
    return list;
  };
  const wx = weights(w, nw), wy = weights(h, nh);
  const tmp = new Float32Array(nw * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < nw; x++) {
      let s = 0;
      for (const [i, wt] of wx[x]) s += data[y * w + i] * wt;
      tmp[y * nw + x] = s;
    }
  }
  const out = new Uint8ClampedArray(nw * nh);
  for (let y = 0; y < nh; y++) {
    for (let x = 0; x < nw; x++) {
      let s = 0;
      for (const [i, wt] of wy[y]) s += tmp[i * nw + x] * wt;
      out[y * nw + x] = roundHalfEven(s);
    }
  }
  return { data: out, w: nw, h: nh };
}

// Bigger copy by bilinear interpolation, like cv2.INTER_LINEAR
function resizeLinear(g, nw, nh) {
  const { data, w, h } = g;
  const out = new Uint8ClampedArray(nw * nh);
  for (let y = 0; y < nh; y++) {
    const fy = Math.min(Math.max((y + 0.5) * h / nh - 0.5, 0), h - 1);
    const y0 = Math.floor(fy), y1 = Math.min(y0 + 1, h - 1), ty = fy - y0;
    for (let x = 0; x < nw; x++) {
      const fx = Math.min(Math.max((x + 0.5) * w / nw - 0.5, 0), w - 1);
      const x0 = Math.floor(fx), x1 = Math.min(x0 + 1, w - 1), tx = fx - x0;
      const top = data[y0 * w + x0] * (1 - tx) + data[y0 * w + x1] * tx;
      const bottom = data[y1 * w + x0] * (1 - tx) + data[y1 * w + x1] * tx;
      out[y * nw + x] = roundHalfEven(top * (1 - ty) + bottom * ty);
    }
  }
  return { data: out, w: nw, h: nh };
}

// Pictures bigger than MEASURE_MAX_SIZE on the long side are shrunk before they are measured. Returns the gray copy
// and the factor (so the face box can follow)
function shrink(gray) {
  const limit = k('MEASURE_MAX_SIZE');
  const { w, h } = gray;
  if (Math.max(h, w) <= limit) return { gray, factor: 1 };
  const factor = limit / Math.max(h, w);
  return { gray: resizeArea(gray, Math.round(w * factor), Math.round(h * factor)), factor };
}

// The part of the picture inside the box (never empty, and never outside the picture)
function crop(g, box) {
  let [left, top, right, bottom] = box;
  left = Math.min(Math.max(0, Math.trunc(left)), g.w - 1);
  top = Math.min(Math.max(0, Math.trunc(top)), g.h - 1);
  right = Math.min(g.w, Math.max(Math.trunc(right), left + 1));
  bottom = Math.min(g.h, Math.max(Math.trunc(bottom), top + 1));
  const w = right - left, h = bottom - top, out = new Uint8ClampedArray(w * h);
  for (let y = 0; y < h; y++) {
    const from = (top + y) * g.w + left;
    out.set(g.data.subarray(from, from + w), y * w);
  }
  return { data: out, w, h };
}

// ---- the measurements ----

// (left, top, right, bottom) around the landmarks. The Python app measures the 162 landmarks of SUBSET, so the box
// uses the same points out of the 478 (points are {x, y} objects or [x, y] pairs, in the pixels of the picture)
export function faceBox(points) {
  const useSubset = Array.isArray(SUBSET) && SUBSET.length > 0 && points.length > Math.max(...SUBSET);
  const used = useSubset ? SUBSET.map((i) => points[i]) : points;
  let left = Infinity, top = Infinity, right = -Infinity, bottom = -Infinity;
  for (const p of used) {
    const x = Array.isArray(p) ? p[0] : p.x, y = Array.isArray(p) ? p[1] : p.y;
    if (x < left) left = x;
    if (x > right) right = x;
    if (y < top) top = y;
    if (y > bottom) bottom = y;
  }
  return [left, top, right, bottom];
}

// Average brightness of the face, 0 to 255
function brightness(g, box) {
  const { data } = crop(g, box);
  let s = 0;
  for (let i = 0; i < data.length; i++) s += data[i];
  return s / data.length;
}

// How different the left and right halves of the face are in brightness, as a share of the average
function lightImbalance(g, box) {
  const f = crop(g, box), half = Math.floor(f.w / 2);
  if (half < 1) return 0;
  let l = 0, r = 0;
  for (let y = 0; y < f.h; y++) {
    const row = y * f.w;
    for (let x = 0; x < half; x++) {
      l += f.data[row + x];
      r += f.data[row + f.w - half + x];
    }
  }
  const n = f.h * half;
  l /= n; r /= n;
  return Math.abs(l - r) / Math.max((l + r) / 2, 1.0);
}

// Variance of the Laplacian of the face (cv2.Laplacian, ksize 1, reflect-101 border), on a copy of the face
// SHARPNESS_WIDTH wide, so the number doesn't depend on how many pixels the camera has
function sharpness(g, box) {
  let face = crop(g, box);
  const width = k('SHARPNESS_WIDTH');
  if (face.w !== width) {
    const nh = Math.max(1, roundHalfEven(face.h * width / face.w));
    face = face.w > width ? resizeArea(face, width, nh) : resizeLinear(face, width, nh);
  }
  const { data, w, h } = face;
  const at = (x, y) => {
    x = x < 0 ? (w > 1 ? -x : 0) : x >= w ? (w > 1 ? 2 * w - 2 - x : 0) : x;
    y = y < 0 ? (h > 1 ? -y : 0) : y >= h ? (h > 1 ? 2 * h - 2 - y : 0) : y;
    return data[y * w + x];
  };
  const lap = new Float64Array(w * h);
  let sum = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = at(x - 1, y) + at(x + 1, y) + at(x, y - 1) + at(x, y + 1) - 4 * data[y * w + x];
      lap[y * w + x] = v;
      sum += v;
    }
  }
  const mean = sum / lap.length;
  let sq = 0;
  for (let i = 0; i < lap.length; i++) sq += (lap[i] - mean) ** 2;
  return sq / lap.length;
}

// How much of the picture the face fills: its width or height as a share of the picture's, the bigger of the two
function faceSize(box, w, h) {
  const [left, top, right, bottom] = box;
  return Math.max((right - left) / w, (bottom - top) / h);
}

// angles: {turn, tilt} in degrees (headAngles), a [turn, tilt] pair, or null/undefined when unknown
function anglesOf(angles) {
  if (!angles) return { turn: null, tilt: null };
  if (Array.isArray(angles)) return { turn: angles[0] ?? null, tilt: angles[1] ?? null };
  return { turn: angles.turn ?? null, tilt: angles.tilt ?? null };
}

// Everything the tips look at (phototips.measure)
export function measure(imageData, points, angles) {
  const { gray, factor } = shrink(toGray(imageData));
  const raw = faceBox(points);
  const box = factor === 1 ? raw : raw.map((v) => v * factor);
  const a = anglesOf(angles);
  return {
    turn: a.turn, tilt: a.tilt,
    sharpness: sharpness(gray, box), brightness: brightness(gray, box),
    imbalance: lightImbalance(gray, box), size: faceSize(box, gray.w, gray.h),
  };
}

// Which tips (keys of TIPS) to show, the most important first and at most maxTips. [] means nothing is wrong
export function pickTips(m, maxTips = k('MAX_TIPS')) {
  const found = [];
  if (m.turn != null && Math.abs(m.turn) > k('TURN_TIP')) found.push('turn');
  if (m.tilt != null && Math.abs(m.tilt) > k('TILT_TIP')) found.push('tilt');
  if (m.sharpness < k('BLURRY_BELOW')) found.push('blurry');
  if (m.brightness < k('TOO_DARK_BELOW')) found.push('dark');
  else if (m.brightness > k('TOO_BRIGHT_ABOVE')) found.push('bright');
  if (m.imbalance > k('UNEVEN_LIGHT_ABOVE')) found.push('uneven');
  if (m.size > k('TOO_CLOSE_ABOVE')) found.push('close');
  else if (m.size < k('TOO_FAR_BELOW')) found.push('far');
  return found.slice(0, maxTips);
}

// The texts to show: the tips, or one positive line when there is nothing to fix
export function tipLines(m, maxTips = k('MAX_TIPS')) {
  const keys = pickTips(m, maxTips);
  return keys.length ? keys.map((key) => TIPS[key]) : [GREAT];
}

// The module's entry point. Never throws: with no face or any other trouble there are just no tips
export function photoTips(imageData, points478px, angles) {
  try {
    if (!imageData || !points478px || !points478px.length) return [];
    return tipLines(measure(imageData, points478px, angles));
  } catch (e) {
    console.warn('photoTips failed', e);
    return [];
  }
}
