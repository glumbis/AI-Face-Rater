// Small image operations that behave like the OpenCV ones the Python app uses, in plain JS (no libraries).
// Images are flat typed arrays with interleaved channels: index = (y * width + x) * channels + c.
// scoring.js and headpose.js use these; they are exported in case another module (like the photo tips) needs the same.

// Python's round(): halves go to the even number (OpenCV's cvRound does the same)
export function pyRound(x) {
  const f = Math.floor(x);
  const d = x - f;
  if (d < 0.5) return f;
  if (d > 0.5) return f + 1;
  return f % 2 === 0 ? f : f + 1;
}

// Where the pixel p outside 0..len-1 is read from, for OpenCV's default border BORDER_REFLECT_101 (gfedcb|abcdefgh|gfedcba)
export function reflect101(p, len) {
  if (len === 1) return 0;
  while (p < 0 || p >= len) {
    p = p < 0 ? -p : 2 * (len - 1) - p;
  }
  return p;
}

// ---------- sRGB, linear light, Lab ----------

// sRGB byte (0..255) to linear light (0..1), as float32 like to_linear() of landmarkdetect.py
export const LINEAR_LUT = (() => {
  const lut = new Float32Array(256);
  for (let i = 0; i < 256; i++) {
    const s = Math.fround(i / 255);
    lut[i] = s <= 0.04045 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
  }
  return lut;
})();

// cv2.COLOR_LBGR2Lab for float32: linear B, G, R (clipped to 0..1) to L (0..100), a, b
const XN = 0.950456, ZN = 1.088754;
const CX = [0.412453 / XN, 0.357580 / XN, 0.180423 / XN];
const CY = [0.212671, 0.715160, 0.072169];
const CZ = [0.019334 / ZN, 0.119193 / ZN, 0.950227 / ZN];
const LAB_THRESH = 0.008856;
const LAB_A = 16 / 116;

function labCbrt(t) {
  return t > LAB_THRESH ? Math.cbrt(t) : 7.787 * t + LAB_A;
}

export function lbgrToLab(src, n, out) {
  // src: Float32Array of n * 3 (B, G, R), out: Float32Array of n * 3 (L, a, b)
  for (let i = 0; i < n; i++) {
    const b = Math.min(Math.max(src[i * 3], 0), 1);
    const g = Math.min(Math.max(src[i * 3 + 1], 0), 1);
    const r = Math.min(Math.max(src[i * 3 + 2], 0), 1);
    const x = r * CX[0] + g * CX[1] + b * CX[2];
    const y = r * CY[0] + g * CY[1] + b * CY[2];
    const z = r * CZ[0] + g * CZ[1] + b * CZ[2];
    const fx = labCbrt(x), fy = labCbrt(y), fz = labCbrt(z);
    out[i * 3] = y > LAB_THRESH ? 116 * fy - 16 : 903.3 * y;
    out[i * 3 + 1] = 500 * (fx - fy);
    out[i * 3 + 2] = 200 * (fy - fz);
  }
  return out;
}

// ---------- Resizing ----------

// OpenCV's table for INTER_AREA: for each destination pixel, which source pixels (si) count and how much (alpha)
function areaTable(ssize, dsize) {
  const scale = ssize / dsize;
  const si = [], di = [], alpha = [];
  for (let dx = 0; dx < dsize; dx++) {
    const fsx1 = dx * scale;
    const fsx2 = fsx1 + scale;
    const cellWidth = Math.min(scale, ssize - fsx1);
    let sx1 = Math.ceil(fsx1);
    let sx2 = Math.floor(fsx2);
    sx2 = Math.min(sx2, ssize - 1);
    sx1 = Math.min(sx1, sx2);
    if (sx1 - fsx1 > 1e-3) {
      di.push(dx); si.push(sx1 - 1); alpha.push(Math.fround((sx1 - fsx1) / cellWidth));
    }
    for (let sx = sx1; sx < sx2; sx++) {
      di.push(dx); si.push(sx); alpha.push(Math.fround(1 / cellWidth));
    }
    if (fsx2 - sx2 > 1e-3) {
      di.push(dx); si.push(sx2); alpha.push(Math.fround(Math.min(Math.min(fsx2 - sx2, 1), cellWidth) / cellWidth));
    }
  }
  return { si, di, alpha };
}

// cv2.resize(src, (dw, dh), interpolation=cv2.INTER_AREA) for uint8 pictures with cn channels
export function resizeAreaU8(src, sw, sh, cn, dw, dh) {
  const out = new Uint8Array(dw * dh * cn);
  const fastX = sw % dw === 0, fastY = sh % dh === 0;
  if (fastX && fastY) {
    // whole-number shrink: plain average of the blocks (OpenCV's fast path)
    const kx = sw / dw, ky = sh / dh;
    const area = kx * ky;
    const inv = Math.fround(1 / area);
    for (let y = 0; y < dh; y++) {
      for (let x = 0; x < dw; x++) {
        for (let c = 0; c < cn; c++) {
          let sum = 0;
          for (let j = 0; j < ky; j++) {
            for (let i = 0; i < kx; i++) sum += src[((y * ky + j) * sw + x * kx + i) * cn + c];
          }
          out[(y * dw + x) * cn + c] = kx === 2 && ky === 2 ? (sum + 2) >> 2 : pyRound(Math.fround(sum * inv));
        }
      }
    }
    return out;
  }
  const xt = areaTable(sw, dw), yt = areaTable(sh, dh);
  // horizontal: every source row to dw columns (float), then vertical
  const rows = new Float32Array(sh * dw * cn);
  for (let y = 0; y < sh; y++) {
    for (let k = 0; k < xt.si.length; k++) {
      const a = xt.alpha[k];
      const s = (y * sw + xt.si[k]) * cn, d = (y * dw + xt.di[k]) * cn;
      for (let c = 0; c < cn; c++) rows[d + c] += src[s + c] * a;
    }
  }
  const acc = new Float32Array(dw * cn);
  for (let dy = 0; dy < dh; dy++) {
    acc.fill(0);
    for (let k = 0; k < yt.si.length; k++) {
      if (yt.di[k] !== dy) continue;
      const a = yt.alpha[k];
      const base = yt.si[k] * dw * cn;
      for (let i = 0; i < dw * cn; i++) acc[i] += rows[base + i] * a;
    }
    for (let i = 0; i < dw * cn; i++) out[dy * dw * cn + i] = Math.min(255, Math.max(0, pyRound(acc[i])));
  }
  return out;
}

// cv2.resize(src, (dw, dh)) (bilinear, the default) for a one-channel float picture
export function resizeLinearF32(src, sw, sh, dw, dh) {
  const scaleX = sw / dw, scaleY = sh / dh;
  const axis = (dsize, scale, ssize) => {
    const i0 = new Int32Array(dsize), i1 = new Int32Array(dsize), w1 = new Float32Array(dsize);
    for (let d = 0; d < dsize; d++) {
      let f = Math.fround((d + 0.5) * scale - 0.5);
      let s = Math.floor(f);
      f -= s;
      if (s < 0) { s = 0; f = 0; }
      if (s >= ssize - 1) { s = ssize - 1; f = 0; }
      i0[d] = s;
      i1[d] = Math.min(s + 1, ssize - 1);
      w1[d] = f;
    }
    return { i0, i1, w1 };
  };
  const xa = axis(dw, scaleX, sw), ya = axis(dh, scaleY, sh);
  const mid = new Float32Array(sh * dw);
  for (let y = 0; y < sh; y++) {
    for (let x = 0; x < dw; x++) {
      const w = xa.w1[x];
      mid[y * dw + x] = src[y * sw + xa.i0[x]] * (1 - w) + src[y * sw + xa.i1[x]] * w;
    }
  }
  const out = new Float32Array(dw * dh);
  for (let y = 0; y < dh; y++) {
    const w = ya.w1[y];
    for (let x = 0; x < dw; x++) {
      out[y * dw + x] = mid[ya.i0[y] * dw + x] * (1 - w) + mid[ya.i1[y] * dw + x] * w;
    }
  }
  return out;
}

// ---------- Blur ----------

// cv2.GaussianBlur(src, (0, 0), sigma) for a float32 picture with cn channels (kernel size and border like OpenCV)
export function gaussianBlurF32(src, w, h, cn, sigma) {
  const ksize = pyRound(sigma * 4 * 2 + 1) | 1;
  const half = (ksize - 1) / 2;
  const kernel = new Float32Array(ksize);
  let sum = 0;
  const raw = [];
  for (let i = 0; i < ksize; i++) {
    const d = i - half;
    raw.push(Math.exp(-(d * d) / (2 * sigma * sigma)));
    sum += raw[i];
  }
  for (let i = 0; i < ksize; i++) kernel[i] = raw[i] / sum;

  const xIndex = new Int32Array(w + 2 * half), yIndex = new Int32Array(h + 2 * half);
  for (let i = 0; i < xIndex.length; i++) xIndex[i] = reflect101(i - half, w);
  for (let i = 0; i < yIndex.length; i++) yIndex[i] = reflect101(i - half, h);

  const tmp = new Float32Array(w * h * cn);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      for (let c = 0; c < cn; c++) {
        let s = 0;
        for (let k = 0; k < ksize; k++) s += src[(y * w + xIndex[x + k]) * cn + c] * kernel[k];
        tmp[(y * w + x) * cn + c] = s;
      }
    }
  }
  const out = new Float32Array(w * h * cn);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      for (let c = 0; c < cn; c++) {
        let s = 0;
        for (let k = 0; k < ksize; k++) s += tmp[(yIndex[y + k] * w + x) * cn + c] * kernel[k];
        out[(y * w + x) * cn + c] = s;
      }
    }
  }
  return out;
}

// ---------- Brightness and sharpness ----------

// cv2.cvtColor(..., COLOR_BGR2GRAY) for RGBA pixels (ImageData.data): OpenCV's integer weights
export function grayFromRGBA(rgba, w, h) {
  const gray = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) {
    gray[i] = (rgba[i * 4 + 2] * 1868 + rgba[i * 4 + 1] * 9617 + rgba[i * 4] * 4899 + 8192) >> 14;
  }
  return gray;
}

// Variance of the Laplacian (cv2.Laplacian(gray, cv2.CV_64F).var()) of a one-channel uint8 picture
export function laplacianVariance(gray, w, h) {
  const at = (x, y) => gray[reflect101(y, h) * w + reflect101(x, w)];
  let sum = 0, sumSq = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = at(x - 1, y) + at(x + 1, y) + at(x, y - 1) + at(x, y + 1) - 4 * at(x, y);
      sum += v;
      sumSq += v * v;
    }
  }
  const n = w * h;
  const mean = sum / n;
  return Math.max(0, sumSq / n - mean * mean);
}
