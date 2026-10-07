// Draws the face overlay on a canvas 2D context: thin anti-aliased contour lines with a soft halo, a few accent key
// points, subtle cheek outlines and a soft red tint on uneven skin. The look is set by the constants below, which are
// meant to be synced with web/OVERLAY_STYLE.md (the desktop app's overlay).
import { connections } from './landmarker.js';

export const STYLE = {
  // Colours (the same as the desktop legend: mint for the landmarks, soft red for uneven skin)
  mint: '#4ade80',
  amber: '#f5a028', // live contours while there is a tip to show
  keyPointRing: '#ffffff',
  cheekOutline: 'rgba(240, 240, 240, 0.55)',
  skinTint: [248, 113, 113], // RGB of the uneven-skin tint
  // Sizes are for a picture 720 px on its long side and grow or shrink with the picture size
  referenceSize: 720,
  minScale: 0.85,
  maxScale: 3,
  contourWidth: 1.4,
  contourAlpha: 0.92,
  ovalAlpha: 0.7, // the face outline is a little softer than the features
  haloWidth: 4.5, // the halo is a wider, very faint copy of the same lines
  haloAlpha: 0.16,
  liveAlpha: 0.85, // overall alpha of the live preview overlay
  keyPointRadius: 2.6,
  keyPointHaloRadius: 6.5,
  keyPointHaloAlpha: 0.22,
  keyPointRingWidth: 1,
  cheekOutlineWidth: 1.2,
  cheekRadius: 6,
  skinTintAlpha: 0.5, // alpha of the tint on the spots, 0 to 1
  skinTintBlur: 1.2, // softens the edges of the spots (skipped where the browser can't blur canvas drawings)
};

// Iris centres, nose tip, chin and the mouth corners
export const KEY_POINTS = [468, 473, 1, 152, 61, 291];

const FEATURES = ['leftEye', 'rightEye', 'leftBrow', 'rightBrow', 'lips', 'leftIris', 'rightIris'];

function scaleFor(width, height) {
  const s = Math.max(width, height) / STYLE.referenceSize;
  return Math.min(STYLE.maxScale, Math.max(STYLE.minScale, s));
}

function pathFor(set, points) {
  const path = new Path2D();
  for (const { start, end } of set ?? []) {
    const a = points[start], b = points[end];
    if (!a || !b) continue;
    path.moveTo(a[0], a[1]);
    path.lineTo(b[0], b[1]);
  }
  return path;
}

function strokeWithHalo(ctx, path, colour, width, alpha, scale) {
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = colour;
  ctx.globalAlpha = STYLE.haloAlpha * (alpha / STYLE.contourAlpha);
  ctx.lineWidth = STYLE.haloWidth * scale;
  ctx.stroke(path);
  ctx.globalAlpha = alpha;
  ctx.lineWidth = width * scale;
  ctx.stroke(path);
}

// The contour lines and key points of a face. points is [[x, y] * 478] in the canvas' pixels.
export function drawFace(ctx, points, { colour = STYLE.mint, alpha = 1, width, height } = {}) {
  if (!points || !connections.oval) return;
  const scale = scaleFor(width ?? ctx.canvas.width, height ?? ctx.canvas.height);
  ctx.save();
  strokeWithHalo(ctx, pathFor(connections.oval, points), colour, STYLE.contourWidth, STYLE.ovalAlpha * alpha, scale);
  const features = new Path2D();
  for (const name of FEATURES) features.addPath(pathFor(connections[name], points));
  strokeWithHalo(ctx, features, colour, STYLE.contourWidth, STYLE.contourAlpha * alpha, scale);

  for (const i of KEY_POINTS) {
    const p = points[i];
    if (!p) continue;
    ctx.globalAlpha = STYLE.keyPointHaloAlpha * alpha;
    ctx.fillStyle = colour;
    ctx.beginPath();
    ctx.arc(p[0], p[1], STYLE.keyPointHaloRadius * scale, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = alpha;
    ctx.beginPath();
    ctx.arc(p[0], p[1], STYLE.keyPointRadius * scale, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = STYLE.keyPointRing;
    ctx.lineWidth = STYLE.keyPointRingWidth * scale;
    ctx.stroke();
  }
  ctx.restore();
}

// The cheek squares: a subtle rounded outline and a soft red tint where the skin is uneven.
// Each cheek is { x1, y1, x2, y2, w, h, mask } with a mask of w * h bytes (not 0 where the skin is uneven).
export function drawCheeks(ctx, cheeks, { width, height } = {}) {
  if (!cheeks?.length) return;
  const scale = scaleFor(width ?? ctx.canvas.width, height ?? ctx.canvas.height);
  ctx.save();
  for (const c of cheeks) {
    if (c.mask && c.w > 0 && c.h > 0) {
      const tint = document.createElement('canvas');
      tint.width = c.w;
      tint.height = c.h;
      const tctx = tint.getContext('2d');
      const image = tctx.createImageData(c.w, c.h);
      const [r, g, b] = STYLE.skinTint;
      for (let i = 0; i < c.w * c.h; i++) {
        if (!c.mask[i]) continue;
        image.data[i * 4] = r;
        image.data[i * 4 + 1] = g;
        image.data[i * 4 + 2] = b;
        image.data[i * 4 + 3] = 255;
      }
      tctx.putImageData(image, 0, 0);
      ctx.globalAlpha = STYLE.skinTintAlpha;
      if ('filter' in ctx) ctx.filter = `blur(${STYLE.skinTintBlur * scale}px)`;
      ctx.drawImage(tint, c.x1, c.y1, c.w, c.h);
      if ('filter' in ctx) ctx.filter = 'none';
    }
    ctx.globalAlpha = 1;
    ctx.strokeStyle = STYLE.cheekOutline;
    ctx.lineWidth = STYLE.cheekOutlineWidth * scale;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(c.x1, c.y1, c.x2 - c.x1, c.y2 - c.y1, STYLE.cheekRadius * scale);
    else ctx.rect(c.x1, c.y1, c.x2 - c.x1, c.y2 - c.y1);
    ctx.stroke();
  }
  ctx.restore();
}

// The live preview: clears the canvas and draws the contours, in amber while there is a tip to show.
export function drawLive(ctx, points, { tip = false } = {}) {
  const { width, height } = ctx.canvas;
  ctx.clearRect(0, 0, width, height);
  if (!points) return;
  drawFace(ctx, points, { colour: tip ? STYLE.amber : STYLE.mint, alpha: STYLE.liveAlpha });
}

// A rated picture: the picture, then the contours and the cheeks on top of it.
export function drawResult(ctx, picture, points, cheeks) {
  const { width, height } = ctx.canvas;
  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(picture, 0, 0, width, height);
  drawCheeks(ctx, cheeks);
  drawFace(ctx, points);
}

export function clear(ctx) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
}
