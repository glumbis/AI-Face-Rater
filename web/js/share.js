// Share card: a 1080x1350 PNG drawn on a canvas with the score and the stats, and never a photo. Same clean style as
// the app: one accent colour, a rounded card, system-ui / Segoe UI Variable, light or dark.

export const CARD_WIDTH = 1080;
export const CARD_HEIGHT = 1350;

// The same palettes as the desktop app (face_rater_app.py LIGHT / DARK)
const THEMES = {
  light: { bg: '#f3f3f3', card: '#ffffff', edge: '#dcdcdc', text: '#1a1a1a', muted: '#666666', faint: '#9c9c9c',
    track: '#dcdcdc', chip: '#f0f0f0', accent: '#0067c0', onAccent: '#ffffff', low: '#b45309' },
  dark: { bg: '#202020', card: '#2b2b2b', edge: '#3c3c3c', text: '#f5f5f5', muted: '#a3a3a3', faint: '#6f6f6f',
    track: '#3c3c3c', chip: '#363636', accent: '#60cdff', onAccent: '#000000', low: '#f3c969' },
};
const FONT = "'Segoe UI Variable Display', 'Segoe UI Variable', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif";
const REFERENCES = { boy: 'Boy model face', girl: 'Girl model face', average: 'Average model face' };

function roundedRect(ctx, x, y, w, h, r) {
  r = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function font(weight, size) {
  return `${weight} ${size}px ${FONT}`;
}

// A thin bar with a rounded track and fill. value is 0..1, or null for "not counted" (empty track)
function bar(ctx, t, x, y, w, h, value) {
  ctx.fillStyle = t.track;
  roundedRect(ctx, x, y, w, h, h / 2);
  ctx.fill();
  if (value !== null) {
    ctx.fillStyle = t.accent;
    roundedRect(ctx, x, y, Math.max(h, w * Math.min(1, Math.max(0, value))), h, h / 2);
    ctx.fill();
  }
}

// The small mark of the wordmark: a score ring with a gap
function ring(ctx, t, cx, cy, r) {
  ctx.lineWidth = r * 0.28;
  ctx.lineCap = 'round';
  ctx.strokeStyle = t.track;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.stroke();
  ctx.strokeStyle = t.accent;
  ctx.beginPath();
  ctx.arc(cx, cy, r, -Math.PI / 2, Math.PI * 0.75);
  ctx.stroke();
}

function makeCanvas() {
  if (typeof document !== 'undefined') {
    const c = document.createElement('canvas');
    c.width = CARD_WIDTH;
    c.height = CARD_HEIGHT;
    return c;
  }
  return new OffscreenCanvas(CARD_WIDTH, CARD_HEIGHT);
}

function toBlob(canvas) {
  if (canvas.convertToBlob) return canvas.convertToBlob({ type: 'image/png' });
  return new Promise((resolve, reject) => {
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('Could not make the share card'))), 'image/png');
  });
}

const percent = (v) => `${Math.round(v * 100)}%`;
const one = (v) => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(1) : '–');

// {score, clarity, symmetry, reference, stats, theme} -> Promise<Blob> (image/png, 1080x1350).
// clarity is 0..1 or null (not counted), symmetry 0..1, reference "boy" | "girl" | "average",
// stats is what history.stats(reference) returns (or null), theme is "light" or "dark".
export function makeShareCard({ score, clarity = null, symmetry = null, reference = 'average', stats = null, theme } = {}) {
  const t = THEMES[theme] || THEMES[(typeof matchMedia !== 'undefined' && matchMedia('(prefers-color-scheme: dark)').matches) ? 'dark' : 'light'];
  const canvas = makeCanvas();
  const ctx = canvas.getContext('2d');
  const W = CARD_WIDTH, H = CARD_HEIGHT, cx = W / 2;

  // Page and card
  ctx.fillStyle = t.bg;
  ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = t.card;
  roundedRect(ctx, 60, 60, W - 120, H - 120, 64);
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = t.edge;
  roundedRect(ctx, 61, 61, W - 122, H - 122, 63);
  ctx.stroke();

  ctx.textBaseline = 'alphabetic';
  ctx.textAlign = 'center';

  // Caption, then the big score with its "/ 10", tinted by band like in the app
  ctx.fillStyle = t.muted;
  ctx.font = font(500, 40);
  ctx.fillText('Beauty score', cx, 190);

  const text = one(score);
  const big = typeof score === 'number' ? (score >= 8 ? t.accent : score < 4 ? t.low : t.text) : t.muted;
  let size = 360;
  ctx.font = font(600, size);
  let scoreW = ctx.measureText(text).width;
  ctx.font = font(500, 96);
  const unitW = ctx.measureText('/ 10').width;
  const gap = 24, room = W - 300;
  if (scoreW + gap + unitW > room) {
    // A score of 10.0 is wider: shrink the digits so the line fits inside the card
    size = Math.floor(size * (room - gap - unitW) / scoreW);
    ctx.font = font(600, size);
    scoreW = ctx.measureText(text).width;
  }
  const left = cx - (scoreW + gap + unitW) / 2;
  ctx.textAlign = 'left';
  ctx.fillStyle = big;
  ctx.font = font(600, size);
  ctx.fillText(text, left, 530);
  ctx.fillStyle = t.muted;
  ctx.font = font(500, 96);
  ctx.fillText('/ 10', left + scoreW + gap, 530);

  // The reference as a quiet chip
  ctx.textAlign = 'center';
  ctx.font = font(500, 36);
  const refText = REFERENCES[reference] || REFERENCES.average;
  const chipW = ctx.measureText(refText).width + 80;
  ctx.fillStyle = t.chip;
  roundedRect(ctx, cx - chipW / 2, 580, chipW, 72, 36);
  ctx.fill();
  ctx.fillStyle = t.muted;
  ctx.fillText(refText, cx, 628);

  // Skin clarity and symmetry
  const bx = 150, bw = W - 300;
  const rows = [['Skin clarity', clarity], ['Symmetry', symmetry]];
  rows.forEach(([label, value], i) => {
    const y = 760 + i * 130;
    const has = typeof value === 'number' && Number.isFinite(value);
    ctx.textAlign = 'left';
    ctx.fillStyle = t.muted;
    ctx.font = font(500, 40);
    ctx.fillText(label, bx, y);
    ctx.textAlign = 'right';
    ctx.fillStyle = has ? t.text : t.muted;
    ctx.font = font(600, 40);
    ctx.fillText(has ? percent(value) : 'Not counted', bx + bw, y);
    bar(ctx, t, bx, y + 28, bw, 16, has ? value : null);
  });

  // Best, average and day streak
  ctx.fillStyle = t.edge;
  ctx.fillRect(bx, 1000, bw, 2);
  const have = stats && stats.count >= 2;
  const streak = stats && stats.streak >= 1 ? String(stats.streak) : '–';
  const cols = [
    [have ? one(stats.best) : '–', 'Best'],
    [have ? one(stats.average) : '–', have ? `Average (${stats.count})` : 'Average'],
    [streak, 'Day streak'],
  ];
  cols.forEach(([value, label], i) => {
    const x = bx + bw * (i + 0.5) / 3;
    ctx.textAlign = 'center';
    ctx.fillStyle = value === '–' ? t.faint : t.text;
    ctx.font = font(600, 84);
    ctx.fillText(value, x, 1105);
    ctx.fillStyle = t.muted;
    ctx.font = font(500, 34);
    ctx.fillText(label, x, 1160);
  });

  // The wordmark
  ctx.font = font(600, 40);
  const mark = 'AI Face Rater', markW = ctx.measureText(mark).width, ring_r = 20, ringGap = 20;
  const startX = cx - (markW + ring_r * 2 + ringGap) / 2;
  ring(ctx, t, startX + ring_r, 1224, ring_r * 0.85);
  ctx.textAlign = 'left';
  ctx.fillStyle = t.text;
  ctx.fillText(mark, startX + ring_r * 2 + ringGap, 1238);

  return toBlob(canvas);
}

// Opens the share sheet with the picture where the browser can share files, otherwise downloads it.
// Returns "shared", "cancelled" (the person closed the share sheet) or "downloaded".
export async function shareOrDownload(blob, filename = 'ai-face-rater.png') {
  const file = new File([blob], filename, { type: 'image/png' });
  try {
    if (typeof navigator !== 'undefined' && navigator.share && navigator.canShare && navigator.canShare({ files: [file] })) {
      await navigator.share({ files: [file], title: 'AI Face Rater' });
      return 'shared';
    }
  } catch (e) {
    if (e && e.name === 'AbortError') return 'cancelled';
    // Anything else (not allowed, not supported): fall back to the download
  }
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.rel = 'noopener';
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
  return 'downloaded';
}
