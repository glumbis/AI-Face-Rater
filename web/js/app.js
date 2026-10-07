// Face Rater: the page. Camera, file picking, states and the wiring of all modules (see FEATURES.md for the contract).
import { scoring, headpose, facedata, tips, history, share } from './modules.js';
import { createLandmarker, detect as detectSafe } from './landmarker.js';
import * as overlay from './overlay.js';
import { registerServiceWorker } from './config.js';

// ---------- Settings ----------
// The live preview looks for the face in a copy this wide, which is much faster than the full frame
const PREVIEW_DETECT_WIDTH = 320;
// "No face found" only shows when no face has been seen for this many milliseconds
const FACE_LOST_GRACE = 400;
// Bigger pictures are shrunk to this many pixels on the long side before rating, like the desktop app
const MAX_PICTURE_SIZE = 1280;
// How often (ms) a full-size frame is kept for Take photo to choose from
const RECENT_INTERVAL = 100;
// The score counts up to its value over this many ms
const SCORE_ANIMATION_MS = 1200;
const STREAK_SHOWN_FROM = 2;
const MAX_HINT_LINES = 3;

const TIP_TEXT = 'Look straight at the camera.';
const NO_CAMERA_TEXT = 'Pick a photo to get started.';
const NO_FACE_TEXT = 'No face found. Face the camera!';
const HINT_MESSAGES = {
  side: 'Turn a little towards the camera.',
  down: 'Lift your chin a little.',
  up: 'Lower your chin a little.',
};
// The tip for how far away the face is (landmarkdetect.SIZE_MESSAGES). The limits come from scoring.js / facedata.js
const SIZE_MESSAGES = {
  near: 'Move a little closer.',
  far: 'Move closer to the camera.',
};
// A webcam face smaller than this can't be rated, so such frames aren't kept for Take photo
const { MIN_FACE_SIZE = 0.30 } = facedata.CONST ?? {};

// ---------- Elements ----------
const $ = (id) => document.getElementById(id);
const el = {
  app: $('app'), stage: $('stage'), video: $('video'), overlay: $('overlay'), picture: $('picture'),
  message: $('message'), messageTitle: $('messageTitle'), messageText: $('messageText'),
  progress: $('progress'), progressBar: $('progressBar'), messageAction: $('messageAction'), drop: $('drop'),
  status: $('status'), statusText: $('statusText'),
  score: $('score'), scoreNum: $('scoreNum'), scoreUnit: $('scoreUnit'), trend: $('trend'),
  streak: $('streak'), historyLine: $('historyLine'),
  clarityVal: $('clarityVal'), clarityBar: $('clarityBar'), symmetryVal: $('symmetryVal'), symmetryBar: $('symmetryBar'),
  legend: $('legend'), hint: $('hint'), primary: $('primaryBtn'), pick: $('pickBtn'), shareBtn: $('shareBtn'),
  file: $('file'), about: $('about'), aboutBtn: $('aboutBtn'), clearBtn: $('clearBtn'), clearNote: $('clearNote'),
  reference: $('reference'),
};
const overlayCtx = el.overlay.getContext('2d');
const pictureCtx = el.picture.getContext('2d');
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');

// ---------- State ----------
const S = {
  mode: 'camera', // "camera" or "result"
  reference: 'boy',
  supported: true,
  modelState: 'loading', // "loading", "ready" or "failed"
  modelProgress: null,
  modelError: '',
  videoReady: false, // the VIDEO-mode landmarker is ready
  cam: 'idle', // "idle", "starting", "live", "denied", "none", "insecure", "error" or "lost"
  image: null, // IMAGE-mode landmarker
  video: null, // VIDEO-mode landmarker
};
let current = null; // the picture on show in result mode
let pictureCounter = 0;
let scoreAnimation = 0;
let scoreTimer = 0;

// ---------- Small helpers ----------
const fmt = (n) => n.toFixed(1);
const pct = (n) => `${Math.round(n * 100)}%`;
// Lets the browser paint ("Rating...") before heavy work. The timer covers hidden tabs, where frames don't come
const nextFrame = () => new Promise((resolve) => { requestAnimationFrame(() => setTimeout(resolve, 0)); setTimeout(resolve, 80); });
const theme = () => (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
const store = {
  get(key) { try { return localStorage.getItem(key); } catch { return null; } },
  set(key, value) { try { localStorage.setItem(key, value); } catch { /* storage can be blocked */ } },
};

function isFaceError(e) {
  return e && (e.name === 'FaceError' || e.constructor?.name === 'FaceError' || e.isFaceError === true);
}

function canvasOf(item) {
  if (!item) return null;
  if (item instanceof HTMLCanvasElement) return item;
  return item.canvas ?? item.frame ?? item.image ?? null;
}

// ---------- Status, hint, message ----------
function setStatus(text, kind = 'neutral') {
  el.status.hidden = !text;
  if (!text) return;
  el.status.dataset.kind = kind;
  if (el.statusText.textContent !== text) el.statusText.textContent = text;
}

function setHint(lines) {
  const wanted = Array.isArray(lines) ? lines : [lines];
  if (el.hint.dataset.text === wanted.join('\n')) return;
  el.hint.dataset.text = wanted.join('\n');
  el.hint.replaceChildren(...wanted.filter(Boolean).slice(0, MAX_HINT_LINES).map((line) => {
    const p = document.createElement('p');
    p.textContent = line;
    return p;
  }));
}

function showMessage({ title, text = '', progress = undefined, action = null }) {
  el.message.hidden = false;
  el.messageTitle.textContent = title;
  el.messageText.textContent = text;
  el.messageText.hidden = !text;
  el.progress.hidden = progress === undefined;
  el.progress.classList.toggle('indeterminate', progress === null);
  el.progressBar.style.width = typeof progress === 'number' ? `${Math.round(progress * 100)}%` : '';
  el.messageAction.hidden = !action;
  if (action) {
    el.messageAction.textContent = action.label;
    el.messageAction.onclick = action.run;
  }
}

function hideMessage() { el.message.hidden = true; }

// What the stage and the buttons show in camera mode, depending on how far the start-up has come
function refreshCameraUI() {
  if (S.mode !== 'camera') return;
  let view = 'none';
  const retry = { label: 'Try again', run: () => startCamera() };
  if (!S.supported) {
    showMessage({ title: 'This browser is not supported', text: 'Face Rater needs WebAssembly and a recent browser. Try the latest Chrome, Edge, Safari or Firefox.' });
  } else if (S.modelState === 'failed') {
    showMessage({ title: 'Could not load the face model', text: `${S.modelError} Check your connection and try again.`, action: { label: 'Try again', run: () => location.reload() } });
  } else if (S.modelState === 'loading') {
    showMessage({ title: 'Loading the face model', text: S.modelProgress === null ? '' : `${Math.round(S.modelProgress * 100)}%`, progress: S.modelProgress });
  } else if (S.cam === 'starting' || S.cam === 'idle' || (S.cam === 'live' && !S.videoReady)) {
    showMessage({ title: S.cam === 'live' ? 'Loading the face model' : 'Starting the camera', text: S.cam === 'live' ? '' : 'Allow the camera when your browser asks.', progress: null });
  } else if (S.cam === 'denied') {
    showMessage({ title: 'Camera access is off', text: 'Allow the camera for this site in your browser settings, or pick a photo instead.', action: retry });
  } else if (S.cam === 'none') {
    showMessage({ title: 'No camera found', text: 'Plug one in, or pick a photo instead.', action: retry });
  } else if (S.cam === 'insecure') {
    showMessage({ title: 'The camera needs a secure connection', text: 'Open this page over https, or pick a photo instead.' });
  } else if (S.cam === 'lost') {
    showMessage({ title: 'Camera disconnected', text: 'Plug it back in to continue, or pick a photo instead.', action: retry });
  } else if (S.cam === 'error') {
    showMessage({ title: 'The camera did not start', text: 'Another app may be using it. Pick a photo instead.', action: retry });
  } else {
    hideMessage();
    view = 'live';
  }
  el.stage.dataset.view = view;
  const live = view === 'live';
  el.primary.disabled = !live;
  el.primary.textContent = 'Take photo';
  el.pick.disabled = !(S.supported && S.modelState === 'ready');
  el.shareBtn.hidden = true;
  if (!live) {
    setStatus('');
    setHint(S.supported && (S.cam === 'denied' || S.cam === 'none' || S.cam === 'insecure' || S.cam === 'lost' || S.cam === 'error') ? NO_CAMERA_TEXT : '');
  } else if (!el.hint.dataset.text) {
    setHint(TIP_TEXT);
  }
}

// ---------- Result panel ----------
function setBar(valueEl, barEl, value, text) {
  valueEl.textContent = text ?? (value === null ? '–' : pct(value));
  barEl.style.width = value === null ? '0%' : `${Math.round(value * 100)}%`;
  valueEl.closest('.stat').dataset.set = value === null ? 'false' : 'true';
}

function resetResultPanel() {
  cancelAnimationFrame(scoreAnimation);
  clearTimeout(scoreTimer);
  el.scoreNum.textContent = '–';
  el.scoreUnit.hidden = true;
  el.score.dataset.band = 'none';
  el.trend.hidden = true;
  el.streak.textContent = '';
  el.historyLine.textContent = '';
  setBar(el.clarityVal, el.clarityBar, null);
  setBar(el.symmetryVal, el.symmetryBar, null);
  el.legend.classList.remove('on');
  el.shareBtn.hidden = true;
  el.shareBtn.disabled = false;
}

function animateScore(score) {
  const band = score >= 8 ? 'high' : score < 4 ? 'low' : 'mid';
  el.score.dataset.band = band;
  el.scoreUnit.hidden = false;
  cancelAnimationFrame(scoreAnimation);
  clearTimeout(scoreTimer);
  if (reducedMotion.matches) { el.scoreNum.textContent = fmt(score); return; }
  const start = performance.now();
  const step = (now) => {
    const t = Math.min((now - start) / SCORE_ANIMATION_MS, 1);
    el.scoreNum.textContent = fmt(score * (1 - (1 - t) ** 3));
    if (t < 1) scoreAnimation = requestAnimationFrame(step);
  };
  scoreAnimation = requestAnimationFrame(step);
  // Frames don't come in a hidden tab, so the final number is also set by a timer
  scoreTimer = setTimeout(() => { cancelAnimationFrame(scoreAnimation); el.scoreNum.textContent = fmt(score); }, SCORE_ANIMATION_MS + 50);
}

function showHistory(st) {
  if (st && st.count >= 2 && st.best != null) {
    el.historyLine.textContent = `Your best: ${fmt(st.best)}  ·  Average: ${fmt(st.average)} (${st.count} photos)`;
  } else {
    el.historyLine.textContent = '';
  }
  if (st?.trend === 'up' || st?.trend === 'down') {
    // "since last photo" is added under it by the CSS
    el.trend.textContent = `${st.trend === 'up' ? '▲' : '▼'} ${fmt(Math.abs(st.diff))}`;
    el.trend.dataset.trend = st.trend;
    el.trend.hidden = false;
  } else {
    el.trend.hidden = true;
  }
  el.streak.textContent = st && st.streak >= STREAK_SHOWN_FROM ? `Day streak: ${st.streak}` : '';
}

// ---------- Modes ----------
function setMode(mode) {
  S.mode = mode;
  el.app.dataset.mode = mode;
  if (mode === 'result') {
    el.stage.dataset.view = 'result';
    hideMessage();
    el.primary.disabled = false;
    el.primary.textContent = 'Back to camera';
  } else {
    resetResultPanel();
    el.hint.dataset.text = '';
    refreshCameraUI();
  }
}

function backToCamera() {
  if (S.mode !== 'result') return;
  current = null;
  headTracker.reset();
  recent.reset();
  setMode('camera');
  setStatus('');
}

// ---------- Rating ----------
function showPicture(canvas, mirrored, points = null, cheeks = null) {
  if (el.picture.width !== canvas.width || el.picture.height !== canvas.height) {
    el.picture.width = canvas.width;
    el.picture.height = canvas.height;
  }
  el.picture.classList.toggle('mirrored', mirrored);
  if (points) overlay.drawResult(pictureCtx, canvas, points, cheeks);
  else { pictureCtx.clearRect(0, 0, canvas.width, canvas.height); pictureCtx.drawImage(canvas, 0, 0); }
}

function showError(e, id) {
  if (current?.id !== id) return;
  let message = 'Something went wrong while rating. Try another photo.';
  if (isFaceError(e)) message = e.message;
  else console.error(e);
  current.failed = true;
  cancelAnimationFrame(scoreAnimation);
  showPicture(current.canvas, current.mirrored);
  setStatus(message, 'bad');
  setHint('Try again, or pick another photo.');
  el.legend.classList.remove('on');
  el.shareBtn.hidden = true;
  el.scoreNum.textContent = '–';
  el.scoreUnit.hidden = true;
  el.score.dataset.band = 'none';
  setBar(el.clarityVal, el.clarityBar, null);
  setBar(el.symmetryVal, el.symmetryBar, null);
}

// Rates a new picture (a canvas) from the camera or a file
async function rateNew(canvas, { mirrored = false, source = 'file' } = {}) {
  const id = ++pictureCounter;
  current = { id, canvas, mirrored, source, tips: null, result: null, failed: false };
  setMode('result');
  resetResultPanel();
  showPicture(canvas, mirrored);
  setStatus('Rating…', 'neutral');
  setHint('');
  await nextFrame();
  try {
    const det = await detectSafe(S.image, canvas);
    if (current?.id !== id) return;
    if (!det) throw new scoring.FaceError('No face found. Face the camera with a straight face.');
    // rateFace refuses a face that is too small or turned too far, with the message to show
    current.det = det;
    current.angles = headpose.headAngles(det.matrix);
    current.imageData = canvas.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, canvas.width, canvas.height);
    renderResult();
  } catch (e) {
    showError(e, id);
  }
}

// Rates the current picture against the chosen model face (also when the model face changes)
function renderResult() {
  const cur = current;
  if (!cur || !cur.det) return;
  const reference = S.reference;
  let result;
  try {
    result = scoring.rateFace(cur.imageData, cur.det.points478px, cur.det.matrix, reference, cur.source);
  } catch (e) {
    showError(e, cur.id);
    return;
  }
  cur.failed = false;
  cur.result = result;
  showPicture(cur.canvas, cur.mirrored, cur.det.points478px, result.cheeks);
  setStatus(`Rated against the ${reference} model`, 'neutral');

  const clarity = result.clarity ?? null;
  setBar(el.clarityVal, el.clarityBar, clarity, clarity === null ? 'Not counted' : undefined);
  setBar(el.symmetryVal, el.symmetryBar, result.symmetry);
  el.legend.classList.add('on');

  if (cur.tips === null) {
    try { cur.tips = tips.photoTips(cur.imageData, cur.det.points478px, cur.angles, cur.source) ?? []; } catch { cur.tips = []; }
  }
  const lines = [];
  if (clarity === null) lines.push('Skin clarity is not counted: cheeks hidden, bearded or black-and-white.');
  setHint([...lines, ...cur.tips]);

  // Saved once per picture and model face (the same pictureId makes history.js skip a re-rating)
  let stats = null;
  try {
    history.addRating({ reference, score: result.score, clarity, symmetry: result.symmetry, source: cur.source, pictureId: cur.id });
    stats = history.stats(reference);
  } catch (e) {
    console.warn('History is not available', e);
  }
  showHistory(stats);
  animateScore(result.score);
  el.shareBtn.hidden = false;
}

// ---------- Pictures from files, drops and pastes ----------
async function decodeImage(file) {
  try {
    return await createImageBitmap(file, { imageOrientation: 'from-image' });
  } catch {
    const url = URL.createObjectURL(file);
    try {
      const img = new Image();
      img.src = url;
      await img.decode();
      return img;
    } finally {
      URL.revokeObjectURL(url);
    }
  }
}

async function ratePhotoFile(file) {
  if (!file) return;
  if (!S.supported || S.modelState !== 'ready' || !S.image) {
    if (S.mode === 'camera') setStatus('Still loading the face model. Try again in a moment.', 'neutral');
    return;
  }
  let bitmap;
  try {
    if (!file.type.startsWith('image/')) throw new Error('not an image');
    bitmap = await decodeImage(file);
  } catch {
    showNotice('Could not open that file as a picture. Try a JPG, PNG or WebP.');
    return;
  }
  const w = bitmap.width || bitmap.naturalWidth, h = bitmap.height || bitmap.naturalHeight;
  const k = Math.min(1, MAX_PICTURE_SIZE / Math.max(w, h));
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(w * k));
  canvas.height = Math.max(1, Math.round(h * k));
  canvas.getContext('2d', { willReadFrequently: true }).drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close?.();
  await rateNew(canvas, { mirrored: false, source: 'file' });
}

// A problem with a file shows in the status pill, in whichever mode is on
function showNotice(text) {
  setStatus(text, 'bad');
}

// ---------- Camera ----------
const headTracker = new headpose.HeadTracker();
const recent = new headpose.RecentFrames();
const detectCanvas = document.createElement('canvas');
const detectCtx = detectCanvas.getContext('2d', { willReadFrequently: true });
const live = { lastVideoTime: -1, lastFace: 0, lastRecent: 0, sizeTip: null, stream: null };

async function startCamera() {
  S.cam = 'starting';
  refreshCameraUI();
  if (!navigator.mediaDevices?.getUserMedia) {
    S.cam = window.isSecureContext ? 'none' : 'insecure';
    refreshCameraUI();
    return;
  }
  try {
    live.stream?.getTracks().forEach((t) => t.stop());
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    live.stream = stream;
    stream.getVideoTracks()[0]?.addEventListener('ended', () => {
      S.cam = 'lost';
      el.overlay.getContext('2d').clearRect(0, 0, el.overlay.width, el.overlay.height);
      refreshCameraUI();
    });
    el.video.srcObject = stream;
    await el.video.play();
    S.cam = 'live';
    live.lastVideoTime = -1;
  } catch (e) {
    S.cam = e?.name === 'NotAllowedError' || e?.name === 'SecurityError' ? 'denied'
      : e?.name === 'NotFoundError' || e?.name === 'OverconstrainedError' || e?.name === 'DevicesNotFoundError' ? 'none'
      : 'error';
    if (S.cam === 'error') console.warn(e);
  }
  refreshCameraUI();
}

function captureFrame() {
  const vw = el.video.videoWidth, vh = el.video.videoHeight;
  const k = Math.min(1, MAX_PICTURE_SIZE / Math.max(vw, vh));
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(vw * k);
  canvas.height = Math.round(vh * k);
  canvas.getContext('2d', { willReadFrequently: true }).drawImage(el.video, 0, 0, canvas.width, canvas.height);
  return canvas;
}

function loop(now) {
  requestAnimationFrame(loop);
  if (S.mode !== 'camera' || S.cam !== 'live' || !S.video || document.hidden) return;
  const video = el.video;
  // Only a new camera frame is worth looking at. When detection is slow, the frames in between are skipped
  if (video.readyState < 2 || !video.videoWidth || video.currentTime === live.lastVideoTime) return;
  live.lastVideoTime = video.currentTime;

  const vw = video.videoWidth, vh = video.videoHeight;
  const dw = Math.min(PREVIEW_DETECT_WIDTH, vw), dh = Math.round((dw * vh) / vw);
  if (detectCanvas.width !== dw || detectCanvas.height !== dh) { detectCanvas.width = dw; detectCanvas.height = dh; }
  if (el.overlay.width !== vw || el.overlay.height !== vh) { el.overlay.width = vw; el.overlay.height = vh; }
  detectCtx.drawImage(video, 0, 0, dw, dh);

  let found = null;
  try {
    found = S.video.detect(detectCanvas, now);
  } catch (e) {
    if (e?.gpuFailed) { S.video.useCpu().catch(console.error); return; }
    console.error(e);
    return;
  }

  if (!found) {
    if (now - live.lastFace > FACE_LOST_GRACE) {
      headTracker.reset();
      live.sizeTip = null;
      overlay.clear(overlayCtx);
      setStatus(NO_FACE_TEXT, 'warn');
    }
    return;
  }
  live.lastFace = now;

  // Too far away comes first: it is what to fix first. A frame with a face that is too small can't be rated, so it
  // isn't kept for Take photo
  const size = scoring.faceSize(found.points478px, dw, dh);
  live.sizeTip = scoring.sizeProblem(size, live.sizeTip);

  const angles = headpose.headAngles(found.matrix);
  let hint = null;
  if (angles) {
    headTracker.add(angles);
    hint = headTracker.hint;
    if (size >= MIN_FACE_SIZE && now - live.lastRecent >= RECENT_INTERVAL) {
      live.lastRecent = now;
      recent.add(captureFrame(), angles, headpose.sharpness(detectCtx.getImageData(0, 0, dw, dh)));
    }
  }

  let text = 'Looking good! Press Take photo or Space.', kind = 'good';
  if (live.sizeTip) { text = SIZE_MESSAGES[live.sizeTip]; kind = 'warn'; }
  else if (hint) { text = HINT_MESSAGES[hint]; kind = 'warn'; }
  setStatus(text, kind);

  const sx = vw / dw, sy = vh / dh;
  const points = found.points478px.map(([x, y]) => [(x + 0.5) * sx - 0.5, (y + 0.5) * sy - 0.5]);
  overlay.drawLive(overlayCtx, points, { tip: kind !== 'good' });
}

async function takePhoto() {
  if (S.mode !== 'camera' || S.cam !== 'live' || !S.image || el.video.readyState < 2) return;
  const canvas = canvasOf(recent.best()) ?? captureFrame();
  recent.reset();
  headTracker.reset();
  // Rated the way the camera sent it, and only shown mirrored, like the preview
  await rateNew(canvas, { mirrored: true, source: 'camera' });
}

// ---------- Buttons and keys ----------
function primaryClicked() {
  if (S.mode === 'result') backToCamera();
  else takePhoto();
}

function pickPhoto() {
  if (el.pick.disabled) return;
  el.file.value = '';
  el.file.click();
}

el.primary.addEventListener('click', primaryClicked);
el.pick.addEventListener('click', pickPhoto);
el.file.addEventListener('change', () => ratePhotoFile(el.file.files[0]));

el.reference.addEventListener('change', (e) => {
  S.reference = e.target.value;
  store.set('afr-reference', S.reference);
  if (S.mode === 'result' && current?.det) renderResult();
});

el.shareBtn.addEventListener('click', async () => {
  if (!current?.result) return;
  el.shareBtn.disabled = true;
  try {
    let stats = null;
    try { stats = history.stats(S.reference); } catch { /* the card works without it */ }
    const blob = await share.makeShareCard({
      score: current.result.score, clarity: current.result.clarity ?? null, symmetry: current.result.symmetry,
      reference: S.reference, stats, theme: theme(),
    });
    const how = await share.shareOrDownload(blob);
    if (how === 'downloaded') setStatus('The share card was saved as a picture.', 'neutral');
  } catch (e) {
    if (e?.name !== 'AbortError') showNotice('Could not share the card.');
  } finally {
    el.shareBtn.disabled = false;
  }
});

el.aboutBtn.addEventListener('click', () => {
  el.clearNote.textContent = '';
  try { el.clearBtn.disabled = history.entryCount() === 0; } catch { el.clearBtn.disabled = false; }
  el.about.showModal();
});
el.about.addEventListener('click', (e) => { if (e.target === el.about) el.about.close(); });
el.clearBtn.addEventListener('click', () => {
  history.clearHistory();
  if (current?.result) showHistory(null);
  el.clearBtn.disabled = true;
  el.clearNote.textContent = 'History cleared.';
});

document.addEventListener('keydown', (e) => {
  if (el.about.open || e.defaultPrevented) return;
  const tag = e.target?.tagName;
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'o') { e.preventDefault(); pickPhoto(); return; }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.key === 'Escape') { backToCamera(); return; }
  if (e.key === ' ' && S.mode === 'camera' && tag !== 'BUTTON' && tag !== 'INPUT') { e.preventDefault(); takePhoto(); }
});

// Drag and drop
let dragDepth = 0;
const hasFiles = (e) => [...(e.dataTransfer?.types ?? [])].includes('Files');
window.addEventListener('dragenter', (e) => { if (hasFiles(e)) { dragDepth++; el.drop.hidden = false; } });
window.addEventListener('dragover', (e) => { if (hasFiles(e)) e.preventDefault(); });
window.addEventListener('dragleave', (e) => { if (hasFiles(e) && --dragDepth <= 0) { dragDepth = 0; el.drop.hidden = true; } });
window.addEventListener('drop', (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  dragDepth = 0;
  el.drop.hidden = true;
  ratePhotoFile([...e.dataTransfer.files].find((f) => f.type.startsWith('image/')) ?? e.dataTransfer.files[0]);
});

// Paste
window.addEventListener('paste', (e) => {
  const file = [...(e.clipboardData?.files ?? [])].find((f) => f.type.startsWith('image/'));
  if (file) { e.preventDefault(); ratePhotoFile(file); }
});

// ---------- Start ----------
function checkSupport() {
  return typeof WebAssembly === 'object' && typeof Path2D === 'function' && typeof createImageBitmap === 'function'
    && typeof HTMLDialogElement === 'function';
}

async function loadModels() {
  S.modelState = 'loading';
  refreshCameraUI();
  const onProgress = (p) => { S.modelProgress = p; if (S.modelState === 'loading') refreshCameraUI(); };
  try {
    S.image = await createLandmarker('IMAGE', onProgress);
    S.modelState = 'ready';
    refreshCameraUI();
    S.video = await createLandmarker('VIDEO');
    S.videoReady = true;
    refreshCameraUI();
  } catch (e) {
    console.error(e);
    if (!S.image) { S.modelState = 'failed'; S.modelError = e?.message ?? ''; }
    refreshCameraUI();
  }
}

// A tiny hook for testing on localhost only: ?test=perBoy / perGirl rates ../perBoy.jpg, ?test=denied / nocam makes the
// camera fail, ?test=fakecam serves a still picture as the "camera". It never touches a real camera.
function testHook() {
  const name = new URLSearchParams(location.search).get('test');
  if (!name || !['localhost', '127.0.0.1'].includes(location.hostname)) return null;
  const fail = (n) => () => Promise.reject(new DOMException('test', n));
  if (name !== 'fakecam') {
    // every test mode except fakecam keeps the real camera out of it
    navigator.mediaDevices.getUserMedia = fail(name === 'denied' ? 'NotAllowedError' : 'NotFoundError');
  } else {
    navigator.mediaDevices.getUserMedia = async () => {
      const img = new Image();
      img.src = '../perBoy.jpg';
      await img.decode();
      const c = document.createElement('canvas');
      c.width = 640; c.height = 480;
      const g = c.getContext('2d');
      const draw = () => { g.fillStyle = '#222'; g.fillRect(0, 0, 640, 480); const k = 480 / img.height; g.drawImage(img, (640 - img.width * k) / 2, 0, img.width * k, 480); };
      draw(); setInterval(draw, 66);
      return c.captureStream(15);
    };
  }
  return name;
}

async function main() {
  const test = testHook();
  const saved = store.get('afr-reference');
  if (['boy', 'girl', 'average'].includes(saved)) {
    S.reference = saved;
    el.reference.querySelector(`input[value="${saved}"]`).checked = true;
  }
  S.supported = checkSupport();
  setMode('camera');
  if (!S.supported) return;

  // Works offline after the first visit (not in the test modes, which would cache the test files too)
  if (!test) registerServiceWorker();
  requestAnimationFrame(loop);
  startCamera();
  await loadModels();

  if (test === 'perBoy' || test === 'perGirl') {
    const blob = await (await fetch(`../${test}.jpg`)).blob();
    await ratePhotoFile(new File([blob], `${test}.jpg`, { type: 'image/jpeg' }));
  }
}

main();
