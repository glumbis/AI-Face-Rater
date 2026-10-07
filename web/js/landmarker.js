// MediaPipe Face Landmarker setup. The library, its WASM files and the model URL all come from config.js (the one
// place that pins the MediaPipe version).
import { MP_BUNDLE_URL, MP_WASM_URL, MODEL_URL } from './config.js';

// Same as MAX_FACES in the desktop app, only the biggest face is used
const NUM_FACES = 3;

// FaceLandmarker's connection sets ({start, end} pairs), filled in once the library has loaded. overlay.js reads this.
export const connections = {};

let libraryPromise = null;
let modelPromise = null;

function loadLibrary() {
  libraryPromise ??= (async () => {
    const mp = await import(/* @vite-ignore */ MP_BUNDLE_URL);
    const fileset = await mp.FilesetResolver.forVisionTasks(MP_WASM_URL);
    const L = mp.FaceLandmarker;
    Object.assign(connections, {
      oval: L.FACE_LANDMARKS_FACE_OVAL,
      leftEye: L.FACE_LANDMARKS_LEFT_EYE,
      rightEye: L.FACE_LANDMARKS_RIGHT_EYE,
      leftBrow: L.FACE_LANDMARKS_LEFT_EYEBROW,
      rightBrow: L.FACE_LANDMARKS_RIGHT_EYEBROW,
      leftIris: L.FACE_LANDMARKS_LEFT_IRIS,
      rightIris: L.FACE_LANDMARKS_RIGHT_IRIS,
      lips: L.FACE_LANDMARKS_LIPS,
    });
    return { mp, fileset };
  })();
  libraryPromise.catch(() => { libraryPromise = null; });
  return libraryPromise;
}

// Downloads the model with progress (onProgress gets a number 0..1, or null when the size isn't known)
function loadModel(onProgress) {
  modelPromise ??= (async () => {
    const response = await fetch(MODEL_URL);
    if (!response.ok) throw new Error(`Could not load the face model (${response.status}).`);
    const total = Number(response.headers.get('content-length')) || 0;
    if (!response.body) return new Uint8Array(await response.arrayBuffer());
    const reader = response.body.getReader();
    const chunks = [];
    let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      onProgress?.(total ? Math.min(1, received / total) : null);
    }
    const bytes = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
    return bytes;
  })();
  modelPromise.catch(() => { modelPromise = null; });
  return modelPromise;
}

function sourceSize(source) {
  return [
    source.videoWidth || source.naturalWidth || source.displayWidth || source.width,
    source.videoHeight || source.naturalHeight || source.displayHeight || source.height,
  ];
}

// MediaPipe's 4x4 matrix as rows, like numpy gives it in the Python app (the translation is in the last column)
function matrixRows(m) {
  const d = m?.data;
  if (!d || d.length !== 16) return null;
  // MediaPipe's data is column-major, so element (row r, column c) is d[c * 4 + r]
  return [0, 1, 2, 3].map((r) => [0, 1, 2, 3].map((c) => d[c * 4 + r]));
}

async function create(mode, delegate, mp, fileset, model) {
  return mp.FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetBuffer: model.slice(), delegate },
    runningMode: mode,
    numFaces: NUM_FACES,
    outputFacialTransformationMatrixes: true,
    outputFaceBlendshapes: false,
  });
}

// mode is "IMAGE" or "VIDEO". Returns { mode, delegate, detect(source, timestampMs), close() }.
// IMAGE (rating a still photo) always runs on the CPU: the GPU delegate gives landmarks that are off by about 0.9 % of
// the eye distance (1.6 px on a 640 px photo), and the model faces and the desktop app's numbers come from the CPU
// path, so the GPU would rate the same photo up to a point lower. VIDEO (the live preview) uses the GPU when there is
// one, because it only draws the dots and needs the speed.
// detect gives { points478px: [[x, y, depth] * 478], matrix: 4x4 rows or null } for the biggest face, or null if there is none.
// The points are in the pixels of the source: x = x_norm * width - 0.5, like the Python app (depth = z_norm * width).
// VIDEO mode needs a timestamp in milliseconds that grows with every call.
export async function createLandmarker(mode, onProgress) {
  const { mp, fileset } = await loadLibrary();
  const model = await loadModel(onProgress);
  let delegate = mode === 'IMAGE' ? 'CPU' : 'GPU';
  let landmarker;
  try {
    landmarker = await create(mode, delegate, mp, fileset, model);
  } catch (e) {
    if (delegate === 'CPU') throw e;
    console.info('GPU delegate not available, using the CPU.', e);
    delegate = 'CPU';
    landmarker = await create(mode, 'CPU', mp, fileset, model);
  }
  let lastTs = 0;

  const run = (source, timestampMs) => {
    if (mode === 'VIDEO') {
      lastTs = Math.max(lastTs + 1, Math.floor(timestampMs ?? performance.now()));
      return landmarker.detectForVideo(source, lastTs);
    }
    return landmarker.detect(source);
  };

  const api = {
    mode,
    get delegate() { return delegate; },
    detect(source, timestampMs) {
      let result;
      try {
        result = run(source, timestampMs);
      } catch (e) {
        if (delegate !== 'GPU') throw e;
        // The GPU worked at start but not for this picture: carry on with the CPU
        console.info('GPU detection failed, switching to the CPU.', e);
        throw Object.assign(e, { gpuFailed: true });
      }
      const faces = result.faceLandmarks;
      if (!faces || !faces.length) return null;
      const [w, h] = sourceSize(source);
      const boxes = faces.map((face) => {
        let x0 = 1, x1 = 0, y0 = 1, y1 = 0;
        for (const p of face) { x0 = Math.min(x0, p.x); x1 = Math.max(x1, p.x); y0 = Math.min(y0, p.y); y1 = Math.max(y1, p.y); }
        return (x1 - x0) * w * (y1 - y0) * h;
      });
      const which = boxes.indexOf(Math.max(...boxes));
      return {
        points478px: faces[which].map((p) => [p.x * w - 0.5, p.y * h - 0.5, p.z * w]),
        matrix: matrixRows(result.facialTransformationMatrixes?.[which]),
      };
    },
    close() { landmarker.close(); },
    async useCpu() {
      if (delegate === 'CPU') return;
      landmarker.close();
      delegate = 'CPU';
      landmarker = await create(mode, 'CPU', mp, fileset, model);
    },
  };
  return api;
}

// detect(landmarker, source, timestampMs) does the same as landmarker.detect, and retries once on the CPU if the GPU
// delegate fails on a picture.
export async function detect(landmarker, source, timestampMs) {
  try {
    return landmarker.detect(source, timestampMs);
  } catch (e) {
    if (!e?.gpuFailed) throw e;
    await landmarker.useCpu();
    return landmarker.detect(source, timestampMs);
  }
}
