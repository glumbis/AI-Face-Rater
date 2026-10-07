// Runs the website's scoring.js and headpose.js on golden.json (made by tools/make_golden.py from the Python app).
// Tolerances: score 0.05 (also each region's score), clarity 0.02, angles 0.2 degrees (the real differences are far smaller, see the Max rows).
import { SOURCE_HASH } from '../js/facedata.js';
import {
  HeadTracker, RecentFrames, facingProblem, headAngles, sharpness,
} from '../js/headpose.js';
import {
  gaussianBlurF32, lbgrToLab, resizeAreaU8, resizeLinearF32,
} from '../js/imageops.js';
import {
  FaceError, cheekInconsistencies, faceSize, modelErrors, rateFace, shapeError, sizeProblem, skinClarity, subsetPoints,
  symmetry,
} from '../js/scoring.js';

const TOL = { score: 0.05, region: 0.05, clarity: 0.02, angle: 0.2, shapeError: 1e-4, symmetry: 0.002, size: 1e-9, sharp: 1e-3 };
const MASK_MISMATCH = 0.01; // share of a cheek's pixels that may differ

const rows = [];
const maxDev = {};
let passed = 0, failed = 0;

function record(group, name, ok, detail = '') {
  rows.push({ group, name, ok, detail });
  if (ok) passed++; else failed++;
}

function dev(kind, value) {
  maxDev[kind] = Math.max(maxDev[kind] ?? 0, value);
}

// Compares numbers (both null counts as equal)
function near(group, name, kind, got, want, tol) {
  if (got === null || want === null || got === undefined || want === undefined) {
    record(group, name, got === want, `got ${got}, want ${want}`);
    return;
  }
  const d = Math.abs(got - want);
  dev(kind, d);
  record(group, name, d <= tol, `got ${got}, want ${want}, off by ${d.toExponential(2)} (limit ${tol})`);
}

const bytes = (b64) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));

async function decodePicture(b64) {
  const image = new Image();
  image.src = `data:image/png;base64,${b64}`;
  await image.decode();
  const canvas = document.createElement('canvas');
  canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(image, 0, 0);
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

const maxAbsDiff = (a, b) => {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
};

// ---------- Image operations ----------
function testOps(ops) {
  for (const [i, c] of ops.area.entries()) {
    const out = resizeAreaU8(Uint8Array.from(c.bgr), c.w, c.h, 3, c.dw, c.dh);
    const diffs = out.filter((v, k) => v !== c.out[k]).length;
    const m = maxAbsDiff(out, c.out);
    dev('area resize (levels)', m);
    // shrinking is exact; enlarging (small cheek squares) differs by 1 level in a few percent of the bytes (OpenCV's 8-bit path)
    record('ops', `INTER_AREA ${c.w}x${c.h} to ${c.dw}x${c.dh}`, m <= 1 && diffs <= (c.dw > c.w ? 0.1 : 0.01) * out.length,
      `${diffs} of ${out.length} bytes differ, max ${m}`);
    const g = resizeAreaU8(Uint8Array.from(c.gray), c.w, c.h, 1, c.dw, c.dh);
    const gm = maxAbsDiff(g, c.grayOut);
    record('ops', `INTER_AREA gray ${c.w}x${c.h} to ${c.dw}x${c.dh}`, gm <= 1, `max ${gm}`);
  }
  for (const c of ops.blur) {
    const out = gaussianBlurF32(Float32Array.from(c.src), c.w, c.h, 3, c.sigma);
    const m = maxAbsDiff(out, c.out);
    dev('gaussian blur', m);
    record('ops', `GaussianBlur ${c.w}x${c.h} sigma ${c.sigma}`, m < 1e-5, `max ${m.toExponential(2)}`);
  }
  {
    const n = ops.lab.src.length / 3;
    const out = lbgrToLab(Float32Array.from(ops.lab.src), n, new Float32Array(n * 3));
    const m = maxAbsDiff(out, ops.lab.out);
    dev('Lab', m);
    record('ops', 'LBGR to Lab', m < 2e-3, `max ${m.toExponential(2)}`);
  }
  for (const c of ops.linear) {
    const out = resizeLinearF32(Float32Array.from(c.src), c.w, c.h, c.dw, c.dh);
    const m = maxAbsDiff(out, c.out);
    dev('bilinear resize', m);
    record('ops', `bilinear ${c.w}x${c.h} to ${c.dw}x${c.dh}`, m < 1e-4, `max ${m.toExponential(2)}`);
  }
}

// ---------- One picture ----------
async function testCase(c) {
  const g = `case ${c.name}`;
  const e = c.expected;
  const img = await decodePicture(c.png);
  record(g, 'picture size', img.width === c.width && img.height === c.height, `${img.width}x${img.height}`);
  const pts = subsetPoints(c.points);

  // head angles
  const angles = headAngles(c.matrix);
  near(g, 'turn', 'angle (degrees)', angles.turn, e.angles.turn, TOL.angle);
  near(g, 'tilt', 'angle (degrees)', angles.tilt, e.angles.tilt, TOL.angle);
  record(g, 'facing problem', facingProblem(angles) === e.problem, `got ${facingProblem(angles)}, want ${e.problem}`);
  const flat = [0, 1, 2, 3].flatMap((j) => [0, 1, 2, 3].map((i) => c.matrix[i][j])); // column-major, like MediaPipe JS
  const viaFlat = headAngles({ rows: 4, columns: 4, data: flat });
  near(g, 'turn from MediaPipe-style data', 'angle (degrees)', viaFlat.turn, e.angles.turn, TOL.angle);
  const viaRowFlat = headAngles(c.matrix.flat());
  near(g, 'tilt from flat row-major array', 'angle (degrees)', viaRowFlat.tilt, e.angles.tilt, TOL.angle);

  // size and sharpness
  near(g, 'face size', 'face size', faceSize(c.points, img), e.size, TOL.size);
  near(g, 'sharpness', 'sharpness (relative)', sharpness(img) / e.sharpness, 1, TOL.sharp);

  // skin
  const skin = skinClarity(img, pts);
  record(g, 'cheeks judged', skin.cheeks.length === e.cheeks.length, `got ${skin.cheeks.length}, want ${e.cheeks.length}`);
  near(g, 'clarity', 'clarity', skin.clarity, e.clarity, TOL.clarity);
  skin.cheeks.forEach((cheek, i) => {
    const want = e.cheeks[i];
    if (!want) return;
    record(g, `cheek ${i} square`, cheek.square.join() === [want.x1, want.y1, want.x2, want.y2].join()
      && cheek.w === want.w && cheek.h === want.h && cheek.x2 - cheek.x1 === cheek.w && cheek.y2 - cheek.y1 === cheek.h,
    `got ${cheek.square} ${cheek.w}x${cheek.h}`);
    near(g, `cheek ${i} uneven share`, 'uneven share', cheek.fraction, want.fraction, 0.01);
    const bits = bytes(want.mask);
    let diff = 0;
    for (let k = 0; k < want.w * want.h; k++) {
      const wantBit = (bits[k >> 3] >> (7 - (k & 7))) & 1;
      if (cheek.mask[k] !== wantBit) diff++;
    }
    const share = diff / (want.w * want.h);
    dev('uneven-skin mask mismatch (share of pixels)', share);
    record(g, `cheek ${i} mask`, share <= MASK_MISMATCH, `${diff} of ${want.w * want.h} pixels differ`);
  });

  // symmetry and shape
  near(g, 'symmetry', 'symmetry', symmetry(pts), e.symmetry, TOL.symmetry);
  near(g, 'symmetry from 478 points', 'symmetry', symmetry(c.points), e.symmetry, TOL.symmetry);
  for (const [ref, want] of Object.entries(e.refs)) {
    near(g, `shape error vs ${ref}`, 'shape error', shapeError(pts, ref), want.shapeError, TOL.shapeError);
    modelErrors(pts, ref).forEach(({ error }, i) => {
      near(g, `shape error vs ${ref} model face ${i + 1}`, 'shape error', error, want.errors[i], TOL.shapeError);
    });
    for (const source of ['camera', 'file']) {
      const label = `rateFace ${ref}/${source}`;
      const refusal = e.refusal[source];
      try {
        const r = rateFace(img, c.points, c.matrix, ref, source);
        if (refusal) { record(g, label, false, `should have been refused: ${refusal}`); continue; }
        near(g, `${label} score`, 'score', r.score, want.score, TOL.score);
        record(g, `${label} closest model face`, r.modelFace === want.modelFace && r.modelFaceCount === want.modelFaceCount,
          `got ${r.modelFace} of ${r.modelFaceCount}, want ${want.modelFace} of ${want.modelFaceCount}`);
        record(g, `${label} regions`, Object.keys(r.regions).join() === Object.keys(want.regions).join(), Object.keys(r.regions).join());
        for (const [region, wantScore] of Object.entries(want.regions)) {
          near(g, `${label} ${region} score`, 'region score', r.regions[region], wantScore, TOL.region);
        }
        near(g, `${label} clarity`, 'clarity', r.clarity, e.clarity, TOL.clarity);
        near(g, `${label} skin penalty`, 'penalty', r.skinPenalty, e.skinPenalty, 1e-3);
        near(g, `${label} symmetry penalty`, 'penalty', r.symmetryPenalty, e.symmetryPenalty, 1e-3);
      } catch (error) {
        const ok = refusal && error instanceof FaceError && error.message === refusal;
        record(g, label, ok, ok ? `refused: ${error.message}` : `${error.name}: ${error.message}, want ${refusal}`);
      }
    }
  }
}

// ---------- The rest ----------
function testHeadCases(cases, base) {
  for (const c of cases) {
    const matrix = c.matrix.map((r) => r.map((v) => (v === null ? NaN : v)));
    const angles = headAngles(matrix);
    const g = `head ${c.label}`;
    if (c.angles === null) {
      record(g, 'no angles', angles === null, `got ${JSON.stringify(angles)}`);
    } else {
      near(g, 'turn', 'angle (degrees)', angles.turn, c.angles.turn, TOL.angle);
      near(g, 'tilt', 'angle (degrees)', angles.tilt, c.angles.tilt, TOL.angle);
    }
    record(g, 'facing problem', facingProblem(angles) === c.problem, `got ${facingProblem(angles)}, want ${c.problem}`);
    // The same matrix on a real picture: refused for the same reason (the face is big enough)
    if (base && c.angles !== null && c.matrix.length === 4) {
      try {
        rateFace(base.img, base.points, matrix, 'boy', 'camera');
        record(g, 'rateFace refusal', c.problem === null, c.problem ? `should be refused (${c.problem})` : 'rated');
      } catch (error) {
        record(g, 'rateFace refusal', error instanceof FaceError && error.message === c.message, error.message);
      }
    }
  }
}

function testSizes(cases) {
  for (const c of cases) {
    const got = sizeProblem(c.size, c.current);
    record('size hint', `size ${c.size} showing ${c.current}`, got === c.problem, `got ${got}, want ${c.problem}`);
  }
}

function testTrackers(sequences) {
  sequences.forEach((steps, n) => {
    const tracker = new HeadTracker();
    let wrong = 0;
    for (const s of steps) {
      const got = tracker.add({ turn: s.turn, tilt: s.tilt }, s.t);
      if (got !== s.problem || tracker.hint !== s.problem) wrong++;
    }
    record('HeadTracker', `sequence ${n} (${steps.length} pictures)`, wrong === 0, `${wrong} different answers`);
  });
}

function testRecents(sequences) {
  sequences.forEach((steps, n) => {
    const recent = new RecentFrames();
    let wrong = 0, asked = 0;
    for (const s of steps) {
      recent.add(s.id, { turn: s.turn, tilt: s.tilt }, s.sharp, s.t);
      if (s.bestAt !== undefined) {
        asked++;
        if (recent.best(s.bestAt) !== s.best) wrong++;
      }
    }
    record('RecentFrames', `sequence ${n} (${asked} questions)`, wrong === 0, `${wrong} different answers`);
  });
}

// ---------- Run ----------
function render(done) {
  const out = document.getElementById('out');
  const summary = document.getElementById('summary');
  summary.textContent = `${done ? 'DONE' : 'RUNNING'}: ${passed} passed, ${failed} failed, ${passed + failed} checks. `
    + (done ? (failed ? 'RESULT: FAIL' : 'RESULT: ALL PASS') : '');
  summary.className = done ? (failed ? 'fail' : 'pass') : '';
  const lines = [];
  lines.push('Max deviations:');
  for (const [k, v] of Object.entries(maxDev)) lines.push(`  ${k}: ${v.toExponential(3)}`);
  lines.push('');
  for (const r of rows) if (!r.ok) lines.push(`FAIL [${r.group}] ${r.name}: ${r.detail}`);
  if (!failed) lines.push('No failures.');
  lines.push('');
  const groups = new Map();
  for (const r of rows) {
    const s = groups.get(r.group) ?? [0, 0];
    s[r.ok ? 0 : 1]++;
    groups.set(r.group, s);
  }
  lines.push('Per group (passed / failed):');
  for (const [g, [p, f]] of groups) lines.push(`  ${g}: ${p} / ${f}`);
  lines.push('');
  lines.push('All checks:');
  for (const r of rows) lines.push(`${r.ok ? 'pass' : 'FAIL'} [${r.group}] ${r.name}: ${r.detail}`);
  out.textContent = lines.join('\n');
}

async function main() {
  try {
    const golden = await (await fetch('golden.json')).json();
    record('setup', 'facedata.js matches golden.json (run both tools/ scripts after changing the Python files)',
      golden.sourceHash === SOURCE_HASH, `facedata ${SOURCE_HASH}, golden ${golden.sourceHash}`);
    testOps(golden.ops);
    let base = null;
    for (const c of golden.cases) {
      await testCase(c);
      if (!base && c.name === 'perBoy-base') base = { img: await decodePicture(c.png), points: c.points };
      render(false);
    }
    testHeadCases(golden.headCases, base);
    testSizes(golden.sizeCases);
    testTrackers(golden.trackers);
    testRecents(golden.recents);
  } catch (error) {
    record('setup', 'the run itself', false, `${error.name}: ${error.message}\n${error.stack}`);
  }
  render(true);
  window.__done = true;
  window.__results = { passed, failed, maxDev, failures: rows.filter((r) => !r.ok) };
}

main();
