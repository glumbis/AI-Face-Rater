// Runs the website's scoring.js and headpose.js on golden.json (made by tools/make_golden.py from the Python app).
// Tolerances: score 0.05 (also each region's score), clarity 0.02, angles 0.2 degrees (the real differences are far smaller, see the Max rows).
// The ideal face's numbers (measurements, errors, deviations) are compared much tighter: both sides work in double precision.
import { GENDERS, IDEAL, SOURCE_HASH, SUBSET } from '../js/facedata.js';
import {
  HeadTracker, RecentFrames, facingProblem, headAngles, sharpness,
} from '../js/headpose.js';
import {
  gaussianBlurF32, lbgrToLab, resizeAreaU8, resizeLinearF32,
} from '../js/imageops.js';
import {
  ExpressionTracker, FaceError, cheekInconsistencies, deviations, expression, expressionProblem, faceSize, idealError,
  idealErrors, measure, neutralMeasures, rateFace, regionScores, scoreFromError, sizeProblem, skinClarity,
  subsetPoints, symmetry, turned,
} from '../js/scoring.js';

const TOL = {
  score: 0.05, region: 0.05, clarity: 0.02, angle: 0.2, shapeError: 1e-7, measure: 1e-8, turned: 2e-6, symmetry: 0.002,
  size: 1e-9, sharp: 1e-3,
};
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

  // symmetry
  near(g, 'symmetry', 'symmetry', symmetry(pts), e.symmetry, TOL.symmetry);
  near(g, 'symmetry from 478 points', 'symmetry', symmetry(c.points), e.symmetry, TOL.symmetry);

  // the measurements, and the same with the expression taken off
  compareValues(g, 'measure', measure(c.points), e.measures, TOL.measure, 'measurement');
  compareValues(g, 'neutral measure', neutralMeasures(c.points, c.blendshapes), e.neutral, TOL.measure, 'measurement');

  // the ratings
  checkRatings(g, img, c, c.blendshapes, e, e);
  for (const variation of c.expressions ?? []) {
    checkRatings(`${g} / ${variation.label}`, img, c, variation.blendshapes, variation.expected, e);
  }
}

// Numbers by name: the same names in the same order, each within tol
function compareValues(group, label, got, want, tol, kind) {
  record(group, `${label} names`, Object.keys(got).join() === Object.keys(want).join(), Object.keys(got).join());
  for (const [key, value] of Object.entries(want)) near(group, `${label} ${key}`, kind, got[key], value, tol);
}

// The ratings of one picture with some blendshapes against what Python gave (expected: refusal and refs; facts: the
// parts that do not depend on the blendshapes), for both genders and both sources
function checkRatings(g, img, c, blendshapes, expected, facts) {
  for (const [ref, want] of Object.entries(expected.refs)) {
    const values = neutralMeasures(c.points, blendshapes);
    const found = idealErrors(values, ref);
    near(g, `ideal error vs ${ref}`, 'shape error', found.error, want.shapeError, TOL.shapeError);
    compareValues(g, `${ref} region errors`, found.regions, want.regionErrors, TOL.shapeError, 'shape error');
    compareValues(g, `${ref} deviations`, found.deviations, want.deviations, TOL.shapeError, 'deviation');
    near(g, `idealError() vs ${ref}`, 'shape error', idealError(c.points, ref, blendshapes).error, want.shapeError, TOL.shapeError);
    for (const source of ['camera', 'file']) {
      const label = `rateFace ${ref}/${source}`;
      const refusal = expected.refusal[source];
      try {
        const r = rateFace(img, c.points, c.matrix, ref, source, blendshapes);
        if (refusal) { record(g, label, false, `should have been refused: ${refusal}`); continue; }
        near(g, `${label} score`, 'score', r.score, want.score, TOL.score);
        near(g, `${label} shape error`, 'shape error', r.shapeError, want.shapeError, TOL.shapeError);
        record(g, `${label} regions`, Object.keys(r.regions).join() === Object.keys(want.regions).join(), Object.keys(r.regions).join());
        for (const [region, wantScore] of Object.entries(want.regions)) {
          near(g, `${label} ${region} score`, 'region score', r.regions[region], wantScore, TOL.region);
        }
        compareValues(g, `${label} deviations`, r.deviations, want.deviations, TOL.shapeError, 'deviation');
        near(g, `${label} clarity`, 'clarity', r.clarity, facts.clarity, TOL.clarity);
        near(g, `${label} skin penalty`, 'penalty', r.skinPenalty, facts.skinPenalty, 1e-3);
        near(g, `${label} symmetry penalty`, 'penalty', r.symmetryPenalty, facts.symmetryPenalty, 1e-3);
        near(g, `${label} skin factor`, 'factor', r.skinFactor, want.skinFactor, 1e-2);
        near(g, `${label} symmetry factor`, 'factor', r.symmetryFactor, want.symmetryFactor, 1e-2);
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

// ---------- The ideal face ----------
const sameNumbers = (got, want, tol) => got.length === want.length && got.every((v, i) => Math.abs(v - want[i]) <= tol);

// The 478 points of a stored case (the golden file keeps a stored picture's points with the case)
function pointsOf(golden, item) {
  return item.from ? golden.cases.find((c) => c.name === item.from).points : item.points;
}

function testIdealCases(golden) {
  for (const item of golden.idealCases) {
    const g = `ideal ${item.label}`;
    const points = pointsOf(golden, item);
    const got = turned(points);
    let worst = 0;
    got.forEach((p, i) => p.forEach((v, a) => { worst = Math.max(worst, Math.abs(v - item.turned[i][a])); }));
    dev('turned face (reference units)', worst);
    record(g, 'turned face', worst <= TOL.turned, `largest difference ${worst.toExponential(2)}`);
    compareValues(g, 'measure', measure(points), item.measure, TOL.measure, 'measurement');
    compareValues(g, 'neutral measure', neutralMeasures(points, item.blendshapes), item.neutral, TOL.measure, 'measurement');
    for (const [key, want] of Object.entries(item.errors)) {
      const [gender, how] = key.split('/');
      const found = idealError(points, gender, how === 'neutral' ? item.blendshapes : null);
      near(g, `error ${key}`, 'shape error', found.error, want.error, TOL.shapeError);
      compareValues(g, `region errors ${key}`, found.regions, want.regions, TOL.shapeError, 'shape error');
      compareValues(g, `deviations ${key}`, found.deviations, want.deviations, TOL.shapeError, 'deviation');
    }
  }
}

function testExpressionCases(cases) {
  cases.forEach((c, i) => {
    const g = 'expression';
    const name = `case ${i} ${JSON.stringify(c.blendshapes)}`;
    const got = expression(c.blendshapes);
    record(g, `${name} values`, c.expression === null ? got === null : sameNumbers(got, c.expression, 1e-12), `got ${got}, want ${c.expression}`);
    record(g, `${name} problem`, expressionProblem(c.blendshapes) === c.problem, `got ${expressionProblem(c.blendshapes)}, want ${c.problem}`);
    record(g, `${name} hint`, expressionProblem(c.blendshapes, true) === c.hint, `got ${expressionProblem(c.blendshapes, true)}, want ${c.hint}`);
  });
}

function testExpressionTrackers(sequences) {
  sequences.forEach((steps, n) => {
    const tracker = new ExpressionTracker();
    let wrong = 0;
    const seen = new Set();
    for (const s of steps) {
      const got = tracker.add(s.blendshapes, s.t);
      seen.add(got);
      if (got !== s.problem || tracker.hint !== s.problem) wrong++;
    }
    record('ExpressionTracker', `sequence ${n} (${steps.length} pictures)`, wrong === 0, `${wrong} different answers`);
    record('ExpressionTracker', `sequence ${n} shows several tips`, seen.size >= 2, `tips seen: ${[...seen]}`);
  });
}

// ---------- What the ideal face must do (the same checks as the Python tests) ----------
const targetsOf = (gender) => Object.fromEntries(Object.entries(IDEAL.IDEALS[gender]).map(([name, [target]]) => [name, target]));

// The landmarks tilted, turned and rolled in 3D, resized and moved
function turnedIn3d(points, tilt, turn, roll, scale, shift) {
  const [a, b, c] = [tilt, turn, roll].map((d) => (d * Math.PI) / 180);
  const tilting = [[1, 0, 0], [0, Math.cos(a), -Math.sin(a)], [0, Math.sin(a), Math.cos(a)]];
  const turning = [[Math.cos(b), 0, -Math.sin(b)], [0, 1, 0], [Math.sin(b), 0, Math.cos(b)]];
  const rolling = [[Math.cos(c), -Math.sin(c), 0], [Math.sin(c), Math.cos(c), 0], [0, 0, 1]];
  const times = (m, n) => m.map((row) => [0, 1, 2].map((j) => row[0] * n[0][j] + row[1] * n[1][j] + row[2] * n[2][j]));
  const r = times(rolling, times(tilting, turning));
  const mid = [0, 1, 2].map((k) => points.reduce((sum, p) => sum + p[k], 0) / points.length);
  return points.map((p) => {
    const d = [p[0] - mid[0], p[1] - mid[1], p[2] - mid[2]];
    return [0, 1, 2].map((k) => scale * (r[k][0] * d[0] + r[k][1] * d[1] + r[k][2] * d[2]) + mid[k] + shift[k]);
  });
}

// Pairs of MediaPipe landmarks that are each other's left and right partner (from the 162 list's MIRROR), and the
// nostril wings
async function mirrorPartners() {
  const { MIRROR } = await import('../js/facedata.js');
  const partner = Array.from({ length: 478 }, (_, i) => i);
  SUBSET.forEach((landmark, i) => { partner[landmark] = SUBSET[MIRROR[i]]; });
  partner[129] = 358; partner[358] = 129;
  return partner;
}

async function testBehaviour(golden) {
  const g = 'ideal behaviour';
  const boy = golden.cases.find((c) => c.name === 'perBoy-base').points;
  const girl = golden.cases.find((c) => c.name === 'perGirl-base').points;

  // a head tilted, turned or rolled, nearer or further away, measures the same
  const before = measure(boy);
  for (const [tilt, turn, roll] of [[20, 0, 0], [-20, 0, 0], [0, 10, 0], [15, -8, 5], [0, 0, 30]]) {
    const after = measure(turnedIn3d(boy, tilt, turn, roll, 1.3, [40, -25, 7]));
    const worst = Math.max(...Object.keys(before).map((k) => Math.abs(after[k] - before[k])));
    dev('turned head (measurement)', worst);
    record(g, `turned head ${tilt}/${turn}/${roll} measures the same`, worst < 1e-8, `largest difference ${worst.toExponential(2)}`);
  }

  // the mirror image (x the other way round, left and right landmarks swapped) measures the same
  const partner = await mirrorPartners();
  const mirrored = Array.from({ length: 478 }, (_, i) => [-boy[partner[i]][0], boy[partner[i]][1], boy[partner[i]][2]]);
  const mirrorMeasures = measure(mirrored);
  const worstMirror = Math.max(...Object.keys(before).map((k) => Math.abs(mirrorMeasures[k] - before[k])));
  dev('mirrored face (measurement)', worstMirror);
  record(g, 'a mirrored face measures the same', worstMirror < 1e-8, `largest difference ${worstMirror.toExponential(2)}`);

  // the ideal face scores ten, and each region only counts its own measurements
  for (const gender of GENDERS) {
    const targets = targetsOf(gender);
    const { error, regions, deviations: z } = idealErrors(targets, gender);
    record(g, `${gender}: the ideal face has no error`, error === 0 && Object.values(regions).every((e) => e === 0)
      && Object.values(z).every((v) => v === 0), `error ${error}`);
    near(g, `${gender}: the ideal face scores 10`, 'score', scoreFromError(error), 10, 1e-9);
    for (const [name, region, , way] of IDEAL.FEATURES) {
      const changed = { ...targets };
      changed[name] -= 3 * IDEAL.IDEALS[gender][name][1] * (way >= 0 ? 1 : -1);
      const found = idealErrors(changed, gender);
      const others = Object.entries(found.regions).filter(([r]) => r !== region).every(([, e]) => e === 0);
      record(g, `${gender}/${name}: only the ${region} region changes`,
        Math.abs(found.deviations[name] - (way >= 0 ? -3 : 3)) < 1e-9 && found.regions[region] > 0.5 && others,
        JSON.stringify(found.regions));
    }
    // past the target is fine for the one-sided measurements
    for (const [name, , , way] of IDEAL.FEATURES) {
      const changed = { ...targets };
      changed[name] += 2 * IDEAL.IDEALS[gender][name][1] * (way || 1);
      const error = idealErrors(changed, gender).error;
      record(g, `${gender}/${name}: ${way ? 'past the target is fine' : 'both ways count'}`, (error < 1e-12) === (way !== 0), `error ${error}`);
    }
  }

  // one odd measurement can't take the whole score
  const odd = measure(boy);
  odd.noseWidth = 100;
  record(g, 'one odd measurement is cut at MAX_DEVIATION', deviations(odd, 'boy').noseWidth === IDEAL.MAX_DEVIATION, '');

  // a rounded jaw is less sharp and scores lower; a wider face is shorter and scores lower
  const rounded = boy.map((p) => p.slice());
  for (const chain of [IDEAL.JAW_LEFT, IDEAL.JAW_RIGHT]) {
    for (let round = 0; round < 3; round++) {
      const inner = chain.slice(1, -1).map((i) => rounded[i].slice());
      chain.slice(1, -1).forEach((i, k) => {
        rounded[i] = [0, 1, 2].map((a) => 0.4 * inner[k][a] + 0.3 * (rounded[chain[k]][a] + rounded[chain[k + 2]][a]));
      });
    }
  }
  record(g, 'a rounded jaw is less sharp', measure(rounded).jawSharpness < measure(boy).jawSharpness - 0.03, '');
  record(g, 'a rounded jaw lowers the jaw region', idealError(rounded, 'boy').regions.jaw > idealError(boy, 'boy').regions.jaw + 0.3, '');
  const wide = boy.map((p) => [p[0] * 1.12, p[1], p[2] * 1.12]);
  record(g, 'a wider face is shorter', measure(wide).faceLength < measure(boy).faceLength * 0.92, '');
  record(g, 'a wider face lowers the jaw region', idealError(wide, 'boy').regions.jaw > idealError(boy, 'boy').regions.jaw + 0.5, '');

  // region scores go through the same formula as the total
  const { regions } = idealError(girl, 'girl');
  const scores = regionScores(regions);
  record(g, 'region scores use the score formula', Object.entries(regions).every(([r, e]) => scores[r] === scoreFromError(e)), '');

  // the face turned back is in the size of the reference
  const front = turned(boy);
  const eyeSpan = Math.hypot(front[33][0] - front[263][0], front[33][1] - front[263][1]);
  record(g, 'turned() resizes to the reference (outer eye corners about 1 apart)', Math.abs(eyeSpan - 1) < 0.15, `${eyeSpan}`);

  // the expression: a big smile or an open mouth is refused, the tip needs the margin to go away
  const shapes = (smile = 0, mouthOpen = 0) => ({ mouthSmileLeft: smile, mouthSmileRight: smile, eyeSquintLeft: 0.3, eyeSquintRight: 0.3, jawOpen: mouthOpen });
  record(g, 'no blendshapes: no problem', expressionProblem(null) === null && expressionProblem({}) === null, '');
  record(g, 'a big smile is a smile problem', expressionProblem(shapes(0.6)) === 'smile', '');
  record(g, 'an open mouth is a mouthOpen problem', expressionProblem(shapes(0, 0.4)) === 'mouthOpen', '');
  record(g, 'a slight smile is fine', expressionProblem(shapes(0.3)) === null && expressionProblem(shapes(0.3), true) === null, '');
  record(g, 'the hint comes earlier than the refusal', expressionProblem(shapes(0.4), true) === 'smile' && expressionProblem(shapes(0.4)) === null, '');
  const tracker = new ExpressionTracker();
  let t = 100;
  const feed = (n, smile) => { let last = null; for (let i = 0; i < n; i++) { t += 0.05; last = tracker.add(shapes(smile), t); } return last; };
  record(g, 'tracker: calm face gives no tip', feed(5, 0.1) === null, '');
  t += 0.05;
  record(g, 'tracker: one odd picture does not bring the tip', tracker.add(shapes(0.9), t) === null, '');
  feed(12, 0.5);
  record(g, 'tracker: a smile brings the tip', tracker.problem === 'smile', `${tracker.problem}`);
  feed(12, IDEAL.HINT_SMILE - 0.02);
  record(g, 'tracker: just under the limit it stays (the margin)', tracker.problem === 'smile', `${tracker.problem}`);
  feed(12, 0.1);
  record(g, 'tracker: a calm face takes it away', tracker.problem === null, `${tracker.problem}`);
  tracker.add(shapes(0.9), t + 0.05);
  tracker.reset();
  record(g, 'tracker: reset forgets everything', tracker.problem === null && tracker.add(null, t + 0.1) === null, '');
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
    testIdealCases(golden);
    testExpressionCases(golden.expressionCases);
    testExpressionTrackers(golden.expressionTrackers);
    await testBehaviour(golden);
  } catch (error) {
    record('setup', 'the run itself', false, `${error.name}: ${error.message}\n${error.stack}`);
  }
  render(true);
  window.__done = true;
  window.__results = { passed, failed, maxDev, failures: rows.filter((r) => !r.ok) };
}

main();
