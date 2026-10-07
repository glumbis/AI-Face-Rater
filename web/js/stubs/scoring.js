// STUB with the contract's exports. It returns believable fake numbers so the UI can be built and tested.
// Replaced by the real js/scoring.js.
import { headAngles, facingProblem } from './headpose.js';

const MIN_FACE_SIZE = 0.30, HINT_FACE_SIZE = 0.35, HINT_SIZE_MARGIN = 0.02, MIN_FACE_SIZE_FILE = 0.15;
const REFUSED = {
  side: 'Head turned too far. Look at the camera.',
  down: 'Head tilted too far down. Lift your chin.',
  up: 'Head tilted too far up. Lower your chin.',
};

export class FaceError extends Error {
  constructor(message) { super(message); this.name = 'FaceError'; }
}

// Face height (landmark 10 to 152) as a share of the picture's shorter side
export function faceSize(points, width, height) {
  return Math.abs(points[152][1] - points[10][1]) / Math.min(width, height);
}

// null, "near" or "far", with the hysteresis of landmarkdetect.size_problem
export function sizeProblem(size, current = null) {
  if (size < MIN_FACE_SIZE + (current === 'far' ? HINT_SIZE_MARGIN : 0)) return 'far';
  if (size < HINT_FACE_SIZE + (current === 'near' || current === 'far' ? HINT_SIZE_MARGIN : 0)) return 'near';
  return null;
}

export function rateFace(imageData, points, matrix, reference, source = 'camera') {
  if (!points || points.length < 468) throw new FaceError('No face found. Face the camera with a straight face.');
  const size = faceSize(points, imageData.width, imageData.height);
  if (source === 'camera' ? size < MIN_FACE_SIZE : size < MIN_FACE_SIZE_FILE) {
    throw new FaceError(source === 'camera'
      ? 'Move closer to the camera.'
      : 'The face is too small in this photo. Use a photo where the face fills more of the picture.');
  }
  const problem = facingProblem(headAngles(matrix));
  if (problem) throw new FaceError(REFUSED[problem]);

  const xs = points.map((p) => p[0]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const mid = (points[10][0] + points[152][0]) / 2;
  const sym = Math.max(0, Math.min(1, 1 - (Math.abs(points[1][0] - mid) / (maxX - minX)) * 4));
  const bias = { boy: 0, girl: -0.5 }[reference] ?? 0;
  const score = Math.max(0, Math.min(10, 6.2 + 2.6 * sym + bias));
  const side = Math.round((maxX - minX) * 0.2);
  const cheek = (p) => {
    const x1 = Math.round(p[0] - side / 2), y1 = Math.round(p[1] - side / 2);
    const mask = new Uint8Array(side * side);
    for (let i = 0; i < mask.length; i++) mask[i] = ((i * 2654435761) >>> 0) % 97 < 6 ? 1 : 0;
    return { x1, y1, x2: x1 + side, y2: y1 + side, w: side, h: side, mask };
  };
  return {
    score, clarity: 0.82, symmetry: sym, shapeError: 0.1, skinPenalty: 0.003, symmetryPenalty: 0.001,
    cheeks: [cheek(points[50]), cheek(points[280])],
  };
}
export const shapeError = () => 0;
export const symmetry = () => 1;
export const align = () => null;
