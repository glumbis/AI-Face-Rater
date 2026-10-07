// Local history: a port of history.py to localStorage. Every rating is saved on this device as a row of numbers
// (never a picture), so the app can show how a new score compares with the earlier ones. Everything fails quietly:
// unavailable or full storage (private mode, blocked site data) never breaks the app; the history then lives in
// memory until the page is closed.

export const MODELS = ['boy', 'girl', 'average'];
// The up/down arrow only shows when the score is at least this far from your previous photo's score (history.TREND_MIN_DIFF)
export const TREND_MIN_DIFF = 0.1;

const KEY = 'afr.history.v1';
const MAX_ENTRIES = 2000;
// Rating the very same numbers for the same reference again within this long counts as the same picture
const DUPLICATE_MS = 10 * 60 * 1000;

let store; // undefined: not looked at yet, null: no storage, otherwise a localStorage-like object
let memory = []; // the history while there is no usable storage
const seen = new Set(); // "pictureId|reference" for the pictures rated in this page session

function getStore() {
  if (store !== undefined) return store;
  try {
    const s = globalThis.localStorage;
    s.setItem('afr.probe', '1');
    s.removeItem('afr.probe');
    store = s;
  } catch {
    store = null;
  }
  return store;
}

// For tests: use another storage (null for "no storage") and forget what was seen
export function _setStorage(s) {
  store = s;
  memory = [];
  seen.clear();
}

const num = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : null);

function read() {
  const s = getStore();
  if (!s) return memory;
  try {
    const list = JSON.parse(s.getItem(KEY) || '[]');
    if (!Array.isArray(list)) return [];
    // Rows that can't be read are skipped
    return list.filter((e) => e && num(e.t) !== null && MODELS.includes(e.r) && num(e.s) !== null);
  } catch {
    return [];
  }
}

function write(list) {
  if (list.length > MAX_ENTRIES) list = list.slice(list.length - MAX_ENTRIES);
  const s = getStore();
  if (s) {
    try {
      s.setItem(KEY, JSON.stringify(list));
      return true;
    } catch {
      // Full or blocked: carry on in memory
      memory = list;
      store = null;
      return false;
    }
  }
  memory = list;
  return false;
}

const round3 = (v) => Math.round(v * 1000) / 1000;
// A score as the page shows it, with one decimal (toFixed, like fmt in app.js)
const shown = (v) => Number(v.toFixed(1));

// Adds one rating: {reference, score, clarity, symmetry, source, pictureId?, now?}. clarity may be null (not counted).
// The same picture rated again against the same reference is saved once: give the same pictureId for one picture
// (then only the id decides), and without one an identical rating of the same reference within ten minutes counts
// as the same picture.
// Returns true if it was added, false if it was a duplicate or not valid.
export function addRating({ reference, score, clarity = null, symmetry = null, source = '', pictureId, now } = {}) {
  if (!MODELS.includes(reference) || num(score) === null) return false;
  const when = num(now) ?? Date.now();
  const id = pictureId == null ? null : `${pictureId}|${reference}`;
  if (id && seen.has(id)) return false;
  const entry = { t: when, r: reference, s: round3(score), c: num(clarity) === null ? null : round3(clarity),
    y: num(symmetry) === null ? null : round3(symmetry), src: String(source || '') };
  const list = read();
  const last = [...list].reverse().find((e) => e.r === reference);
  if (!id && last && when - last.t >= 0 && when - last.t < DUPLICATE_MS && last.s === entry.s && last.c === entry.c && last.y === entry.y) {
    return false;
  }
  if (id) seen.add(id);
  list.push(entry);
  write(list);
  return true;
}

// Local calendar day of a time, as a number that is one higher for each following day
function dayOf(t) {
  const d = new Date(t);
  return Math.round(Date.UTC(d.getFullYear(), d.getMonth(), d.getDate()) / 86400000);
}

// How many days in a row (ending today, or yesterday if nothing is rated yet today) have at least one rating
function streakOf(list, now) {
  const days = new Set(list.map((e) => dayOf(e.t)));
  const today = dayOf(now);
  let day = days.has(today) ? today : today - 1;
  let count = 0;
  while (days.has(day)) {
    count++;
    day--;
  }
  return count;
}

// How the latest rating of a reference looks next to the earlier ones (history.compare and history.streak):
// {best, average, count} include the latest rating, trend is "up", "down" or null against the PREVIOUS photo, the
// rating before the latest one (diff is the difference of the scores as shown, with one decimal, so 7.04 -> 7.16
// is 7.0 -> 7.2, a difference of 0.2), streak counts days over all references. With a single
// rating there is nothing to compare with (count is 1: show no best/average line then, like the desktop app). With
// no rating at all: best and average are null and count is 0.
export function stats(reference, now = Date.now()) {
  const list = read();
  const streak = streakOf(list, now);
  const scores = list.filter((e) => e.r === reference).map((e) => e.s);
  if (!scores.length) return { best: null, average: null, count: 0, trend: null, diff: 0, streak };
  let trend = null, diff = 0;
  if (scores.length >= 2) {
    diff = shown(shown(scores[scores.length - 1]) - shown(scores[scores.length - 2]));
    trend = diff >= TREND_MIN_DIFF ? 'up' : diff <= -TREND_MIN_DIFF ? 'down' : null;
  }
  return { best: Math.max(...scores), average: scores.reduce((a, b) => a + b, 0) / scores.length,
    count: scores.length, trend, diff, streak };
}

// The latest n scores for a reference, oldest first (for a small chart). Fewer if there are fewer; [] if none.
export function series(reference, n = 12) {
  const scores = read().filter((e) => e.r === reference).map((e) => e.s);
  return n > 0 ? scores.slice(-n) : [];
}

// How many ratings are saved (for example to enable a "Clear history" button)
export function entryCount() {
  return read().length;
}

// Deletes the history. Returns true if there is no history left
export function clearHistory() {
  seen.clear();
  memory = [];
  const s = getStore();
  if (!s) return true;
  try {
    s.removeItem(KEY);
    return true;
  } catch {
    return false;
  }
}
