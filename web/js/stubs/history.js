// STUB (localStorage, numbers only). Replaced by the real js/history.js.
const KEY = 'afr-stub-history';
const load = () => { try { return JSON.parse(localStorage.getItem(KEY)) || []; } catch { return []; } };
const save = (a) => { try { localStorage.setItem(KEY, JSON.stringify(a)); } catch { /* ignore */ } };

export function addRating({ reference, score, clarity, symmetry, source }) {
  const a = load(); a.push({ t: Date.now(), reference, score, clarity, symmetry, source }); save(a);
}
export function stats(reference) {
  const a = load().filter((e) => e.reference === reference);
  if (!a.length) return { best: null, average: null, count: 0, trend: null, diff: 0, streak: 0 };
  const scores = a.map((e) => e.score), last = scores[scores.length - 1];
  const avg = scores.reduce((x, y) => x + y, 0) / scores.length;
  const diff = scores.length > 1 ? last - scores[scores.length - 2] : 0;
  const days = new Set(load().map((e) => new Date(e.t).toDateString()));
  let streak = 0; const d = new Date();
  while (days.has(d.toDateString())) { streak++; d.setDate(d.getDate() - 1); }
  return { best: Math.max(...scores), average: avg, count: scores.length, trend: diff >= 0.1 ? 'up' : diff <= -0.1 ? 'down' : null, diff, streak };
}
export function clearHistory() { try { localStorage.removeItem(KEY); } catch { /* ignore */ } }
