// The optional leaderboard and community average, kept in a Supabase database (see supabase/setup.sql). Only a
// nickname, a score and the five region scores are sent, never a picture or landmarks. Everything goes through the
// database functions with plain fetch; this file does nothing at all while config.js has no URL and key.
import { SUPABASE_URL, SUPABASE_KEY, CONSENT_VERSION } from './config.js';

export const enabled = Boolean(SUPABASE_URL && SUPABASE_KEY);

const BASE = SUPABASE_URL.replace(/\/+$/, '');
const TOKEN_KEY = 'afr-board-token';
const TIMEOUT_MS = 8000;
export const NAME_MIN = 2;
export const NAME_MAX = 20;
// Obvious slurs and profanity, matched as lowercase substrings with spaces and . _ - taken out. It is only a first
// filter; the nickname is a fun label, not an identity.
const BLOCKED = ['fuck', 'shit', 'cunt', 'bitch', 'whore', 'slut', 'nigg', 'fagg', 'retard', 'rapist', 'nazi', 'hitler'];

// Null when the nickname is fine, otherwise what to tell the person. Same rules as the database: 2 to 20 characters,
// letters, digits, spaces and . _ -
export function nameProblem(name) {
  const trimmed = String(name ?? '').trim();
  const length = [...trimmed].length;
  if (length < NAME_MIN || length > NAME_MAX) return `Use ${NAME_MIN} to ${NAME_MAX} characters.`;
  if (!/^[\p{L}\p{N} _.-]+$/u.test(trimmed)) return 'Use letters, numbers, spaces, . _ or - only.';
  const flat = trimmed.toLowerCase().replace(/[ ._-]/g, '');
  if (BLOCKED.some((word) => flat.includes(word))) return 'Please pick a different nickname.';
  return null;
}

// The random secret that proves which entries are yours: 32 random bytes as 64 hex characters. The database only
// keeps its hash. Kept in this browser; without storage it lives until the page closes.
let memoryToken = null;
function storedToken() {
  try {
    const token = localStorage.getItem(TOKEN_KEY);
    return /^[0-9a-f]{64}$/.test(token ?? '') ? token : memoryToken;
  } catch {
    return memoryToken;
  }
}
function myToken() {
  let token = storedToken();
  if (token) return token;
  token = [...crypto.getRandomValues(new Uint8Array(32))].map((b) => b.toString(16).padStart(2, '0')).join('');
  memoryToken = token;
  try { localStorage.setItem(TOKEN_KEY, token); } catch { /* the entry can't be deleted from this browser later */ }
  return token;
}

async function rpc(fn, args) {
  if (!enabled) throw new Error('The leaderboard is not set up.');
  const res = await fetch(`${BASE}/rest/v1/rpc/${fn}`, {
    method: 'POST',
    headers: { apikey: SUPABASE_KEY, Authorization: `Bearer ${SUPABASE_KEY}`, 'Content-Type': 'application/json' },
    body: JSON.stringify(args),
    signal: globalThis.AbortSignal?.timeout?.(TIMEOUT_MS),
  });
  if (!res.ok) throw new Error(`The leaderboard answered ${res.status}.`);
  const text = await res.text();
  return text ? JSON.parse(text) : null;
}

const oneDecimal = (n) => Math.round(n * 10) / 10;

// Adds or replaces this browser's entry for that ideal. regions: {jaw, brows, nose, eyes, outerLips} -> 0 to 10
export async function submitScore({ name, score, reference, regions }) {
  const problem = nameProblem(name);
  if (problem) throw new Error(problem);
  const rounded = {};
  for (const [key, value] of Object.entries(regions ?? {})) if (Number.isFinite(value)) rounded[key] = oneDecimal(value);
  await rpc('submit_score', {
    p_token: myToken(), p_name: name.trim(), p_score: oneDecimal(score), p_reference: reference,
    p_regions: Object.keys(rounded).length ? rounded : null, p_consent_version: CONSENT_VERSION,
  });
}

// The best 20 for an ideal: [{name, score}]
export async function topScores(reference) {
  const rows = await rpc('top_scores', { p_reference: reference, p_limit: 20 });
  return (rows ?? []).map((r) => ({ name: String(r.name), score: Number(r.score) }));
}

// How a score compares with the people who shared theirs: {players, average, below} (below is 0 to 1), or null while
// nobody has shared one
export async function communityStats(reference, score) {
  const rows = await rpc('community_stats', { p_reference: reference, p_score: oneDecimal(score) });
  const r = rows?.[0];
  if (!r || !Number(r.players)) return null;
  return { players: Number(r.players), average: Number(r.average), below: Number(r.below) };
}

// Removes every entry made from this browser; resolves to how many there were
export async function deleteMyEntries() {
  const token = storedToken();
  if (!token) return 0; // nothing was ever submitted from here
  return Number(await rpc('delete_my_entries', { p_token: token })) || 0;
}
