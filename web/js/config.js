// Shared settings. MP_VERSION is the one place that says which MediaPipe build the site loads. Change it here (and
// nowhere else); sw.js caches whatever pinned jsDelivr URL the page asks for, so it follows automatically.
export const MP_VERSION = '0.10.21';

const MP_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
// The JS bundle (import it with: await import(MP_BUNDLE_URL)) and the folder with the WASM files
// (FilesetResolver.forVisionTasks(MP_WASM_URL))
export const MP_BUNDLE_URL = `${MP_BASE}/vision_bundle.mjs`;
export const MP_WASM_URL = `${MP_BASE}/wasm`;
// The face model, served by the site itself (copy of the repo root's face_landmarker.task)
export const MODEL_URL = new URL('../face_landmarker.task', import.meta.url).href;

// What the service worker is asked to keep for offline use (the SIMD build of the WASM, which every current browser uses)
export const MP_PRECACHE_URLS = [MP_BUNDLE_URL, `${MP_WASM_URL}/vision_wasm_internal.js`, `${MP_WASM_URL}/vision_wasm_internal.wasm`];

// Registers the service worker (offline use + install). Safe to call anywhere: it does nothing without service worker
// support or a secure context (https or localhost), and never throws. Call it once, after the page has loaded.
export function registerServiceWorker() {
  if (!('serviceWorker' in navigator) || !globalThis.isSecureContext) return Promise.resolve(null);
  const register = () => navigator.serviceWorker.register(new URL('../sw.js', import.meta.url).href)
    .then(async (registration) => {
      // Ask the worker to keep the MediaPipe files too (it also caches them whenever the page loads them)
      try {
        const ready = await navigator.serviceWorker.ready;
        (ready.active || registration.active)?.postMessage({ type: 'cache', urls: MP_PRECACHE_URLS });
      } catch { /* not needed for the app to work */ }
      return registration;
    })
    .catch((e) => {
      console.warn('Service worker not registered', e);
      return null;
    });
  if (document.readyState === 'complete') return register();
  return new Promise((resolve) => addEventListener('load', () => resolve(register()), { once: true }));
}
