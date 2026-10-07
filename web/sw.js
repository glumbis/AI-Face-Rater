// Service worker: the app works offline after the first visit.
//   - App shell (page, css, scripts, icons): cached when installed, then served cache-first. The cache name carries
//     BUILD, which the deploy workflow replaces with the commit hash, so every deploy gets a fresh cache and the old
//     ones are deleted. Not stamped (local development): the network goes first, so edits show up with a reload.
//   - face_landmarker.task: cache-first in a cache of its own that survives deploys (bump MODEL_CACHE if the model
//     file is ever replaced).
//   - MediaPipe from jsDelivr: cached when first used. Pinned URLs (with @x.y.z) never change, so they are served
//     cache-first; anything unpinned is stale-while-revalidate. The page can also ask for them up front with
//     postMessage({type: 'cache', urls}) (config.js registerServiceWorker does that).
//   - Nothing else is touched: the leaderboard's calls to Supabase (another origin, never jsDelivr) are not handled
//     here at all, so they always go straight to the network and are never cached.
const BUILD = '__BUILD__';
const DEV = BUILD.startsWith('__');
const SHELL_CACHE = `afr-shell-${BUILD}`;
const MODEL_CACHE = 'afr-model-v1';
const CDN_CACHE = 'afr-cdn-v1';
const KEEP = [SHELL_CACHE, MODEL_CACHE, CDN_CACHE];

const SHELL = [
  './', 'index.html', 'manifest.webmanifest', 'css/app.css',
  'js/app.js', 'js/landmarker.js', 'js/overlay.js', 'js/scoring.js', 'js/headpose.js', 'js/facedata.js',
  'js/tips.js', 'js/history.js', 'js/share.js', 'js/config.js', 'js/imageops.js', 'js/leaderboard.js',
  'icons/favicon-32.png', 'icons/icon-192.png', 'icons/icon-512.png', 'icons/icon-maskable-512.png', 'icons/apple-touch-icon.png',
];
const MODEL = 'face_landmarker.task';

const scoped = (path) => new URL(path, self.registration.scope).href;
const cacheable = (res) => res && (res.ok || res.type === 'opaque');

self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    // File by file: one that doesn't exist (yet) must not stop the others from being cached
    const shell = await caches.open(SHELL_CACHE);
    await Promise.allSettled(SHELL.map((p) => shell.add(scoped(p))));
    const models = await caches.open(MODEL_CACHE);
    if (!(await models.match(scoped(MODEL)))) await models.add(scoped(MODEL)).catch(() => {});
    await self.skipWaiting();
  })());
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    for (const name of await caches.keys()) {
      if (name.startsWith('afr-') && !KEEP.includes(name)) await caches.delete(name);
    }
    await self.clients.claim();
  })());
});

self.addEventListener('message', (event) => {
  const data = event.data || {};
  if (data.type === 'skipWaiting') self.skipWaiting();
  if (data.type === 'cache' && Array.isArray(data.urls)) {
    event.waitUntil((async () => {
      const cdn = await caches.open(CDN_CACHE);
      for (const url of data.urls) {
        try {
          if (!(await cdn.match(url))) await cdn.add(url);
        } catch { /* offline or blocked: it is cached when first used instead */ }
      }
    })());
  }
});

async function cacheFirst(request, cacheName, options) {
  const cache = await caches.open(cacheName);
  const hit = await cache.match(request, options);
  if (hit) return hit;
  const res = await fetch(request);
  if (cacheable(res)) cache.put(request, res.clone()).catch(() => {});
  return res;
}

async function staleWhileRevalidate(request, cacheName) {
  const cache = await caches.open(cacheName);
  const hit = await cache.match(request);
  const update = fetch(request).then((res) => {
    if (cacheable(res)) cache.put(request, res.clone()).catch(() => {});
    return res;
  });
  if (hit) {
    update.catch(() => {});
    return hit;
  }
  return update;
}

async function networkFirst(request, cacheName, options) {
  const cache = await caches.open(cacheName);
  try {
    const res = await fetch(request);
    if (cacheable(res) && res.status === 200) cache.put(request, res.clone()).catch(() => {});
    return res;
  } catch (e) {
    const hit = await cache.match(request, options);
    if (hit) return hit;
    throw e;
  }
}

self.addEventListener('fetch', (event) => {
  const request = event.request;
  if (request.method !== 'GET') return;
  const url = new URL(request.url);
  if (url.protocol !== 'https:' && url.protocol !== 'http:') return;

  if (url.origin === self.location.origin) {
    if (url.pathname.endsWith('/' + MODEL)) {
      event.respondWith(cacheFirst(request, MODEL_CACHE, { ignoreSearch: true }));
      return;
    }
    if (url.pathname.endsWith('/sw.js')) return; // the browser checks for a new worker itself
    const options = { ignoreSearch: true };
    event.respondWith((async () => {
      try {
        const res = await (DEV ? networkFirst(request, SHELL_CACHE, options) : cacheFirst(request, SHELL_CACHE, options));
        return res;
      } catch (e) {
        // Offline and not cached: a page visit still gets the app
        if (request.mode === 'navigate') {
          const page = await caches.match(scoped('index.html'), { ignoreSearch: true });
          if (page) return page;
        }
        throw e;
      }
    })());
    return;
  }

  if (url.hostname === 'cdn.jsdelivr.net') {
    const pinned = /@\d+\.\d+\.\d+/.test(url.pathname);
    event.respondWith(pinned ? cacheFirst(request, CDN_CACHE) : staleWhileRevalidate(request, CDN_CACHE));
  }
  // Any other origin (Supabase for the leaderboard included) falls through to the network, uncached
});
