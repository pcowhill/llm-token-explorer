# Model caching and offline use

## Application shell (service worker)

The production build registers a Workbox service worker (scope `/llm-token-explorer/`)
that precaches the HTML, JavaScript (including the inference worker), CSS, icons and the
web manifest. After the first visit the app opens offline and can be installed as a PWA.
The ONNX Runtime WASM binary (27 MB) is cached at runtime the first time a model loads, so
first-time visitors who never load a model do not download it. When a new version is
deployed, the header shows **Update available — reload**.

## Model files (Cache Storage)

- Files downloaded from Hugging Face are stored in Cache Storage under the cache name
  `transformers-cache`, keyed by their pinned `…/resolve/<sha>/<file>` URL.
- Requests are answered **cache-first**, so a cached model loads without any network
  access. Transformers.js performs a few revision-less (`main`) existence checks; the app
  pins those to the revision being loaded.
- **Model → Cache** lists cached models with sizes, removes one or all, shows the storage
  estimate (`navigator.storage.estimate()`) and can request persistent storage
  (`navigator.storage.persist()`), which reduces the chance of eviction.
- If a file cannot be written (e.g. `QuotaExceededError`), the app warns that the model will
  be downloaded again and will not work offline.
- Locally selected model files are never copied into any cache.

## Offline behaviour

| Situation                       | Behaviour                                                               |
| ------------------------------- | ----------------------------------------------------------------------- |
| App visited before, now offline | App shell loads from the service worker; header shows _Offline_.        |
| Model fully cached              | Loads and runs normally (verified by an automated test).                |
| Model not cached                | Loading fails with “You are offline and … is not in the browser cache.” |
| Local files                     | Work offline (they are read from disk).                                 |

## Limits

- Storage quotas differ between browsers and depend on free disk space; Safari is the most
  restrictive. Private/incognito windows use in-memory storage with small limits — very
  large files (observed: > 256 MB in headless Chromium incognito) may silently not be cached.
- Browsers may evict cached data under storage pressure unless persistence was granted.

## Network requests made by the app

- The app's own static files from GitHub Pages.
- `https://huggingface.co/<repo>/resolve/<sha>/…` (and Hugging Face's CDN redirects) — only
  when you load a model, its tokenizer, or check cache state for it.

There is no analytics, telemetry or tracking, and prompts never leave the browser.
