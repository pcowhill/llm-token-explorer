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
- **Cache state per quantization.** A dtype counts as cached only when its graph file, every
  external-data chunk (`*.onnx_data`, `*.onnx_data_1`, …) and the tokenizer/config files are
  cached (`src/models/variants.ts`). The quantization dropdowns mark each option
  “✓ cached” or “↓ <size still to download>” (“partly cached” when some files are there), the
  Load button reads **Load model (cached)** or **Download & load (X MB)**, and the card badge
  names the cached dtypes (“Cached: fp16, q8”).
- **Model → Cache** lists cached models with their quantizations and sizes, removes one or all, shows the storage
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
  when you load a model or its tokenizer.
- `https://huggingface.co/api/models/<repo>/revision/<rev>?blobs=true` — only while you type
  a repository in **Model → Hugging Face repo**, to list its files and sizes. Offline, the
  tab falls back to the quantizations already in the cache.

There is no analytics, telemetry or tracking, and prompts never leave the browser.
