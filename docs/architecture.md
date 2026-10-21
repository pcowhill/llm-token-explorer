# Architecture

LLM Token Explorer is a static single-page application (Vue 3 + TypeScript + Vite) that runs
real neural-network inference in the browser with
[Transformers.js](https://github.com/huggingface/transformers.js) 4.3.1 and
[ONNX Runtime Web](https://onnxruntime.ai/) 1.31 (WebAssembly or WebGPU). There is no backend.

```
┌──────────────────────────── Main thread (UI) ───────────────────────────┐
│ Vue components ── provide/inject ──► Workbench store (src/stores)        │
│   model / context / probability / sampling / generation / history ...    │
│                                   │                                      │
│   pure modules: sampling/ (softmax, pipeline, PRNG)                      │
│                 tokenization/ (sequence ops, display, vocabulary query)  │
│                 history/ (branching tree, experiment JSON)               │
│                 models/ (catalog, variants & cache state, Hub API,       │
│                          local-file validation, cache mgmt)              │
│                                   │ InferenceEngine interface            │
│                    WorkerEngineClient (postMessage RPC)                  │
└───────────────────────────────────┼──────────────────────────────────────┘
                                    ▼
┌──────────────────────────── Inference Web Worker ───────────────────────┐
│ TransformersEngine (src/inference/transformersEngine.ts)                 │
│   AutoTokenizer + AutoModelForCausalLM → model({input_ids, attention_mask})│
│   → logits tensor [1, seq, vocab] → last row as Float32Array (fp16 too)  │
│   fetch shim: local files, cache-first Hub access, revision pinning     │
│   ONNX Runtime Web (self-hosted WASM under <base>/ort/) / WebGPU         │
└──────────────────────────────────────────────────────────────────────────┘
```

## Separation of concerns

| Concern                   | Module                                                                                  | Notes                                                                                                                            |
| ------------------------- | --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| UI                        | `src/components/**`                                                                     | No component imports Transformers.js.                                                                                            |
| State & orchestration     | `src/stores/workbench.ts`                                                               | Single source of truth; stale-result protection; debounced inference.                                                            |
| Inference runtime adapter | `src/inference/transformersEngine.ts`                                                   | The **only** module importing `@huggingface/transformers`.                                                                       |
| Worker transport          | `src/workers/inference.worker.ts`, `src/inference/workerClient.ts`, `protocol.ts`       | Requests are serialized (ORT Web cannot run sessions concurrently); logits are transferred, not copied.                          |
| Engine interface          | `src/inference/engine.ts`                                                               | `InferenceEngine` — implemented by the worker client in the app and by a fake engine in component tests.                         |
| Logits → probabilities    | `src/sampling/softmax.ts`, `logitsHealth.ts`                                            | Float64, max-subtraction, NaN/±∞ handling; NaN/constant (e.g. all-zero) logits are rejected as errors.                           |
| fp16 logits               | `src/inference/logitsRow.ts`                                                            | Last row of `[1, seq, V]` → Float32Array, from Float16Array or raw Uint16 half bits.                                             |
| Sampling                  | `src/sampling/pipeline.ts`, `prng.ts`                                                   | Temperature → top-k → top-p → min-p; SFC32 PRNG.                                                                                 |
| Tokens                    | `src/tokenization/*`                                                                    | Exact-ID sequence ops, invisible-character display, byte decoding, vocabulary search/sort/pagination.                            |
| History                   | `src/history/tree.ts`, `experiment.ts`                                                  | Branching tree of exact token sequences; validated JSON import/export.                                                           |
| Models                    | `src/models/catalog.ts`, `variants.ts`, `hubApi.ts`, `localFiles.ts`, `cacheStorage.ts` | Pinned catalog; dtype ↔ files and per-dtype cache state; Hub API file listing; local layout detection; Cache Storage management. |
| Keyboard shortcuts        | `src/composables/useShortcuts.ts`, `App.vue`                                            | Remappable single keys (persisted), global handler inactive in text fields and dialogs; rate-limited repeat.                     |

## Why a raw forward pass instead of `generate()`

`generate()` hides the logits behind its own logits processors and sampler. The adapter
instead calls the model directly:

```ts
const out = await model({ input_ids, attention_mask }) // ONNX session run
out.logits.dims // [1, sequenceLength, vocabularySize]  e.g. [1, 5, 49152]
```

and copies the last row. This is the complete, unprocessed next-token logits vector; every
probability shown in the UI is computed from it by this app's own, tested code. Before a
result is shown, `checkLogits()` rejects NaN / +∞ / constant vectors (a broken
variant/backend combination would otherwise look like a uniform distribution).

## Loading a model resets the workspace

Token IDs mean different text in different vocabularies, so every successful load (catalog,
Hub repo, local; reload included) clears the context, starts a fresh history tree, restarts
the generator from the seed, resets the Example menu and loads the model's
`defaultExampleId` example if it has one, with a “Context cleared for new model” note. The
one exception: right after importing an experiment (or restoring a saved session), loading
that experiment's own model (same repo and revision, any dtype) keeps the imported history.

## Data integrity

- **Token IDs are authoritative.** The context is the current history node's `number[]`.
  Text editing re-tokenizes the whole text explicitly; token-level edits never round-trip
  through text. The UI shows whether `encode(decode(ids)) === ids`.
- **Stale-result protection.** Each forward pass carries a request id and an exact context
  key (`model identity | id,id,id…`). A result is adopted only if it is the newest request
  _and_ the context still matches.
- **No redundant execution.** Sampling changes re-use the cached logits (`prepared` →
  `adjusted` is a pure computed). Recent results are kept in a small LRU keyed by context, so
  undo/redo/branch navigation shows distributions instantly.
- **Vocabulary vs. logits size.** IDs are validated against the model's input/output size
  (`config.vocab_size`), not the tokenizer size. Logit rows without tokenizer entries are
  shown but never sampled.

## Runtime configuration (worker)

- ONNX Runtime WASM binaries are **self-hosted** (copied from `onnxruntime-web` into
  `dist/ort/` by a Vite plugin) instead of Transformers.js' default jsDelivr CDN. This keeps
  the app working offline, avoids a third-party request and pins the tested runtime.
- WASM threads: 1 unless the page is cross-origin isolated (GitHub Pages cannot send
  COOP/COEP headers).
- Hub downloads go through `env.fetch` → `appFetch`, which (1) pins revision-less lookups
  to the revision being loaded, (2) answers from Cache Storage first so cached models load
  offline, and (3) serves user-selected local files from memory.
- Caching uses a custom Cache Storage adapter (`env.customCache`) that records write
  failures so the UI can warn when a model will not be available offline.

## Build and deployment

- Vite `base` defaults to `/llm-token-explorer/` (override with `BASE_PATH`); every asset,
  the worker, the WASM files, the manifest and the service-worker scope resolve under it.
- `vite-plugin-pwa` (Workbox `generateSW`) precaches the app shell; the 27 MB ORT WASM is
  runtime-cached on first use. Model weights are never handled by the service worker.
- GitHub Actions (`.github/workflows/ci.yml`): quality → e2e + real inference → deploy to
  Pages (main only) → live verification against the deployed URL.
