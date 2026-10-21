# CLAUDE.md — LLM Token Explorer

A fully client-side web app for exploring tokenization, next-token logits, probabilities and
sampling with real small LLMs. It runs inference in the browser with Transformers.js and ONNX
Runtime Web, on WASM or WebGPU.

- Live site: https://pcowhill.github.io/llm-token-explorer/
- Repo: `pcowhill/llm-token-explorer`. `main` deploys to GitHub Pages automatically.
- Owner: Patrick (GitHub `pcowhill`).

## Working agreements (the owner's preferences)

- **Interview when a request is ambiguous.** Ask questions in batches, and give a recommended
  answer for each so the owner can just accept them. Don't guess on product decisions.
- **Show real inference, not claims.**
  - Never say a model or quantization "works" without a real-inference test: load the real
    weights, get the full-vocabulary logits, and check that they are finite, non-degenerate and
    plausible.
  - Catalog entries are pinned to commit SHAs and record the fidelity measured against an
    fp32/fp16 reference.
  - If something can't be verified in this environment (for example f16 on WebGPU in CI), say so
    plainly in the code, the docs and the report.
- **Every bug fix gets a regression test, shown to fail before the fix.**
- **Keep docs and screenshots in sync.** When UI or behavior changes, update the relevant
  `docs/*.md` and `README.md`, and regenerate `docs/screenshots/` with
  `node scripts/capture-screenshots.mjs` (needs `vite preview` on :4173).
- **Default to full autonomy.** Unless told otherwise:
  1. Work on a branch and open a PR.
  2. Drive CI to green and self-review the diff.
  3. Merge to `main` and wait for deploy plus `verify-live`.
  4. Check the live site, then report.

  Ask only for decisions outside agreed preferences, such as removing a whole model.

- **Commits.** Use clear, descriptive messages and the attribution trailer the harness supplies.
  Never put model identifiers in commits, PRs or code.

## Commands

```bash
npm ci                       # install (.npmrc skips onnxruntime-node's CUDA download)
npm run dev                  # Vite dev server
npm run check                # lint + typecheck + unit/component tests + build (run before pushing)
npm run lint                 # ESLint, --max-warnings=0
npm run format:check         # Prettier (CI enforces it; `npm run format` to fix)
npm run typecheck            # vue-tsc --build --force (strict)
npm run test:unit            # Vitest "unit" project (node)
npm run test:component       # Vitest "component" project (jsdom, FakeEngine)

node scripts/fetch-test-models.mjs --smollm2   # tiny GPT-2 fixture + SmolLM2 fp16/q8 into tests/fixtures/models (git-ignored)
npx playwright test --project=e2e              # workflow, a11y (axe), PWA, against the production build
npx playwright test --project=real-inference   # REAL SmolLM2 inference (WASM; WebGPU when an adapter exists)
VERIFY_ALL_MODELS=1 npx playwright test --project=real-inference catalog   # every catalog model (big downloads)
LIVE_ONLY=1 npx playwright test --project=live # smoke test against the deployed site
```

**Environment notes:**

- **Proxy.** Behind the agent proxy, run Node network scripts with `NODE_USE_ENV_PROXY=1`.
- **Chromium.** The cloud container has a preinstalled Chromium. Playwright is pinned to
  **1.56.1** to match it (`chromium-1194`), and `overrides.playwright-core` keeps
  `@axe-core/playwright` on that version. Never run `playwright install` here. Don't bump
  Playwright without checking the browser build.
- **Stale preview server.** Locally, Playwright's `reuseExistingServer` can attach to a stale
  `vite preview` on :4173 or a stale model server on :4174. Kill them after changing code, or the
  e2e tests run old code.
- **TypeScript version.** It is pinned at 6.0.x because typescript-eslint requires `<6.1`.

## Architecture (see `docs/architecture.md` for detail)

- **Stack.**
  - Vue 3 (`<script setup>`), TypeScript strict, Vite 8, vite-plugin-pwa (generateSW).
  - Base path is `/llm-token-explorer/`, overridable with the `BASE_PATH` env var.
- **Inference runs in a Web Worker.**
  - `src/workers/inference.worker.ts` speaks an RPC protocol (`src/inference/protocol.ts`) through
    `src/inference/workerClient.ts`.
  - The UI only knows the `InferenceEngine` interface (`src/inference/engine.ts`).
- **Only `src/inference/transformersEngine.ts` imports `@huggingface/transformers`.**
  - It runs the raw forward pass `model({ input_ids, attention_mask })`, which returns logits
    `[1, seq, V]`. The app uses the last row.
  - The last logits row is extracted by `lastLogitsRow()` (`src/inference/logitsRow.ts`), which
    converts float16 logits (Float16Array or raw Uint16 bits) itself. `checkLogits()`
    (`src/sampling/logitsHealth.ts`) rejects NaN / constant (all-zero) logits before display.
  - An `appFetch` shim handles three things:
    - Local files under the pseudo-path `/__local_models__/local/m<N>/`. This must be a path, not
      a URL, or file probing skips it.
    - Hugging Face requests, answered cache-first.
    - Lookups for the "main" revision, pinned to the loaded revision so offline loads work.
  - An `env.customCache` adapter records cache write failures.
  - ONNX Runtime Web files are self-hosted under `<base>/ort/`.
  - The KV cache is **off by default**: with q8 dynamic quantization, incremental passes differ
    from a full pass.
- **Store: `src/stores/workbench.ts`.** It owns:
  - `loadModel`, `loadCatalog`, `cancelLoad`
  - `setTokens`, `setText`, `runInference`
  - `sampleOne`, `greedyOne`, `startAuto`
  - `undo`, `redo`, `resetSeed`, `importExperiment`
  - `chooseDtype` and `resolveBackend` (a SwiftShader WebGPU adapter is never auto-picked).

  Stale results are dropped using `requestId` plus `contextKey`. A small LRU caches results.

- **Sampling (`src/sampling/`).**
  - Pipeline order: temperature → top-k → top-p → min-p → renormalize.
  - Seeded SFC32 PRNG.
  - Logit rows with no tokenizer entry are excluded from sampling.
- **History (`src/history/`).** History is a branching tree. Experiments use a JSON format that is
  validated on import (`docs/experiment-format.md`).
- **Model catalog: `src/models/catalog.ts`.**
  - It holds the "Recommended" models: `qwen3.5-0.8b` (featured "★ Patrick's Choice", listed
    first, WebGPU only), `smollm2-135m-instruct`, `qwen2.5-0.5b-instruct` and `gpt2`.
  - Each is pinned to a repo plus commit SHA and has per-dtype `variants` with `sizeBytes`,
    `backends`, `fidelity` and `note`.
  - Each also has examples, verification status and limitations.
  - Current defaults (most faithful variant that runs on each backend): Qwen3.5 fp32 on
    WebGPU; SmolLM2 and GPT-2 fp16 on WASM, fp32 on WebGPU; Qwen2.5 bnb4 on both.
  - `excludedVariants` records variants deliberately not offered on a backend (shown on the
    card); `backendNotes` explains a backend a model cannot use at all; `defaultExampleId` is
    loaded after every model load.
  - `scripts/fetch-test-models.mjs` (`CATALOG_FILES`) lists every pinned file per dtype;
    `scripts/measure-fidelity.mjs <key>` measures KL vs fp32 in Node.
  - `docs/adding-a-model.md` describes the process for adding a model.
- **Other model sources.** The Models panel (`src/components/model/ModelPanel.vue`) also offers a
  "Hugging Face repo" tab (`HubRepoLoader.vue`: dtypes and sizes from the Hub API via
  `src/models/hubApi.ts`), local files/folders (`src/models/localFiles.ts`) and a Cache tab
  (`src/models/cacheStorage.ts`). Per-dtype cache state comes from `src/models/variants.ts`.
- **Key components.**
  - `ModelCard.vue`: dtype select with the `userChoseDtype` guard and per-dtype cache marks,
    Load button ("Load model (cached)" / "Download & load (X MB)"), "Cached: …" badge.
  - `ContextWorkspace.vue`: the context editor, the Example action menu (returns to its
    placeholder after each choice) with a "Loaded: …" note, and Clear.
  - `App.vue`: global keyboard shortcuts (S/G remappable via `src/composables/useShortcuts.ts`,
    `?` cheat-sheet, Ctrl+Z/Y, Ctrl+Enter, Alt+S) and session persistence.
  - Loading any model resets the workspace (`resetWorkspace` in the store), except that an
    imported experiment is kept when its own model is loaded next.
  - `SettingsDialog.vue`: settings; persisted preferences live in `src/composables/usePreferences.ts`.
- **Debug hook.** `?debug=1` exposes the store as `window.__lte`, read-only, for e2e tests.

## Tests

- **Unit tests** (`tests/unit`): pure logic for softmax, the sampling pipeline, the PRNG,
  tokenization display, sequences, vocab, history and experiment validation.
- **Component tests** (`tests/component`): mount the real components against `FakeEngine`
  (`fakeEngine.ts`) via `makeWorkbench` and `mountWith` (`helpers.ts`).
- **E2E tests** (`tests/e2e`):
  - `workflow`, `a11y` and `pwa` use the tiny random GPT-2 fixture through the local-files path.
    The ONNX model really runs, but its outputs are meaningless.
  - `real-inference.spec.ts` is mandatory and runs in CI. It loads real SmolLM2-135M through the
    catalog and checks the exact token IDs, `[1, seq, 49152]` logits, probabilities that sum to 1,
    and fp16 " Paris" ≈ 46.9%. It also loads Qwen3.5 fp32 on WebGPU (SwiftShader, ≈5 min,
    ≈11 GB RAM) and checks the starter examples ("6" ≈ 30 %, "Thinking" ≈ 95 %).
  - `catalog-verification.spec.ts` is opt-in and verifies every catalog model × dtype × backend.
  - `live.spec.ts` targets the deployed site.
- **How real-inference tests get model files.**
  - Tests intercept `https://huggingface.co/**` and 302-redirect cached pinned files to
    `scripts/serve-test-models.mjs` on :4174, because `route.fulfill` can't serve 100 MB+ bodies.
  - Files live in `tests/fixtures/models/hf/<owner>/<repo>/<rev>/<path>`.
  - To use another model or dtype in CI, extend `scripts/fetch-test-models.mjs`. The CI cache key
    hashes that script.
- **Test limits.**
  - CI's WebGPU adapter is SwiftShader, which **lacks `shader-f16`**. So fp16, q4f16 and other f16
    variants can't run on WebGPU in CI. Cover them with unit tests plus WASM runs, and document
    the gap.
  - Incognito-like test contexts don't persist cache entries above ~256 MB.

## CI/CD (`.github/workflows/ci.yml`)

- **Pipeline:**
  1. `quality`: lint, format, typecheck, unit and component tests, build.
  2. `e2e`: e2e plus real inference.
  3. `deploy`: Pages, `main` only.
  4. `verify-live`: the live smoke test.
- **Pinning.** Actions are pinned by SHA. Default permissions are `contents: read`.
- **PRs.** Merging a PR to `main` deploys it. After a merge, watch `deploy` and `verify-live` to
  confirm the live site works.

## Gotchas learned the hard way

- **Prettier breaks inline handlers.** It rewrites multi-statement inline Vue event handlers into
  invalid code. Use named functions in `<script setup>`.
- **Late capabilities.** Capability detection finishes after mount, so don't let it override a
  choice the user already made (see `userChoseDtype` in `ModelCard.vue`). This caused a flaky CI
  failure.
- **Large tables.** The vocabulary table takes a `sortable` prop (`CandidateTable.vue`). Explorer views pass parallel
  arrays already in order, and the table must not re-sort them.
- **Quantization fidelity varies a lot.**
  - q8 noticeably distorts SmolLM2's distribution.
  - Measure KL divergence against fp32/fp16 over several prompts before choosing a default
    (`docs/model-compatibility.md`, `docs/research.md`).
  - Pick the most faithful variant that fits each backend.
- **Model size limits.** Very large single-file weights (≳2 GB, external-data ONNX) may not fit
  WASM memory, so check per backend.
- **Degenerate logits.** All-zero or uniform logits mean a broken variant or a bad conversion.
  They are not a valid distribution, so never present them as one (`checkLogits` enforces it).
- **fp16 overflow on WebGPU.** ORT fuses decomposed fp16 RMSNorm into an fp32 kernel only on
  CPU/CUDA. On WebGPU, exports with fp16 `Pow/ReduceMean` norms (SmolLM2, Qwen2.5) overflow
  and give all-zero logits. Check new fp16 exports with `scripts/analysis/fp16_overflow.py`.
- **Offline loads.** Offline loading depends on revision pinning plus cache-first fetch. Loading
  the same model again with no network must make zero `huggingface.co` requests.
