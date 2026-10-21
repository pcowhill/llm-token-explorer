# Adding a recommended model

1. **Find a compatible ONNX export.** It must be a decoder-only causal LM in a repository
   with `config.json`, `tokenizer.json`, `tokenizer_config.json` and `onnx/model*.onnx`
   (typically under `onnx-community/`). Check the license.
2. **Pin a revision.** Get the commit SHA:
   `curl -s https://huggingface.co/api/models/<repo> | jq -r .sha`.
3. **Collect metadata** from the Hub API at that revision:
   `curl -s https://huggingface.co/api/models/<repo>/tree/<sha>/onnx` (file sizes),
   `config.json` (`vocab_size`, `max_position_embeddings`, `model_type`),
   `tokenizer.json` (number of entries), `generation_config.json` (stop tokens).
4. **Download the files and measure fidelity.** Add the model's pinned files (per dtype,
   including every `.onnx_data` chunk) to `CATALOG_FILES` in
   `scripts/fetch-test-models.mjs`, then run
   `node scripts/fetch-test-models.mjs --model <key>` and
   `node scripts/measure-fidelity.mjs <key>` (KL divergence vs fp32 over the reference
   prompts; it also flags degenerate output). Check fp16 exports for the
   [fp16 overflow](model-compatibility.md#fp16-overflow-on-webgpu) with
   `scripts/analysis/fp16_overflow.py` before offering f16 variants on WebGPU.
   Choose per backend the **most faithful variant that actually runs** there (WASM ≲ 1 GB
   of weights, fp16 counts double; try each one in the browser).
5. **Add an entry** to `CATALOG` in `src/models/catalog.ts`: name, publisher, repo,
   revision, upstream repo, license, parameter count, context length, tokenizer and logits
   sizes, variants (`files` incl. data chunks, exact `sizeBytes`, `backends`, `defaultFor`
   (a list of backends), `fidelity`, optional `memoryEstimateBytes`), memory estimates,
   `excludedVariants` (variants deliberately not offered on a backend, with the reason shown
   on the card), `backendNotes` (why the model cannot run on a backend at all), limitations,
   prompt format, examples (`text`, `chat` with optional `templateOptions` such as
   `{ enable_thinking: true }`, `parts` with real special-token names) and
   `defaultExampleId` (loaded after the model loads). Tune examples with real inference so
   their next-token distributions are interesting (not one token at 99 %). `featured` sets
   a highlight badge and lists the model first; keep at most one featured model.
6. **Verify** with real inference — every variant on every backend it is offered for:
   `VERIFY_ALL_MODELS=1 VERIFY_MODELS=<key> npx playwright test --project=real-inference catalog`
   and set the `verification` record to what actually passed. Do not mark a model verified
   because it downloads or loads; f16 variants on WebGPU cannot be verified in CI.
7. Run `npm run check` and update `docs/model-compatibility.md`.
