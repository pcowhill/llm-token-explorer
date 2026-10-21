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
4. **Measure fidelity** of the variants you intend to offer (see
   [model-compatibility.md](model-compatibility.md#quantization-fidelity)) and choose a
   default per backend that is faithful and fits memory (WASM ≲ 1 GB of weights).
5. **Add an entry** to `CATALOG` in `src/models/catalog.ts`: name, publisher, repo,
   revision, upstream repo, license, parameter count, context length, tokenizer and logits
   sizes, variants (`files`, `sizeBytes`, `backends`, `defaultFor`, `fidelity`), memory
   estimates, limitations, prompt format and examples (`text`, `chat`, `parts` with real
   special-token names).
6. **Verify** with real inference:
   `VERIFY_ALL_MODELS=1 npx playwright test --project=real-inference catalog` and set the
   `verification` record to what actually passed. Do not mark a model verified because it
   downloads or loads.
7. Run `npm run check` and update `docs/model-compatibility.md`.
