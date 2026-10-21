# Model compatibility and verification

All results below come from runs on 2026-10-21 with Transformers.js 4.3.1 and
ONNX Runtime Web 1.31.0-dev.20260914 in headless Chromium 141 (Playwright 1.56.1),
plus Node.js 22 (`onnxruntime-node` CPU) for reference measurements.

## Verification status (catalog)

"Verified" means a **real forward pass** produced the full-vocabulary logits vector, the
softmax distribution was finite, non-negative and summed to 1, a token was appended and
another forward pass succeeded — inside the app, in the inference Web Worker.

| Model (pinned revision)                                                     | Precision              | Backend                               | Real inference tested                      | Result                                                                                   |
| --------------------------------------------------------------------------- | ---------------------- | ------------------------------------- | ------------------------------------------ | ---------------------------------------------------------------------------------------- |
| SmolLM2-135M-Instruct (`onnx-community/SmolLM2-135M-Instruct-ONNX@b8a5c0f`) | fp16 (default)         | WASM                                  | Yes — CI smoke test                        | Logits `[1, 5, 49152]`; “The capital of France is” → “ Paris” 46.9 % (identical to fp32) |
|                                                                             | fp32                   | WASM                                  | Yes (manual run)                           | Identical distribution to fp16                                                           |
|                                                                             | q8                     | WASM                                  | Yes — CI offline test                      | Works; distribution differs (“ located” 74.7 %)                                          |
|                                                                             | q8                     | WebGPU (SwiftShader software adapter) | Yes — CI smoke test (when adapter present) | Works; ~1.9–3.8 s per pass on software GPU                                               |
|                                                                             | fp16 / q4f16           | WebGPU (hardware)                     | **Not tested** (no GPU in CI)              | Documented as supported; needs `shader-f16`                                              |
| Qwen2.5-0.5B-Instruct (`onnx-community/Qwen2.5-0.5B-Instruct@22942cb`)      | bnb4 (default on WASM) | WASM                                  | Yes (opt-in catalog verification)          | Logits `[1, 5, 151936]`; ~0.8 s per pass                                                 |
|                                                                             | q8, q4                 | WASM                                  | Yes (manual runs)                          | Work; q4 ~1.5 s per pass                                                                 |
|                                                                             | fp16                   | WASM                                  | Yes — **fails**                            | `std::bad_alloc` (1 GB model exceeds WASM memory) → offered for WebGPU only              |
|                                                                             | q4f16, fp16            | WebGPU (hardware)                     | **Not tested**                             | Documented as supported                                                                  |
| GPT-2 124M (`onnx-community/gpt2-ONNX@1975fb9`)                             | fp16 (default)         | WASM                                  | Yes (opt-in catalog verification)          | Logits `[1, 5, 50257]`; matches Node fp32 reference                                      |
|                                                                             | q8                     | WASM                                  | Yes                                        | Works                                                                                    |
|                                                                             | fp32                   | WebGPU (hardware)                     | **Not tested**                             | Documented as supported                                                                  |

Re-run the full catalog verification (downloads ≈1.3 GB):

```bash
node scripts/fetch-test-models.mjs --smollm2
VERIFY_ALL_MODELS=1 npx playwright test --project=real-inference
```

### Additional models investigated (not in the catalog)

| Model                                                     | Finding                                                                                                                                                                                                                                |
| --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `onnx-community/Qwen3-0.6B-ONNX` (q4f16)                  | Real logits in Node.js (`[1, 5, 151936]`). Not added: larger download, thinking-mode template.                                                                                                                                         |
| `onnx-community/gemma-3-270m-it-ONNX` (q4, external data) | Real logits in Node.js (`[1, 6, 262144]`), but the tokenizer has 262,145 entries (one more than logits). Excluded from the catalog because the upstream model is license-gated (Gemma terms); it can still be loaded as a local model. |
| `hf-internal-testing/tiny-random-GPT2LMHeadModel`         | Used as the e2e fixture (random weights, 1,024 logits, 0.8 MB).                                                                                                                                                                        |

## Quantization fidelity

Quantization makes models smaller but changes the distribution — sometimes a lot. To pick
honest defaults, each variant's next-token distribution was compared with the
full-precision reference over 5 prompts (Node.js CPU, `tests`-independent script):
mean KL(reference ‖ variant) in nats and top-1 agreement.

| Model        | Reference | fp16          | q8        | q4        | bnb4      |
| ------------ | --------- | ------------- | --------- | --------- | --------- |
| SmolLM2-135M | fp32      | < 0.0001, 5/5 | 1.15, 2/5 | 1.57, 1/5 | 0.48, 3/5 |
| GPT-2        | fp32      | < 0.0001, 5/5 | 0.22, 3/5 | —         | —         |
| Qwen2.5-0.5B | fp16      | (reference)   | 1.46, 3/5 | 0.57, 3/5 | 0.22, 5/5 |

Consequences:

- SmolLM2 and GPT-2 default to **fp16** (exact, and for GPT-2 even smaller than q8).
- Qwen2.5 defaults to **bnb4** on WASM: fp16 does not fit in WebAssembly memory and bnb4 is
  the closest browser-feasible variant measured.
- The model card shows each variant's measured fidelity; q8 is labelled approximate.
- Quantized kernels also differ between runtimes: SmolLM2 q8 gives “ located” 47 % in Node.js
  but 75 % in browser WASM, while fp16/fp32 agree across runtimes.

## Backend notes

- **WASM**: works in every modern browser with WebAssembly SIMD. Single-threaded on GitHub
  Pages (no cross-origin isolation). fp16 models run on WASM (as fast as q8 here).
- **WebGPU**: much faster on real GPUs. f16 variants (fp16, q4f16) need the `shader-f16`
  feature; the app detects it and falls back to fp32/q8 variants. Software adapters
  (SwiftShader) are never auto-selected.
- Different backends/precisions give slightly different logits; sampling with the same seed
  can then pick different tokens.

## How to verify another model

1. Load it via **Model → Hugging Face repo** (or Local files) with `?debug=1` appended to the
   URL.
2. Check in **Diagnostics** that _Logits output tensor_ is `[1, n, V]` with `V` equal to
   `config.vocab_size`, and in the distribution stats that Σ original p = 1.000000000.
3. Compare the top tokens for a simple prompt against the reference implementation
   (`transformers` in Python, or Transformers.js in Node with `dtype: 'fp32'`).
4. For an automated check, add the model to the catalog and run the catalog verification
   spec (see above).
