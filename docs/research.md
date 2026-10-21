# Technical research notes (October 2026)

Findings that shaped the design. Versions: `@huggingface/transformers` 4.3.1,
`onnxruntime-web` 1.31.0-dev.20260914, Chromium 141 (Playwright 1.56.1).

1. **Browser-runnable causal LMs.** Transformers.js `AutoModelForCausalLM` runs ONNX exports
   of Llama, Qwen2/3, Gemma 3, GPT-2, Phi, SmolLM, etc. (`onnx-community/*`). Practical
   browser limits: WASM ≈ up to ~800 MB of weights; larger needs WebGPU.
2. **Full-vocabulary logits.** Calling the model directly
   (`model({ input_ids, attention_mask })`) returns `logits` with shape
   `[batch, seq, vocab]`. Proven for
   SmolLM2 (49,152), Qwen2.5 (151,936), Qwen3 (151,936), GPT-2 (50,257), Gemma 3 (262,144).
   `generate()` is not used because it hides the logits.
3. **WebGPU.** Works through ONNX Runtime's WebGPU EP. f16 variants require the
   `shader-f16` feature (the SwiftShader adapter in headless Chromium lacks it: “Program
   Expand requires f16”). q8/fp32 run on any adapter.
4. **WASM/CPU.** All tested variants except Qwen fp16 (1 GB → `std::bad_alloc`) run.
   fp16 weights run on WASM at q8 speed for SmolLM2.
5. **Quantized variants.** Names → files: fp32 `model.onnx`, fp16 `_fp16`, q8
   `_quantized`, int8, uint8, q4, q4f16, bnb4. q8 (dynamic quantization) distorts small
   models' distributions noticeably; see the fidelity table in
   [model-compatibility.md](model-compatibility.md).
6. **Local files.** Transformers.js has no File-object API, but `env.fetch`,
   `env.localModelPath`, `allowLocalModels` and `local_files_only` let a custom fetch serve
   files. Pitfalls found: `localModelPath` must be a path, not a URL (file probing skips
   URLs), and file metadata is memoized per model id.
7. **Multi-file / external data.** External weights are named `<file>.onnx_data`,
   `<file>.onnx_data_1`, … and enabled with `use_external_data_format: <chunks>`.
8. **Hugging Face loading.** `from_pretrained(repo, { revision })` downloads
   `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json` and
   the ONNX file(s). Some internal existence checks ignore `revision` and query `main`.
9. **Tokenizer metadata.** `get_vocab()`, `all_special_ids`, `bos/eos/pad/unk_token_id`,
   `chat_template` are public; the `special` flag of added tokens and the decoder type are
   read from `tokenizer._tokenizerJSON` (isolated in the adapter, covered by tests).
10. **Chat templates.** `apply_chat_template(messages, { tokenize: false, add_generation_prompt })`
    renders the model's real Jinja template; the app then encodes
    without adding special tokens again.
11. **Web Workers.** Everything runs in a module worker; ORT Web cannot run two sessions
    concurrently, so worker requests are serialized.
12. **Caching.** Cache Storage (`transformers-cache`), customizable via `env.customCache`.
    Incognito contexts may not store entries > ~256 MB.
13. **Browser security.** WASM threads need `SharedArrayBuffer` → cross-origin isolation
    (COOP/COEP headers). WebGPU needs a secure context. CORS: Hugging Face serves
    `Access-Control-Allow-Origin`.
14. **GitHub Pages.** Static hosting under `/<repo>/`, HTTPS, correct `application/wasm`
    MIME type, no custom headers (so no cross-origin isolation → single-threaded WASM), no
    server-side routing (single-page app, no client routes needed). 100 MB per-file limit
    (the 27 MB ORT WASM is fine).

## KV cache and exactness

Reusing `present.*` outputs as `past_key_values.*` reproduces full-pass logits to 2e-5 with
fp32 weights. With **q8 (dynamic activation quantization)** the incremental pass differs by
up to ~7 logits, because activation scales are computed over the tokens processed together.
The KV cache is therefore optional and off by default.

## Backend differences

The same q8 model produced top-1 probability 0.747 on browser WASM, 0.474 on Node CPU and a
different value again on WebGPU. fp16/fp32 agree across runtimes.
