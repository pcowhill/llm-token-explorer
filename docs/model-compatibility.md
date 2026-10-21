# Model compatibility and verification

All results below come from runs on 2026-10-21 with Transformers.js 4.3.1 and
ONNX Runtime Web 1.31.0-dev.20260914 in headless Chromium 141 (Playwright 1.56.1),
plus Node.js 22 (`onnxruntime-node` 1.30 CPU) for reference measurements.

## Verification status (catalog)

"Verified" means a **real forward pass** produced the full-vocabulary logits vector, the
logits were finite and not degenerate (they vary, and the top token is far above uniform),
the softmax distribution summed to 1, a token was appended and another forward pass
succeeded — inside the app, in the inference Web Worker, through the model card.

The only GPU available to CI is Chromium's **SwiftShader** software WebGPU adapter. It has
no `shader-f16`, so **f16 variants (fp16, q4f16) on WebGPU are not verified in CI** — the
app labels them “not verified in CI”. See the [manual GPU checklist](#manual-gpu-checklist).

<!-- VERIFICATION-MATRIX -->

Re-run the full matrix (downloads ≈12 GB once into `tests/fixtures/models/hf`):

```bash
node scripts/fetch-test-models.mjs --all
VERIFY_ALL_MODELS=1 npx playwright test --project=real-inference catalog
VERIFY_ALL_MODELS=1 VERIFY_MODELS=gpt2 npx playwright test --project=real-inference catalog   # one model
```

CI runs on every push: SmolLM2 fp16 (WASM), SmolLM2 q8 (WebGPU/SwiftShader), the offline
reload, and **Qwen3.5 fp32 on WebGPU/SwiftShader** (its default; ≈5 min per run).

## Quantization fidelity

Each variant's next-token distribution was compared with the fp32 reference over the same 5
prompts (`REFERENCE_PROMPTS` in [`scripts/measure-fidelity.mjs`](../scripts/measure-fidelity.mjs):
“The capital of France is”, a boiling-point Q&A, a chat-template request for an octopus fact,
a Python `fibonacci` stub and “Once upon a time, in a small village,”) in Node.js CPU:
mean KL(fp32 ‖ variant) in nats and top-1 agreement.

| Model        | fp16          | q8          | q4        | q4f16     | bnb4      |
| ------------ | ------------- | ----------- | --------- | --------- | --------- |
| Qwen3.5-0.8B | < 0.0001, 5/5 | 0.0006, 5/5 | 0.27, 3/5 | 0.27, 3/5 | —         |
| SmolLM2-135M | < 0.0001, 5/5 | 0.64, 3/5   | —         | 0.60, 1/5 | —         |
| Qwen2.5-0.5B | 0.0007, 5/5   | 0.86, 3/5   | 0.34, 5/5 | 0.34, 5/5 | 0.22, 5/5 |
| GPT-2        | < 0.0001, 5/5 | 0.18, 4/5   | —         | —         | —         |

```bash
node scripts/fetch-test-models.mjs --model qwen3.5-0.8b
node scripts/measure-fidelity.mjs qwen3.5-0.8b            # fp32 reference vs every variant
```

Node's CPU kernels are not the browser's (SmolLM2 q8 gives “ located” 47 % in Node.js but
75 % in browser WASM), so this measures the weights; the browser runs above check that each
variant actually works where it is offered.

### Defaults: the most faithful variant that runs on each backend

| Model        | WASM                              | WebGPU                                                   |
| ------------ | --------------------------------- | -------------------------------------------------------- |
| Qwen3.5-0.8B | — (nothing fits; see below)       | **fp32** (2.9 GB; q8 at 0.9 GB is practically identical) |
| SmolLM2-135M | **fp16** (exact, 257 MB)          | **fp32** (fp16 overflows on WebGPU, see below)           |
| Qwen2.5-0.5B | **bnb4** (fp16/fp32 do not fit)   | **bnb4** (fp16 overflows, fp32 does not fit)             |
| GPT-2        | **fp16** (exact, smaller than q8) | **fp32**                                                 |

f16 variants are never auto-selected on a GPU without `shader-f16`.

## fp16 overflow on WebGPU

**Symptom (reported):** SmolLM2 135M Instruct q4f16 on WebGPU returned all-zero logits — a
perfectly uniform distribution over 49,152 tokens.

**Not the logits conversion.** SmolLM2's ONNX graphs end with a `Cast` to float32 for every
variant (`logits` is `FLOAT` in fp16 and q4f16 too), so the app's float16→float32 path is
never used for it. That conversion is now our own, unit-tested code anyway
(`src/inference/logitsRow.ts`, handling both `Float16Array` and raw `Uint16Array` bits); it
matters for Qwen3.5, whose fp16/q4f16 graphs do return float16 logits.

**Root cause: the export, executed by WebGPU in fp16.**

1. The fp16 and q4f16 exports of SmolLM2 (and of Qwen2.5) compute RMSNorm as separate
   **fp16** ops: `Pow(x, 2) → ReduceMean → Add(ε) → Sqrt → Div(1, ·) → Mul`.
2. On CPU/WASM, ONNX Runtime's `SimplifiedLayerNormFusion` turns that pattern into one
   `SimplifiedLayerNormalization` kernel that accumulates in fp32. This fusion is registered
   only for the CPU and CUDA execution providers
   ([`graph_transformer_utils.cc`](https://github.com/microsoft/onnxruntime/blob/8d85527a0/onnxruntime/core/optimizer/graph_transformer_utils.cc),
   `SimplifiedLayerNormFusion(cpu_cuda_eps)`), so on **WebGPU the ops run as written, in fp16**.
3. SmolLM2 has “massive activations”: measured with the fp32 model, the input of the final
   norm reaches |x| ≈ 3,000–4,400 and intermediate norms see up to ≈26,000; 51–53 of its 61
   norm inputs exceed |x| = 255.9, the point where x² overflows fp16 (max 65,504).
4. `x² = +inf` → `mean = inf` → `1 / sqrt(inf) = 0` → the normalized hidden state is exactly
   0 → with tied embeddings every logit is exactly 0.

Emulating the fp16 overflow inside the fp32 graph (every `Pow` output above 65,504 becomes
+inf) reproduces it exactly: **all logits 0.0 on every prompt** for SmolLM2. For Qwen2.5
(38 of 49 norm inputs above 255.9, |x| up to ≈1,600) it gives a corrupted distribution
(KL 4.3 and 0.5 nats, and all-zero logits for a lone `<|endoftext|>`). Reproduce with
[`scripts/analysis/fp16_overflow.py`](../scripts/analysis/fp16_overflow.py).

Unaffected exports: **Qwen3.5** uses fused `SimplifiedLayerNormalization` /
`SkipSimplifiedLayerNormalization` nodes with `stash_type = 1` (fp32 accumulation, and the
WebGPU kernel accumulates in f32), and its activations stay below |x| ≈ 15. **GPT-2**'s
decomposed LayerNorm is fused for WebGPU (`LayerNormFusion` includes the WebGPU EP, whose
LayerNorm shader accumulates in f32); unfused it would be distorted (emulated KL 0.44), so
GPT-2 fp16 on WebGPU stays on the manual checklist.

**Consequences in the app**

- SmolLM2: fp16 and q4f16 are **not offered on WebGPU** (WASM keeps both: fp16 is exact
  there, q4f16 works). WebGPU defaults to fp32.
- Qwen2.5: fp16 is removed (it never fit WASM memory, and overflows on WebGPU); q4f16 is
  WASM-only; WebGPU defaults to bnb4.
- Each card shows a “Not offered on WebGPU: … — why?” explanation.
- A **degenerate-logits guard** (`src/sampling/logitsHealth.ts`) rejects NaN, +Infinity,
  all -Infinity and constant (incl. all-zero) logits with a clear error instead of showing a
  uniform distribution, for any model, including Hub and local ones.

## Qwen3.5-0.8B (★ Patrick's Choice)

`onnx-community/Qwen3.5-0.8B-Text-ONNX@25cdf33` (Apache-2.0, base model `Qwen/Qwen3.5-0.8B`):
`model_type` `qwen3_5_text`, 24 layers (18 Gated DeltaNet linear-attention + 6 full-attention),
vocabulary 248,320 logits / 248,077 tokenizer entries, EOS `<|endoftext|>` 248044 (the app
also stops at `<|im_end|>` 248046), 262,144-token context. Files at that revision:

| Variant | Files                                                  | Size                               |
| ------- | ------------------------------------------------------ | ---------------------------------- |
| fp32    | `model.onnx` + `model.onnx_data` + `model.onnx_data_1` | 3,077,060,202 B (3.1 GB / 2.9 GiB) |
| fp16    | `model_fp16.onnx` + `model_fp16.onnx_data`             | 1,538,945,869 B (1.5 GB)           |
| q8      | `model_quantized.onnx` + `model_quantized.onnx_data`   | 938,947,493 B (939 MB)             |
| q4      | `model_q4.onnx` + `model_q4.onnx_data`                 | 551,247,327 B (551 MB)             |
| q4f16   | `model_q4f16.onnx` + `model_q4f16.onnx_data`           | 469,929,410 B (470 MB)             |
| —       | `tokenizer.json`                                       | 19,226,111 B (19.2 MB)             |

**WASM: no variant runs.** fp32 and fp16 fail with `std::bad_alloc` (the 32-bit WebAssembly
heap; fp16 is additionally expanded to fp32 by the CPU execution provider). q8, q4 and q4f16
fail at session creation: their embedding uses `com.microsoft.GatherBlockQuantized`, which
ONNX Runtime Web's WASM build does not include (Node's `onnxruntime-node` has it, the
WebGPU EP has it). The card explains this and offers “Switch to WebGPU”; the tokenizer alone
loads on WASM.

**WebGPU:** fp32, q8 and q4 verified on SwiftShader. fp32 peaks at ≈11 GB host memory on
the software adapter (≈3 GB of GPU memory on real hardware).

**Starter examples** (measured with fp32, Node.js; the browser fp32 run matches to 0.1 %):

| Example                                               | Next token                                           |
| ----------------------------------------------------- | ---------------------------------------------------- |
| Chat, thinking off: pick a number (default)           | “6” 30.1 %, “7” 21.8 %, “5” 18.0 %, “4” 12.6 %       |
| Raw completion: “My favorite programming language is” | “ Python” 15.1 %, “ Java” 14.3 %, “ C” 12.3 %        |
| Chat, thinking on (`enable_thinking=true`)            | “Thinking” 95.4 % (after the open `<think>`)         |
| One-word answer: name a planet                        | “Earth” 30.8 %, “J” 27.8 %, “M” 14.3 %, “Mer” 11.5 % |
| Special tokens, no think block                        | `<think>` ≈ 100 % — the model asks to think          |

The template's thinking-off form is an empty `<think>\n\n</think>\n\n` block, so the next
token after it is the answer itself.

### Additional models investigated (not in the catalog)

| Model                                                     | Finding                                                                                                                                                                                                                                |
| --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `onnx-community/Qwen3-0.6B-ONNX` (q4f16)                  | Real logits in Node.js (`[1, 5, 151936]`). Not added: larger download, thinking-mode template.                                                                                                                                         |
| `onnx-community/gemma-3-270m-it-ONNX` (q4, external data) | Real logits in Node.js (`[1, 6, 262144]`), but the tokenizer has 262,145 entries (one more than logits). Excluded from the catalog because the upstream model is license-gated (Gemma terms); it can still be loaded as a local model. |
| `hf-internal-testing/tiny-random-GPT2LMHeadModel`         | Used as the e2e fixture (random weights, 1,024 logits, 0.8 MB).                                                                                                                                                                        |

## Backend notes

- **WASM**: works in every modern browser with WebAssembly SIMD. Single-threaded on GitHub
  Pages (no cross-origin isolation). 32-bit heap: in practice ≲ 1 GB of weights, and fp16
  weights count double (expanded to fp32).
- **WebGPU**: much faster on real GPUs. f16 variants need `shader-f16`; the app detects it
  and falls back to non-f16 variants. Software adapters (SwiftShader) are never
  auto-selected. Exports that compute normalization in fp16 can overflow (above).
- Different backends/precisions give slightly different logits; sampling with the same seed
  can then pick different tokens.

## Manual GPU checklist

Run on a machine with a hardware GPU that has `shader-f16` (Model panel → “shader-f16: yes”),
in Chrome/Edge, with **Inference backend = WebGPU**. For each row: load the variant, check
that the distribution is not an error, and compare the top tokens with the reference.

1. **Qwen3.5 fp32** (default): chat example → “6” ≈ 30 %, “7” ≈ 22 %, “5” ≈ 18 %.
2. **Qwen3.5 fp16**: same example → within ≈1 % of fp32 (Node.js reference: “6” 30.1 %,
   “7” 21.7 %, “5” 18.0 %). If it matches, fp16 could become the WebGPU default (half the
   download).
3. **Qwen3.5 q4f16**: same example → Node.js reference “7” 26.6 %, “5” 26.2 %, “6” 21.1 %
   (q4 on WebGPU/SwiftShader should agree). Thinking example → “Thinking” ≈ 94 %.
4. **GPT-2 fp16**: “The capital of France is” → same top tokens and probabilities as GPT-2
   fp16 on WASM.
5. **SmolLM2 fp32 / q8** and **Qwen2.5 bnb4 / q4 / q8**: completion example → no error,
   plausible tokens (“ Paris” for France).
6. Optional confirmation of the root cause: load
   `onnx-community/SmolLM2-135M-Instruct-ONNX` via **Hugging Face repo** with dtype
   `q4f16` → the app must now show “returned invalid next-token logits: Every finite logit
   has the same value (0)” instead of a uniform distribution.
7. Speed sanity check: Qwen3.5 fp32 forward pass on the chat example should take well under
   a second on a discrete GPU.

## How to verify another model

1. Load it via **Model → Hugging Face repo** (or Local files) with `?debug=1` appended to the
   URL.
2. Check in **Diagnostics** that _Logits output tensor_ is `[1, n, V]` with `V` equal to
   `config.vocab_size`, and in the distribution stats that Σ original p = 1.000000000.
3. Compare the top tokens for a simple prompt against the reference implementation
   (`transformers` in Python, or Transformers.js in Node with `dtype: 'fp32'`).
4. For an automated check, add the model to the catalog and run the catalog verification
   spec (see above).
