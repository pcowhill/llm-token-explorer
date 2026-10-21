import type { Backend, CatalogModel, ModelDtype } from '@/types'

/**
 * Curated, pinned model catalog.
 *
 * Every entry is pinned to an exact commit of its ONNX repository so that the
 * files we verified are the files users download. Sizes come from the Hugging Face
 * API for that revision. Memory figures are ESTIMATES (weights + KV cache + logits
 * buffers + runtime overhead) and are labelled as such in the UI.
 *
 * Verification statuses are updated from real test runs; see docs/model-compatibility.md.
 */

const MB = 1024 * 1024
const HF = 'https://huggingface.co'

export const DTYPE_FILE_SUFFIX: Record<ModelDtype, string> = {
  fp32: '',
  fp16: '_fp16',
  q8: '_quantized',
  int8: '_int8',
  uint8: '_uint8',
  q4: '_q4',
  q4f16: '_q4f16',
  bnb4: '_bnb4',
}

export const DTYPE_DESCRIPTIONS: Record<ModelDtype, string> = {
  fp32: '32-bit floating point weights (no quantization, largest, reference accuracy).',
  fp16: '16-bit floating point weights (half the size of fp32; essentially identical outputs). Runs on WASM; on WebGPU it needs shader-f16.',
  q8: '8-bit dynamically quantized weights and activations (≈¼ of fp32). Small, but can distort the distribution noticeably.',
  int8: 'Signed 8-bit quantized weights.',
  uint8: 'Unsigned 8-bit quantized weights.',
  q4: '4-bit block-quantized weights with fp32 activations.',
  q4f16: '4-bit block-quantized weights with fp16 activations (smallest; WebGPU).',
  bnb4: '4-bit NF4/FP4 (bitsandbytes-style) weight quantization; activations stay in fp32.',
}

const FP16_RMSNORM_OVERFLOW =
  'this export computes RMSNorm with fp16 Pow/ReduceMean ops. ONNX Runtime fuses them into an fp32-accumulating kernel on WASM, but runs them as-is on WebGPU, where'

export const CATALOG: CatalogModel[] = [
  {
    key: 'qwen3.5-0.8b',
    name: 'Qwen3.5 0.8B',
    publisher: 'Alibaba Cloud (Qwen)',
    repo: 'onnx-community/Qwen3.5-0.8B-Text-ONNX',
    revision: '25cdf33f386b487a25f3d743e86442517bdb77ff',
    upstreamRepo: 'Qwen/Qwen3.5-0.8B',
    parameters: '0.8B',
    modelType: 'instruct',
    architecture:
      'Qwen3.5 (Qwen3_5ForCausalLM), 24 layers: 18 Gated DeltaNet linear-attention + 6 full-attention (GQA 8/2), hidden 1024',
    license: { name: 'Apache-2.0', url: `${HF}/Qwen/Qwen3.5-0.8B` },
    contextLength: 262144,
    tokenizerSize: 248077,
    logitsSize: 248320,
    featured: '★ Patrick’s Choice',
    defaultExampleId: 'chat',
    variants: [
      {
        dtype: 'fp32',
        files: ['onnx/model.onnx', 'onnx/model.onnx_data', 'onnx/model.onnx_data_1'],
        sizeBytes: 3_077_060_202,
        backends: ['webgpu'],
        defaultFor: ['webgpu'],
        memoryEstimateBytes: 3600 * MB,
        fidelity: 'Reference precision (2 external-data chunks).',
        note: 'Default: the exact distribution, verified with real inference on WebGPU. Large download.',
      },
      {
        dtype: 'fp16',
        files: ['onnx/model_fp16.onnx', 'onnx/model_fp16.onnx_data'],
        sizeBytes: 1_538_945_869,
        backends: ['webgpu'],
        memoryEstimateBytes: 2000 * MB,
        fidelity: 'Matches fp32 (KL < 0.0001 nats, top-1 5/5; measured in Node.js).',
        note: 'Half the download of fp32. Needs shader-f16; not verified on WebGPU in CI (no f16-capable GPU there).',
      },
      {
        dtype: 'q8',
        files: ['onnx/model_quantized.onnx', 'onnx/model_quantized.onnx_data'],
        sizeBytes: 938_947_493,
        backends: ['webgpu'],
        memoryEstimateBytes: 1400 * MB,
        fidelity: 'Near-exact: KL 0.0006 nats vs fp32, top-1 agreement 5/5.',
        note: 'A third of the fp32 download with practically the same distribution.',
      },
      {
        dtype: 'q4',
        files: ['onnx/model_q4.onnx', 'onnx/model_q4.onnx_data'],
        sizeBytes: 551_247_327,
        backends: ['webgpu'],
        memoryEstimateBytes: 1000 * MB,
        fidelity: 'Approximate: KL 0.27 nats vs fp32, top-1 agreement 3/5.',
      },
      {
        dtype: 'q4f16',
        files: ['onnx/model_q4f16.onnx', 'onnx/model_q4f16.onnx_data'],
        sizeBytes: 469_929_410,
        backends: ['webgpu'],
        memoryEstimateBytes: 900 * MB,
        fidelity: 'Approximate: KL 0.27 nats vs fp32, top-1 agreement 3/5 (Node.js).',
        note: 'Smallest. Needs shader-f16; not verified on WebGPU in CI.',
      },
    ],
    supportFilesBytes: 19_238_247,
    memoryEstimateBytes: { wasm: 3600 * MB, webgpu: 3600 * MB },
    backendNotes: {
      wasm: 'Qwen3.5 needs the WebGPU backend. On WASM, fp32 and fp16 exceed WebAssembly memory (std::bad_alloc; fp16 is also expanded to fp32 on the CPU), and the quantized files use the GatherBlockQuantized embedding kernel, which ONNX Runtime Web’s WASM build does not include. The tokenizer alone works on WASM.',
    },
    verification: {
      status: 'verified',
      summary:
        'Real forward passes in headless Chromium inside the app’s Web Worker on WebGPU (SwiftShader software adapter) with fp32, q8 and q4: full 248,320-dim logits, softmax sums to 1, token append + further passes. fp32 in the browser reproduces the Node.js fp32 reference (“The capital of France is” → “ Paris” 13.0 %). fp16 and q4f16 were measured in Node.js only: CI’s GPU adapter lacks shader-f16. Every variant fails on WASM (see the WASM note).',
      verifiedOn: [
        'Chromium WebGPU/SwiftShader (fp32, q8, q4)',
        'Node.js CPU (fp32, fp16, q8, q4, q4f16)',
      ],
      date: '2026-10-21',
    },
    description:
      'A hybrid linear-attention model with a 248k-token vocabulary and an optional thinking mode. Much stronger than the other models here; needs WebGPU.',
    limitations: [
      'WebGPU only (see the WASM note). The default fp32 is a ≈2.9 GB download; q8 (≈0.9 GB) gives practically the same distribution.',
      'fp16 and q4f16 need the shader-f16 GPU feature and have not been verified on WebGPU in CI.',
      'Logits dimension (248,320) is larger than the tokenizer (248,077): IDs 248077–248319 have no token text and are excluded from sampling.',
      'Thinking is OFF in the chat template unless enable_thinking=true: the assistant turn then starts with an empty <think>\\n\\n</think> block.',
      'generation_config.json lists only <|endoftext|> as end of sequence; the app also stops at the tokenizer’s <|im_end|>.',
      'Hybrid (linear + full attention) architecture: every forward pass reprocesses the whole context (the app’s optional KV-cache reuse does not apply).',
    ],
    promptFormat:
      'ChatML with a think block: <|im_start|>user\\n…<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n (thinking off, the template default) or …assistant\\n<think>\\n (enable_thinking=true). No BOS token, no default system prompt.',
    examples: [
      {
        id: 'chat',
        label: 'Chat, thinking off: pick a number',
        description:
          'Real chat template with thinking off (empty <think></think> block), so the next token is the answer itself: “6” ≈ 30 %, “7” ≈ 22 %, “5” ≈ 18 %, “4” ≈ 13 %.',
        kind: 'chat',
        messages: [
          {
            role: 'user',
            content: 'Pick a random number between 1 and 10. Reply with just the number.',
          },
        ],
        addGenerationPrompt: true,
      },
      {
        id: 'completion',
        label: 'Raw completion (no template)',
        description:
          'Plain text continued by the model: “ Python” ≈ 15 %, “ Java” ≈ 14 %, “ C” ≈ 12 % — a close race.',
        kind: 'text',
        text: 'My favorite programming language is',
      },
      {
        id: 'thinking',
        label: 'Chat, thinking on (<think>)',
        description:
          'enable_thinking=true: the template ends with an open <think> block (token 248068), so the model starts reasoning (“Thinking” ≈ 95 %) before it answers.',
        kind: 'chat',
        messages: [{ role: 'user', content: 'Which is bigger, 9.11 or 9.9?' }],
        addGenerationPrompt: true,
        templateOptions: { enable_thinking: true },
      },
      {
        id: 'planet',
        label: 'Chat: one-word answer (sub-word tokens)',
        description:
          'Planet names split into pieces: “Earth” ≈ 31 %, “J” ≈ 28 %, “M” ≈ 14 %, “Mer” ≈ 12 %.',
        kind: 'chat',
        messages: [
          {
            role: 'user',
            content: 'Name one planet in our solar system. Answer with just the name.',
          },
        ],
        addGenerationPrompt: true,
      },
      {
        id: 'special',
        label: 'Special tokens: the model opens <think>',
        description:
          'ChatML from explicit special-token IDs, with no think block: the model itself predicts <think> (≈ 100 %). The template’s empty think block is what turns thinking off.',
        kind: 'parts',
        parts: [
          { special: '<|im_start|>' },
          { text: 'user\nWhich is bigger, 9.11 or 9.9?' },
          { special: '<|im_end|>' },
          { text: '\n' },
          { special: '<|im_start|>' },
          { text: 'assistant\n' },
        ],
      },
      {
        id: 'minimal',
        label: 'Minimal: <|endoftext|>',
        description:
          'Qwen has no BOS token; <|endoftext|> (248044) is the document separator. A very flat first-token distribution.',
        kind: 'parts',
        parts: [{ special: '<|endoftext|>' }],
      },
    ],
  },
  {
    key: 'smollm2-135m-instruct',
    name: 'SmolLM2 135M Instruct',
    publisher: 'Hugging Face (HuggingFaceTB)',
    repo: 'onnx-community/SmolLM2-135M-Instruct-ONNX',
    revision: 'b8a5c0f183b78c55955a5364f610c36668b5e681',
    upstreamRepo: 'HuggingFaceTB/SmolLM2-135M-Instruct',
    parameters: '135M',
    modelType: 'instruct',
    architecture: 'Llama (LlamaForCausalLM), 30 layers, hidden 576, GQA 9/3 heads',
    license: { name: 'Apache-2.0', url: `${HF}/HuggingFaceTB/SmolLM2-135M-Instruct` },
    contextLength: 8192,
    tokenizerSize: 49152,
    logitsSize: 49152,
    defaultExampleId: 'completion',
    variants: [
      {
        dtype: 'fp16',
        files: ['onnx/model_fp16.onnx'],
        sizeBytes: 269_851_815,
        backends: ['wasm'],
        defaultFor: ['wasm'],
        fidelity: 'Matches fp32 (KL < 0.0001 nats, top-1 5/5).',
        note: 'Default on WASM: exact distribution at half the size of fp32. Not offered on WebGPU (see “Not offered”).',
      },
      {
        dtype: 'fp32',
        files: ['onnx/model.onnx'],
        sizeBytes: 538_868_107,
        backends: ['wasm', 'webgpu'],
        defaultFor: ['webgpu'],
        fidelity: 'Reference precision.',
        note: 'Default on WebGPU.',
      },
      {
        dtype: 'q8',
        files: ['onnx/model_quantized.onnx'],
        sizeBytes: 135_658_354,
        backends: ['wasm', 'webgpu'],
        fidelity: 'Approximate: KL 0.64 nats vs fp32, top-1 agreement 3/5.',
        note: 'Small, but dynamic 8-bit quantization visibly reshapes the distribution of this small model.',
      },
      {
        dtype: 'q4f16',
        files: ['onnx/model_q4f16.onnx'],
        sizeBytes: 117_266_133,
        backends: ['wasm'],
        fidelity: 'Approximate: KL 0.60 nats vs fp32, top-1 agreement 1/5.',
        note: 'Smallest download; the least faithful variant. WASM only (see “Not offered”).',
      },
    ],
    excludedVariants: [
      {
        dtype: 'fp16',
        backends: ['webgpu'],
        reason: `Returns all-zero logits on WebGPU: ${FP16_RMSNORM_OVERFLOW} SmolLM2’s activations (|x| up to ≈4,400) overflow fp16 when squared, so the normalized hidden state and every logit become exactly 0.`,
      },
      {
        dtype: 'q4f16',
        backends: ['webgpu'],
        reason:
          'Same fp16 RMSNorm overflow as fp16: on WebGPU every logit is exactly 0 (a uniform distribution). This was the reported bug.',
      },
    ],
    supportFilesBytes: 3_527_558,
    memoryEstimateBytes: { wasm: 650 * MB, webgpu: 500 * MB },
    verification: {
      status: 'verified',
      summary:
        'Real forward passes in headless Chromium inside the app’s Web Worker: full 49,152-dim logits, softmax sums to 1, token append + further passes (automated CI smoke test). Verified on WASM with fp16, fp32, q8 and q4f16, and on WebGPU (SwiftShader software adapter) with fp32 and q8. fp16 reproduces fp32 exactly (“The capital of France is” → “ Paris” 46.9%).',
      verifiedOn: [
        'Chromium WASM (fp16, fp32, q8, q4f16)',
        'Chromium WebGPU/SwiftShader (fp32, q8)',
        'Node.js CPU',
      ],
      date: '2026-10-21',
    },
    description:
      'A very small instruction-tuned model. Fast enough on CPU for interactive exploration; the recommended starting point.',
    limitations: [
      'Small model: factual answers are often wrong and text quickly becomes repetitive.',
      'q8 and q4f16 visibly change this model’s distribution (KL ≈0.6 nats vs fp32); use fp16/fp32 when exact probabilities matter.',
      'fp16 and q4f16 are not offered on WebGPU: they return all-zero logits there (fp16 overflow in RMSNorm, see “Not offered”).',
      'The tokenizer config names <|im_start|> as "bos_token" but encode() does not add it automatically; the chat template adds role markers explicitly.',
    ],
    promptFormat:
      'ChatML: <|im_start|>role\\n…<|im_end|>\\n. The real Jinja chat template from tokenizer_config.json is used for chat examples (it inserts a default system prompt).',
    examples: [
      {
        id: 'minimal',
        label: 'Minimal: document start',
        description:
          'Just the <|endoftext|> document separator (ID 0) used between pre-training documents. Shows the model’s unconditional first-token distribution.',
        kind: 'parts',
        parts: [{ special: '<|endoftext|>' }],
      },
      {
        id: 'completion',
        label: 'Text completion',
        description: 'Plain text, no special tokens.',
        kind: 'text',
        text: 'The capital of France is',
      },
      {
        id: 'qa',
        label: 'Question & answer',
        description: 'A plain-text Q/A pattern.',
        kind: 'text',
        text: 'Question: What is the boiling point of water at sea level?\nAnswer:',
      },
      {
        id: 'chat',
        label: 'Chat (real chat template)',
        description:
          'Rendered with the tokenizer’s chat template, ending with the assistant generation prefix.',
        kind: 'chat',
        messages: [{ role: 'user', content: 'Give me one fun fact about octopuses.' }],
        addGenerationPrompt: true,
      },
      {
        id: 'special',
        label: 'Special-token demo',
        description:
          'ChatML built from explicit special-token IDs (<|im_start|>=1, <|im_end|>=2) instead of text, without a system prompt.',
        kind: 'parts',
        parts: [
          { special: '<|im_start|>' },
          { text: 'user\nWhat is 2 + 2?' },
          { special: '<|im_end|>' },
          { text: '\n' },
          { special: '<|im_start|>' },
          { text: 'assistant\n' },
        ],
      },
    ],
  },
  {
    key: 'qwen2.5-0.5b-instruct',
    name: 'Qwen2.5 0.5B Instruct',
    publisher: 'Alibaba Cloud (Qwen)',
    repo: 'onnx-community/Qwen2.5-0.5B-Instruct',
    revision: '22942cb7d7ba4cc81bb4673549ca4d4614469b5e',
    upstreamRepo: 'Qwen/Qwen2.5-0.5B-Instruct',
    parameters: '494M',
    modelType: 'instruct',
    architecture: 'Qwen2 (Qwen2ForCausalLM), 24 layers, hidden 896, GQA 14/2 heads',
    license: { name: 'Apache-2.0', url: `${HF}/Qwen/Qwen2.5-0.5B-Instruct/blob/main/LICENSE` },
    contextLength: 32768,
    tokenizerSize: 151665,
    logitsSize: 151936,
    defaultExampleId: 'completion',
    variants: [
      {
        dtype: 'bnb4',
        files: ['onnx/model_bnb4.onnx'],
        sizeBytes: 763_794_004,
        backends: ['wasm', 'webgpu'],
        defaultFor: ['wasm', 'webgpu'],
        fidelity:
          'Most faithful variant that runs in the browser: KL 0.22 nats vs fp32, top-1 agreement 5/5.',
        note: 'Default: fp32 (2 GB) and fp16 do not work in the browser (see “Not offered”).',
      },
      {
        dtype: 'q4',
        files: ['onnx/model_q4.onnx'],
        sizeBytes: 786_156_820,
        backends: ['wasm', 'webgpu'],
        fidelity: 'Approximate: KL 0.34 nats vs fp32, top-1 agreement 5/5.',
      },
      {
        dtype: 'q4f16',
        files: ['onnx/model_q4f16.onnx'],
        sizeBytes: 483_003_582,
        backends: ['wasm'],
        fidelity: 'Approximate: KL 0.34 nats vs fp32, top-1 agreement 5/5.',
        note: 'Smallest download. WASM only (see “Not offered”).',
      },
      {
        dtype: 'q8',
        files: ['onnx/model_quantized.onnx'],
        sizeBytes: 512_096_557,
        backends: ['wasm', 'webgpu'],
        fidelity: 'Approximate: KL 0.86 nats vs fp32, top-1 agreement 3/5.',
        note: 'Noticeably distorted distribution.',
      },
    ],
    excludedVariants: [
      {
        dtype: 'fp16',
        backends: ['wasm', 'webgpu'],
        reason: `On WASM the 1 GB file exceeds WebAssembly memory (std::bad_alloc). On WebGPU ${FP16_RMSNORM_OVERFLOW} Qwen2.5’s activations (|x| up to ≈1,600) overflow fp16 when squared and corrupt the result (emulated: KL 2.4 nats, all-zero logits for some prompts).`,
      },
      {
        dtype: 'q4f16',
        backends: ['webgpu'],
        reason:
          'Same fp16 RMSNorm overflow as fp16 on WebGPU. It works on WASM, where it is offered.',
      },
      {
        dtype: 'fp32',
        backends: ['wasm', 'webgpu'],
        reason:
          'The single 2 GB file exceeds WebAssembly memory on both backends (std::bad_alloc).',
      },
    ],
    supportFilesBytes: 7_040_442,
    memoryEstimateBytes: { wasm: 1800 * MB, webgpu: 1100 * MB },
    verification: {
      status: 'verified',
      summary:
        'Real forward passes in headless Chromium with bnb4, q4, q4f16 and q8 on WASM, and bnb4, q4 and q8 on WebGPU (SwiftShader software adapter): 151,936-dim logits (271 more output rows than the 151,665 tokenizer entries — these padding rows are shown but excluded from sampling). Fidelity was measured against fp32 in Node.js.',
      verifiedOn: [
        'Chromium WASM (bnb4, q4, q4f16, q8)',
        'Chromium WebGPU/SwiftShader (bnb4, q4, q8)',
        'Node.js CPU (fp32, fp16, bnb4, q4, q4f16, q8)',
      ],
      date: '2026-10-21',
    },
    description:
      'A stronger multilingual instruct model with a large 151k-token byte-level BPE vocabulary. Heavier download.',
    limitations: [
      '≈730 MB download (bnb4); one WASM forward pass takes roughly 1 s on a typical CPU.',
      'Every variant that runs in the browser is quantized and therefore approximate: fp32 and fp16 do not work (see “Not offered”).',
      'Logits dimension (151,936) is larger than the tokenizer (151,665): IDs 151665–151935 have no token text.',
      'The model card recommends repetition_penalty=1.1, which this app does not implement (it exposes the raw distribution).',
    ],
    promptFormat:
      'ChatML with Qwen system prompt: <|im_start|>system\\n…<|im_end|>\\n<|im_start|>user\\n…<|im_end|>\\n<|im_start|>assistant\\n. No BOS token.',
    examples: [
      {
        id: 'minimal',
        label: 'Minimal: <|endoftext|>',
        description: 'Qwen has no BOS token; <|endoftext|> (151643) is the document separator.',
        kind: 'parts',
        parts: [{ special: '<|endoftext|>' }],
      },
      {
        id: 'completion',
        label: 'Text completion',
        description: 'Plain text, no special tokens.',
        kind: 'text',
        text: 'The capital of France is',
      },
      {
        id: 'qa',
        label: 'Question & answer',
        description: 'A plain-text Q/A pattern.',
        kind: 'text',
        text: 'Q: How many legs does a spider have?\nA:',
      },
      {
        id: 'chat',
        label: 'Chat (real chat template)',
        description:
          'Rendered with the tokenizer’s chat template (adds the default Qwen system prompt).',
        kind: 'chat',
        messages: [{ role: 'user', content: 'Translate "good morning" into French.' }],
        addGenerationPrompt: true,
      },
      {
        id: 'special',
        label: 'Special-token demo',
        description: 'ChatML from explicit special-token IDs (151644 / 151645), no system prompt.',
        kind: 'parts',
        parts: [
          { special: '<|im_start|>' },
          { text: 'user\nName a primary color.' },
          { special: '<|im_end|>' },
          { text: '\n' },
          { special: '<|im_start|>' },
          { text: 'assistant\n' },
        ],
      },
    ],
  },
  {
    key: 'gpt2',
    name: 'GPT-2 (124M, base)',
    publisher: 'OpenAI',
    repo: 'onnx-community/gpt2-ONNX',
    revision: '1975fb9b4a2a92bd99f457407337ee25ab6cfde0',
    upstreamRepo: 'openai-community/gpt2',
    parameters: '124M',
    modelType: 'base',
    architecture: 'GPT-2 (GPT2LMHeadModel), 12 layers, hidden 768',
    license: { name: 'MIT', url: `${HF}/openai-community/gpt2` },
    contextLength: 1024,
    tokenizerSize: 50257,
    logitsSize: 50257,
    defaultExampleId: 'completion',
    variants: [
      {
        dtype: 'fp16',
        files: ['onnx/model_fp16.onnx'],
        sizeBytes: 249_252_545,
        backends: ['wasm', 'webgpu'],
        defaultFor: ['wasm'],
        fidelity: 'Matches fp32 (KL < 0.0001 nats, top-1 5/5).',
        note: 'Default on WASM: exact and smaller than q8 (whose embedding matrix is not quantized). On WebGPU it needs shader-f16 and is not verified in CI.',
      },
      {
        dtype: 'fp32',
        files: ['onnx/model.onnx'],
        sizeBytes: 498_126_358,
        backends: ['wasm', 'webgpu'],
        defaultFor: ['webgpu'],
        fidelity: 'Reference precision.',
        note: 'Default on WebGPU.',
      },
      {
        dtype: 'q8',
        files: ['onnx/model_quantized.onnx'],
        sizeBytes: 279_633_549,
        backends: ['wasm', 'webgpu'],
        fidelity: 'Approximate: KL 0.18 nats vs fp32, top-1 agreement 4/5.',
      },
    ],
    supportFilesBytes: 2_109_139,
    memoryEstimateBytes: { wasm: 600 * MB, webgpu: 800 * MB },
    verification: {
      status: 'verified',
      summary:
        'Real forward passes in headless Chromium with fp16, fp32 and q8 on WASM and fp32 and q8 on WebGPU (SwiftShader software adapter): 50,257-dim logits, softmax sums to 1. fp16 matches fp32 in Node.js; fp16 on WebGPU is not verified (CI’s GPU adapter lacks shader-f16).',
      verifiedOn: [
        'Chromium WASM (fp16, fp32, q8)',
        'Chromium WebGPU/SwiftShader (fp32, q8)',
        'Node.js CPU (fp32, fp16, q8)',
      ],
      date: '2026-10-21',
    },
    description:
      'The classic 2019 base (non-chat) model. A pure next-token predictor with the original byte-level BPE tokenizer and a single special token.',
    limitations: [
      'Base model: no chat template and no instruction following.',
      'Short 1,024-token context window.',
      'The q8 file is larger than fp16 because the tied embedding matrix is not quantized.',
    ],
    promptFormat:
      'Plain text. <|endoftext|> (50256) is the only special token: it separates documents and serves as BOS/EOS.',
    examples: [
      {
        id: 'minimal',
        label: 'Minimal: <|endoftext|>',
        description: 'Unconditional generation starts from <|endoftext|> (50256).',
        kind: 'parts',
        parts: [{ special: '<|endoftext|>' }],
      },
      {
        id: 'completion',
        label: 'Text completion',
        description: 'Plain text.',
        kind: 'text',
        text: 'The capital of France is',
      },
      {
        id: 'qa',
        label: 'Question & answer',
        description: 'Base models can follow simple patterns without a chat format.',
        kind: 'text',
        text: 'Q: What is the largest planet in the solar system?\nA:',
      },
      {
        id: 'special',
        label: 'Special-token demo',
        description: 'Two documents separated by the real <|endoftext|> token ID.',
        kind: 'parts',
        parts: [{ text: 'The end.' }, { special: '<|endoftext|>' }, { text: 'Once upon a time' }],
      },
    ],
  },
]

export function catalogModel(key: string): CatalogModel | undefined {
  return CATALOG.find((m) => m.key === key)
}

export function defaultVariant(model: CatalogModel, backend: Backend) {
  return (
    model.variants.find((v) => v.defaultFor?.includes(backend)) ??
    model.variants.find((v) => v.backends.includes(backend)) ??
    model.variants[0]
  )
}

/** Whether any variant of the model runs on the backend. */
export function supportsBackend(model: CatalogModel, backend: Backend): boolean {
  return model.variants.some((v) => v.backends.includes(backend))
}

/** Catalog order for display: featured models first, otherwise as listed. */
export function catalogInDisplayOrder(): CatalogModel[] {
  return [...CATALOG].sort((a, b) => Number(!!b.featured) - Number(!!a.featured))
}

/**
 * f16 variants on WebGPU need the shader-f16 GPU feature, which CI's software adapter
 * (SwiftShader) lacks, so no automated test has run them. The UI labels them accordingly.
 */
export function notVerifiedInCi(dtype: ModelDtype, backend: Backend): boolean {
  return backend === 'webgpu' && dtype.includes('16')
}

export function repoUrl(repo: string, revision?: string): string {
  return revision ? `${HF}/${repo}/tree/${revision}` : `${HF}/${repo}`
}

export function fileUrl(repo: string, revision: string, file: string): string {
  return `${HF}/${repo}/resolve/${revision}/${file}`
}

/** Files Transformers.js fetches for a causal LM besides the ONNX weights. */
export const SUPPORT_FILES = [
  'config.json',
  'generation_config.json',
  'tokenizer.json',
  'tokenizer_config.json',
]

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined || !Number.isFinite(bytes)) return '—'
  if (bytes < 1024) return `${bytes} B`
  const units = ['KB', 'MB', 'GB', 'TB']
  let v = bytes / 1024
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${v.toFixed(v >= 100 ? 0 : 1)} ${units[i]}`
}
