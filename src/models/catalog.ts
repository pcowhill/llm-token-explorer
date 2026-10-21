import type { CatalogModel, ModelDtype } from '@/types'

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

export const CATALOG: CatalogModel[] = [
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
    variants: [
      {
        dtype: 'fp16',
        files: ['onnx/model_fp16.onnx'],
        sizeBytes: 269_900_000,
        backends: ['wasm', 'webgpu'],
        defaultFor: 'wasm',
        fidelity: 'Matches fp32 (KL < 0.0001 nats, top-1 5/5).',
        note: 'Default: exact distribution at half the size of fp32. On WebGPU it needs shader-f16.',
      },
      {
        dtype: 'fp32',
        files: ['onnx/model.onnx'],
        sizeBytes: 538_900_000,
        backends: ['wasm', 'webgpu'],
        fidelity: 'Reference precision.',
      },
      {
        dtype: 'q8',
        files: ['onnx/model_quantized.onnx'],
        sizeBytes: 135_700_000,
        backends: ['wasm', 'webgpu'],
        fidelity: 'Approximate: KL 1.15 nats vs fp32, top-1 agreement 2/5.',
        note: 'Smallest CPU option, but dynamic 8-bit quantization visibly reshapes the distribution of this small model.',
      },
      {
        dtype: 'q4f16',
        files: ['onnx/model_q4f16.onnx'],
        sizeBytes: 117_300_000,
        backends: ['webgpu'],
        defaultFor: 'webgpu',
        fidelity: 'Approximate (4-bit weights); not measured.',
      },
    ],
    supportFilesBytes: 3_527_558,
    memoryEstimateBytes: { wasm: 650 * MB, webgpu: 500 * MB },
    verification: {
      status: 'verified',
      summary:
        'Real forward passes in headless Chromium inside the app’s Web Worker: full 49,152-dim logits, softmax sums to 1, token append + further passes (automated CI smoke test). Verified on WASM with fp16, fp32 and q8, and on WebGPU (SwiftShader software adapter) with q8. fp16 reproduces fp32 exactly (“The capital of France is” → “ Paris” 46.9%).',
      verifiedOn: [
        'Chromium WASM (fp16, fp32, q8)',
        'Chromium WebGPU/SwiftShader (q8)',
        'Node.js CPU',
      ],
      date: '2026-10-21',
    },
    description:
      'A very small instruction-tuned model. Fast enough on CPU for interactive exploration; the recommended starting point.',
    limitations: [
      'Small model: factual answers are often wrong and text quickly becomes repetitive.',
      'q8 visibly changes this model’s distribution (KL 1.15 nats vs fp32); use fp16/fp32 when exact probabilities matter.',
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
    variants: [
      {
        dtype: 'bnb4',
        files: ['onnx/model_bnb4.onnx'],
        sizeBytes: 763_800_000,
        backends: ['wasm'],
        defaultFor: 'wasm',
        fidelity: 'Closest browser-feasible variant: KL 0.22 nats vs fp16, top-1 agreement 5/5.',
        note: 'fp16 (1 GB) does not fit in WebAssembly memory, so a 4-bit variant is the default on WASM.',
      },
      {
        dtype: 'q8',
        files: ['onnx/model_quantized.onnx'],
        sizeBytes: 512_100_000,
        backends: ['wasm', 'webgpu'],
        fidelity: 'Approximate: KL 1.46 nats vs fp16, top-1 agreement 3/5.',
        note: 'Smaller download, noticeably distorted distribution.',
      },
      {
        dtype: 'q4',
        files: ['onnx/model_q4.onnx'],
        sizeBytes: 786_200_000,
        backends: ['wasm', 'webgpu'],
        fidelity: 'Approximate: KL 0.57 nats vs fp16, top-1 agreement 3/5.',
      },
      {
        dtype: 'q4f16',
        files: ['onnx/model_q4f16.onnx'],
        sizeBytes: 483_000_000,
        backends: ['webgpu'],
        defaultFor: 'webgpu',
        fidelity: 'Approximate (4-bit weights); not measured.',
      },
      {
        dtype: 'fp16',
        files: ['onnx/model_fp16.onnx'],
        sizeBytes: 997_400_000,
        backends: ['webgpu'],
        fidelity: 'Reference (fp16).',
        note: 'WebGPU only (too large for WASM memory); needs shader-f16.',
      },
    ],
    supportFilesBytes: 7_040_442,
    memoryEstimateBytes: { wasm: 1800 * MB, webgpu: 1100 * MB },
    verification: {
      status: 'verified',
      summary:
        'Real forward passes in headless Chromium (WASM) with bnb4, q4 and q8: 151,936-dim logits (271 more output rows than the 151,665 tokenizer entries — these padding rows are shown but excluded from sampling). fp16 was verified only in Node.js (it exceeds WASM memory in the browser); WebGPU variants were not tested (no hardware GPU in CI).',
      verifiedOn: ['Chromium WASM (bnb4, q4, q8)', 'Node.js CPU (fp16, q8, q4, bnb4)'],
      date: '2026-10-21',
    },
    description:
      'A stronger multilingual instruct model with a large 151k-token byte-level BPE vocabulary. Heavier download.',
    limitations: [
      '≈764 MB download (bnb4); one WASM forward pass takes roughly 1 s on a typical CPU.',
      'fp16 (1 GB) does not fit in WebAssembly memory; every browser-WASM variant is quantized and therefore approximate.',
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
    variants: [
      {
        dtype: 'fp16',
        files: ['onnx/model_fp16.onnx'],
        sizeBytes: 249_300_000,
        backends: ['wasm', 'webgpu'],
        defaultFor: 'wasm',
        fidelity: 'Matches fp32 (KL < 0.0001 nats, top-1 5/5).',
        note: 'Default: exact and smaller than q8 (whose embedding matrix is not quantized).',
      },
      {
        dtype: 'fp32',
        files: ['onnx/model.onnx'],
        sizeBytes: 498_100_000,
        backends: ['wasm', 'webgpu'],
        defaultFor: 'webgpu',
        fidelity: 'Reference precision.',
      },
      {
        dtype: 'q8',
        files: ['onnx/model_quantized.onnx'],
        sizeBytes: 279_600_000,
        backends: ['wasm', 'webgpu'],
        fidelity: 'Approximate: KL 0.22 nats vs fp32, top-1 agreement 3/5.',
      },
    ],
    supportFilesBytes: 2_109_139,
    memoryEstimateBytes: { wasm: 600 * MB, webgpu: 800 * MB },
    verification: {
      status: 'verified',
      summary:
        'Real forward passes in headless Chromium (WASM) with fp16 and q8: 50,257-dim logits, softmax sums to 1. fp16 matches fp32 in Node.js.',
      verifiedOn: ['Chromium WASM (fp16, q8)', 'Node.js CPU (fp32, fp16, q8)'],
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

export function defaultVariant(model: CatalogModel, backend: 'wasm' | 'webgpu') {
  return (
    model.variants.find((v) => v.defaultFor === backend) ??
    model.variants.find((v) => v.backends.includes(backend)) ??
    model.variants[0]
  )
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
