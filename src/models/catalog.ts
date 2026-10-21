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
  fp16: '16-bit floating point weights (half the size of fp32; best on WebGPU with shader-f16).',
  q8: '8-bit dynamically quantized weights (≈¼ of fp32; recommended for WASM/CPU).',
  int8: 'Signed 8-bit quantized weights.',
  uint8: 'Unsigned 8-bit quantized weights.',
  q4: '4-bit block-quantized weights with fp32 activations.',
  q4f16: '4-bit block-quantized weights with fp16 activations (smallest; WebGPU).',
  bnb4: '4-bit bitsandbytes-style quantization.',
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
        dtype: 'q8',
        files: ['onnx/model_quantized.onnx'],
        sizeBytes: 135_700_000,
        backends: ['wasm', 'webgpu'],
        defaultFor: 'wasm',
      },
      {
        dtype: 'q4f16',
        files: ['onnx/model_q4f16.onnx'],
        sizeBytes: 117_300_000,
        backends: ['webgpu'],
        defaultFor: 'webgpu',
      },
      {
        dtype: 'fp16',
        files: ['onnx/model_fp16.onnx'],
        sizeBytes: 269_900_000,
        backends: ['webgpu'],
      },
      {
        dtype: 'fp32',
        files: ['onnx/model.onnx'],
        sizeBytes: 538_900_000,
        backends: ['wasm', 'webgpu'],
      },
    ],
    supportFilesBytes: 3_527_558,
    memoryEstimateBytes: { wasm: 400 * MB, webgpu: 350 * MB },
    verification: {
      status: 'verified',
      summary:
        'Real forward passes in headless Chromium (WASM backend, q8) inside a Web Worker: full 49,152-dim logits, softmax sums to 1, append + second forward pass. Also verified in Node.js (onnxruntime-node CPU).',
      verifiedOn: ['Chromium WASM (q8)', 'Node.js CPU (q8)'],
      date: '2026-10-21',
    },
    description:
      'A very small instruction-tuned model. Fast enough on CPU for interactive exploration; the recommended starting point.',
    limitations: [
      'Small model: factual answers are often wrong and text quickly becomes repetitive.',
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
        dtype: 'q8',
        files: ['onnx/model_quantized.onnx'],
        sizeBytes: 512_100_000,
        backends: ['wasm', 'webgpu'],
        defaultFor: 'wasm',
      },
      {
        dtype: 'q4f16',
        files: ['onnx/model_q4f16.onnx'],
        sizeBytes: 483_000_000,
        backends: ['webgpu'],
        defaultFor: 'webgpu',
      },
      {
        dtype: 'fp16',
        files: ['onnx/model_fp16.onnx'],
        sizeBytes: 997_400_000,
        backends: ['webgpu'],
      },
    ],
    supportFilesBytes: 7_040_442,
    memoryEstimateBytes: { wasm: 1200 * MB, webgpu: 900 * MB },
    verification: {
      status: 'verified',
      summary:
        'Real forward passes in headless Chromium (WASM, q8): 151,936-dim logits (271 more output rows than the 151,665 tokenizer entries — these padding rows are shown but excluded from sampling). Also verified in Node.js.',
      verifiedOn: ['Chromium WASM (q8)', 'Node.js CPU (q8)'],
      date: '2026-10-21',
    },
    description:
      'A stronger multilingual instruct model with a large 151k-token byte-level BPE vocabulary. Heavier download.',
    limitations: [
      '≈512 MB download; WASM inference is noticeably slower than SmolLM2.',
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
        dtype: 'q8',
        files: ['onnx/model_quantized.onnx'],
        sizeBytes: 279_600_000,
        backends: ['wasm', 'webgpu'],
        defaultFor: 'wasm',
      },
      {
        dtype: 'fp16',
        files: ['onnx/model_fp16.onnx'],
        sizeBytes: 249_300_000,
        backends: ['webgpu'],
        defaultFor: 'webgpu',
      },
      {
        dtype: 'fp32',
        files: ['onnx/model.onnx'],
        sizeBytes: 498_100_000,
        backends: ['wasm', 'webgpu'],
      },
    ],
    supportFilesBytes: 2_109_139,
    memoryEstimateBytes: { wasm: 700 * MB, webgpu: 600 * MB },
    verification: {
      status: 'verified',
      summary:
        'Real forward passes in headless Chromium (WASM, q8): 50,257-dim logits. Also verified in Node.js.',
      verifiedOn: ['Chromium WASM (q8)', 'Node.js CPU (q8)'],
      date: '2026-10-21',
    },
    description:
      'The classic 2019 base (non-chat) model. A pure next-token predictor with the original byte-level BPE tokenizer and a single special token.',
    limitations: [
      'Base model: no chat template and no instruction following.',
      'Short 1,024-token context window.',
      'The q8 file is larger than you might expect because the tied embedding matrix is not quantized.',
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
