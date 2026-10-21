/**
 * Shared domain types for LLM Token Explorer.
 *
 * The authoritative model context is ALWAYS an ordered array of integer token IDs
 * (`TokenSequence`). Text is only ever an editing convenience / derived preview.
 */

export type TokenId = number
export type TokenSequence = readonly TokenId[]

/** Concrete inference backends exposed by ONNX Runtime Web. */
export type Backend = 'webgpu' | 'wasm'
export type BackendPreference = 'auto' | Backend

/** Quantization / precision variants as named by Transformers.js `dtype`. */
export type ModelDtype = 'fp32' | 'fp16' | 'q8' | 'int8' | 'uint8' | 'q4' | 'q4f16' | 'bnb4'

export type VerificationStatus =
  /** A real forward pass produced full-vocabulary logits in an automated test. */
  | 'verified'
  /** Some stages verified (e.g. real logits in Node / one backend only). */
  | 'partial'
  /** Supported according to documentation but not tested by this project. */
  | 'documented'
  /** Known issues or not expected to work everywhere. */
  | 'experimental'

export interface ModelVariant {
  dtype: ModelDtype
  /** ONNX file(s) inside the repository for this dtype. */
  files: string[]
  /** Size of the ONNX weight file(s) in bytes (from the Hugging Face API). */
  sizeBytes: number
  backends: Backend[]
  /** Which backend this variant is the default for. */
  defaultFor?: Backend
  /**
   * Measured fidelity vs. the full-precision model: mean KL divergence (nats) of the
   * next-token distribution over 5 reference prompts, plus top-1 agreement. See
   * docs/model-compatibility.md#quantization-fidelity.
   */
  fidelity?: string
  note?: string
}

export interface VerificationRecord {
  status: VerificationStatus
  /** Human-readable summary of what was verified and how. */
  summary: string
  /** Backends (and runtime) on which real inference was observed. */
  verifiedOn: string[]
  /** ISO date of the verification run. */
  date: string
}

export type PromptPart = { text: string } | { special: string } | { tokenId: number }

export interface PromptExample {
  id: string
  label: string
  description: string
  /**
   * - `text`: plain text tokenized with the tokenizer's default special-token handling.
   * - `chat`: messages rendered by the tokenizer's real chat template.
   * - `parts`: explicit sequence of text segments and named special tokens (exact IDs).
   */
  kind: 'text' | 'chat' | 'parts'
  text?: string
  messages?: { role: string; content: string }[]
  addGenerationPrompt?: boolean
  parts?: PromptPart[]
}

export interface CatalogModel {
  /** Stable slug used in experiment files. */
  key: string
  name: string
  publisher: string
  /** ONNX repository on the Hugging Face Hub. */
  repo: string
  /** Pinned git revision (commit SHA) of `repo`. */
  revision: string
  /** Original (non-ONNX) model repository / model card. */
  upstreamRepo: string
  parameters: string
  modelType: 'base' | 'instruct'
  architecture: string
  license: { name: string; url: string }
  contextLength: number
  /** Number of entries in tokenizer.json (vocab + added tokens). */
  tokenizerSize: number
  /** Size of the logits / embedding dimension (config.vocab_size). */
  logitsSize: number
  variants: ModelVariant[]
  /** Bytes of config/tokenizer files downloaded in addition to the weights. */
  supportFilesBytes: number
  /** Rough working-memory estimate in bytes, per backend, clearly an estimate. */
  memoryEstimateBytes: { wasm: number; webgpu: number }
  verification: VerificationRecord
  description: string
  limitations: string[]
  promptFormat: string
  examples: PromptExample[]
}

/** How a special token is used, derived from tokenizer metadata. */
export type SpecialTokenRole =
  'bos' | 'eos' | 'pad' | 'unk' | 'sep' | 'cls' | 'mask' | 'stop' | 'chat' | 'other'

export interface SpecialTokenInfo {
  id: TokenId
  content: string
  roles: SpecialTokenRole[]
  /** Short documented / inferred purpose. */
  purpose: string
  /** Whether the model's generation config treats this as a stop token. */
  isStop: boolean
}

/** Everything the UI needs to know about the loaded tokenizer, transferred once. */
export interface TokenizerSnapshot {
  /** raw[id] = the token's string in tokenizer.json ('' when the ID is unmapped). */
  raw: string[]
  /** decoded[id] = tokenizer.decode([id]) without cleanup. */
  decoded: string[]
  /** IDs flagged `special: true` in tokenizer.json added_tokens or listed as special. */
  specialIds: TokenId[]
  /** IDs that are added tokens (special or not). */
  addedIds: TokenId[]
  specialTokens: SpecialTokenInfo[]
  /** Pre-tokenizer / decoder family, used to explain raw token spellings. */
  scheme: 'byte-level-bpe' | 'sentencepiece' | 'wordpiece' | 'unknown'
  bosTokenId: TokenId | null
  eosTokenIds: TokenId[]
  padTokenId: TokenId | null
  unkTokenId: TokenId | null
  /** Whether `encode(text)` adds special tokens (e.g. BOS) by default. */
  addsSpecialTokensByDefault: boolean
  hasChatTemplate: boolean
  modelMaxLength: number | null
  /** Stable fingerprint of the vocabulary (FNV-1a of raw tokens) for experiment matching. */
  fingerprint: string
}

export interface LoadedModelInfo {
  /** Unique identity for experiment files, e.g. `hub:onnx-community/X@sha#q8` or `local:<name>`. */
  identity: string
  displayName: string
  source: 'catalog' | 'hub' | 'local'
  catalogKey?: string
  repo?: string
  revision?: string
  dtype: string
  backend: Backend
  modelType: string
  /** config.vocab_size = number of rows in the embedding / logits dimension. */
  logitsSize: number
  tokenizerSize: number
  contextLength: number | null
  /** Token IDs that the generation config treats as end of generation. */
  stopTokenIds: TokenId[]
  downloadBytes: number | null
  loadTimeMs: number
  fromCache: boolean | null
  /** True when only the tokenizer (no weights) was loaded. */
  tokenizerOnly: boolean
}

/** One next-token forward pass result (raw output of the neural network). */
export interface ForwardResult {
  /** Monotonic request id; used to drop stale results. */
  requestId: number
  /** Key of the token sequence this result belongs to. */
  contextKey: string
  /** Raw next-token logits for the final position: length = logitsSize. */
  logits: Float32Array
  /** Full output tensor dims, e.g. [1, seqLen, vocab]. */
  outputDims: number[]
  outputDtype: string
  inferenceMs: number
  usedKvCache: boolean
}

export interface SamplingSettings {
  /** 0 is treated as greedy. Valid range (0, 5]. */
  temperature: number
  /** 0 disables top-k. */
  topK: number
  /** 1 disables top-p. */
  topP: number
  /** 0 disables min-p. */
  minP: number
  greedy: boolean
  seed: number
}

export const DEFAULT_SAMPLING: Readonly<SamplingSettings> = Object.freeze({
  temperature: 1,
  topK: 0,
  topP: 1,
  minP: 0,
  greedy: false,
  seed: 42,
})

export interface GenerationSettings {
  maxNewTokens: number
  stopAtStopTokens: boolean
  /** ms between automatic inference after edits. */
  debounceMs: number
  autoInfer: boolean
  useKvCache: boolean
}

export const DEFAULT_GENERATION: Readonly<GenerationSettings> = Object.freeze({
  maxNewTokens: 32,
  stopAtStopTokens: true,
  debounceMs: 400,
  autoInfer: true,
  // Off by default: with dynamically quantized (q8) models, incremental KV-cached passes
  // produce slightly different logits than a full pass over the same tokens, because
  // activation quantization scales depend on which tokens are processed together.
  useKvCache: false,
})

export type ProgressEvent =
  | { stage: 'initiate'; file: string }
  | { stage: 'download'; file: string }
  | { stage: 'progress'; file: string; loaded: number; total: number }
  | { stage: 'done'; file: string }
  | { stage: 'total'; loaded: number; total: number }
  | { stage: 'session'; message: string }
  | { stage: 'tokenizer'; message: string }

export interface Capabilities {
  webgpu: boolean
  webgpuAdapter: {
    vendor?: string
    architecture?: string
    description?: string
    maxBufferSize?: number
    maxStorageBufferBindingSize?: number
    isFallbackAdapter?: boolean
    shaderF16?: boolean
  } | null
  webgpuError?: string
  wasm: boolean
  wasmSimd: boolean
  crossOriginIsolated: boolean
  hardwareConcurrency: number | null
  /** navigator.deviceMemory (GB, coarse, Chromium only). */
  deviceMemoryGb: number | null
  cacheStorage: boolean
  opfs: boolean
  serviceWorker: boolean
}
