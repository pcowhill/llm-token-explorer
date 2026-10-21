/**
 * Runtime-agnostic inference engine interface.
 *
 * The UI and stores only ever talk to this interface. The production implementation
 * (`TransformersEngine`, in transformersEngine.ts) runs inside a Web Worker and is
 * reached through `WorkerEngineClient`; component tests use an in-memory fake.
 * Nothing outside src/inference/ imports Transformers.js.
 */

import type {
  Backend,
  Capabilities,
  LoadedModelInfo,
  ModelDtype,
  ProgressEvent,
  TokenizerSnapshot,
} from '@/types'

export interface HubModelSource {
  kind: 'hub'
  repo: string
  revision: string
  catalogKey?: string
  displayName: string
  /** Expected total download size in bytes (for display), if known. */
  expectedBytes?: number
}

export interface LocalModelSource {
  kind: 'local'
  displayName: string
  /** Files with paths relative to the model root (as produced by analyzeLocalFiles). */
  files: { path: string; file: File }[]
  subfolder: string
  baseName: string
  externalDataChunks: number
}

export type ModelSource = HubModelSource | LocalModelSource

export interface LoadRequest {
  source: ModelSource
  device: Backend
  dtype: ModelDtype
  /** Load only config + tokenizer (a few MB) for tokenization demos; no inference. */
  tokenizerOnly?: boolean
}

export interface LoadResult {
  info: LoadedModelInfo
  tokenizer: TokenizerSnapshot
  /** Sampling defaults suggested by generation_config.json, if any. */
  suggestedSampling: { temperature?: number; topK?: number; topP?: number; repetitionPenalty?: number } | null
}

export interface ForwardOptions {
  /** Reuse the KV cache of the previous call when the new sequence extends it. */
  useKvCache: boolean
}

export interface ForwardOutput {
  logits: Float32Array
  outputDims: number[]
  outputDtype: string
  inferenceMs: number
  usedKvCache: boolean
  /** Number of new tokens actually fed to the network this call. */
  tokensProcessed: number
}

export interface ChatMessage {
  role: string
  content: string
}

export interface InferenceEngine {
  capabilities(): Promise<Capabilities>
  load(request: LoadRequest, onProgress: (e: ProgressEvent) => void): Promise<LoadResult>
  tokenize(text: string, addSpecialTokens: boolean): Promise<{ ids: number[]; ms: number }>
  decode(ids: number[]): Promise<string>
  applyChatTemplate(messages: ChatMessage[], addGenerationPrompt: boolean): Promise<{ ids: number[]; text: string }>
  forward(ids: number[], options: ForwardOptions): Promise<ForwardOutput>
  unload(): Promise<void>
  /** Permanently stop the engine (terminates the worker). */
  terminate(): void
}

export class EngineError extends Error {
  constructor(
    message: string,
    public readonly kind:
      | 'unsupported-architecture'
      | 'missing-file'
      | 'network'
      | 'out-of-memory'
      | 'backend'
      | 'quota'
      | 'invalid-input'
      | 'no-model'
      | 'aborted'
      | 'unknown' = 'unknown',
    public readonly hint?: string,
  ) {
    super(message)
    this.name = 'EngineError'
  }
}

/** Map low-level runtime errors to actionable categories + hints for the UI. */
export function classifyError(err: unknown): EngineError {
  if (err instanceof EngineError) return err
  const message = err instanceof Error ? err.message : String(err)
  const m = message.toLowerCase()
  if (/unsupported model type|not supported|unknown model class|unrecognized/.test(m))
    return new EngineError(message, 'unsupported-architecture', 'This architecture is not supported by Transformers.js as a causal language model.')
  if (/could not locate file|file was not found|404|not found locally|missing required file/.test(m))
    return new EngineError(message, 'missing-file', 'A required model file is missing. Check the file list and try again.')
  if (/quota|storage/.test(m) && /exceed|full|quota/.test(m))
    return new EngineError(message, 'quota', 'Browser storage is full. Remove cached models in the Model panel, or load without caching.')
  if (/out of memory|oom|allocation failed|cannot allocate|memory access out of bounds|array buffer allocation|rangeerror: array buffer/.test(m))
    return new EngineError(message, 'out-of-memory', 'The model did not fit in memory. Try a smaller model or a more aggressively quantized variant (q4/q4f16).')
  if (/failed to fetch|networkerror|network error|load failed|err_|aborted|timeout/.test(m))
    return new EngineError(message, 'network', 'The download failed or was interrupted. Check your connection and retry; completed files are cached.')
  if (/webgpu|gpu|adapter|device lost|jsep|wgsl/.test(m))
    return new EngineError(message, 'backend', 'The WebGPU backend failed. Try the WASM/CPU backend instead.')
  if (/wasm|webassembly|instantiate/.test(m))
    return new EngineError(message, 'backend', 'WebAssembly initialization failed. Your browser may block WASM or lack SIMD support.')
  return new EngineError(message, 'unknown')
}
