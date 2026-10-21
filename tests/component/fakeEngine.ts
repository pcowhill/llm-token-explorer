/**
 * FAKE inference engine for component tests ONLY.
 *
 * It implements the InferenceEngine interface with a tiny hand-written vocabulary and
 * deterministic synthetic logits so UI behaviour can be tested in isolation (jsdom has no
 * WebAssembly ONNX runtime). It is never bundled into the application, and results from
 * it are NOT evidence of real model inference — see tests/e2e for real inference tests.
 */
import type {
  ChatMessage,
  ForwardOptions,
  ForwardOutput,
  InferenceEngine,
  LoadRequest,
  LoadResult,
} from '@/inference/engine'
import { EngineError } from '@/inference/engine'
import type { Capabilities, ProgressEvent, TokenizerSnapshot } from '@/types'

export const FAKE_RAW = [
  '<|endoftext|>', // 0 special
  '<|im_start|>', // 1 special
  '<|im_end|>', // 2 special (stop)
  'Ġthe',
  'Ġcat',
  'Ġsat',
  'Ġon',
  'Ġmat',
  'The',
  'Ċ',
  'Ġdog',
  'Ġran',
  '.',
  'Ġa',
  'Ġhat',
  'user',
  'assistant',
]
export const FAKE_DECODED = FAKE_RAW.map((r) => r.replace(/Ġ/g, ' ').replace(/Ċ/g, '\n'))
/** Two extra logit rows without tokenizer entries (like Qwen's padding rows). */
export const FAKE_LOGITS_SIZE = FAKE_RAW.length + 2

export function fakeTokenizer(): TokenizerSnapshot {
  return {
    raw: [...FAKE_RAW],
    decoded: [...FAKE_DECODED],
    specialIds: [0, 1, 2],
    addedIds: [0, 1, 2],
    specialTokens: [
      {
        id: 0,
        content: '<|endoftext|>',
        roles: ['other'],
        purpose: 'End of text / document separator.',
        isStop: false,
      },
      {
        id: 1,
        content: '<|im_start|>',
        roles: ['bos', 'chat'],
        purpose: 'ChatML: starts a message.',
        isStop: false,
      },
      {
        id: 2,
        content: '<|im_end|>',
        roles: ['eos', 'stop', 'chat'],
        purpose: 'ChatML: ends a message.',
        isStop: true,
      },
    ],
    scheme: 'byte-level-bpe',
    bosTokenId: 1,
    eosTokenIds: [2],
    padTokenId: null,
    unkTokenId: null,
    addsSpecialTokensByDefault: false,
    hasChatTemplate: true,
    modelMaxLength: 64,
    fingerprint: 'fake0001',
  }
}

/** Deterministic synthetic logits: depends only on the last token and sequence length. */
export function fakeLogits(ids: number[]): Float32Array {
  const out = new Float32Array(FAKE_LOGITS_SIZE)
  const last = ids[ids.length - 1] ?? 0
  for (let i = 0; i < out.length; i++) out[i] = -((i * 7 + last * 3 + ids.length) % 11) / 2
  // A clear favourite: token 4 (" cat") after "the", else token 3 (" the")
  out[last === 3 ? 4 : 3] = 4
  out[2] = 1 // stop token has moderate probability
  return out
}

export class FakeEngine implements InferenceEngine {
  forwardCalls: number[][] = []
  loadCalls: LoadRequest[] = []
  failLoad: string | null = null
  failForward: string | null = null
  forwardDelayMs = 0
  logitsFn: (ids: number[]) => Float32Array = fakeLogits

  async capabilities(): Promise<Capabilities> {
    return {
      webgpu: false,
      webgpuAdapter: null,
      webgpuError: 'Fake engine',
      wasm: true,
      wasmSimd: true,
      crossOriginIsolated: false,
      hardwareConcurrency: 4,
      deviceMemoryGb: null,
      cacheStorage: false,
      opfs: false,
      serviceWorker: false,
    }
  }

  async load(request: LoadRequest, onProgress: (e: ProgressEvent) => void): Promise<LoadResult> {
    this.loadCalls.push(request)
    onProgress({ stage: 'initiate', file: 'config.json' })
    onProgress({ stage: 'progress', file: 'onnx/model_quantized.onnx', loaded: 50, total: 100 })
    if (this.failLoad) throw new EngineError(this.failLoad, 'network', 'Check your connection.')
    onProgress({ stage: 'done', file: 'onnx/model_quantized.onnx' })
    const repo = request.source.kind === 'hub' ? request.source.repo : 'local'
    return {
      info: {
        identity: `fake:${repo}#${request.dtype}`,
        displayName: request.source.displayName,
        source:
          request.source.kind === 'hub' ? (request.source.catalogKey ? 'catalog' : 'hub') : 'local',
        ...(request.source.kind === 'hub'
          ? { repo, revision: request.source.revision, catalogKey: request.source.catalogKey }
          : {}),
        dtype: request.dtype,
        backend: request.device,
        modelType: 'fake',
        logitsSize: FAKE_LOGITS_SIZE,
        tokenizerSize: FAKE_RAW.length,
        contextLength: 64,
        stopTokenIds: [2],
        downloadBytes: 100,
        loadTimeMs: 5,
        fromCache: false,
        tokenizerOnly: !!request.tokenizerOnly,
      },
      tokenizer: fakeTokenizer(),
      suggestedSampling: { temperature: 0.7, topP: 0.8, topK: 20 },
    }
  }

  async tokenize(text: string): Promise<{ ids: number[]; ms: number }> {
    const ids: number[] = []
    const re = /<\|[a-z_]+\|>| ?[A-Za-z]+|\n|\./g
    for (const m of text.match(re) ?? []) {
      const raw = m.replace(/^ /, 'Ġ').replace(/\n/, 'Ċ')
      const id = FAKE_RAW.indexOf(raw)
      if (id >= 0) ids.push(id)
    }
    return { ids, ms: 0.1 }
  }

  async decode(ids: number[]): Promise<string> {
    return ids.map((i) => FAKE_DECODED[i] ?? '').join('')
  }

  async applyChatTemplate(messages: ChatMessage[], addGenerationPrompt: boolean) {
    const ids: number[] = []
    for (const m of messages) {
      ids.push(1, FAKE_RAW.indexOf(m.role) >= 0 ? FAKE_RAW.indexOf(m.role) : 15, 9)
      ids.push(...(await this.tokenize(m.content)).ids, 2, 9)
    }
    if (addGenerationPrompt) ids.push(1, 16, 9)
    return { ids, text: await this.decode(ids) }
  }

  async forward(ids: number[], _options: ForwardOptions): Promise<ForwardOutput> {
    this.forwardCalls.push([...ids])
    if (this.forwardDelayMs) await new Promise((r) => setTimeout(r, this.forwardDelayMs))
    if (this.failForward) throw new EngineError(this.failForward, 'backend')
    return {
      logits: this.logitsFn(ids),
      outputDims: [1, ids.length, FAKE_LOGITS_SIZE],
      outputDtype: 'float32',
      inferenceMs: 1,
      usedKvCache: false,
      tokensProcessed: ids.length,
    }
  }

  async unload(): Promise<void> {}
  terminate(): void {}
}
