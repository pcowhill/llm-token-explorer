/**
 * Transformers.js adapter. Runs inside the inference Web Worker.
 *
 * This is the ONLY module that imports @huggingface/transformers. It:
 *  - configures the runtime (self-hosted ONNX Runtime WASM, caching, local files),
 *  - loads a tokenizer + AutoModelForCausalLM,
 *  - executes a raw forward pass `model({ input_ids, attention_mask })` and returns the
 *    complete logits row for the final position (NOT `generate()`, which hides logits),
 *  - optionally reuses the KV cache when the new sequence extends the previous one.
 *
 * Internal Transformers.js fields used (isolated here, covered by the real-inference
 * tests): `tokenizer._tokenizerJSON` (added-token flags + decoder type) and the ONNX
 * session input/output names `present.*` ↔ `past_key_values.*`.
 */

import {
  AutoConfig,
  AutoModelForCausalLM,
  AutoTokenizer,
  DynamicCache,
  ModelRegistry,
  Tensor,
  env,
  type PreTrainedModel,
  type PreTrainedTokenizer,
  type ProgressInfo,
} from '@huggingface/transformers'
import type {
  Backend,
  Capabilities,
  LoadedModelInfo,
  ProgressEvent,
  SpecialTokenInfo,
  SpecialTokenRole,
  TokenizerSnapshot,
} from '@/types'
import { fnv1a } from '@/tokenization/sequence'
import {
  EngineError,
  classifyError,
  type ChatMessage,
  type ForwardOptions,
  type ForwardOutput,
  type InferenceEngine,
  type LoadRequest,
  type LoadResult,
  type LocalModelSource,
} from './engine'
import { detectCapabilities } from './capabilities'

/**
 * Local files are exposed to Transformers.js under this pseudo path. It must NOT be an
 * http(s) URL: the library only probes `localModelPath` for files when it is a path.
 * Requests for it never reach the network; `appFetch` answers them from the File objects.
 */
const LOCAL_PREFIX = '/__local_models__/'
let localLoadCounter = 0
const BROWSER_CACHE_KEY = 'transformers-cache'

let localFiles: Map<string, File> | null = null

/** fetch() wrapper: serves user-selected local files from memory; everything else goes to the network. */
async function appFetch(input: string | URL, init?: RequestInit): Promise<Response> {
  const url = typeof input === 'string' ? input : input.href
  const path = url.startsWith(self.location.origin) ? url.slice(self.location.origin.length) : url
  if (path.startsWith(LOCAL_PREFIX)) {
    // <prefix>local/<loadId>/<relative path>
    const rel = decodeURIComponent(
      path.slice(LOCAL_PREFIX.length).split('/').filter(Boolean).slice(2).join('/'),
    )
    const file = localFiles?.get(rel)
    if (!file)
      return new Response(null, { status: 404, statusText: `Local file not selected: ${rel}` })
    return new Response(file, {
      status: 200,
      headers: { 'content-length': String(file.size), 'content-type': 'application/octet-stream' },
    })
  }
  return fetch(input, init)
}

let envConfigured = false
export function configureRuntime(): void {
  if (envConfigured) return
  envConfigured = true
  env.allowRemoteModels = true
  env.allowLocalModels = false
  env.useBrowserCache = typeof caches !== 'undefined'
  env.cacheKey = BROWSER_CACHE_KEY
  env.useWasmCache = typeof caches !== 'undefined'
  env.fetch = appFetch
  const onnx = env.backends.onnx
  if (onnx.wasm) {
    // Self-hosted ONNX Runtime WASM (copied into <base>/ort/ at build time).
    const base = new URL(`${import.meta.env.BASE_URL}ort/`, self.location.href).href
    onnx.wasm.wasmPaths = {
      mjs: `${base}ort-wasm-simd-threaded.asyncify.mjs`,
      wasm: `${base}ort-wasm-simd-threaded.asyncify.wasm`,
    }
    // Multi-threading needs SharedArrayBuffer (cross-origin isolation), which
    // GitHub Pages cannot enable via headers; fall back to one thread otherwise.
    onnx.wasm.numThreads = self.crossOriginIsolated
      ? Math.max(1, Math.min(4, (navigator.hardwareConcurrency ?? 2) - 1))
      : 1
  }
}

function mapProgress(p: ProgressInfo): ProgressEvent | null {
  switch (p.status) {
    case 'initiate':
      return { stage: 'initiate', file: p.file }
    case 'download':
      return { stage: 'download', file: p.file }
    case 'progress':
      return { stage: 'progress', file: p.file, loaded: p.loaded, total: p.total }
    case 'done':
      return { stage: 'done', file: p.file }
    case 'progress_total':
      return { stage: 'total', loaded: p.loaded, total: p.total }
    default:
      return null
  }
}

interface KvState {
  ids: number[]
  cache: DynamicCache
  logits: Float32Array
  dims: number[]
  dtype: string
}

const TOKEN_PURPOSES: [RegExp, string][] = [
  [
    /^<\|endoftext\|>$/,
    'End of text / document separator. Many models were trained with it between documents.',
  ],
  [
    /^<\|im_start\|>$/,
    'ChatML: starts a message; followed by the role name (system/user/assistant) and a newline.',
  ],
  [/^<\|im_end\|>$/, 'ChatML: ends a message (end of turn).'],
  [/^<start_of_turn>$/, 'Gemma: starts a conversation turn.'],
  [/^<end_of_turn>$/, 'Gemma: ends a conversation turn.'],
  [/^<\|begin_of_text\|>$/, 'Llama 3: beginning of text.'],
  [/^<\|eot_id\|>$/, 'Llama 3: end of turn.'],
  [
    /^<\|start_header_id\|>$|^<\|end_header_id\|>$/,
    'Llama 3: delimits the role header of a message.',
  ],
  [/^<s>$|^<bos>$/, 'Beginning of sequence.'],
  [/^<\/s>$|^<eos>$/, 'End of sequence.'],
  [/^<pad>$|^\[PAD\]$/, 'Padding (used to batch sequences of different lengths).'],
  [/^<unk>$|^\[UNK\]$/, 'Unknown token (placeholder for text the vocabulary cannot represent).'],
  [/^<mask>$|^\[MASK\]$/, 'Mask token (masked-language-model training).'],
  [
    /^<\|?(object_ref|box|quad)_(start|end)\|?>$/,
    'Qwen-VL grounding markers (object references / bounding boxes).',
  ],
  [
    /^<\|?(vision|image|video)_(start|end|pad)\|?>$|^<(start|end)_of_image>$|^<image_soft_token>$/,
    'Multimodal placeholder (images / video); unused by text-only models.',
  ],
  [/^<tool_call>$|^<\/tool_call>$/, 'Tool/function-call delimiters.'],
  [/^<think>$|^<\/think>$/, 'Reasoning ("thinking") block delimiters.'],
  [
    /^<(repo_name|reponame|file_sep|filename|gh_stars)>$/,
    'Code-repository metadata markers from code pre-training data.',
  ],
  [
    /^<(issue_start|issue_comment|issue_closed)>$/,
    'GitHub-issue structure markers from pre-training data.',
  ],
  [
    /^<(jupyter_start|jupyter_text|jupyter_code|jupyter_output|jupyter_script|empty_output)>$/,
    'Jupyter-notebook structure markers from pre-training data.',
  ],
  [/^<\|fim_(prefix|middle|suffix|pad)\|>$/, 'Fill-in-the-middle code-completion markers.'],
]

function purposeOf(content: string, roles: SpecialTokenRole[]): string {
  for (const [re, text] of TOKEN_PURPOSES) if (re.test(content)) return text
  if (roles.includes('bos')) return 'Beginning of sequence.'
  if (roles.includes('eos')) return 'End of sequence.'
  if (roles.includes('pad')) return 'Padding.'
  return 'Model-specific control token (purpose not documented in tokenizer metadata).'
}

type AddedTokenJson = { id: number; content: string; special?: boolean }

interface TokenizerInternals {
  _tokenizerJSON?: {
    added_tokens?: AddedTokenJson[]
    decoder?: { type?: string; decoders?: { type?: string }[] }
    pre_tokenizer?: { type?: string; pretokenizers?: { type?: string }[] }
    model?: { type?: string; byte_fallback?: boolean }
  }
}

function detectScheme(tok: PreTrainedTokenizer, raw: string[]): TokenizerSnapshot['scheme'] {
  const j = (tok as unknown as TokenizerInternals)._tokenizerJSON
  const types = [
    j?.decoder?.type,
    ...(j?.decoder?.decoders?.map((d) => d.type) ?? []),
    j?.pre_tokenizer?.type,
    ...(j?.pre_tokenizer?.pretokenizers?.map((d) => d.type) ?? []),
  ]
  if (types.includes('ByteLevel')) return 'byte-level-bpe'
  if (types.includes('Metaspace') || types.includes('ByteFallback') || j?.model?.byte_fallback)
    return 'sentencepiece'
  if (types.includes('WordPiece') || j?.model?.type === 'WordPiece') return 'wordpiece'
  // Heuristic fallback
  if (raw.some((r) => r.startsWith('Ġ'))) return 'byte-level-bpe'
  if (raw.some((r) => r.startsWith('▁'))) return 'sentencepiece'
  return 'unknown'
}

function toIdList(v: unknown): number[] {
  if (typeof v === 'number') return [v]
  if (Array.isArray(v)) return v.filter((x): x is number => typeof x === 'number')
  return []
}

export class TransformersEngine implements InferenceEngine {
  private tokenizer: PreTrainedTokenizer | null = null
  private model: PreTrainedModel | null = null
  private info: LoadedModelInfo | null = null
  private kv: KvState | null = null
  private kvSupported = false
  private config: Record<string, unknown> | null = null

  async capabilities(): Promise<Capabilities> {
    return detectCapabilities()
  }

  async load(request: LoadRequest, onProgress: (e: ProgressEvent) => void): Promise<LoadResult> {
    configureRuntime()
    await this.unload()
    const t0 = performance.now()
    const { source, device, dtype } = request
    const isLocal = source.kind === 'local'
    let modelId: string
    let fromCache: boolean | null = null
    const options: Record<string, unknown> = {
      progress_callback: (p: ProgressInfo) => {
        const e = mapProgress(p)
        if (e) onProgress(e)
      },
    }
    let totalBytes = 0
    const seen = new Map<string, number>()
    const trackProgress = (e: ProgressEvent) => {
      if (e.stage === 'progress') seen.set(e.file, e.total)
    }
    const origCb = options.progress_callback as (p: ProgressInfo) => void
    options.progress_callback = (p: ProgressInfo) => {
      const e = mapProgress(p)
      if (e) trackProgress(e)
      origCb(p)
    }

    if (isLocal) {
      const src = source as LocalModelSource
      localFiles = new Map(src.files.map((f) => [f.path, f.file]))
      env.allowLocalModels = true
      env.allowRemoteModels = false
      env.localModelPath = LOCAL_PREFIX
      // Never copy local files into Cache Storage.
      env.useBrowserCache = false
      // Unique id per load: Transformers.js memoizes file metadata per model id.
      modelId = `local/m${++localLoadCounter}`
      Object.assign(options, {
        local_files_only: true,
        subfolder: src.subfolder,
        model_file_name: src.baseName,
        use_external_data_format: src.externalDataChunks > 0 ? src.externalDataChunks : false,
      })
    } else {
      localFiles = null
      env.allowLocalModels = false
      env.allowRemoteModels = true
      env.useBrowserCache = typeof caches !== 'undefined'
      modelId = source.repo
      options.revision = source.revision
      try {
        fromCache = await ModelRegistry.is_cached(source.repo, {
          revision: source.revision,
          dtype,
          device,
        })
      } catch {
        fromCache = null
      }
    }

    try {
      onProgress({ stage: 'tokenizer', message: 'Loading tokenizer…' })
      const tokenizer = await AutoTokenizer.from_pretrained(modelId, options)
      this.tokenizer = tokenizer
      if (request.tokenizerOnly) {
        this.config = (await AutoConfig.from_pretrained(modelId, options)) as unknown as Record<
          string,
          unknown
        >
      } else {
        onProgress({
          stage: 'session',
          message: `Downloading weights and creating ONNX Runtime session (${device}, ${dtype})…`,
        })
        const model = await AutoModelForCausalLM.from_pretrained(modelId, {
          ...options,
          device,
          dtype,
        })
        this.model = model
        this.config = model.config as unknown as Record<string, unknown>
      }
    } catch (err) {
      this.tokenizer = null
      this.model = null
      throw classifyError(err)
    } finally {
      if (isLocal) {
        env.allowLocalModels = false
        env.allowRemoteModels = true
        env.useBrowserCache = typeof caches !== 'undefined'
      }
    }

    for (const v of seen.values()) totalBytes += v
    const model = this.model
    const tokenizer = this.tokenizer!
    const config = this.config ?? {}
    const textConfig = (config.text_config ?? {}) as Record<string, unknown>
    const logitsSizeCfg = Number(config.vocab_size ?? textConfig.vocab_size ?? 0)
    const session = model
      ? (
          model as unknown as {
            sessions: Record<string, { inputNames: string[]; outputNames: string[] }>
          }
        ).sessions
      : {}
    const main = session.model ?? Object.values(session)[0]
    this.kvSupported =
      !!main &&
      main.outputNames.some((n) => n.startsWith('present.')) &&
      main.outputNames
        .filter((n) => n.startsWith('present'))
        .every((n) => /^present\.\d+\.(key|value)$/.test(n)) &&
      main.inputNames
        .filter((n) => n.startsWith('past'))
        .every((n) => /^past_key_values\.\d+\.(key|value)$/.test(n))

    const generationConfig = (model?.generation_config ?? null) as Record<string, unknown> | null
    const tokenizerSnap = this.snapshotTokenizer(tokenizer, generationConfig)
    // Determine logits size by a probe forward pass when config is silent.
    let logitsSize = logitsSizeCfg
    if (!logitsSize && model) {
      const probeId = tokenizerSnap.bosTokenId ?? tokenizerSnap.eosTokenIds[0] ?? 0
      const out = await this.forward([probeId], { useKvCache: false })
      logitsSize = out.logits.length
    }

    if (!logitsSize) logitsSize = tokenizerSnap.raw.length
    const gen = generationConfig ?? {}
    const stopIds = new Set<number>([
      ...toIdList(gen.eos_token_id),
      ...toIdList(config.eos_token_id),
    ])
    if (stopIds.size === 0) for (const id of tokenizerSnap.eosTokenIds) stopIds.add(id)
    for (const s of tokenizerSnap.specialTokens) s.isStop = stopIds.has(s.id)
    for (const s of tokenizerSnap.specialTokens)
      if (s.isStop && !s.roles.includes('stop')) s.roles.push('stop')

    const contextLength =
      Number(
        config.max_position_embeddings ??
          config.n_positions ??
          textConfig.max_position_embeddings ??
          0,
      ) || null
    const info: LoadedModelInfo = {
      identity: isLocal
        ? `local:${source.displayName}#${dtype}#${tokenizerSnap.fingerprint}`
        : `hub:${source.repo}@${source.revision}#${dtype}`,
      displayName: source.displayName,
      source: isLocal ? 'local' : source.catalogKey ? 'catalog' : 'hub',
      ...(source.kind === 'hub'
        ? { repo: source.repo, revision: source.revision, catalogKey: source.catalogKey }
        : {}),
      dtype,
      backend: device,
      modelType: String(config.model_type ?? 'unknown'),
      logitsSize,
      tokenizerSize: tokenizerSnap.raw.length,
      contextLength,
      stopTokenIds: [...stopIds].sort((a, b) => a - b),
      downloadBytes: isLocal ? null : totalBytes || null,
      loadTimeMs: performance.now() - t0,
      fromCache,
      tokenizerOnly: !!request.tokenizerOnly,
    }
    this.info = info
    const suggested =
      gen.do_sample || gen.temperature || gen.top_p || gen.top_k
        ? {
            ...(typeof gen.temperature === 'number' ? { temperature: gen.temperature } : {}),
            ...(typeof gen.top_k === 'number' ? { topK: gen.top_k } : {}),
            ...(typeof gen.top_p === 'number' ? { topP: gen.top_p } : {}),
            ...(typeof gen.repetition_penalty === 'number'
              ? { repetitionPenalty: gen.repetition_penalty }
              : {}),
          }
        : null
    return { info, tokenizer: tokenizerSnap, suggestedSampling: suggested }
  }

  private snapshotTokenizer(
    tok: PreTrainedTokenizer,
    generationConfig: Record<string, unknown> | null,
  ): TokenizerSnapshot {
    const vocab = tok.get_vocab()
    const internals = (tok as unknown as TokenizerInternals)._tokenizerJSON
    const added: AddedTokenJson[] = internals?.added_tokens ?? []
    let maxId = -1
    for (const id of vocab.values()) if (id > maxId) maxId = id
    for (const a of added) if (a.id > maxId) maxId = a.id
    const size = maxId + 1
    const raw = new Array<string>(size).fill('')
    for (const [t, id] of vocab) raw[id] = t
    for (const a of added) raw[a.id] = a.content

    const specialSet = new Set<number>(tok.all_special_ids ?? [])
    for (const a of added) if (a.special) specialSet.add(a.id)
    const addedIds = added.map((a) => a.id)

    const decoded = new Array<string>(size).fill('')
    for (let id = 0; id < size; id++) {
      if (!raw[id]) continue
      try {
        decoded[id] = tok.decode([id], {
          skip_special_tokens: false,
          clean_up_tokenization_spaces: false,
        })
      } catch {
        decoded[id] = raw[id] as string
      }
    }

    const t = tok as unknown as Record<string, unknown>
    const idOf = (v: unknown) => (typeof v === 'number' && v >= 0 ? v : null)
    const bos = idOf(t.bos_token_id)
    const eos = idOf(t.eos_token_id)
    const roleMap = new Map<number, Set<SpecialTokenRole>>()
    const addRole = (id: number | null, r: SpecialTokenRole) => {
      if (id === null) return
      if (!roleMap.has(id)) roleMap.set(id, new Set())
      roleMap.get(id)!.add(r)
    }
    addRole(bos, 'bos')
    addRole(eos, 'eos')
    addRole(idOf(t.pad_token_id), 'pad')
    addRole(idOf(t.unk_token_id), 'unk')
    addRole(idOf(t.sep_token_id), 'sep')
    addRole(idOf(t.mask_token_id), 'mask')
    const genEos = toIdList(generationConfig?.eos_token_id)
    for (const id of genEos) addRole(id, 'stop')
    const template = String(tok.chat_template ?? '')
    for (const id of specialSet) {
      const c = raw[id] ?? ''
      if (c && template.includes(c)) addRole(id, 'chat')
    }

    const specialTokens: SpecialTokenInfo[] = [...specialSet]
      .filter((id) => id < size)
      .sort((a, b) => a - b)
      .map((id) => {
        const roles = [...(roleMap.get(id) ?? new Set<SpecialTokenRole>())]
        if (roles.length === 0) roles.push('other')
        const content = raw[id] ?? ''
        return {
          id,
          content,
          roles,
          purpose: purposeOf(content, roles),
          isStop: genEos.includes(id),
        }
      })

    let addsSpecial = false
    try {
      const a = tok.encode('a', { add_special_tokens: true })
      const b = tok.encode('a', { add_special_tokens: false })
      addsSpecial = a.length !== b.length
    } catch {
      addsSpecial = false
    }
    const mml = Number(tok.model_max_length)
    return {
      raw,
      decoded,
      specialIds: [...specialSet].sort((a, b) => a - b),
      addedIds,
      specialTokens,
      scheme: detectScheme(tok, raw),
      bosTokenId: bos,
      eosTokenIds: eos === null ? [] : [eos],
      padTokenId: idOf(t.pad_token_id),
      unkTokenId: idOf(t.unk_token_id),
      addsSpecialTokensByDefault: addsSpecial,
      hasChatTemplate: !!tok.chat_template,
      modelMaxLength: Number.isFinite(mml) && mml > 0 && mml < 1e8 ? mml : null,
      fingerprint: fnv1a(raw),
    }
  }

  private requireTokenizer(): PreTrainedTokenizer {
    if (!this.tokenizer) throw new EngineError('No tokenizer is loaded', 'no-model')
    return this.tokenizer
  }

  private requireModel(): PreTrainedModel {
    if (!this.model)
      throw new EngineError(
        this.tokenizer
          ? 'Only the tokenizer is loaded. Load the model weights to run inference.'
          : 'No model is loaded',
        'no-model',
      )
    return this.model
  }

  async tokenize(text: string, addSpecialTokens: boolean): Promise<{ ids: number[]; ms: number }> {
    const tokenizer = this.requireTokenizer()
    const t0 = performance.now()
    const ids = tokenizer.encode(text, { add_special_tokens: addSpecialTokens })
    return { ids: Array.from(ids, Number), ms: performance.now() - t0 }
  }

  async decode(ids: number[]): Promise<string> {
    const tokenizer = this.requireTokenizer()
    return tokenizer.decode(ids, {
      skip_special_tokens: false,
      clean_up_tokenization_spaces: false,
    })
  }

  async applyChatTemplate(
    messages: ChatMessage[],
    addGenerationPrompt: boolean,
  ): Promise<{ ids: number[]; text: string }> {
    const tokenizer = this.requireTokenizer()
    if (!tokenizer.chat_template)
      throw new EngineError('This tokenizer has no chat template', 'invalid-input')
    const text = tokenizer.apply_chat_template(messages, {
      tokenize: false,
      add_generation_prompt: addGenerationPrompt,
    }) as unknown as string
    // Templates already contain BOS etc. as text; encode without adding them again.
    const ids = Array.from(tokenizer.encode(text, { add_special_tokens: false }), Number)
    return { ids, text }
  }

  async forward(ids: number[], options: ForwardOptions): Promise<ForwardOutput> {
    const model = this.requireModel()
    if (ids.length === 0)
      throw new EngineError(
        'The context is empty. Add at least one token (e.g. a BOS / <|endoftext|> token).',
        'invalid-input',
      )
    const limit =
      this.info?.logitsSize ??
      Number((model.config as unknown as Record<string, unknown>).vocab_size ?? Infinity)
    for (const id of ids) {
      if (!Number.isInteger(id) || id < 0 || id >= limit)
        throw new EngineError(
          `Token ID ${id} is outside the model's input range 0–${limit - 1}`,
          'invalid-input',
        )
    }
    const ctx = this.info?.contextLength
    if (ctx && ids.length > ctx)
      throw new EngineError(
        `The context has ${ids.length} tokens but the model supports at most ${ctx}.`,
        'invalid-input',
      )

    // Exact repeat of the last call: return the cached logits (no network execution).
    if (this.kv && this.kv.ids.length === ids.length && this.kv.ids.every((v, i) => v === ids[i])) {
      return {
        logits: this.kv.logits.slice(),
        outputDims: this.kv.dims,
        outputDtype: this.kv.dtype,
        inferenceMs: 0,
        usedKvCache: true,
        tokensProcessed: 0,
      }
    }

    let past: DynamicCache | null = null
    let feed = ids
    if (
      options.useKvCache &&
      this.kvSupported &&
      this.kv &&
      ids.length > this.kv.ids.length &&
      this.kv.ids.every((v, i) => v === ids[i])
    ) {
      past = this.kv.cache
      feed = ids.slice(this.kv.ids.length)
    } else {
      await this.disposeKv()
    }

    const input_ids = new Tensor(
      'int64',
      BigInt64Array.from(feed, (v) => BigInt(v)),
      [1, feed.length],
    )
    const attention_mask = new Tensor('int64', new BigInt64Array(ids.length).fill(1n), [
      1,
      ids.length,
    ])
    const t0 = performance.now()
    let out: Record<string, Tensor>
    try {
      out = await (
        model as unknown as (x: Record<string, unknown>) => Promise<Record<string, Tensor>>
      )({
        input_ids,
        attention_mask,
        ...(past ? { past_key_values: past } : {}),
      })
    } catch (err) {
      await this.disposeKv()
      throw classifyError(err)
    }
    const inferenceMs = performance.now() - t0
    const logitsT = out.logits
    if (!logitsT)
      throw new EngineError(
        'The model did not return a "logits" output',
        'unsupported-architecture',
      )
    const dims = [...logitsT.dims]
    const V = dims[dims.length - 1] as number
    const rows = logitsT.size / V
    const f32 = logitsT.type === 'float32' ? logitsT : logitsT.to('float32')
    const data = f32.data as Float32Array
    const logits = data.slice((rows - 1) * V, rows * V)

    // Build / replace the KV cache from present.* outputs.
    const presentNames = Object.keys(out).filter((k) => k.startsWith('present'))
    if (options.useKvCache && this.kvSupported && presentNames.length) {
      const entries: Record<string, Tensor> = {}
      for (const k of presentNames)
        entries[k.replace(/^present\./, 'past_key_values.')] = out[k] as Tensor
      const cache = new DynamicCache(entries)
      if (past && past !== cache) await disposeCache(past, cache)
      this.kv = { ids: [...ids], cache, logits: logits.slice(), dims, dtype: logitsT.type }
    } else {
      for (const k of presentNames) disposeTensor(out[k] as Tensor)
      if (past) await disposeCache(past, null)
      this.kv = null
    }
    return {
      logits,
      outputDims: dims,
      outputDtype: logitsT.type,
      inferenceMs,
      usedKvCache: past !== null,
      tokensProcessed: feed.length,
    }
  }

  private async disposeKv(): Promise<void> {
    if (this.kv) {
      await disposeCache(this.kv.cache, null)
      this.kv = null
    }
  }

  async unload(): Promise<void> {
    await this.disposeKv()
    if (this.model) {
      try {
        await this.model.dispose()
      } catch {
        /* ignore */
      }
    }
    this.model = null
    this.tokenizer = null
    this.config = null
    this.info = null
    localFiles = null
  }

  terminate(): void {
    void this.unload()
  }
}

function disposeTensor(t: Tensor): void {
  try {
    if (t.location === 'gpu-buffer') t.dispose()
  } catch {
    /* ignore */
  }
}

async function disposeCache(cache: DynamicCache, keep: DynamicCache | null): Promise<void> {
  const keepSet = new Set(keep ? Object.values(keep as unknown as Record<string, Tensor>) : [])
  for (const t of Object.values(cache as unknown as Record<string, Tensor>)) {
    if (!keepSet.has(t)) disposeTensor(t)
  }
}

export type { Backend }
