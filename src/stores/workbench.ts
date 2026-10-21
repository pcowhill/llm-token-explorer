/**
 * Workbench store: the single source of truth for UI state.
 *
 * Design rules
 * - The token-ID sequence of the current history node is the authoritative context.
 * - Neural-network execution (forward pass → raw logits) is separated from token
 *   selection (sampling over an adjusted distribution). Changing sampling settings
 *   recomputes only the adjusted distribution from cached logits.
 * - Every forward pass carries a request id and an exact context key; results that
 *   do not match the *current* context when they arrive are discarded (stale).
 * - Large typed arrays are kept in shallowRefs / markRaw objects to avoid deep reactivity.
 */

import { computed, reactive, ref, shallowRef, watch, markRaw } from 'vue'
import type { InferenceEngine, ModelSource, LoadResult, ChatMessage } from '@/inference/engine'
import { EngineError, classifyError } from '@/inference/engine'
import {
  DEFAULT_GENERATION,
  DEFAULT_SAMPLING,
  type Backend,
  type BackendPreference,
  type Capabilities,
  type CatalogModel,
  type GenerationSettings,
  type LoadedModelInfo,
  type ModelDtype,
  type PromptExample,
  type SamplingSettings,
  type TokenizerSnapshot,
} from '@/types'
import {
  adjustDistribution,
  isGreedy,
  prepareLogits,
  sampleFromAdjusted,
  validateSamplingSettings,
  type AdjustedDistribution,
  type PreparedLogits,
} from '@/sampling/pipeline'
import { Sfc32, seedToState, type PrngState } from '@/sampling/prng'
import {
  appendToken as treeAppend,
  canRedo,
  canUndo,
  commitEdit,
  createTree,
  current as currentNode,
  goTo,
  redo as treeRedo,
  switchBranch as treeSwitchBranch,
  undo as treeUndo,
  type AppendSource,
  type HistoryTree,
} from '@/history/tree'
import {
  checkCompatibility,
  parseExperiment,
  serializeExperiment,
  type CompatibilityReport,
  type ExperimentModelRef,
} from '@/history/experiment'
import { contextKey, insertAt, invalidIds, removeAt, sequencesEqual, assertValidTokenId } from '@/tokenization/sequence'
import { catalogModel, defaultVariant } from '@/models/catalog'

export type ModelStatus = 'idle' | 'loading' | 'ready' | 'error'
export type InferenceStatus = 'idle' | 'pending' | 'running' | 'ready' | 'error'

export interface FileProgress {
  loaded: number
  total: number
  done: boolean
}

export interface Notice {
  id: number
  level: 'info' | 'warning' | 'error' | 'success'
  message: string
}

export interface InferenceResult {
  requestId: number
  contextKey: string
  tokens: number[]
  logits: Float32Array
  outputDims: number[]
  outputDtype: string
  inferenceMs: number
  usedKvCache: boolean
  tokensProcessed: number
  fromResultCache: boolean
}

export interface LastPick {
  tokenId: number
  source: AppendSource
  originalProbability: number
  adjustedProbability: number
  logit: number
  uniform: number | null
  position: number
}

const RESULT_CACHE_SIZE = 24

export function resolveBackend(pref: BackendPreference, caps: Capabilities | null): Backend {
  if (pref === 'wasm') return 'wasm'
  if (pref === 'webgpu') return 'webgpu'
  return caps?.webgpu && !caps.webgpuAdapter?.isFallbackAdapter ? 'webgpu' : 'wasm'
}

/** Choose a dtype that the backend + adapter can run (f16 variants need shader-f16 on WebGPU). */
export function chooseDtype(model: CatalogModel, backend: Backend, caps: Capabilities | null): ModelDtype {
  const f16ok = backend !== 'webgpu' || !!caps?.webgpuAdapter?.shaderF16
  const preferred = defaultVariant(model, backend)
  if (preferred && (f16ok || !preferred.dtype.includes('16'))) return preferred.dtype
  const alt = model.variants.find((v) => v.backends.includes(backend) && !v.dtype.includes('16'))
  return (alt ?? model.variants[0]!).dtype
}

export function createWorkbench(engine: InferenceEngine, opts: { appVersion?: string } = {}) {
  // ---------------------------------------------------------------- model
  const capabilities = shallowRef<Capabilities | null>(null)
  const backendPref = ref<BackendPreference>('auto')
  const modelStatus = ref<ModelStatus>('idle')
  const modelInfo = shallowRef<LoadedModelInfo | null>(null)
  const tokenizer = shallowRef<TokenizerSnapshot | null>(null)
  const suggestedSampling = shallowRef<LoadResult['suggestedSampling']>(null)
  const loadingLabel = ref('')
  const loadingStage = ref('')
  const loadingTarget = shallowRef<{ label: string; backend: Backend; dtype: ModelDtype; expectedBytes?: number } | null>(null)
  const fileProgress = reactive<Record<string, FileProgress>>({})
  const modelError = shallowRef<EngineError | null>(null)
  const activeCatalogKey = ref<string | null>(null)

  // ------------------------------------------------------------- settings
  const sampling = reactive<SamplingSettings>({ ...DEFAULT_SAMPLING })
  const generation = reactive<GenerationSettings>({ ...DEFAULT_GENERATION })
  const addSpecialTokens = ref(true)

  // -------------------------------------------------------------- history
  const historyVersion = ref(0)
  let tree: HistoryTree = createTree([], null, seedToState(sampling.seed))
  const touch = () => historyVersion.value++
  const history = computed(() => {
    void historyVersion.value
    return tree
  })
  const tokens = computed<number[]>(() => {
    void historyVersion.value
    return currentNode(tree).tokens
  })
  const prng = shallowRef<PrngState>(seedToState(sampling.seed))

  // --------------------------------------------------------- text editing
  const textDraft = ref('')
  /** 'tokens' = textDraft was decoded from the token sequence; 'user' = typed by the user. */
  const textSource = ref<'tokens' | 'user'>('tokens')
  /** Whether re-tokenizing the decoded text reproduces the exact token IDs (null = unknown). */
  const roundTrips = ref<boolean | null>(null)
  const roundTripDiff = ref(0)
  const textLocked = ref(false)
  const tokenizeMs = ref<number | null>(null)
  const tokenizing = ref(false)
  let textTimer: ReturnType<typeof setTimeout> | null = null
  let textRequest = 0
  let decodeRequest = 0

  // ------------------------------------------------------------ inference
  const inferenceStatus = ref<InferenceStatus>('idle')
  const inferenceError = shallowRef<EngineError | null>(null)
  const result = shallowRef<InferenceResult | null>(null)
  const prepared = shallowRef<PreparedLogits | null>(null)
  const resultCache = new Map<string, InferenceResult>()
  let requestCounter = 0
  let latestRequest = 0
  let inferTimer: ReturnType<typeof setTimeout> | null = null
  let inflight: Promise<InferenceResult | null> | null = null
  let inflightKey: string | null = null
  const lastForwardMs = ref<number | null>(null)
  const forwardCount = ref(0)

  // ----------------------------------------------------------- generation
  const autoRunning = ref(false)
  const autoGenerated = ref(0)
  const autoStartedAt = ref<number | null>(null)
  const autoTokensPerSecond = ref<number | null>(null)
  const autoStopReason = ref<string | null>(null)
  const lastPick = shallowRef<LastPick | null>(null)
  let stopRequested = false

  // -------------------------------------------------------------- notices
  const notices = ref<Notice[]>([])
  let noticeId = 0
  function notify(level: Notice['level'], message: string, timeoutMs = level === 'error' ? 0 : 6000) {
    const id = ++noticeId
    notices.value = [...notices.value, { id, level, message }]
    if (timeoutMs > 0) setTimeout(() => dismiss(id), timeoutMs)
    return id
  }
  function dismiss(id: number) {
    notices.value = notices.value.filter((n) => n.id !== id)
  }

  // ------------------------------------------------------------ derived
  const identity = computed(() => modelInfo.value?.identity ?? 'none')
  const currentKey = computed(() => contextKey(identity.value, tokens.value))
  const canInfer = computed(() => modelStatus.value === 'ready' && !!modelInfo.value && !modelInfo.value.tokenizerOnly)
  const invalidTokenIds = computed(() => (modelInfo.value ? invalidIds(tokens.value, modelInfo.value.logitsSize) : []))
  const samplingErrors = computed(() => validateSamplingSettings(sampling))
  const resultIsCurrent = computed(() => !!result.value && result.value.contextKey === currentKey.value)

  const samplingMs = ref<number | null>(null)
  const adjusted = computed<AdjustedDistribution | null>(() => {
    const p = prepared.value
    if (!p || samplingErrors.value.length) return null
    const t0 = performance.now()
    const a = adjustDistribution(p, { ...sampling })
    samplingMs.value = performance.now() - t0
    return markRaw(a)
  })

  const stopTokenIds = computed(() => modelInfo.value?.stopTokenIds ?? [])
  const canUndoNow = computed(() => {
    void historyVersion.value
    return canUndo(tree)
  })
  const canRedoNow = computed(() => {
    void historyVersion.value
    return canRedo(tree)
  })

  // ===================================================================
  // Model loading
  // ===================================================================

  async function init() {
    try {
      capabilities.value = markRaw(await engine.capabilities())
    } catch (e) {
      notify('warning', `Could not detect browser capabilities: ${(e as Error).message}`)
    }
  }

  function resetProgress() {
    for (const k of Object.keys(fileProgress)) delete fileProgress[k]
  }

  async function loadModel(
    source: ModelSource,
    backend: Backend,
    dtype: ModelDtype,
    options: { tokenizerOnly?: boolean } = {},
  ): Promise<boolean> {
    if (modelStatus.value === 'loading') return false
    stopAuto('model change')
    modelStatus.value = 'loading'
    modelError.value = null
    loadingLabel.value = source.displayName
    loadingStage.value = 'Starting…'
    loadingTarget.value = {
      label: source.displayName,
      backend,
      dtype,
      ...(source.kind === 'hub' && source.expectedBytes ? { expectedBytes: source.expectedBytes } : {}),
    }
    resetProgress()
    clearInference()
    try {
      const res = await engine.load({ source, device: backend, dtype, tokenizerOnly: !!options.tokenizerOnly }, (e) => {
        switch (e.stage) {
          case 'initiate':
          case 'download':
            if (!fileProgress[e.file]) fileProgress[e.file] = { loaded: 0, total: 0, done: false }
            loadingStage.value = `Fetching ${e.file}`
            break
          case 'progress':
            fileProgress[e.file] = { loaded: e.loaded, total: e.total, done: false }
            loadingStage.value = `Downloading ${e.file}`
            break
          case 'done':
            if (fileProgress[e.file]) fileProgress[e.file]!.done = true
            else fileProgress[e.file] = { loaded: 0, total: 0, done: true }
            break
          case 'session':
          case 'tokenizer':
            loadingStage.value = e.message
            break
          case 'total':
            break
        }
      })
      modelInfo.value = markRaw(res.info)
      tokenizer.value = markRaw(res.tokenizer)
      suggestedSampling.value = res.suggestedSampling
      activeCatalogKey.value = source.kind === 'hub' ? (source.catalogKey ?? null) : null
      modelStatus.value = 'ready'
      loadingStage.value = ''
      addSpecialTokens.value = true
      resultCache.clear()
      // Re-validate the existing context against the new model.
      const bad = invalidIds(tokens.value, res.info.logitsSize)
      if (bad.length) {
        notify('warning', `${bad.length} token ID(s) in the current context are outside this model's range. Clear the context or load a matching model.`)
      } else if (tokens.value.length) {
        notify('info', 'The context keeps its token IDs. Different models use different vocabularies, so the same IDs may mean different text.')
      }
      await refreshDecodedText()
      if (!res.info.tokenizerOnly) scheduleInference(0)
      return true
    } catch (e) {
      const err = classifyError(e)
      modelError.value = markRaw(err)
      if (err.kind !== 'aborted') notify('error', `Loading failed: ${err.message}`)
      // The engine unloads the previous model before loading a new one.
      modelInfo.value = null
      tokenizer.value = null
      modelStatus.value = 'error'
      return false
    } finally {
      loadingTarget.value = null
    }
  }

  async function loadCatalog(key: string, options: { dtype?: ModelDtype; backend?: Backend; tokenizerOnly?: boolean } = {}) {
    const model = catalogModel(key)
    if (!model) throw new Error(`Unknown catalog model ${key}`)
    const backend = options.backend ?? resolveBackend(backendPref.value, capabilities.value)
    const dtype = options.dtype ?? chooseDtype(model, backend, capabilities.value)
    const variant = model.variants.find((v) => v.dtype === dtype)
    const ok = await loadModel(
      {
        kind: 'hub',
        repo: model.repo,
        revision: model.revision,
        catalogKey: model.key,
        displayName: model.name,
        ...(variant ? { expectedBytes: variant.sizeBytes } : {}),
      },
      backend,
      dtype,
      { tokenizerOnly: options.tokenizerOnly },
    )
    if (ok && tokens.value.length === 0) {
      const ex = model.examples.find((e) => e.id === 'completion') ?? model.examples[0]
      if (ex) await loadExample(ex)
    }
    return ok
  }

  function cancelLoad() {
    if (modelStatus.value !== 'loading') return
    engine.terminate()
    modelStatus.value = 'idle'
    modelInfo.value = null
    tokenizer.value = null
    loadingStage.value = ''
    notify('info', 'Model loading cancelled. Files that finished downloading remain cached.')
  }

  async function unloadModel() {
    stopAuto('model unloaded')
    await engine.unload()
    modelInfo.value = null
    tokenizer.value = null
    modelStatus.value = 'idle'
    clearInference()
  }

  // ===================================================================
  // Context editing
  // ===================================================================

  function setTokens(next: number[], label: string, opts: { coalesce?: boolean; fromText?: boolean } = {}) {
    commitEdit(tree, next, {
      modelIdentity: modelInfo.value?.identity ?? null,
      prngState: prng.value,
      label,
      coalesce: opts.coalesce ?? false,
    })
    touch()
    if (!opts.fromText) void refreshDecodedText()
    scheduleInference()
  }

  /** User typed in the text editor: debounce, re-tokenize the whole text, replace the context. */
  function setText(text: string) {
    textDraft.value = text
    textSource.value = 'user'
    roundTrips.value = null
    if (textTimer) clearTimeout(textTimer)
    textTimer = setTimeout(() => void tokenizeDraft(), Math.min(250, generation.debounceMs))
  }

  async function tokenizeDraft(): Promise<void> {
    if (!tokenizer.value) return
    const req = ++textRequest
    const text = textDraft.value
    tokenizing.value = true
    try {
      const { ids, ms } = await engine.tokenize(text, addSpecialTokens.value)
      if (req !== textRequest) return
      tokenizeMs.value = ms
      if (!sequencesEqual(ids, tokens.value)) setTokens(ids, 'Edited text (re-tokenized)', { coalesce: true, fromText: true })
      roundTrips.value = true
      roundTripDiff.value = 0
    } catch (e) {
      notify('error', `Tokenization failed: ${classifyError(e).message}`)
    } finally {
      if (req === textRequest) tokenizing.value = false
    }
  }

  /** Decode the authoritative tokens into the text editor and check round-tripping. */
  async function refreshDecodedText(): Promise<void> {
    if (!tokenizer.value) return
    const req = ++decodeRequest
    const ids = tokens.value
    try {
      const text = ids.length ? await engine.decode(ids) : ''
      if (req !== decodeRequest) return
      textDraft.value = text
      textSource.value = 'tokens'
      const { ids: re, ms } = await engine.tokenize(text, addSpecialTokens.value)
      if (req !== decodeRequest) return
      tokenizeMs.value = ms
      const same = sequencesEqual(re, ids)
      roundTrips.value = same
      roundTripDiff.value = same ? 0 : countDiff(re, ids)
      textLocked.value = !same
    } catch {
      if (req === decodeRequest) roundTrips.value = null
    }
  }

  function countDiff(a: number[], b: number[]) {
    let d = Math.abs(a.length - b.length)
    for (let i = 0; i < Math.min(a.length, b.length); i++) if (a[i] !== b[i]) d++
    return d
  }

  function unlockText() {
    textLocked.value = false
  }

  function validateId(id: number) {
    const size = modelInfo.value?.logitsSize ?? tokenizer.value?.raw.length ?? Infinity
    assertValidTokenId(id, size)
  }

  function insertToken(index: number, id: number) {
    validateId(id)
    setTokens(insertAt(tokens.value, index, [id]), `Inserted token ${id} at ${index}`)
  }

  function appendTokens(ids: number[], label: string) {
    ids.forEach(validateId)
    setTokens([...tokens.value, ...ids], label)
  }

  function removeToken(index: number) {
    setTokens(removeAt(tokens.value, index), `Removed token at ${index}`)
  }

  function clearContext() {
    setTokens([], 'Cleared context')
  }

  async function exampleTokens(ex: PromptExample): Promise<number[]> {
    const tok = tokenizer.value
    if (!tok) throw new EngineError('Load a model or tokenizer first', 'no-model')
    if (ex.kind === 'text') return (await engine.tokenize(ex.text ?? '', true)).ids
    if (ex.kind === 'chat') {
      const r = await engine.applyChatTemplate((ex.messages ?? []) as ChatMessage[], ex.addGenerationPrompt ?? true)
      return r.ids
    }
    const out: number[] = []
    for (const part of ex.parts ?? []) {
      if ('tokenId' in part) out.push(part.tokenId)
      else if ('special' in part) {
        const id = tok.raw.indexOf(part.special)
        if (id < 0) throw new EngineError(`Special token ${part.special} is not in this tokenizer`, 'invalid-input')
        out.push(id)
      } else out.push(...(await engine.tokenize(part.text, false)).ids)
    }
    return out
  }

  async function loadExample(ex: PromptExample) {
    try {
      const ids = await exampleTokens(ex)
      setTokens(ids, `Example: ${ex.label}`)
    } catch (e) {
      notify('error', `Could not build example: ${classifyError(e).message}`)
    }
  }

  async function applyChat(messages: ChatMessage[], addGenerationPrompt: boolean) {
    try {
      const r = await engine.applyChatTemplate(messages, addGenerationPrompt)
      setTokens(r.ids, 'Chat template')
    } catch (e) {
      notify('error', `Chat template failed: ${classifyError(e).message}`)
    }
  }

  // ===================================================================
  // Inference
  // ===================================================================

  function clearInference() {
    if (inferTimer) clearTimeout(inferTimer)
    latestRequest = ++requestCounter
    result.value = null
    prepared.value = null
    inferenceStatus.value = 'idle'
    inferenceError.value = null
  }

  function scheduleInference(delay = generation.debounceMs) {
    if (!canInfer.value) return
    if (inferTimer) clearTimeout(inferTimer)
    if (!generation.autoInfer && delay !== 0) {
      inferenceStatus.value = resultIsCurrent.value ? 'ready' : 'idle'
      return
    }
    inferenceStatus.value = resultIsCurrent.value ? 'ready' : 'pending'
    inferTimer = setTimeout(() => void runInference(), delay)
  }

  function adopt(r: InferenceResult) {
    result.value = markRaw(r)
    const tok = tokenizer.value
    const unmapped: number[] = []
    if (tok) for (let i = 0; i < Math.min(tok.raw.length, r.logits.length); i++) if (!tok.raw[i]) unmapped.push(i)
    prepared.value = markRaw(prepareLogits(r.logits, { mappedCount: tok?.raw.length ?? r.logits.length, unmappedIds: unmapped }))
    inferenceStatus.value = 'ready'
    inferenceError.value = null
  }

  /** Run (or reuse) a forward pass for the current context. Returns null if stale/blocked. */
  async function runInference(force = false): Promise<InferenceResult | null> {
    if (inferTimer) {
      clearTimeout(inferTimer)
      inferTimer = null
    }
    if (!canInfer.value) return null
    const ids = [...tokens.value]
    const key = contextKey(identity.value, ids)
    if (!force && result.value?.contextKey === key) {
      inferenceStatus.value = 'ready'
      return result.value
    }
    const cached = resultCache.get(key)
    if (!force && cached) {
      adopt({ ...cached, fromResultCache: true })
      return result.value
    }
    if (invalidTokenIds.value.length) {
      inferenceStatus.value = 'error'
      inferenceError.value = markRaw(new EngineError(`Context contains token IDs outside the model's range: ${invalidTokenIds.value.slice(0, 5).join(', ')}`, 'invalid-input'))
      return null
    }
    if (ids.length === 0) {
      inferenceStatus.value = 'idle'
      result.value = null
      prepared.value = null
      return null
    }
    if (inflight && inflightKey === key) return inflight
    const requestId = ++requestCounter
    latestRequest = requestId
    inferenceStatus.value = 'running'
    const run = (async () => {
      try {
        const out = await engine.forward(ids, { useKvCache: generation.useKvCache })
        forwardCount.value++
        lastForwardMs.value = out.inferenceMs
        const r: InferenceResult = {
          requestId,
          contextKey: key,
          tokens: ids,
          logits: out.logits,
          outputDims: out.outputDims,
          outputDtype: out.outputDtype,
          inferenceMs: out.inferenceMs,
          usedKvCache: out.usedKvCache,
          tokensProcessed: out.tokensProcessed,
          fromResultCache: false,
        }
        resultCache.set(key, r)
        if (resultCache.size > RESULT_CACHE_SIZE) resultCache.delete(resultCache.keys().next().value as string)
        // Stale-result protection: only adopt if this is still the newest request
        // AND the context has not changed while the network was running.
        if (requestId !== latestRequest || key !== currentKey.value) return null
        adopt(r)
        return r
      } catch (e) {
        if (requestId !== latestRequest) return null
        const err = classifyError(e)
        inferenceStatus.value = 'error'
        inferenceError.value = markRaw(err)
        return null
      } finally {
        if (inflightKey === key) {
          inflight = null
          inflightKey = null
        }
      }
    })()
    inflight = run
    inflightKey = key
    return run
  }

  // ===================================================================
  // Token selection
  // ===================================================================

  function probabilityInfo(id: number) {
    const p = prepared.value
    const a = adjusted.value
    return {
      originalProbability: p?.original.probs[id] ?? 0,
      adjustedProbability: a?.probs[id] ?? 0,
      logit: p?.logits[id] ?? NaN,
    }
  }

  function append(id: number, source: AppendSource, uniform: number | null, prngBefore: PrngState | null) {
    validateId(id)
    const info = probabilityInfo(id)
    const position = tokens.value.length
    treeAppend(
      tree,
      {
        tokenId: id,
        source,
        originalProbability: resultIsCurrent.value ? info.originalProbability : null,
        adjustedProbability: resultIsCurrent.value ? info.adjustedProbability : null,
        logit: resultIsCurrent.value && Number.isFinite(info.logit) ? info.logit : null,
        sampling: source === 'manual' ? null : { ...sampling },
        prngBefore,
        uniform,
      },
      {
        modelIdentity: modelInfo.value?.identity ?? null,
        prngState: prng.value,
        label: `${source === 'manual' ? 'Picked' : source === 'greedy' ? 'Greedy' : source === 'auto' ? 'Auto' : 'Sampled'} ${id}`,
      },
    )
    touch()
    lastPick.value = {
      tokenId: id,
      source,
      originalProbability: info.originalProbability,
      adjustedProbability: info.adjustedProbability,
      logit: info.logit,
      uniform,
      position,
    }
    void refreshDecodedText()
    scheduleInference(0)
  }

  /** Manually choose ANY valid token (not limited to the top candidates). */
  function pickToken(id: number) {
    try {
      append(id, 'manual', null, null)
    } catch (e) {
      notify('error', (e as Error).message)
    }
  }

  async function ensureResult(): Promise<boolean> {
    if (resultIsCurrent.value) return true
    const r = await runInference()
    return !!r && r.contextKey === currentKey.value
  }

  /** Sample exactly one token from the adjusted distribution (greedy if configured). */
  async function sampleOne(source: AppendSource = 'sample'): Promise<number | null> {
    if (!(await ensureResult())) {
      if (!inferenceError.value) notify('warning', 'No next-token distribution available for this context.')
      return null
    }
    const a = adjusted.value
    if (!a || a.keptCount === 0) {
      notify('error', samplingErrors.value[0] ?? 'No eligible tokens to sample.')
      return null
    }
    if (isGreedy(sampling)) {
      const id = a.kept[0] as number
      append(id, source === 'sample' ? 'greedy' : source, null, null)
      return id
    }
    const before = prng.value
    const g = new Sfc32(before)
    const u = g.nextFloat()
    const id = sampleFromAdjusted(a, u)
    prng.value = g.getState()
    append(id, source, u, before)
    return id
  }

  /** Append the single most likely token (argmax of the raw logits over eligible IDs). */
  async function greedyOne(): Promise<number | null> {
    if (!(await ensureResult())) return null
    const id = prepared.value?.order[0]
    if (id === undefined) return null
    append(id, 'greedy', null, null)
    return id
  }

  // ===================================================================
  // Automatic generation
  // ===================================================================

  async function startAuto() {
    if (autoRunning.value || !canInfer.value) return
    autoRunning.value = true
    stopRequested = false
    autoGenerated.value = 0
    autoStopReason.value = null
    autoStartedAt.value = performance.now()
    autoTokensPerSecond.value = null
    const max = generation.maxNewTokens
    try {
      while (!stopRequested && autoGenerated.value < max) {
        const id = await sampleOne('auto')
        if (id === null) {
          autoStopReason.value = inferenceError.value ? `Inference error: ${inferenceError.value.message}` : 'No distribution available'
          break
        }
        autoGenerated.value++
        const elapsed = (performance.now() - (autoStartedAt.value ?? 0)) / 1000
        autoTokensPerSecond.value = elapsed > 0 ? autoGenerated.value / elapsed : null
        if (generation.stopAtStopTokens && stopTokenIds.value.includes(id)) {
          autoStopReason.value = `Stop token ${id} (${tokenizer.value?.raw[id] ?? ''}) generated`
          break
        }
        const ctx = modelInfo.value?.contextLength
        if (ctx && tokens.value.length >= ctx) {
          autoStopReason.value = 'Context length limit reached'
          break
        }
      }
      if (!autoStopReason.value) autoStopReason.value = stopRequested ? 'Stopped by user' : `Reached maximum of ${max} tokens`
      // Compute the distribution for the final context so the user can continue manually.
      if (!stopRequested) void runInference()
    } finally {
      autoRunning.value = false
    }
  }

  function stopAuto(reason = 'Stopped by user') {
    if (!autoRunning.value) return
    stopRequested = true
    autoStopReason.value = reason
  }

  // ===================================================================
  // History navigation
  // ===================================================================

  function afterNavigate() {
    const n = currentNode(tree)
    if (n.prngState) prng.value = n.prngState
    touch()
    void refreshDecodedText()
    scheduleInference(0)
  }

  function undo() {
    stopAuto()
    treeUndo(tree)
    afterNavigate()
  }
  function redo() {
    stopAuto()
    treeRedo(tree)
    afterNavigate()
  }
  function goToNode(id: string) {
    stopAuto()
    goTo(tree, id)
    afterNavigate()
  }
  function switchBranch(delta: number) {
    stopAuto()
    treeSwitchBranch(tree, delta)
    afterNavigate()
  }

  function resetSeed(seed = sampling.seed) {
    sampling.seed = seed
    prng.value = seedToState(seed)
  }

  watch(
    () => sampling.seed,
    (s) => {
      if (Number.isInteger(s) && s >= 0 && s <= 0xffffffff) prng.value = seedToState(s)
    },
  )

  watch(
    () => generation.autoInfer,
    (on) => {
      if (on) scheduleInference(0)
    },
  )

  watch(addSpecialTokens, () => {
    if (textSource.value === 'user') void tokenizeDraft()
    else void refreshDecodedText()
  })

  // ===================================================================
  // Experiments
  // ===================================================================

  function modelRef(): ExperimentModelRef | null {
    const info = modelInfo.value
    if (!info) return null
    return {
      identity: info.identity,
      displayName: info.displayName,
      source: info.source,
      ...(info.catalogKey ? { catalogKey: info.catalogKey } : {}),
      ...(info.repo ? { repo: info.repo } : {}),
      ...(info.revision ? { revision: info.revision } : {}),
      dtype: info.dtype,
      logitsSize: info.logitsSize,
      tokenizerSize: info.tokenizerSize,
      ...(tokenizer.value ? { tokenizerFingerprint: tokenizer.value.fingerprint } : {}),
    }
  }

  function exportExperiment(): string {
    const file = serializeExperiment({
      tree,
      model: modelRef(),
      sampling: { ...sampling },
      generation: { ...generation },
      appVersion: opts.appVersion ?? 'dev',
    })
    return JSON.stringify(file, null, 2)
  }

  const importReport = shallowRef<CompatibilityReport | null>(null)
  const importedModel = shallowRef<ExperimentModelRef | null>(null)

  function importExperiment(text: string): CompatibilityReport {
    const { file, tree: imported } = parseExperiment(text)
    stopAuto()
    tree = imported
    Object.assign(sampling, file.sampling)
    Object.assign(generation, file.generation)
    const n = currentNode(tree)
    prng.value = n.prngState ?? seedToState(sampling.seed)
    importedModel.value = file.model
    const report = checkCompatibility(
      file,
      tree,
      modelInfo.value && tokenizer.value
        ? { identity: modelInfo.value.identity, logitsSize: modelInfo.value.logitsSize, tokenizerFingerprint: tokenizer.value.fingerprint }
        : null,
    )
    importReport.value = report
    resultCache.clear()
    clearInference()
    touch()
    void refreshDecodedText()
    scheduleInference(0)
    return report
  }

  function restoreHistory(t: HistoryTree) {
    tree = t
    touch()
  }

  return {
    // state
    capabilities,
    backendPref,
    modelStatus,
    modelInfo,
    tokenizer,
    suggestedSampling,
    loadingLabel,
    loadingStage,
    loadingTarget,
    fileProgress,
    modelError,
    activeCatalogKey,
    sampling,
    generation,
    addSpecialTokens,
    history,
    historyVersion,
    tokens,
    prng,
    textDraft,
    textSource,
    roundTrips,
    roundTripDiff,
    textLocked,
    tokenizeMs,
    tokenizing,
    inferenceStatus,
    inferenceError,
    result,
    prepared,
    adjusted,
    samplingMs,
    samplingErrors,
    resultIsCurrent,
    canInfer,
    invalidTokenIds,
    lastForwardMs,
    forwardCount,
    autoRunning,
    autoGenerated,
    autoStartedAt,
    autoTokensPerSecond,
    autoStopReason,
    lastPick,
    notices,
    stopTokenIds,
    canUndo: canUndoNow,
    canRedo: canRedoNow,
    importReport,
    importedModel,
    currentKey,
    // actions
    init,
    notify,
    dismiss,
    loadModel,
    loadCatalog,
    cancelLoad,
    unloadModel,
    setText,
    tokenizeDraft,
    refreshDecodedText,
    unlockText,
    setTokens,
    insertToken,
    appendTokens,
    removeToken,
    clearContext,
    loadExample,
    exampleTokens,
    applyChat,
    scheduleInference,
    runInference,
    pickToken,
    sampleOne,
    greedyOne,
    startAuto,
    stopAuto,
    undo,
    redo,
    goToNode,
    switchBranch,
    resetSeed,
    exportExperiment,
    importExperiment,
    restoreHistory,
    getTree: () => tree,
  }
}

export type Workbench = ReturnType<typeof createWorkbench>
