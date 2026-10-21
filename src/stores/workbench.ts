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
import { checkLogits } from '@/sampling/logitsHealth'
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
import {
  contextKey,
  insertAt,
  invalidIds,
  removeAt,
  sequencesEqual,
  assertValidTokenId,
} from '@/tokenization/sequence'
import { catalogModel, defaultVariant, supportsBackend } from '@/models/catalog'

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
  const a = caps?.webgpuAdapter
  // Software adapters (e.g. SwiftShader) are slower than WASM; only auto-pick real GPUs.
  const software =
    !!a?.isFallbackAdapter ||
    /swiftshader/i.test(`${a?.architecture ?? ''} ${a?.description ?? ''}`)
  return caps?.webgpu && !software ? 'webgpu' : 'wasm'
}

/** Choose a dtype that the backend + adapter can run (f16 variants need shader-f16 on WebGPU). */
export function chooseDtype(
  model: CatalogModel,
  backend: Backend,
  caps: Capabilities | null,
): ModelDtype {
  const f16ok = backend !== 'webgpu' || !!caps?.webgpuAdapter?.shaderF16
  const preferred = defaultVariant(model, backend)
  if (preferred && (f16ok || !preferred.dtype.includes('16'))) return preferred.dtype
  // Variants are listed most faithful first, so this is the best one without f16.
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
  const loadingTarget = shallowRef<{
    label: string
    backend: Backend
    dtype: ModelDtype
    expectedBytes?: number
  } | null>(null)
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
  /** Context length when the current/last automatic run started. */
  const autoStartIndex = ref(0)
  const lastPick = shallowRef<LastPick | null>(null)
  let stopRequested = false

  /** The example last loaded into the context (shown as "Loaded: …"); null after other resets. */
  const loadedExample = shallowRef<{ id: string; label: string; tokens: number[] } | null>(null)
  /**
   * Model of a history tree that was imported (experiment file or saved session) and not yet
   * paired with a loaded model. Loading that same model keeps the imported history instead
   * of resetting the workspace.
   */
  let pendingImport: { model: ExperimentModelRef | null } | null = null

  // -------------------------------------------------------------- notices
  const notices = ref<Notice[]>([])
  let noticeId = 0
  function notify(
    level: Notice['level'],
    message: string,
    timeoutMs = level === 'error' ? 0 : 6000,
  ) {
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
  const canInfer = computed(
    () => modelStatus.value === 'ready' && !!modelInfo.value && !modelInfo.value.tokenizerOnly,
  )
  const invalidTokenIds = computed(() =>
    modelInfo.value ? invalidIds(tokens.value, modelInfo.value.logitsSize) : [],
  )
  const samplingErrors = computed(() => validateSamplingSettings(sampling))
  const resultIsCurrent = computed(
    () => !!result.value && result.value.contextKey === currentKey.value,
  )

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
    options: { tokenizerOnly?: boolean; defaultExample?: PromptExample } = {},
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
      ...(source.kind === 'hub' && source.expectedBytes
        ? { expectedBytes: source.expectedBytes }
        : {}),
    }
    resetProgress()
    clearInference()
    try {
      const res = await engine.load(
        { source, device: backend, dtype, tokenizerOnly: !!options.tokenizerOnly },
        (e) => {
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
        },
      )
      modelInfo.value = markRaw(res.info)
      tokenizer.value = markRaw(res.tokenizer)
      suggestedSampling.value = res.suggestedSampling
      activeCatalogKey.value = source.kind === 'hub' ? (source.catalogKey ?? null) : null
      modelStatus.value = 'ready'
      loadingStage.value = ''
      addSpecialTokens.value = true
      resultCache.clear()
      const keepImported =
        !!pendingImport && (!pendingImport.model || sameModel(pendingImport.model, res.info))
      pendingImport = null
      if (keepImported) {
        const bad = invalidIds(tokens.value, res.info.logitsSize)
        if (bad.length)
          notify(
            'warning',
            `${bad.length} token ID(s) in the imported context are outside this model's range.`,
          )
      } else {
        // A new model starts from a clean workspace: token IDs mean different text in
        // different vocabularies, so the old context and history would be misleading.
        resetWorkspace()
        notify('info', 'Context cleared for new model.')
      }
      if (res.cacheWarnings?.length) {
        notify(
          'warning',
          `Some model files could not be saved in the browser cache, so the model will be downloaded again next time and is not available offline (${res.cacheWarnings[0]}).`,
          0,
        )
      }
      if (!keepImported && options.defaultExample) await loadExample(options.defaultExample)
      await refreshDecodedText()
      if (!res.info.tokenizerOnly) scheduleInference(0)
      return true
    } catch (e) {
      const err = classifyError(e)
      // The engine unloads the previous model before loading a new one.
      modelInfo.value = null
      tokenizer.value = null
      if (err.kind === 'aborted') {
        // Cancelled by the user (cancelLoad already reported it).
        modelError.value = null
        modelStatus.value = 'idle'
        return false
      }
      modelError.value = markRaw(err)
      notify('error', `Loading failed: ${err.message}`)
      modelStatus.value = 'error'
      return false
    } finally {
      loadingTarget.value = null
    }
  }

  async function loadCatalog(
    key: string,
    options: { dtype?: ModelDtype; backend?: Backend; tokenizerOnly?: boolean } = {},
  ) {
    const model = catalogModel(key)
    if (!model) throw new Error(`Unknown catalog model ${key}`)
    const backend = options.backend ?? resolveBackend(backendPref.value, capabilities.value)
    if (!options.tokenizerOnly && !supportsBackend(model, backend)) {
      notify(
        'error',
        model.backendNotes?.[backend] ??
          `${model.name} cannot run on ${backend === 'webgpu' ? 'WebGPU' : 'WASM'}.`,
      )
      return false
    }
    const dtype = options.dtype ?? chooseDtype(model, backend, capabilities.value)
    const variant = model.variants.find((v) => v.dtype === dtype)
    const defaultExample = model.defaultExampleId
      ? model.examples.find((e) => e.id === model.defaultExampleId)
      : undefined
    return loadModel(
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
      { tokenizerOnly: options.tokenizerOnly, ...(defaultExample ? { defaultExample } : {}) },
    )
  }

  /** Same model (repo + revision, or the same local identity), ignoring the dtype. */
  function sameModel(ref: ExperimentModelRef, info: LoadedModelInfo): boolean {
    if (ref.repo && info.repo) return ref.repo === info.repo && ref.revision === info.revision
    return ref.identity === info.identity
  }

  /** Empty context, a fresh history tree, generator restarted from the seed. */
  function resetWorkspace() {
    stopAuto('model change')
    cancelPendingText()
    decodeRequest++
    prng.value = seedToState(sampling.seed)
    tree = createTree([], modelInfo.value?.identity ?? null, prng.value)
    touch()
    textDraft.value = ''
    textSource.value = 'tokens'
    roundTrips.value = null
    textLocked.value = false
    lastPick.value = null
    loadedExample.value = null
    autoGenerated.value = 0
    autoStopReason.value = null
    clearInference()
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

  function setTokens(
    next: number[],
    label: string,
    opts: { coalesce?: boolean; fromText?: boolean } = {},
  ) {
    if (!opts.fromText) cancelPendingText()
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

  /** Drop a pending/in-flight re-tokenization of typed text (superseded by a token-level change). */
  function cancelPendingText() {
    if (textTimer) clearTimeout(textTimer)
    textTimer = null
    textRequest++
    tokenizing.value = false
  }

  /** User typed in the text editor: debounce, re-tokenize the whole text, replace the context. */
  function setText(text: string) {
    decodeRequest++ // an in-flight decode must not overwrite what the user just typed
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
      if (!sequencesEqual(ids, tokens.value))
        setTokens(ids, 'Edited text (re-tokenized)', { coalesce: true, fromText: true })
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
    loadedExample.value = null
  }

  async function exampleTokens(ex: PromptExample): Promise<number[]> {
    const tok = tokenizer.value
    if (!tok) throw new EngineError('Load a model or tokenizer first', 'no-model')
    if (ex.kind === 'text') return (await engine.tokenize(ex.text ?? '', true)).ids
    if (ex.kind === 'chat') {
      const r = await engine.applyChatTemplate(
        (ex.messages ?? []) as ChatMessage[],
        ex.addGenerationPrompt ?? true,
        ex.templateOptions,
      )
      return r.ids
    }
    const out: number[] = []
    for (const part of ex.parts ?? []) {
      if ('tokenId' in part) out.push(part.tokenId)
      else if ('special' in part) {
        const id = tok.raw.indexOf(part.special)
        if (id < 0)
          throw new EngineError(
            `Special token ${part.special} is not in this tokenizer`,
            'invalid-input',
          )
        out.push(id)
      } else out.push(...(await engine.tokenize(part.text, false)).ids)
    }
    return out
  }

  async function loadExample(ex: PromptExample) {
    try {
      const ids = await exampleTokens(ex)
      setTokens(ids, `Example: ${ex.label}`)
      loadedExample.value = { id: ex.id, label: ex.label, tokens: ids }
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
    inflight = null
    inflightKey = null
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
    if (tok)
      for (let i = 0; i < Math.min(tok.raw.length, r.logits.length); i++)
        if (!tok.raw[i]) unmapped.push(i)
    prepared.value = markRaw(
      prepareLogits(r.logits, {
        mappedCount: tok?.raw.length ?? r.logits.length,
        unmappedIds: unmapped,
      }),
    )
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
      inferenceError.value = markRaw(
        new EngineError(
          `Context contains token IDs outside the model's range: ${invalidTokenIds.value.slice(0, 5).join(', ')}`,
          'invalid-input',
        ),
      )
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
    // Declared first so the finally block can tell whether it is still the in-flight run.
    let run: Promise<InferenceResult | null> | undefined = undefined
    run = (async () => {
      try {
        const out = await engine.forward(ids, { useKvCache: generation.useKvCache })
        forwardCount.value++
        lastForwardMs.value = out.inferenceMs
        const problem = checkLogits(out.logits)
        if (problem) {
          // Never present NaN / constant logits as a distribution (constant → uniform).
          if (requestId !== latestRequest || key !== currentKey.value) return null
          const info = modelInfo.value
          result.value = null
          prepared.value = null
          inferenceStatus.value = 'error'
          inferenceError.value = markRaw(
            new EngineError(
              `${info ? `${info.dtype} on ${info.backend === 'webgpu' ? 'WebGPU' : 'WASM'}` : 'The model'} returned invalid next-token logits: ${problem.message}`,
              'degenerate-output',
              'This variant does not run correctly on this backend, so no distribution is shown. Try another quantization or backend.',
            ),
          )
          return null
        }
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
        if (resultCache.size > RESULT_CACHE_SIZE)
          resultCache.delete(resultCache.keys().next().value as string)
        // Stale-result protection: only adopt if this is still the newest request
        // AND the context has not changed while the network was running.
        if (requestId !== latestRequest || key !== currentKey.value) return null
        adopt(r)
        return r
      } catch (e) {
        if (requestId !== latestRequest || key !== currentKey.value) return null
        const err = classifyError(e)
        inferenceStatus.value = 'error'
        inferenceError.value = markRaw(err)
        return null
      } finally {
        if (inflight === run) {
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

  function append(
    id: number,
    source: AppendSource,
    uniform: number | null,
    prngBefore: PrngState | null,
  ) {
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
    const startKey = currentKey.value
    if (!(await ensureResult())) {
      if (!inferenceError.value)
        notify('warning', 'No next-token distribution available for this context.')
      return null
    }
    // Another action may have changed the context while we waited (e.g. key repeat).
    if (currentKey.value !== startKey || !resultIsCurrent.value) return null
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
    const startKey = currentKey.value
    if (!(await ensureResult())) return null
    if (currentKey.value !== startKey || !resultIsCurrent.value) return null
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
    autoStartIndex.value = tokens.value.length
    autoStopReason.value = null
    autoStartedAt.value = performance.now()
    autoTokensPerSecond.value = null
    const max = generation.maxNewTokens
    try {
      while (!stopRequested && autoGenerated.value < max) {
        const id = await sampleOne('auto')
        if (id === null) {
          autoStopReason.value = inferenceError.value
            ? `Inference error: ${inferenceError.value.message}`
            : 'No distribution available'
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
      if (!autoStopReason.value)
        autoStopReason.value = stopRequested
          ? 'Stopped by user'
          : `Reached maximum of ${max} tokens`
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
    cancelPendingText()
    const n = currentNode(tree)
    if (n.prngState) prng.value = n.prngState
    // Show the seed that produced this node's generator state, when recorded.
    const seed = n.append?.sampling?.seed
    if (seed !== undefined) sampling.seed = seed
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

  /**
   * Restart the generator from a seed. The new state is also stored on the current
   * history node, so navigating back here resumes the stream that matches the shown seed.
   * (Seed changes are explicit actions; importing settings never resets the generator.)
   */
  function resetSeed(seed = sampling.seed) {
    sampling.seed = seed
    prng.value = seedToState(seed)
    currentNode(tree).prngState = prng.value
  }

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
    cancelPendingText()
    tree = imported
    Object.assign(sampling, file.sampling)
    Object.assign(generation, file.generation)
    const n = currentNode(tree)
    prng.value = n.prngState ?? seedToState(sampling.seed)
    importedModel.value = file.model
    loadedExample.value = null
    // Loading the experiment's model next keeps the imported history (see loadModel).
    pendingImport =
      modelInfo.value && (!file.model || sameModel(file.model, modelInfo.value))
        ? null
        : { model: file.model }
    const report = checkCompatibility(
      file,
      tree,
      modelInfo.value && tokenizer.value
        ? {
            identity: modelInfo.value.identity,
            logitsSize: modelInfo.value.logitsSize,
            tokenizerFingerprint: tokenizer.value.fingerprint,
          }
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
    autoStartIndex,
    lastPick,
    loadedExample,
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
