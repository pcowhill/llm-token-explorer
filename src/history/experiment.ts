/**
 * Experiment file format (JSON). Documented in docs/experiment-format.md.
 *
 * Experiments contain token IDs, settings and history — never model weights and
 * never executable content. Imported files are validated field-by-field; unknown
 * fields are ignored; token IDs are preserved exactly (never re-tokenized).
 */

import { isPrngState, type PrngState } from '@/sampling/prng'
import { validateSamplingSettings } from '@/sampling/pipeline'
import { DEFAULT_GENERATION, type GenerationSettings, type SamplingSettings } from '@/types'
import type { AppendDetails, AppendSource, HistoryNode, HistoryTree, NodeKind } from './tree'

export const EXPERIMENT_FORMAT = 'llm-token-explorer/experiment'
export const EXPERIMENT_VERSION = 1

export const LIMITS = Object.freeze({
  maxFileBytes: 50 * 1024 * 1024,
  maxNodes: 20_000,
  maxTokensPerNode: 200_000,
  maxTokenId: 10_000_000,
  maxStringLength: 10_000,
})

export interface ExperimentModelRef {
  identity: string
  displayName: string
  source: 'catalog' | 'hub' | 'local'
  catalogKey?: string
  repo?: string
  revision?: string
  dtype?: string
  logitsSize?: number
  tokenizerSize?: number
  tokenizerFingerprint?: string
}

export interface ExperimentFile {
  format: typeof EXPERIMENT_FORMAT
  version: number
  createdAt: string
  app: { name: 'LLM Token Explorer'; version: string }
  model: ExperimentModelRef | null
  sampling: SamplingSettings
  generation: GenerationSettings
  history: {
    rootId: string
    currentId: string
    nodes: SerializedNode[]
  }
  notes?: string
}

export interface SerializedNode {
  id: string
  parentId: string | null
  kind: NodeKind
  tokens: number[]
  children: string[]
  preferredChild: string | null
  createdAt: string
  label: string
  modelIdentity: string | null
  prngState: [number, number, number, number] | null
  append?: AppendDetails
}

export function serializeExperiment(params: {
  tree: HistoryTree
  model: ExperimentModelRef | null
  sampling: SamplingSettings
  generation: GenerationSettings
  appVersion: string
  notes?: string
}): ExperimentFile {
  const nodes: SerializedNode[] = []
  for (const n of params.tree.nodes.values()) {
    nodes.push({
      id: n.id,
      parentId: n.parentId,
      kind: n.kind,
      tokens: [...n.tokens],
      children: [...n.children],
      preferredChild: n.preferredChild,
      createdAt: n.createdAt,
      label: n.label,
      modelIdentity: n.modelIdentity,
      prngState: n.prngState ? [...n.prngState] : null,
      ...(n.append
        ? {
            append: {
              ...n.append,
              prngBefore: n.append.prngBefore
                ? ([...n.append.prngBefore] as unknown as PrngState)
                : null,
            },
          }
        : {}),
    })
  }
  return {
    format: EXPERIMENT_FORMAT,
    version: EXPERIMENT_VERSION,
    createdAt: new Date().toISOString(),
    app: { name: 'LLM Token Explorer', version: params.appVersion },
    model: params.model,
    sampling: { ...params.sampling },
    generation: { ...params.generation },
    history: { rootId: params.tree.rootId, currentId: params.tree.currentId, nodes },
    ...(params.notes ? { notes: params.notes } : {}),
  }
}

export class ExperimentValidationError extends Error {
  constructor(public readonly problems: string[]) {
    super(
      `Invalid experiment file:\n- ${problems.slice(0, 12).join('\n- ')}${problems.length > 12 ? `\n… and ${problems.length - 12} more` : ''}`,
    )
    this.name = 'ExperimentValidationError'
  }
}

type Obj = Record<string, unknown>
const isObj = (v: unknown): v is Obj => typeof v === 'object' && v !== null && !Array.isArray(v)
const isStr = (v: unknown, max: number = LIMITS.maxStringLength): v is string =>
  typeof v === 'string' && v.length <= max
const isTokenId = (v: unknown): v is number =>
  Number.isInteger(v) && (v as number) >= 0 && (v as number) <= LIMITS.maxTokenId
const NODE_KINDS: NodeKind[] = ['root', 'edit', 'append']
const SOURCES: AppendSource[] = ['manual', 'sample', 'greedy', 'auto']
const ID_RE = /^[A-Za-z0-9_-]{1,64}$/

function numOrNull(v: unknown): number | null | undefined {
  if (v === null) return null
  if (typeof v === 'number' && Number.isFinite(v)) return v
  return undefined
}

function parseSampling(v: unknown, problems: string[]): SamplingSettings {
  if (!isObj(v)) {
    problems.push('sampling: must be an object')
    return { temperature: 1, topK: 0, topP: 1, minP: 0, greedy: false, seed: 0 }
  }
  const s: SamplingSettings = {
    temperature: Number(v.temperature),
    topK: Number(v.topK),
    topP: Number(v.topP),
    minP: Number(v.minP),
    greedy: v.greedy === true,
    seed: Number(v.seed),
  }
  if (typeof v.greedy !== 'boolean') problems.push('sampling.greedy: must be a boolean')
  for (const e of validateSamplingSettings(s)) problems.push(`sampling: ${e}`)
  return s
}

function parseGeneration(v: unknown, problems: string[]): GenerationSettings {
  if (!isObj(v)) return { ...DEFAULT_GENERATION }
  const g: GenerationSettings = {
    maxNewTokens: Number.isInteger(v.maxNewTokens)
      ? (v.maxNewTokens as number)
      : DEFAULT_GENERATION.maxNewTokens,
    stopAtStopTokens:
      typeof v.stopAtStopTokens === 'boolean'
        ? v.stopAtStopTokens
        : DEFAULT_GENERATION.stopAtStopTokens,
    debounceMs: Number.isInteger(v.debounceMs)
      ? (v.debounceMs as number)
      : DEFAULT_GENERATION.debounceMs,
    autoInfer: typeof v.autoInfer === 'boolean' ? v.autoInfer : DEFAULT_GENERATION.autoInfer,
    useKvCache: typeof v.useKvCache === 'boolean' ? v.useKvCache : DEFAULT_GENERATION.useKvCache,
  }
  if (g.maxNewTokens < 1 || g.maxNewTokens > 4096)
    problems.push('generation.maxNewTokens: must be 1–4096')
  if (g.debounceMs < 0 || g.debounceMs > 10_000)
    problems.push('generation.debounceMs: must be 0–10000')
  return g
}

function parseModel(v: unknown, problems: string[]): ExperimentModelRef | null {
  if (v === null || v === undefined) return null
  if (!isObj(v)) {
    problems.push('model: must be an object or null')
    return null
  }
  if (!isStr(v.identity, 512)) problems.push('model.identity: must be a string')
  if (!isStr(v.displayName, 512)) problems.push('model.displayName: must be a string')
  if (!['catalog', 'hub', 'local'].includes(v.source as string))
    problems.push('model.source: must be catalog | hub | local')
  const ref: ExperimentModelRef = {
    identity: String(v.identity ?? ''),
    displayName: String(v.displayName ?? ''),
    source: v.source as ExperimentModelRef['source'],
  }
  for (const k of ['catalogKey', 'repo', 'revision', 'dtype', 'tokenizerFingerprint'] as const) {
    if (v[k] !== undefined) {
      if (!isStr(v[k], 512)) problems.push(`model.${k}: must be a string`)
      else ref[k] = v[k] as string
    }
  }
  for (const k of ['logitsSize', 'tokenizerSize'] as const) {
    if (v[k] !== undefined) {
      if (!isTokenId(v[k])) problems.push(`model.${k}: must be a non-negative integer`)
      else ref[k] = v[k] as number
    }
  }
  return ref
}

function parseAppend(v: unknown, where: string, problems: string[]): AppendDetails | undefined {
  if (v === undefined) return undefined
  if (!isObj(v)) {
    problems.push(`${where}.append: must be an object`)
    return undefined
  }
  if (!isTokenId(v.tokenId))
    problems.push(`${where}.append.tokenId: must be a non-negative integer`)
  if (!SOURCES.includes(v.source as AppendSource)) problems.push(`${where}.append.source: invalid`)
  const prngBefore =
    v.prngBefore === null || v.prngBefore === undefined
      ? null
      : isPrngState(v.prngBefore)
        ? (v.prngBefore as PrngState)
        : undefined
  if (prngBefore === undefined)
    problems.push(`${where}.append.prngBefore: must be null or 4 uint32 values`)
  let sampling: SamplingSettings | null = null
  if (v.sampling !== null && v.sampling !== undefined)
    sampling = parseSampling(v.sampling, problems)
  const fields = ['originalProbability', 'adjustedProbability', 'logit', 'uniform'] as const
  const nums: Partial<Record<(typeof fields)[number], number | null>> = {}
  for (const f of fields) {
    const n = numOrNull(v[f] ?? null)
    if (n === undefined) problems.push(`${where}.append.${f}: must be a finite number or null`)
    nums[f] = n ?? null
  }
  return {
    tokenId: v.tokenId as number,
    source: v.source as AppendSource,
    originalProbability: nums.originalProbability ?? null,
    adjustedProbability: nums.adjustedProbability ?? null,
    logit: nums.logit ?? null,
    uniform: nums.uniform ?? null,
    sampling,
    prngBefore: prngBefore ?? null,
  }
}

export interface ParsedExperiment {
  file: ExperimentFile
  tree: HistoryTree
}

/** Parse + validate JSON text. Throws ExperimentValidationError with all problems found. */
export function parseExperiment(text: string): ParsedExperiment {
  if (text.length > LIMITS.maxFileBytes) throw new ExperimentValidationError(['File is too large'])
  let data: unknown
  try {
    data = JSON.parse(text)
  } catch (e) {
    throw new ExperimentValidationError([`Not valid JSON: ${(e as Error).message}`])
  }
  return validateExperiment(data)
}

export function validateExperiment(data: unknown): ParsedExperiment {
  const problems: string[] = []
  if (!isObj(data)) throw new ExperimentValidationError(['Top level must be a JSON object'])
  if (data.format !== EXPERIMENT_FORMAT) problems.push(`format: expected "${EXPERIMENT_FORMAT}"`)
  if (data.version !== EXPERIMENT_VERSION)
    problems.push(
      `version: unsupported version ${String(data.version)} (expected ${EXPERIMENT_VERSION})`,
    )
  const model = parseModel(data.model, problems)
  const sampling = parseSampling(data.sampling, problems)
  const generation = parseGeneration(data.generation, problems)
  const history = data.history
  if (!isObj(history)) {
    problems.push('history: must be an object')
    throw new ExperimentValidationError(problems)
  }
  if (!Array.isArray(history.nodes)) {
    problems.push('history.nodes: must be an array')
    throw new ExperimentValidationError(problems)
  }
  if (history.nodes.length === 0) problems.push('history.nodes: must not be empty')
  if (history.nodes.length > LIMITS.maxNodes)
    problems.push(`history.nodes: too many nodes (max ${LIMITS.maxNodes})`)

  const nodes = new Map<string, HistoryNode>()
  history.nodes.slice(0, LIMITS.maxNodes).forEach((raw: unknown, i: number) => {
    const where = `history.nodes[${i}]`
    if (!isObj(raw)) {
      problems.push(`${where}: must be an object`)
      return
    }
    if (!isStr(raw.id) || !ID_RE.test(raw.id)) {
      problems.push(`${where}.id: invalid`)
      return
    }
    if (nodes.has(raw.id)) problems.push(`${where}.id: duplicate id ${raw.id}`)
    if (!(raw.parentId === null || (isStr(raw.parentId) && ID_RE.test(raw.parentId))))
      problems.push(`${where}.parentId: invalid`)
    if (!NODE_KINDS.includes(raw.kind as NodeKind)) problems.push(`${where}.kind: invalid`)
    if (
      !Array.isArray(raw.tokens) ||
      raw.tokens.length > LIMITS.maxTokensPerNode ||
      !raw.tokens.every(isTokenId)
    ) {
      problems.push(`${where}.tokens: must be an array of non-negative integer token IDs`)
    }
    if (!Array.isArray(raw.children) || !raw.children.every((c) => isStr(c) && ID_RE.test(c)))
      problems.push(`${where}.children: invalid`)
    if (!(
      raw.preferredChild === null ||
      raw.preferredChild === undefined ||
      (isStr(raw.preferredChild) && ID_RE.test(raw.preferredChild))
    ))
      problems.push(`${where}.preferredChild: invalid`)
    const prng =
      raw.prngState === null || raw.prngState === undefined
        ? null
        : isPrngState(raw.prngState)
          ? (raw.prngState as PrngState)
          : undefined
    if (prng === undefined) problems.push(`${where}.prngState: must be null or 4 uint32 values`)
    const append = parseAppend(raw.append, where, problems)
    nodes.set(raw.id, {
      id: raw.id,
      parentId: (raw.parentId as string | null) ?? null,
      kind: raw.kind as NodeKind,
      tokens: Array.isArray(raw.tokens) ? (raw.tokens as number[]).slice() : [],
      children: Array.isArray(raw.children) ? (raw.children as string[]).slice() : [],
      preferredChild: (raw.preferredChild as string | null | undefined) ?? null,
      createdAt: isStr(raw.createdAt, 64) ? raw.createdAt : '',
      label: isStr(raw.label, 500) ? raw.label : '',
      modelIdentity: isStr(raw.modelIdentity, 512) ? raw.modelIdentity : null,
      prngState: prng ?? null,
      ...(append ? { append } : {}),
    })
  })

  // Structural checks: single root, consistent parent/child links, no cycles.
  const rootId = history.rootId
  const currentId = history.currentId
  if (!isStr(rootId) || !nodes.has(rootId)) problems.push('history.rootId: must reference a node')
  if (!isStr(currentId) || !nodes.has(currentId))
    problems.push('history.currentId: must reference a node')
  for (const n of nodes.values()) {
    if (n.id === rootId) {
      if (n.parentId !== null) problems.push(`node ${n.id}: root must have parentId null`)
    } else if (n.parentId === null) {
      problems.push(`node ${n.id}: only the root may have parentId null`)
    } else {
      const p = nodes.get(n.parentId)
      if (!p) problems.push(`node ${n.id}: parent ${n.parentId} does not exist`)
      else if (!p.children.includes(n.id))
        problems.push(`node ${n.id}: parent ${n.parentId} does not list it as a child`)
    }
    for (const c of n.children) {
      const child = nodes.get(c)
      if (!child) problems.push(`node ${n.id}: child ${c} does not exist`)
      else if (child.parentId !== n.id)
        problems.push(`node ${n.id}: child ${c} has a different parent`)
    }
    if (n.preferredChild && !n.children.includes(n.preferredChild)) n.preferredChild = null
    if (n.kind === 'root' && n.id !== rootId)
      problems.push(`node ${n.id}: only the root may be kind "root"`)
    if (n.kind === 'append') {
      if (!n.append) problems.push(`node ${n.id}: append node without append details`)
      else if (n.tokens[n.tokens.length - 1] !== n.append.tokenId)
        problems.push(`node ${n.id}: last token does not match append.tokenId`)
      const parent = n.parentId ? nodes.get(n.parentId) : undefined
      if (
        parent &&
        (n.tokens.length !== parent.tokens.length + 1 ||
          parent.tokens.some((t, i) => n.tokens[i] !== t))
      )
        problems.push(`node ${n.id}: tokens must equal the parent's tokens plus the appended token`)
    }
  }
  if (isStr(rootId) && nodes.has(rootId)) {
    const seen = new Set<string>()
    const stack = [rootId]
    while (stack.length) {
      const id = stack.pop() as string
      if (seen.has(id)) {
        problems.push(`history: cycle or shared child at ${id}`)
        break
      }
      seen.add(id)
      for (const c of nodes.get(id)?.children ?? []) stack.push(c)
    }
    if (seen.size !== nodes.size)
      problems.push('history: some nodes are not reachable from the root')
  }

  if (problems.length) throw new ExperimentValidationError(problems)

  let maxNum = 0
  for (const id of nodes.keys()) {
    // Only small numeric ids participate in numbering; huge ones cannot collide with it.
    const m = /^n(\d{1,9})$/.exec(id)
    if (m) maxNum = Math.max(maxNum, Number(m[1]))
  }
  const tree: HistoryTree = {
    rootId: rootId as string,
    currentId: currentId as string,
    nodes,
    nextId: maxNum + 1,
  }
  // Avoid id clashes with imported non-"nN" ids.
  while (nodes.has(`n${tree.nextId}`)) tree.nextId++

  const file: ExperimentFile = {
    format: EXPERIMENT_FORMAT,
    version: EXPERIMENT_VERSION,
    createdAt: isStr(data.createdAt, 64) ? data.createdAt : '',
    app: {
      name: 'LLM Token Explorer',
      version: isObj(data.app) && isStr(data.app.version, 64) ? data.app.version : 'unknown',
    },
    model,
    sampling,
    generation,
    history: { rootId: rootId as string, currentId: currentId as string, nodes: [] },
    ...(isStr(data.notes) ? { notes: data.notes } : {}),
  }
  return { file, tree }
}

export interface CompatibilityReport {
  sameModel: boolean
  sameTokenizer: boolean | null
  /** Token IDs that are outside the loaded model's valid input range. */
  outOfRangeIds: number[]
  messages: string[]
}

/** Compare an imported experiment with the currently loaded model (or none). */
export function checkCompatibility(
  file: ExperimentFile,
  tree: HistoryTree,
  loaded: { identity: string; logitsSize: number; tokenizerFingerprint: string } | null,
): CompatibilityReport {
  const messages: string[] = []
  if (!loaded) {
    return {
      sameModel: false,
      sameTokenizer: null,
      outOfRangeIds: [],
      messages: [
        file.model
          ? `This experiment was created with “${file.model.displayName}”. Load that model to run inference; token IDs are preserved as-is.`
          : 'No model is loaded. Token IDs are preserved; load a model to run inference.',
      ],
    }
  }
  const sameModel = file.model?.identity === loaded.identity
  const sameTokenizer = file.model?.tokenizerFingerprint
    ? file.model.tokenizerFingerprint === loaded.tokenizerFingerprint
    : null
  const bad = new Set<number>()
  for (const n of tree.nodes.values())
    for (const t of n.tokens) if (t >= loaded.logitsSize) bad.add(t)
  if (!sameModel && file.model)
    messages.push(
      `Experiment model “${file.model.displayName}” differs from the loaded model. Token IDs are NOT re-tokenized and may mean different text.`,
    )
  if (sameTokenizer === false)
    messages.push('The tokenizer vocabulary differs from the one used to create this experiment.')
  if (bad.size)
    messages.push(
      `${bad.size} token ID(s) are outside the loaded model's range (0–${loaded.logitsSize - 1}). Inference is blocked for contexts containing them.`,
    )
  return { sameModel, sameTokenizer, outOfRangeIds: [...bad].sort((a, b) => a - b), messages }
}
