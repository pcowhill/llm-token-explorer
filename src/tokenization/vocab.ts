/**
 * Full-vocabulary search, sort and pagination.
 *
 * Works on parallel arrays (no per-row objects for 260k-entry vocabularies) and only
 * materializes rows for the visible page. Probabilities are never truncated: the
 * full distribution arrays are passed in and only *displayed* rows are sliced.
 */

export type VocabSortKey = 'probability' | 'adjusted' | 'logit' | 'id' | 'alphabetical'
export type SortDirection = 'asc' | 'desc'
export type VocabSearchField = 'any' | 'decoded' | 'raw' | 'id'
export type VocabFilter = 'all' | 'special' | 'regular' | 'eligible' | 'filtered-out'

export interface VocabSource {
  /** Total number of IDs to browse: max(tokenizerSize, logitsSize). */
  size: number
  raw: readonly string[]
  decoded: readonly string[]
  isSpecial: (id: number) => boolean
}

export interface VocabDistribution {
  logits: Float32Array
  original: Float64Array
  adjusted: Float64Array
  /** IDs sorted by logit descending (ties: lower ID first), covering all logits. */
  logitOrder: Uint32Array
}

export interface VocabQuery {
  text: string
  field: VocabSearchField
  caseSensitive: boolean
  filter: VocabFilter
  sort: VocabSortKey
  direction: SortDirection
}

export const DEFAULT_VOCAB_QUERY: Readonly<VocabQuery> = Object.freeze({
  text: '',
  field: 'any',
  caseSensitive: false,
  filter: 'all',
  sort: 'probability',
  direction: 'desc',
})

/** Interpret simple escapes so users can search for "\n", "\t" or a leading space "\s". */
export function unescapeQuery(q: string): string {
  return q.replace(/\\([ntrs\\])/g, (_, c: string) =>
    c === 'n' ? '\n' : c === 't' ? '\t' : c === 'r' ? '\r' : c === 's' ? ' ' : '\\',
  )
}

export function matchesQuery(id: number, source: VocabSource, query: VocabQuery): boolean {
  const raw = query.text
  if (!raw) return true
  const trimmed = raw.trim()
  const asId = /^#?\d+$/.test(trimmed) ? Number(trimmed.replace('#', '')) : null
  if (query.field === 'id') return asId !== null && id === asId
  if (query.field === 'any' && asId !== null && id === asId) return true
  const needle = unescapeQuery(raw)
  const norm = (s: string) => (query.caseSensitive ? s : s.toLowerCase())
  const n = norm(needle)
  if (query.field === 'decoded' || query.field === 'any') {
    if (norm(source.decoded[id] ?? '').includes(n)) return true
  }
  if (query.field === 'raw' || query.field === 'any') {
    if (norm(source.raw[id] ?? '').includes(n)) return true
  }
  return false
}

function passesFilter(
  id: number,
  source: VocabSource,
  dist: VocabDistribution | null,
  f: VocabFilter,
) {
  switch (f) {
    case 'all':
      return true
    case 'special':
      return source.isSpecial(id)
    case 'regular':
      return !source.isSpecial(id)
    case 'eligible':
      return !!dist && (dist.adjusted[id] ?? 0) > 0
    case 'filtered-out':
      return !!dist && id < dist.adjusted.length && (dist.adjusted[id] ?? 0) === 0
  }
}

const alphaCache = new WeakMap<readonly string[], Uint32Array>()

/** Alphabetical order by decoded text (code-unit order, deterministic), cached per vocab. */
export function alphabeticalOrder(decoded: readonly string[], size: number): Uint32Array {
  const cached = alphaCache.get(decoded)
  if (cached && cached.length === size) return cached
  const idx = new Uint32Array(size)
  for (let i = 0; i < size; i++) idx[i] = i
  const keys = new Array<string>(size)
  for (let i = 0; i < size; i++) keys[i] = (decoded[i] ?? '').toLowerCase()
  idx.sort((a, b) => {
    const ka = keys[a] as string
    const kb = keys[b] as string
    return ka < kb ? -1 : ka > kb ? 1 : a - b
  })
  alphaCache.set(decoded, idx)
  return idx
}

/** Base ordering of all IDs for a sort key, in DESCENDING (or natural ascending for id/alpha) order. */
function baseOrder(
  source: VocabSource,
  dist: VocabDistribution | null,
  key: VocabSortKey,
): Uint32Array {
  const n = source.size
  if (key === 'alphabetical') return alphabeticalOrder(source.decoded, n)
  if (key === 'id' || !dist) {
    const idx = new Uint32Array(n)
    for (let i = 0; i < n; i++) idx[i] = i
    return idx
  }
  // probability and logit share the same order (softmax is monotonic).
  // IDs without a logit (tokenizer entries beyond logitsSize) go last.
  const out = new Uint32Array(n)
  let w = 0
  if (key === 'adjusted') {
    // Survivors (adjusted > 0) first in logit order, then the rest in logit order.
    for (const id of dist.logitOrder) if ((dist.adjusted[id] ?? 0) > 0) out[w++] = id
    for (const id of dist.logitOrder) if (!((dist.adjusted[id] ?? 0) > 0)) out[w++] = id
  } else {
    for (const id of dist.logitOrder) out[w++] = id
  }
  for (let id = dist.logitOrder.length; id < n; id++) out[w++] = id
  return out
}

export interface VocabQueryResult {
  /** Matching IDs in display order. */
  ids: Uint32Array
  total: number
}

export function queryVocab(
  source: VocabSource,
  dist: VocabDistribution | null,
  query: VocabQuery,
): VocabQueryResult {
  const naturalAsc = query.sort === 'id' || query.sort === 'alphabetical'
  const order = baseOrder(source, dist, query.sort)
  const reverse = naturalAsc ? query.direction === 'desc' : query.direction === 'asc'
  const out: number[] = []
  const n = order.length
  for (let k = 0; k < n; k++) {
    const id = order[reverse ? n - 1 - k : k] as number
    if (!passesFilter(id, source, dist, query.filter)) continue
    if (!matchesQuery(id, source, query)) continue
    out.push(id)
  }
  return { ids: Uint32Array.from(out), total: out.length }
}

export interface Page {
  page: number
  pageSize: number
  pageCount: number
  start: number
  end: number
  ids: number[]
}

export function paginate(ids: ArrayLike<number>, page: number, pageSize: number): Page {
  const size = Math.max(1, Math.floor(pageSize))
  const pageCount = Math.max(1, Math.ceil(ids.length / size))
  const p = Math.min(Math.max(0, Math.floor(page)), pageCount - 1)
  const start = p * size
  const end = Math.min(ids.length, start + size)
  const out: number[] = []
  for (let i = start; i < end; i++) out.push(ids[i] as number)
  return { page: p, pageSize: size, pageCount, start, end, ids: out }
}

/** 1-based rank of `id` under the probability ordering (for display). */
export function rankOf(id: number, logitOrder: Uint32Array): number {
  for (let i = 0; i < logitOrder.length; i++) if (logitOrder[i] === id) return i + 1
  return -1
}
