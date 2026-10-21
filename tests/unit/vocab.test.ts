import { describe, expect, it } from 'vitest'
import {
  DEFAULT_VOCAB_QUERY,
  alphabeticalOrder,
  matchesQuery,
  paginate,
  queryVocab,
  unescapeQuery,
  type VocabQuery,
  type VocabSource,
} from '@/tokenization/vocab'
import { adjustDistribution, prepareLogits } from '@/sampling/pipeline'
import { DEFAULT_SAMPLING } from '@/types'

const raw = ['<|endoftext|>', 'Ġthe', 'the', 'Ċ', 'Ġcat', 'Hello', '']
const decoded = ['<|endoftext|>', ' the', 'the', '\n', ' cat', 'Hello', '']
const source: VocabSource = { size: 8, raw, decoded, isSpecial: (id) => id === 0 }
const logits = Float32Array.from([0, 5, 1, 2, 4, 3, -1]) // 7 logits; ID 7 has none
const prepared = prepareLogits(logits, { mappedCount: 6 })
const adjusted = adjustDistribution(prepared, { ...DEFAULT_SAMPLING, topK: 2 })
const dist = { logits, original: prepared.original.probs, adjusted: adjusted.probs, logitOrder: prepared.fullOrder }
const Q = (o: Partial<VocabQuery>): VocabQuery => ({ ...DEFAULT_VOCAB_QUERY, ...o })

describe('vocabulary search', () => {
  it('searches decoded text case-insensitively', () => {
    expect(Array.from(queryVocab(source, dist, Q({ text: 'THE', field: 'decoded', sort: 'id', direction: 'asc' })).ids)).toEqual([1, 2])
  })
  it('searches raw token spellings', () => {
    expect(Array.from(queryVocab(source, dist, Q({ text: 'Ġ', field: 'raw', sort: 'id', direction: 'asc' })).ids)).toEqual([1, 4])
  })
  it('searches by numeric ID (with or without #)', () => {
    expect(Array.from(queryVocab(source, dist, Q({ text: '#4', field: 'id' })).ids)).toEqual([4])
    expect(matchesQuery(5, source, Q({ text: '5' }))).toBe(true)
  })
  it('supports escapes for invisible characters', () => {
    expect(unescapeQuery('\\n\\t\\s\\\\')).toBe('\n\t \\')
    expect(Array.from(queryVocab(source, dist, Q({ text: '\\n', field: 'decoded' })).ids)).toEqual([3])
  })
  it('respects case sensitivity', () => {
    expect(queryVocab(source, dist, Q({ text: 'hello', caseSensitive: true })).total).toBe(0)
    expect(queryVocab(source, dist, Q({ text: 'Hello', caseSensitive: true })).total).toBe(1)
  })
})

describe('vocabulary sorting and filtering', () => {
  it('sorts by probability descending by default, IDs without logits last', () => {
    expect(Array.from(queryVocab(source, dist, Q({})).ids)).toEqual([1, 4, 5, 3, 2, 0, 6, 7])
  })
  it('logit sort equals probability sort; ascending reverses it', () => {
    expect(Array.from(queryVocab(source, dist, Q({ sort: 'logit' })).ids)).toEqual([1, 4, 5, 3, 2, 0, 6, 7])
    expect(Array.from(queryVocab(source, dist, Q({ sort: 'probability', direction: 'asc' })).ids)).toEqual([7, 6, 0, 2, 3, 5, 4, 1])
  })
  it('sorts by token ID', () => {
    expect(Array.from(queryVocab(source, dist, Q({ sort: 'id', direction: 'desc' })).ids)).toEqual([7, 6, 5, 4, 3, 2, 1, 0])
  })
  it('sorts alphabetically by decoded text', () => {
    const ids = Array.from(queryVocab(source, dist, Q({ sort: 'alphabetical', direction: 'asc' })).ids)
    expect(ids.map((i) => decoded[i] ?? '')).toEqual(['', '', '\n', ' cat', ' the', '<|endoftext|>', 'Hello', 'the'])
    expect(alphabeticalOrder(decoded, 8)).toBe(alphabeticalOrder(decoded, 8)) // cached
  })
  it('sorts by adjusted probability with eligible survivors first', () => {
    expect(Array.from(queryVocab(source, dist, Q({ sort: 'adjusted' })).ids).slice(0, 2)).toEqual([1, 4])
  })
  it('filters special, regular, eligible and filtered-out tokens', () => {
    expect(Array.from(queryVocab(source, dist, Q({ filter: 'special' })).ids)).toEqual([0])
    expect(queryVocab(source, dist, Q({ filter: 'regular' })).total).toBe(7)
    expect(Array.from(queryVocab(source, dist, Q({ filter: 'eligible', sort: 'id', direction: 'asc' })).ids)).toEqual([1, 4])
    expect(queryVocab(source, dist, Q({ filter: 'filtered-out' })).total).toBe(5)
  })
  it('works without a distribution (no model output yet)', () => {
    expect(queryVocab(source, null, Q({ sort: 'probability' })).total).toBe(8)
  })
})

describe('paginate', () => {
  const ids = Array.from({ length: 45 }, (_, i) => i)
  it('slices pages and clamps out-of-range page numbers', () => {
    expect(paginate(ids, 0, 20)).toMatchObject({ page: 0, pageCount: 3, start: 0, end: 20 })
    expect(paginate(ids, 2, 20).ids).toEqual([40, 41, 42, 43, 44])
    expect(paginate(ids, 99, 20).page).toBe(2)
    expect(paginate(ids, -4, 20).page).toBe(0)
    expect(paginate([], 0, 20)).toMatchObject({ pageCount: 1, ids: [] })
  })
  it('pages through a large vocabulary without materializing all rows', () => {
    const big = new Uint32Array(262_144).map((_, i) => i)
    const p = paginate(big, 13106, 20)
    expect(p.ids).toEqual(Array.from({ length: 20 }, (_, i) => 262_120 + i))
    expect(p.pageCount).toBe(13108)
  })
})
