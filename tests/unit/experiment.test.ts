import { describe, expect, it } from 'vitest'
import { appendToken, createTree, undo } from '@/history/tree'
import {
  EXPERIMENT_FORMAT,
  ExperimentValidationError,
  checkCompatibility,
  parseExperiment,
  serializeExperiment,
} from '@/history/experiment'
import { DEFAULT_GENERATION, DEFAULT_SAMPLING } from '@/types'
import { seedToState } from '@/sampling/prng'

function sample() {
  const t = createTree([1, 50256], 'hub:x@y#q8', seedToState(42))
  appendToken(
    t,
    {
      tokenId: 262,
      source: 'sample',
      originalProbability: 0.12,
      adjustedProbability: 0.3,
      logit: -1.5,
      sampling: { ...DEFAULT_SAMPLING, temperature: 0.7, topK: 40 },
      prngBefore: seedToState(42),
      uniform: 0.25,
    },
    { modelIdentity: 'hub:x@y#q8', prngState: seedToState(43), label: 'Sampled 262' },
  )
  undo(t)
  appendToken(
    t,
    { tokenId: 99, source: 'manual', originalProbability: null, adjustedProbability: null, logit: null, sampling: null, prngBefore: null, uniform: null },
    { modelIdentity: 'hub:x@y#q8', prngState: null, label: 'Picked 99' },
  )
  return t
}

const serialize = () =>
  serializeExperiment({
    tree: sample(),
    model: { identity: 'hub:x@y#q8', displayName: 'X', source: 'catalog', catalogKey: 'x', logitsSize: 60000, tokenizerFingerprint: 'abcd' },
    sampling: { ...DEFAULT_SAMPLING, seed: 7, temperature: 0.8 },
    generation: { ...DEFAULT_GENERATION, maxNewTokens: 64 },
    appVersion: '1.0.0',
  })

describe('experiment serialization', () => {
  it('round-trips token IDs, branches, settings and PRNG states exactly', () => {
    const file = serialize()
    const json = JSON.stringify(file)
    const { file: back, tree } = parseExperiment(json)
    expect(back.format).toBe(EXPERIMENT_FORMAT)
    expect(back.sampling.seed).toBe(7)
    expect(back.sampling.temperature).toBe(0.8)
    expect(back.generation.maxNewTokens).toBe(64)
    expect(back.model?.catalogKey).toBe('x')
    expect(tree.nodes.size).toBe(3)
    const cur = tree.nodes.get(tree.currentId)!
    expect(cur.tokens).toEqual([1, 50256, 99])
    const sampled = [...tree.nodes.values()].find((n) => n.append?.source === 'sample')!
    expect(sampled.tokens).toEqual([1, 50256, 262])
    expect(sampled.append?.sampling?.topK).toBe(40)
    expect(sampled.append?.prngBefore).toEqual([...seedToState(42)])
    expect(sampled.prngState).toEqual([...seedToState(43)])
    // New nodes do not clash with imported ids
    expect(tree.nodes.has(`n${tree.nextId}`)).toBe(false)
  })

  it('rejects non-JSON and wrong formats', () => {
    expect(() => parseExperiment('not json')).toThrow(ExperimentValidationError)
    expect(() => parseExperiment('[]')).toThrow(/Top level/)
    expect(() => parseExperiment(JSON.stringify({ ...serialize(), format: 'other' }))).toThrow(/format/)
    expect(() => parseExperiment(JSON.stringify({ ...serialize(), version: 99 }))).toThrow(/version/)
  })

  it('rejects invalid token IDs', () => {
    const f = serialize()
    f.history.nodes[0]!.tokens = [1, -5, 2.5 as number]
    expect(() => parseExperiment(JSON.stringify(f))).toThrow(/tokens/)
  })

  it('rejects broken tree structure', () => {
    const f = serialize()
    f.history.nodes[1]!.parentId = 'n999'
    expect(() => parseExperiment(JSON.stringify(f))).toThrow(/parent/)
    const g = serialize()
    g.history.currentId = 'nope'
    expect(() => parseExperiment(JSON.stringify(g))).toThrow(/currentId/)
    const h = serialize()
    h.history.nodes[1]!.tokens = [1, 50256, 5] // last token ≠ append.tokenId
    expect(() => parseExperiment(JSON.stringify(h))).toThrow(/append.tokenId/)
  })

  it('rejects invalid sampling settings and PRNG states', () => {
    const f = serialize()
    ;(f.sampling as { topP: number }).topP = 5
    expect(() => parseExperiment(JSON.stringify(f))).toThrow(/Top-p/)
    const g = serialize()
    ;(g.history.nodes[0] as { prngState: unknown }).prngState = [1, 2]
    expect(() => parseExperiment(JSON.stringify(g))).toThrow(/prngState/)
  })

  it('ignores unknown fields and never evaluates content', () => {
    const f = serialize() as unknown as Record<string, unknown>
    f.__proto__polluted = { evil: true }
    f.script = 'alert(1)'
    const { file } = parseExperiment(JSON.stringify(f))
    expect((file as unknown as Record<string, unknown>).script).toBeUndefined()
    expect(({} as Record<string, unknown>).evil).toBeUndefined()
  })

  it('reports model compatibility without re-tokenizing', () => {
    const { file, tree } = parseExperiment(JSON.stringify(serialize()))
    const none = checkCompatibility(file, tree, null)
    expect(none.messages[0]).toMatch(/created with “X”/)
    const same = checkCompatibility(file, tree, { identity: 'hub:x@y#q8', logitsSize: 60000, tokenizerFingerprint: 'abcd' })
    expect(same.sameModel).toBe(true)
    expect(same.messages).toEqual([])
    const other = checkCompatibility(file, tree, { identity: 'other', logitsSize: 1000, tokenizerFingerprint: 'zz' })
    expect(other.sameModel).toBe(false)
    expect(other.sameTokenizer).toBe(false)
    expect(other.outOfRangeIds).toEqual([50256])
  })
})
