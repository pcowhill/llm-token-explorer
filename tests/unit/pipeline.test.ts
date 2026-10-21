import { describe, expect, it } from 'vitest'
import {
  adjustDistribution,
  isGreedy,
  prepareLogits,
  sampleFromAdjusted,
  validateSamplingSettings,
} from '@/sampling/pipeline'
import { stableSum } from '@/sampling/softmax'
import { DEFAULT_SAMPLING, type SamplingSettings } from '@/types'

const S = (o: Partial<SamplingSettings> = {}): SamplingSettings => ({ ...DEFAULT_SAMPLING, ...o })
/** Logits whose T=1 softmax is exactly [0.5, 0.25, 0.125, 0.0625, 0.0625]. */
const LOGITS = Float32Array.from([0.5, 0.25, 0.125, 0.0625, 0.0625].map(Math.log))
const close = (a: ArrayLike<number>, b: number[], digits = 6) =>
  b.forEach((v, i) => expect(a[i] as number).toBeCloseTo(v, digits))

describe('prepareLogits', () => {
  it('computes the original distribution over all IDs and never mutates logits', () => {
    const logits = LOGITS.slice()
    const p = prepareLogits(logits)
    close(p.original.probs, [0.5, 0.25, 0.125, 0.0625, 0.0625])
    expect(Array.from(p.order)).toEqual([0, 1, 2, 3, 4])
    adjustDistribution(p, S({ temperature: 0.3, topK: 2 }))
    expect(Array.from(logits)).toEqual(Array.from(LOGITS))
  })

  it('excludes unmapped IDs (beyond the tokenizer) from sampling but keeps them in the original', () => {
    const p = prepareLogits(Float32Array.from([0, 0, 0, 10]), { mappedCount: 3 })
    expect(p.original.probs[3]).toBeGreaterThan(0.99)
    expect(p.eligible[3]).toBe(0)
    const a = adjustDistribution(p, S())
    expect(a.probs[3]).toBe(0)
    close(a.probs, [1 / 3, 1 / 3, 1 / 3, 0])
  })

  it('excludes explicitly listed unmapped IDs and NaN logits', () => {
    const p = prepareLogits(Float32Array.from([1, NaN, 1, 1]), { unmappedIds: [2] })
    expect(p.eligibleCount).toBe(2)
    const a = adjustDistribution(p, S())
    close(a.probs, [0.5, 0, 0, 0.5])
  })
})

describe('adjustDistribution', () => {
  const p = prepareLogits(LOGITS)

  it('T=1 with no filters equals the original distribution', () => {
    const a = adjustDistribution(p, S())
    // float32 logits: probabilities agree with the float64 original to ~1e-9
    close(a.probs, Array.from(p.original.probs), 12)
    close(a.probs, [0.5, 0.25, 0.125, 0.0625, 0.0625], 8)
    expect(a.keptCount).toBe(5)
  })

  it('temperature 0.5 squares and renormalizes probabilities', () => {
    // p^(1/T) = p^2 = [.25,.0625,.015625,.00390625,.00390625] sum=.3359375
    const a = adjustDistribution(p, S({ temperature: 0.5 }))
    const sum = 0.3359375
    close(a.probs, [0.25 / sum, 0.0625 / sum, 0.015625 / sum, 0.00390625 / sum, 0.00390625 / sum], 6)
  })

  it('temperature 2 flattens the distribution (sqrt)', () => {
    const r = [0.5, 0.25, 0.125, 0.0625, 0.0625].map(Math.sqrt)
    const sum = r.reduce((x, y) => x + y)
    close(adjustDistribution(p, S({ temperature: 2 })).probs, r.map((v) => v / sum), 6)
  })

  it('temperature 0 takes the explicit greedy path (no division by zero)', () => {
    const a = adjustDistribution(p, S({ temperature: 0 }))
    expect(a.greedy).toBe(true)
    expect(Array.from(a.probs)).toEqual([1, 0, 0, 0, 0])
    expect(isGreedy(S({ temperature: 0 }))).toBe(true)
  })

  it('greedy mode overrides temperature and breaks ties by lowest ID', () => {
    const tie = prepareLogits(Float32Array.from([1, 3, 3, 2]))
    const a = adjustDistribution(tie, S({ greedy: true, temperature: 1.7 }))
    expect(Array.from(a.probs)).toEqual([0, 1, 0, 0])
  })

  it('top-k keeps exactly k tokens and renormalizes', () => {
    const a = adjustDistribution(p, S({ topK: 2 }))
    close(a.probs, [2 / 3, 1 / 3, 0, 0, 0], 6)
    expect(a.keptCount).toBe(2)
    expect(a.stages.afterTopK).toBe(2)
  })

  it('top-k larger than the vocabulary is a no-op', () => {
    expect(adjustDistribution(p, S({ topK: 1000 })).keptCount).toBe(5)
  })

  it('top-p keeps the boundary token that crosses the threshold', () => {
    // cumulative: .5, .75, .875, .9375, 1
    expect(adjustDistribution(p, S({ topP: 0.6 })).keptCount).toBe(2) // .5 < .6 → include .25
    expect(adjustDistribution(p, S({ topP: 0.75 })).keptCount).toBe(2) // exactly reaches .75
    expect(adjustDistribution(p, S({ topP: 0.76 })).keptCount).toBe(3)
    expect(adjustDistribution(p, S({ topP: 0.01 })).keptCount).toBe(1) // always ≥ 1
    close(adjustDistribution(p, S({ topP: 0.8 })).probs, [0.5 / 0.875, 0.25 / 0.875, 0.125 / 0.875, 0, 0], 6)
  })

  it('min-p keeps tokens with p >= min_p * p_max', () => {
    // threshold 0.3 * 0.5 = 0.15 → keep .5 and .25
    const a = adjustDistribution(p, S({ minP: 0.3 }))
    close(a.probs, [2 / 3, 1 / 3, 0, 0, 0], 6)
    // threshold 0.12 * 0.5 = 0.06 → keep all
    expect(adjustDistribution(p, S({ minP: 0.12 })).keptCount).toBe(5)
    // inclusive comparison at an exactly representable boundary
    // inclusive: equal logits have bit-identical probabilities, so min-p = 1 keeps all of them
    const exact = prepareLogits(Float32Array.from([0, 0, 0, -1]))
    expect(adjustDistribution(exact, S({ minP: 1 })).keptCount).toBe(3)
    // min-p 1 keeps only the max
    expect(adjustDistribution(p, S({ minP: 1 })).keptCount).toBe(1)
  })

  it('applies temperature → top-k → top-p → min-p in that order', () => {
    // T=1, top-k 4 → [.5,.25,.125,.0625]/.9375 ; top-p .8 over renormalized:
    // cum .5333, .8 → keep 2 ; min-p .4 → threshold .4*.6667=.2667 → .3333 kept
    const a = adjustDistribution(p, S({ topK: 4, topP: 0.8, minP: 0.4 }))
    expect(a.stages).toEqual({ afterTemperature: 5, afterTopK: 4, afterTopP: 2, afterMinP: 2 })
    close(a.probs, [2 / 3, 1 / 3, 0, 0, 0], 6)
    // Higher min-p removes the second token as well.
    const b = adjustDistribution(p, S({ topK: 4, topP: 0.8, minP: 0.6 }))
    expect(b.keptCount).toBe(1)
    expect(b.probs[0]).toBe(1)
  })

  it('always leaves at least one token under aggressive filtering', () => {
    const a = adjustDistribution(p, S({ topK: 1, topP: 0.0001, minP: 1, temperature: 0.01 }))
    expect(a.keptCount).toBe(1)
    expect(a.probs[0]).toBe(1)
  })

  it('always produces a normalized distribution', () => {
    const big = new Float32Array(50_000)
    for (let i = 0; i < big.length; i++) big[i] = Math.sin(i * 12.9898) * 15
    const bp = prepareLogits(big)
    for (const s of [S(), S({ temperature: 0.2 }), S({ temperature: 3, topK: 40 }), S({ topP: 0.9, minP: 0.05 })]) {
      const a = adjustDistribution(bp, s)
      expect(Math.abs(stableSum(a.probs) - 1)).toBeLessThan(1e-9)
      expect(a.probs.every((x) => x >= 0 && Number.isFinite(x))).toBe(true)
    }
  })

  it('very low temperature does not produce NaN and approaches greedy', () => {
    const a = adjustDistribution(p, S({ temperature: 1e-6 }))
    expect(a.probs[0]).toBe(1)
    expect(a.probs.every(Number.isFinite)).toBe(true)
  })
})

describe('sampleFromAdjusted', () => {
  const a = adjustDistribution(prepareLogits(LOGITS), S())
  it('maps uniform draws through the CDF in descending-probability order', () => {
    expect(sampleFromAdjusted(a, 0)).toBe(0)
    expect(sampleFromAdjusted(a, 0.4999)).toBe(0)
    expect(sampleFromAdjusted(a, 0.5001)).toBe(1)
    expect(sampleFromAdjusted(a, 0.7499)).toBe(1)
    expect(sampleFromAdjusted(a, 0.8)).toBe(2)
    expect(sampleFromAdjusted(a, 0.9)).toBe(3)
    expect(sampleFromAdjusted(a, 0.99999)).toBe(4)
  })
  it('rejects invalid uniforms', () => {
    expect(() => sampleFromAdjusted(a, 1)).toThrow()
    expect(() => sampleFromAdjusted(a, -0.1)).toThrow()
  })
  it('matches the target distribution empirically', () => {
    const counts = [0, 0, 0, 0, 0]
    const N = 20000
    for (let i = 0; i < N; i++) counts[sampleFromAdjusted(a, (i + 0.5) / N)]!++
    close(counts.map((c) => c / N), [0.5, 0.25, 0.125, 0.0625, 0.0625], 3)
  })
})

describe('validateSamplingSettings', () => {
  it('accepts defaults and rejects out-of-range values', () => {
    expect(validateSamplingSettings(S())).toEqual([])
    expect(validateSamplingSettings(S({ temperature: -1 }))).toHaveLength(1)
    expect(validateSamplingSettings(S({ topK: 1.5 }))).toHaveLength(1)
    expect(validateSamplingSettings(S({ topP: 0 }))).toHaveLength(1)
    expect(validateSamplingSettings(S({ minP: 2 }))).toHaveLength(1)
    expect(validateSamplingSettings(S({ seed: -3 }))).toHaveLength(1)
    expect(validateSamplingSettings(S({ temperature: NaN }))).toHaveLength(1)
  })
})
