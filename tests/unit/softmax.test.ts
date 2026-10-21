import { describe, expect, it } from 'vitest'
import { argsortDescending, softmax, stableSum } from '@/sampling/softmax'

describe('softmax', () => {
  it('matches hand-computed values for a small vector', () => {
    // exp([1,2,3]) / sum = [0.09003057, 0.24472847, 0.66524096]
    const r = softmax([1, 2, 3])
    expect(r.ok).toBe(true)
    expect(r.probs[0]).toBeCloseTo(0.0900305731703805, 14)
    expect(r.probs[1]).toBeCloseTo(0.2447284710547976, 14)
    expect(r.probs[2]).toBeCloseTo(0.6652409557748219, 14)
    expect(r.logSumExp).toBeCloseTo(3.40760596444438, 12)
    expect(r.maxIndex).toBe(2)
  })

  it('is invariant to adding a constant (stability with huge logits)', () => {
    const a = softmax([1, 2, 3])
    const b = softmax([1001, 1002, 1003])
    const c = softmax([-1e4 + 1, -1e4 + 2, -1e4 + 3])
    for (let i = 0; i < 3; i++) {
      expect(b.probs[i]).toBeCloseTo(a.probs[i] as number, 12)
      expect(c.probs[i]).toBeCloseTo(a.probs[i] as number, 12)
    }
    expect(b.probs.every(Number.isFinite)).toBe(true)
  })

  it('sums to 1 over a large random vocabulary', () => {
    const n = 151_936
    const logits = new Float32Array(n)
    let s = 12345
    for (let i = 0; i < n; i++) {
      s = (s * 1103515245 + 12345) >>> 0
      logits[i] = ((s / 2 ** 32) - 0.5) * 40
    }
    const r = softmax(logits)
    expect(Math.abs(stableSum(r.probs) - 1)).toBeLessThan(1e-12)
    expect(r.probs.every((p) => p >= 0 && Number.isFinite(p))).toBe(true)
  })

  it('keeps extremely small probabilities in log space without underflowing to NaN', () => {
    const r = softmax([0, -800])
    // exp(-800) underflows double precision: probability is 0 but log-prob is exact.
    expect(r.probs[1]).toBe(0)
    expect(r.logProbs[1]).toBeCloseTo(-800, 10)
    expect(r.probs[0]).toBe(1)
    const tiny = softmax([0, -50])
    expect(tiny.probs[1]).toBeCloseTo(Math.exp(-50) / (1 + Math.exp(-50)), 30)
    expect(tiny.probs[1]).toBeGreaterThan(0)
  })

  it('treats NaN as excluded and reports it', () => {
    const r = softmax([NaN, 0, 0])
    expect(r.nanCount).toBe(1)
    expect(r.probs[0]).toBe(0)
    expect(r.probs[1]).toBeCloseTo(0.5, 15)
  })

  it('treats -Infinity as probability 0 (masking)', () => {
    const r = softmax([-Infinity, Math.log(3), 0])
    expect(r.probs[0]).toBe(0)
    expect(r.logProbs[0]).toBe(-Infinity)
    expect(r.probs[1]).toBeCloseTo(0.75, 15)
  })

  it('splits mass evenly among +Infinity logits', () => {
    const r = softmax([Infinity, 5, Infinity])
    expect(r.posInfCount).toBe(2)
    expect(r.probs[0]).toBe(0.5)
    expect(r.probs[1]).toBe(0)
    expect(r.probs[2]).toBe(0.5)
  })

  it('fails clearly when no logit is finite', () => {
    expect(softmax([]).ok).toBe(false)
    const r = softmax([NaN, -Infinity])
    expect(r.ok).toBe(false)
    expect(r.error).toMatch(/All logits/)
  })

  it('applies a scale (1/T) before exponentiation', () => {
    const r = softmax([0, Math.log(4)], 0.5) // logits/2 → [0, ln2] → [1/3, 2/3]
    expect(r.probs[0]).toBeCloseTo(1 / 3, 14)
    expect(r.probs[1]).toBeCloseTo(2 / 3, 14)
  })
})

describe('argsortDescending', () => {
  it('sorts descending with ties broken by lower index and NaN last', () => {
    expect(Array.from(argsortDescending([1, 3, NaN, 3, -Infinity, 2]))).toEqual([1, 3, 5, 0, 2, 4])
  })
})
