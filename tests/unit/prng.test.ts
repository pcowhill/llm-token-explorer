import { describe, expect, it } from 'vitest'
import { Sfc32, isPrngState, seedToState } from '@/sampling/prng'

describe('Sfc32 PRNG', () => {
  it('is reproducible for the same seed', () => {
    const a = Sfc32.fromSeed(42)
    const b = Sfc32.fromSeed(42)
    const xs = Array.from({ length: 100 }, () => a.nextFloat())
    const ys = Array.from({ length: 100 }, () => b.nextFloat())
    expect(xs).toEqual(ys)
  })

  it('differs for different seeds', () => {
    expect(Sfc32.fromSeed(1).nextFloat()).not.toBe(Sfc32.fromSeed(2).nextFloat())
  })

  it('matches the reference SFC32 output for a known state', () => {
    // Reference: sfc32(a=1,b=2,c=3,d=4) first outputs computed with the canonical algorithm.
    const g = new Sfc32([1, 2, 3, 4])
    const out = [g.nextUint32(), g.nextUint32(), g.nextUint32()]
    // Step 1: t = a+b+d = 7; then a=2, b=3+(3<<3)=27, c=rotl(3,21)+7=6291463, d=5
    expect(out[0]).toBe(7)
    // Step 2: t = 2+27+5 = 34; then a=27, b=6291463*9=56623167, d=6
    expect(out[1]).toBe(34)
    // Step 3: t = 27+56623167+6
    expect(out[2]).toBe(56623200)
  })

  it('can resume from a saved state', () => {
    const g = Sfc32.fromSeed(7)
    g.nextFloat()
    const saved = g.getState()
    const next = [g.nextFloat(), g.nextFloat()]
    const r = new Sfc32(saved)
    expect([r.nextFloat(), r.nextFloat()]).toEqual(next)
  })

  it('produces floats in [0,1) with a plausible mean', () => {
    const g = Sfc32.fromSeed(123)
    let sum = 0
    for (let i = 0; i < 100_000; i++) {
      const u = g.nextFloat()
      expect(u >= 0 && u < 1).toBe(true)
      sum += u
    }
    expect(sum / 100_000).toBeCloseTo(0.5, 2)
  })

  it('validates seeds and states', () => {
    expect(() => seedToState(-1)).toThrow()
    expect(() => seedToState(1.5)).toThrow()
    expect(() => seedToState(2 ** 32)).toThrow()
    expect(isPrngState(seedToState(0))).toBe(true)
    expect(isPrngState([1, 2, 3])).toBe(false)
    expect(isPrngState([1, 2, 3, -1])).toBe(false)
  })
})
