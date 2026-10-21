import { describe, expect, it } from 'vitest'
import { halfToFloat, lastLogitsRow } from '@/inference/logitsRow'
import { checkLogits } from '@/sampling/logitsHealth'

/** Encode a float as IEEE-754 binary16 bits (round to nearest even), for test fixtures. */
function floatToHalf(v: number): number {
  const f32 = new Float32Array([v])
  const u = new Uint32Array(f32.buffer)[0]!
  const sign = (u >>> 16) & 0x8000
  const exp = ((u >>> 23) & 0xff) - 127 + 15
  const mant = u & 0x7fffff
  if (Number.isNaN(v)) return 0x7e00
  if (exp >= 0x1f) return sign | 0x7c00
  if (exp <= 0) return sign | Math.round((mant | 0x800000) / 2 ** (14 - exp))
  return sign | (exp << 10) | Math.round(mant / 8192)
}

describe('halfToFloat', () => {
  it('decodes normal, subnormal and special binary16 values', () => {
    expect(halfToFloat(0x3c00)).toBe(1)
    expect(halfToFloat(0xc000)).toBe(-2)
    expect(halfToFloat(0x7bff)).toBe(65504) // fp16 max
    expect(halfToFloat(0x0001)).toBe(2 ** -24) // smallest subnormal
    expect(halfToFloat(0x0000)).toBe(0)
    expect(Object.is(halfToFloat(0x8000), -0)).toBe(true)
    expect(halfToFloat(0x7c00)).toBe(Infinity)
    expect(halfToFloat(0xfc00)).toBe(-Infinity)
    expect(halfToFloat(0x7e00)).toBeNaN()
  })
})

describe('lastLogitsRow', () => {
  const V = 4
  const rows = [
    [0.5, -1, 2, 3.25],
    [10, -20.5, 0.125, 7],
  ]
  it('returns the final row of float32 logits', () => {
    const data = Float32Array.from(rows.flat())
    expect(Array.from(lastLogitsRow([1, 2, V], 'float32', data))).toEqual(rows[1])
  })
  it('converts float16 stored as raw Uint16 bits (no Float16Array)', () => {
    const data = Uint16Array.from(rows.flat(), floatToHalf)
    const out = lastLogitsRow([1, 2, V], 'float16', data)
    expect(out).toBeInstanceOf(Float32Array)
    expect(Array.from(out)).toEqual(rows[1])
  })
  it('converts float16 stored as numbers (Float16Array-like)', () => {
    const data = Float64Array.from(rows.flat())
    expect(Array.from(lastLogitsRow([1, 2, V], 'float16', data))).toEqual(rows[1])
  })
  it('handles a single-row [1, 1, V] output (num_logits_to_keep = 1)', () => {
    expect(Array.from(lastLogitsRow([1, 1, V], 'float32', Float32Array.from(rows[0]!)))).toEqual(
      rows[0],
    )
  })
  it('rejects unsupported logit types', () => {
    expect(() => lastLogitsRow([1, 1, 2], 'int8', Int8Array.from([1, 2]))).toThrow(/Unsupported/)
  })
})

describe('checkLogits', () => {
  it('accepts an informative vector, including -Infinity (masked) entries', () => {
    expect(checkLogits(Float32Array.from([1, 2, 3]))).toBeNull()
    expect(checkLogits(Float32Array.from([-Infinity, 0, 0.5]))).toBeNull()
  })
  it('rejects all-zero and constant logits (they would become a uniform distribution)', () => {
    expect(checkLogits(new Float32Array(49152))?.kind).toBe('constant')
    expect(checkLogits(new Float32Array(10).fill(-3))?.kind).toBe('constant')
    expect(checkLogits(Float32Array.from([2, 2, -Infinity]))?.kind).toBe('constant')
  })
  it('rejects NaN, +Infinity, all -Infinity and empty vectors', () => {
    expect(checkLogits(Float32Array.from([1, NaN, 3]))).toMatchObject({ kind: 'nan', count: 1 })
    expect(checkLogits(Float32Array.from([1, Infinity]))?.kind).toBe('infinite')
    expect(checkLogits(Float32Array.from([-Infinity, -Infinity]))?.kind).toBe('infinite')
    expect(checkLogits(new Float32Array(0))?.kind).toBe('empty')
  })
})
