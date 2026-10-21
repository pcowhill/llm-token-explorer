import { describe, expect, it } from 'vitest'
import {
  TokenIdError,
  assertValidTokenId,
  commonPrefixLength,
  contextKey,
  fnv1a,
  insertAt,
  invalidIds,
  moveToken,
  parseTokenId,
  removeAt,
  replaceAt,
  sequencesEqual,
} from '@/tokenization/sequence'

describe('token sequence operations', () => {
  const seq = Object.freeze([10, 20, 30])
  it('inserts, removes, replaces and moves without mutating the input', () => {
    expect(insertAt(seq, 0, [1])).toEqual([1, 10, 20, 30])
    expect(insertAt(seq, 3, [1, 2])).toEqual([10, 20, 30, 1, 2])
    expect(removeAt(seq, 1)).toEqual([10, 30])
    expect(replaceAt(seq, 2, 99)).toEqual([10, 20, 99])
    expect(moveToken(seq, 0, 2)).toEqual([20, 30, 10])
    expect(seq).toEqual([10, 20, 30])
  })
  it('rejects out-of-range positions', () => {
    expect(() => insertAt(seq, 4, [1])).toThrow(RangeError)
    expect(() => removeAt(seq, 3)).toThrow(RangeError)
    expect(() => removeAt([], 0)).toThrow(RangeError)
  })
  it('validates token IDs against the model input range', () => {
    expect(() => assertValidTokenId(5, 10)).not.toThrow()
    expect(() => assertValidTokenId(10, 10)).toThrow(TokenIdError)
    expect(() => assertValidTokenId(-1, 10)).toThrow(/≥ 0/)
    expect(() => assertValidTokenId(1.5, 10)).toThrow(/integer/)
    expect(invalidIds([1, 2, 99, -1], 10)).toEqual([99, -1])
  })
  it('parses user-entered IDs strictly', () => {
    expect(parseTokenId(' 42 ')).toBe(42)
    expect(parseTokenId('#7')).toBe(7)
    expect(parseTokenId('4.2')).toBeNull()
    expect(parseTokenId('abc')).toBeNull()
    expect(parseTokenId('')).toBeNull()
  })
  it('compares sequences and prefixes', () => {
    expect(sequencesEqual([1, 2], [1, 2])).toBe(true)
    expect(sequencesEqual([1, 2], [1, 3])).toBe(false)
    expect(commonPrefixLength([1, 2, 3], [1, 2, 4])).toBe(2)
  })
  it('builds exact context keys and stable fingerprints', () => {
    expect(contextKey('m', [1, 23])).not.toBe(contextKey('m', [12, 3]))
    expect(contextKey('a', [1])).not.toBe(contextKey('b', [1]))
    expect(fnv1a(['a', 'b'])).toBe(fnv1a(['a', 'b']))
    expect(fnv1a(['ab'])).not.toBe(fnv1a(['a', 'b']))
  })
})
