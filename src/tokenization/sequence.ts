/**
 * Pure operations on the authoritative token-ID sequence. Every function returns a
 * new array; inputs are never mutated. IDs are validated against the model's
 * input range (the embedding / logits size), not the tokenizer size, because the two
 * can differ (see docs/special-tokens.md#vocabulary-vs-logits).
 */

import type { TokenId, TokenSequence } from '@/types'

export class TokenIdError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'TokenIdError'
  }
}

export function assertValidTokenId(id: number, modelSize: number): void {
  if (!Number.isInteger(id))
    throw new TokenIdError(`Token ID must be an integer (got ${String(id)})`)
  if (id < 0) throw new TokenIdError(`Token ID must be ≥ 0 (got ${id})`)
  if (id >= modelSize)
    throw new TokenIdError(`Token ID ${id} is outside the model's input range 0–${modelSize - 1}`)
}

export function parseTokenId(input: string): number | null {
  const t = input.trim().replace(/^#/, '')
  if (!/^\d+$/.test(t)) return null
  const n = Number(t)
  return Number.isSafeInteger(n) ? n : null
}

export function insertAt(seq: TokenSequence, index: number, ids: readonly TokenId[]): TokenId[] {
  if (index < 0 || index > seq.length)
    throw new RangeError(`Insert position ${index} out of range 0–${seq.length}`)
  return [...seq.slice(0, index), ...ids, ...seq.slice(index)]
}

export function removeAt(seq: TokenSequence, index: number, count = 1): TokenId[] {
  if (index < 0 || index >= seq.length)
    throw new RangeError(`Remove position ${index} out of range`)
  return [...seq.slice(0, index), ...seq.slice(index + count)]
}

export function replaceAt(seq: TokenSequence, index: number, id: TokenId): TokenId[] {
  if (index < 0 || index >= seq.length)
    throw new RangeError(`Replace position ${index} out of range`)
  const out = [...seq]
  out[index] = id
  return out
}

export function moveToken(seq: TokenSequence, from: number, to: number): TokenId[] {
  if (from < 0 || from >= seq.length || to < 0 || to >= seq.length)
    throw new RangeError('Move out of range')
  const out = [...seq]
  const [t] = out.splice(from, 1)
  out.splice(to, 0, t as TokenId)
  return out
}

export function invalidIds(seq: TokenSequence, modelSize: number): number[] {
  const bad: number[] = []
  for (const id of seq) if (!Number.isInteger(id) || id < 0 || id >= modelSize) bad.push(id)
  return bad
}

export function sequencesEqual(a: TokenSequence, b: TokenSequence): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false
  return true
}

/** Number of leading tokens two sequences share. */
export function commonPrefixLength(a: TokenSequence, b: TokenSequence): number {
  const n = Math.min(a.length, b.length)
  let i = 0
  while (i < n && a[i] === b[i]) i++
  return i
}

/** FNV-1a 32-bit hash over strings (fast, non-cryptographic fingerprint). */
export function fnv1a(parts: Iterable<string>): string {
  let h = 0x811c9dc5
  for (const s of parts) {
    for (let i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i)
      h = Math.imul(h, 0x01000193)
    }
    h ^= 0xff // separator
    h = Math.imul(h, 0x01000193)
  }
  return (h >>> 0).toString(16).padStart(8, '0')
}

/** Exact key identifying a (model, token sequence) pair for stale-result protection. */
export function contextKey(modelIdentity: string, seq: TokenSequence): string {
  return `${modelIdentity}|${seq.join(',')}`
}
