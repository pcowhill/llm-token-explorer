/**
 * Helpers that make token strings understandable without lying about them.
 *
 * Tokens are frequently not printable words: they can be a leading space, a newline,
 * a single UTF-8 byte that is only part of a character, or a control token.
 * Everything here is pure string processing; nothing is ever rendered as HTML.
 */

import type { TokenizerSnapshot } from '@/types'

export type TokenScheme = TokenizerSnapshot['scheme']

const VISIBLE: Record<string, string> = {
  ' ': '␣',
  '\n': '↵',
  '\r': '␍',
  '\t': '⇥',
  '\v': '␋',
  '\f': '␌',
  ' ': '⍽',
}

/** Characters that render as nothing / ambiguously (controls, format chars, separators). */
const INVISIBLE_RE = /[\p{Cc}\p{Cf}\p{Zl}\p{Zp}\p{Co}\p{Cn}]/u

export function codePointLabel(cp: number): string {
  return 'U+' + cp.toString(16).toUpperCase().padStart(4, '0')
}

/**
 * Replace whitespace and invisible characters with visible symbols so token
 * boundaries are obvious. `�` (replacement character) is preserved because it
 * genuinely appears when a token is an incomplete UTF-8 sequence.
 */
export function escapeVisible(text: string): string {
  let out = ''
  for (const ch of text) {
    const v = VISIBLE[ch]
    if (v !== undefined) out += v
    else if (INVISIBLE_RE.test(ch)) out += `⟨${codePointLabel(ch.codePointAt(0) as number)}⟩`
    else out += ch
  }
  return out
}

/** Escape for single-line code-like display, e.g. "\n" → "\\n". */
export function escapeLiteral(text: string): string {
  let out = ''
  for (const ch of text) {
    switch (ch) {
      case '\n':
        out += '\\n'
        break
      case '\r':
        out += '\\r'
        break
      case '\t':
        out += '\\t'
        break
      case '\\':
        out += '\\\\'
        break
      case '"':
        out += '\\"'
        break
      default: {
        const cp = ch.codePointAt(0) as number
        if (INVISIBLE_RE.test(ch)) {
          out += cp > 0xffff ? `\\u{${cp.toString(16)}}` : `\\u${cp.toString(16).padStart(4, '0')}`
        } else out += ch
      }
    }
  }
  return `"${out}"`
}

const CHAR_NAMES: Record<number, string> = {
  0x09: 'CHARACTER TABULATION (tab)',
  0x0a: 'LINE FEED (newline)',
  0x0d: 'CARRIAGE RETURN',
  0x20: 'SPACE',
  0xa0: 'NO-BREAK SPACE',
  0x200b: 'ZERO WIDTH SPACE',
  0x200c: 'ZERO WIDTH NON-JOINER',
  0x200d: 'ZERO WIDTH JOINER',
  0xfeff: 'ZERO WIDTH NO-BREAK SPACE (BOM)',
  0xfffd: 'REPLACEMENT CHARACTER (incomplete or invalid UTF-8)',
  0x2581: 'LOWER ONE EIGHTH BLOCK (SentencePiece space marker ▁)',
  0x0120: 'LATIN CAPITAL LETTER G WITH DOT ABOVE (byte-level BPE space marker Ġ)',
  0x010a: 'LATIN CAPITAL LETTER C WITH DOT ABOVE (byte-level BPE newline marker Ċ)',
}

export interface CodePointInfo {
  char: string
  visible: string
  codePoint: string
  name: string
  category: string
}

export function describeCodePoints(text: string, limit = 64): CodePointInfo[] {
  const out: CodePointInfo[] = []
  for (const ch of text) {
    if (out.length >= limit) break
    const cp = ch.codePointAt(0) as number
    out.push({
      char: ch,
      visible: escapeVisible(ch),
      codePoint: codePointLabel(cp),
      name: CHAR_NAMES[cp] ?? '',
      category: unicodeCategory(ch),
    })
  }
  return out
}

function unicodeCategory(ch: string): string {
  if (/\p{L}/u.test(ch)) return 'letter'
  if (/\p{N}/u.test(ch)) return 'number'
  if (/\p{P}/u.test(ch)) return 'punctuation'
  if (/\p{S}/u.test(ch)) return 'symbol'
  if (/\p{Z}/u.test(ch) || /\s/u.test(ch)) return 'whitespace'
  if (/\p{M}/u.test(ch)) return 'mark'
  if (/\p{C}/u.test(ch)) return 'control/format'
  return 'other'
}

// ---------------------------------------------------------------------------
// Byte-level BPE (GPT-2, Qwen, SmolLM2): each byte 0–255 is mapped to a printable
// Unicode character. Reversing the map recovers the token's exact bytes.
// ---------------------------------------------------------------------------

let unicodeToByte: Map<string, number> | null = null

function byteLevelTable(): Map<string, number> {
  if (unicodeToByte) return unicodeToByte
  const bs: number[] = []
  for (let b = 0x21; b <= 0x7e; b++) bs.push(b)
  for (let b = 0xa1; b <= 0xac; b++) bs.push(b)
  for (let b = 0xae; b <= 0xff; b++) bs.push(b)
  const cs = bs.slice()
  let n = 0
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b)
      cs.push(256 + n)
      n++
    }
  }
  unicodeToByte = new Map()
  bs.forEach((b, i) => unicodeToByte!.set(String.fromCodePoint(cs[i] as number), b))
  return unicodeToByte
}

export function byteLevelToBytes(raw: string): Uint8Array | null {
  const table = byteLevelTable()
  const out: number[] = []
  for (const ch of raw) {
    const b = table.get(ch)
    if (b === undefined) return null
    out.push(b)
  }
  return Uint8Array.from(out)
}

const SP_BYTE_RE = /^<0x([0-9A-Fa-f]{2})>$/

/** SentencePiece: "▁" is a space; "<0xNN>" byte-fallback tokens are single raw bytes. */
export function sentencePieceToBytes(raw: string): Uint8Array {
  const m = SP_BYTE_RE.exec(raw)
  if (m) return Uint8Array.of(parseInt(m[1] as string, 16))
  return new TextEncoder().encode(raw.replaceAll('▁', ' '))
}

/** Exact bytes a (non-special) token contributes to the decoded text, when knowable. */
export function tokenBytes(raw: string, scheme: TokenScheme): Uint8Array | null {
  if (!raw) return null
  if (scheme === 'byte-level-bpe') return byteLevelToBytes(raw)
  if (scheme === 'sentencepiece') return sentencePieceToBytes(raw)
  return null
}

export function formatBytes(bytes: Uint8Array): string {
  return Array.from(bytes, (b) => b.toString(16).toUpperCase().padStart(2, '0')).join(' ')
}

/** True when the bytes are not a complete, valid UTF-8 sequence. */
export function isIncompleteUtf8(bytes: Uint8Array): boolean {
  try {
    new TextDecoder('utf-8', { fatal: true }).decode(bytes)
    return false
  } catch {
    return true
  }
}

export type TokenCategory =
  | 'special'
  | 'added'
  | 'unmapped'
  | 'partial-utf8'
  | 'byte'
  | 'whitespace'
  | 'word'
  | 'word-continuation'
  | 'number'
  | 'punctuation'
  | 'symbol'
  | 'mixed'

export interface TokenCategoryInput {
  raw: string
  decoded: string
  special: boolean
  added: boolean
  scheme: TokenScheme
}

export function categorizeToken(t: TokenCategoryInput): TokenCategory {
  if (t.special) return 'special'
  if (!t.raw && !t.decoded) return 'unmapped'
  if (t.added) return 'added'
  if (t.scheme === 'sentencepiece' && SP_BYTE_RE.test(t.raw)) return 'byte'
  const bytes = tokenBytes(t.raw, t.scheme)
  if (bytes && isIncompleteUtf8(bytes)) return 'partial-utf8'
  const text = t.decoded
  if (text.length > 0 && /^\s+$/u.test(text)) return 'whitespace'
  const core = text.replace(/^\s+/u, '')
  if (/^\p{L}+$/u.test(core)) return text !== core ? 'word' : 'word-continuation'
  if (/^\p{N}+$/u.test(core)) return 'number'
  if (/^\p{P}+$/u.test(core)) return 'punctuation'
  if (/^\p{S}+$/u.test(core)) return 'symbol'
  return 'mixed'
}

export const CATEGORY_LABELS: Record<TokenCategory, string> = {
  special: 'Special / control token',
  added: 'Added (non-special) token',
  unmapped: 'No tokenizer entry for this ID',
  'partial-utf8': 'Partial UTF-8 byte sequence',
  byte: 'Byte-fallback token',
  whitespace: 'Whitespace',
  word: 'Word (with leading space)',
  'word-continuation': 'Word piece (no leading space)',
  number: 'Number / digits',
  punctuation: 'Punctuation',
  symbol: 'Symbol',
  mixed: 'Mixed characters',
}
