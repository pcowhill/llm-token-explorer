import { describe, expect, it } from 'vitest'
import {
  byteLevelToBytes,
  categorizeToken,
  describeCodePoints,
  escapeLiteral,
  escapeVisible,
  formatBytes,
  isIncompleteUtf8,
  sentencePieceToBytes,
  tokenBytes,
} from '@/tokenization/display'

describe('escapeVisible', () => {
  it('makes whitespace and control characters visible', () => {
    expect(escapeVisible(' a\tb\nc\r')).toBe('␣a⇥b↵c␍')
    expect(escapeVisible('​')).toBe('⟨U+200B⟩')
    expect(escapeVisible('\u0000')).toBe('⟨U+0000⟩')
    expect(escapeVisible('é')).toBe('é')
    expect(escapeVisible('�')).toBe('�')
  })
  it('escapeLiteral produces a JSON-like literal', () => {
    expect(escapeLiteral(' hi\n"x"\\')).toBe('" hi\\n\\"x\\"\\\\"')
    expect(escapeLiteral('​')).toBe('"\\u200b"')
  })
  it('describes code points', () => {
    const d = describeCodePoints(' \n')
    expect(d.map((x) => x.codePoint)).toEqual(['U+0020', 'U+000A'])
    expect(d[0]?.name).toBe('SPACE')
    expect(d[1]?.category).toBe('whitespace')
  })
})

describe('byte-level BPE bytes', () => {
  it('maps GPT-2 style characters back to bytes', () => {
    expect(Array.from(byteLevelToBytes('Ġthe')!)).toEqual([0x20, 0x74, 0x68, 0x65])
    expect(Array.from(byteLevelToBytes('Ċ')!)).toEqual([0x0a])
    // "’" (U+2019) = E2 80 99 → byte-level chars "âĢĻ"
    expect(formatBytes(byteLevelToBytes('âĢĻ')!)).toBe('E2 80 99')
    expect(byteLevelToBytes('中')).toBeNull()
  })
  it('detects incomplete UTF-8 sequences', () => {
    expect(isIncompleteUtf8(byteLevelToBytes('âĢ')!)).toBe(true)
    expect(isIncompleteUtf8(byteLevelToBytes('âĢĻ')!)).toBe(false)
  })
})

describe('SentencePiece bytes', () => {
  it('handles ▁ spaces and <0xNN> byte fallback', () => {
    expect(Array.from(sentencePieceToBytes('▁the'))).toEqual([0x20, 0x74, 0x68, 0x65])
    expect(Array.from(sentencePieceToBytes('<0x0A>'))).toEqual([0x0a])
    expect(tokenBytes('', 'sentencepiece')).toBeNull()
    expect(tokenBytes('abc', 'unknown')).toBeNull()
  })
})

describe('categorizeToken', () => {
  const base = { special: false, added: false, scheme: 'byte-level-bpe' as const }
  it('categorizes common token kinds', () => {
    expect(categorizeToken({ ...base, raw: '<|endoftext|>', decoded: '<|endoftext|>', special: true })).toBe('special')
    expect(categorizeToken({ ...base, raw: 'Ġthe', decoded: ' the' })).toBe('word')
    expect(categorizeToken({ ...base, raw: 'ing', decoded: 'ing' })).toBe('word-continuation')
    expect(categorizeToken({ ...base, raw: 'Ċ', decoded: '\n' })).toBe('whitespace')
    expect(categorizeToken({ ...base, raw: '123', decoded: '123' })).toBe('number')
    expect(categorizeToken({ ...base, raw: '.', decoded: '.' })).toBe('punctuation')
    expect(categorizeToken({ ...base, raw: 'âĢ', decoded: '�' })).toBe('partial-utf8')
    expect(categorizeToken({ ...base, raw: '', decoded: '' })).toBe('unmapped')
    expect(categorizeToken({ ...base, raw: '<0x0A>', decoded: '\n', scheme: 'sentencepiece' })).toBe('byte')
  })
})
