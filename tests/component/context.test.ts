import { describe, expect, it } from 'vitest'
import ContextWorkspace from '@/components/context/ContextWorkspace.vue'
import { makeWorkbench, mountWith, settle } from './helpers'

describe('ContextWorkspace', () => {
  it('tokenizes typed text into individually distinguishable token chips', async () => {
    const { wb } = await makeWorkbench()
    const w = mountWith(ContextWorkspace, wb)
    await w.get('[data-testid="context-text"]').setValue('The cat sat on the mat.')
    await new Promise((r) => setTimeout(r, 30))
    await settle()
    const chips = w.findAll('[data-testid="token-strip"] button')
    expect(chips.map((c) => Number(c.attributes('data-token-id')))).toEqual([8, 4, 5, 6, 3, 7, 12])
    // Leading spaces are visible, not hidden
    expect(chips[1]!.text()).toContain('␣cat')
    // Alternating colors make boundaries obvious
    expect(chips[1]!.classes()).toContain('alt')
    expect(chips[0]!.classes()).not.toContain('alt')
    expect(w.get('[data-testid="token-count"]').text()).toContain('7 tokens')
    w.unmount()
  })

  it('opens an inspector with ID, position, escaped text and bytes', async () => {
    const { wb } = await makeWorkbench({ text: 'The cat\n' })
    const w = mountWith(ContextWorkspace, wb)
    await w.get('[data-index="2"]').trigger('click')
    const ins = w.get('[data-testid="token-inspector"]')
    expect(ins.get('[data-testid="inspector-id"]').text()).toBe('9')
    expect(ins.text()).toContain('position 2')
    expect(ins.get('[data-testid="inspector-decoded"]').text()).toBe('"\\n"')
    expect(ins.text()).toContain('0A')
    expect(ins.text()).toContain('Whitespace')
    w.unmount()
  })

  it('removes tokens with the keyboard and the inspector', async () => {
    const { wb } = await makeWorkbench({ text: 'The cat sat' })
    const w = mountWith(ContextWorkspace, wb)
    await w.get('[data-index="1"]').trigger('keydown', { key: 'Delete' })
    expect(wb.tokens.value).toEqual([8, 5])
    await w.get('[data-index="0"]').trigger('click')
    await w.get('[data-testid="inspector-remove"]').trigger('click')
    expect(wb.tokens.value).toEqual([5])
    w.unmount()
  })

  it('inserts an exact token ID at the end or before the selection', async () => {
    const { wb } = await makeWorkbench({ text: 'The cat' })
    const w = mountWith(ContextWorkspace, wb)
    await w.get('[data-testid="insert-id"]').setValue('2')
    expect(w.get('[data-testid="insert-preview"]').text()).toContain('<|im_end|>')
    await w.get('form[aria-label="Insert token by ID"]').trigger('submit')
    expect(wb.tokens.value).toEqual([8, 4, 2])
    await w.get('[data-index="1"]').trigger('click')
    await w.get('[data-testid="insert-where"]').setValue('before')
    await w.get('[data-testid="insert-id"]').setValue('14')
    await w.get('form[aria-label="Insert token by ID"]').trigger('submit')
    expect(wb.tokens.value).toEqual([8, 14, 4, 2])
    w.unmount()
  })

  it('rejects out-of-range token IDs', async () => {
    const { wb } = await makeWorkbench({ text: 'The' })
    const w = mountWith(ContextWorkspace, wb)
    await w.get('[data-testid="insert-id"]').setValue('9999')
    expect(w.get('[data-testid="insert-preview"]').text()).toContain('Out of range')
    expect(w.get('[data-testid="insert-submit"]').attributes('disabled')).toBeDefined()
    w.unmount()
  })

  it('special-token browser is searchable and inserts the real token ID', async () => {
    const { wb } = await makeWorkbench({ text: 'The' })
    const w = mountWith(ContextWorkspace, wb)
    const browser = w.get('[data-testid="special-browser"]')
    expect(browser.findAll('tbody tr')).toHaveLength(3)
    await browser.get('[data-testid="special-search"]').setValue('im_end')
    expect(browser.findAll('tbody tr')).toHaveLength(1)
    await browser.get('[data-testid="special-row-2"] button').trigger('click')
    expect(wb.tokens.value).toEqual([8, 2])
    // Special tokens are visually distinct
    expect(w.get('[data-index="1"]').classes()).toContain('special')
    w.unmount()
  })

  it('marks the text as a non-round-tripping preview after token-level edits', async () => {
    const { wb } = await makeWorkbench({ text: 'The cat' })
    const w = mountWith(ContextWorkspace, wb)
    // ID 17 is a logit row with no tokenizer entry: decoding then re-encoding cannot reproduce it.
    wb.insertToken(2, 17)
    await settle()
    expect(w.find('[data-testid="roundtrip-warning"]').exists()).toBe(true)
    expect(w.get('[data-testid="context-text"]').attributes('readonly')).toBeDefined()
    await w.get('[data-testid="unlock-text"]').trigger('click')
    expect(w.get('[data-testid="context-text"]').attributes('readonly')).toBeUndefined()
    w.unmount()
  })

  it('loads model-specific examples and applies the chat template', async () => {
    const { wb } = await makeWorkbench()
    const w = mountWith(ContextWorkspace, wb)
    await w.get('[data-testid="example-select"]').setValue('special')
    await settle()
    expect(wb.tokens.value[0]).toBe(1) // <|im_start|> by ID
    await w.get('[data-testid="chat-user"]').setValue('The cat')
    await w.get('[data-testid="apply-chat"]').trigger('click')
    await settle()
    expect(wb.tokens.value).toEqual([1, 15, 9, 8, 4, 2, 9, 1, 16, 9])
    w.unmount()
  })
})
