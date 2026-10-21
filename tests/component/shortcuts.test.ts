import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import App from '@/App.vue'
import SettingsDialog from '@/components/SettingsDialog.vue'
import GenerationControls from '@/components/sampling/GenerationControls.vue'
import { useShortcuts } from '@/composables/useShortcuts'
import { mount } from '@vue/test-utils'
import { makeWorkbench, mountWith, settle } from './helpers'
import { WORKBENCH_KEY } from '@/composables/useWorkbench'

type Wb = Awaited<ReturnType<typeof makeWorkbench>>['wb']

function press(key: string, init: KeyboardEventInit = {}, target: EventTarget = window) {
  target.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, ...init }))
}
async function until(cond: () => boolean, ms = 3000) {
  const t0 = Date.now()
  while (!cond()) {
    if (Date.now() - t0 > ms) throw new Error('timeout')
    await settle(1)
  }
}
async function mountApp(wb: Wb) {
  localStorage.setItem('lte:persist', 'false')
  const w = mount(App, { props: { workbench: wb }, attachTo: document.body })
  await settle()
  return w
}

beforeEach(() => {
  localStorage.clear()
  useShortcuts().resetBindings()
})
afterEach(() => {
  document.body.innerHTML = ''
})

describe('sampling shortcuts', () => {
  it('S samples one token and G appends the most likely one', async () => {
    const { wb } = await makeWorkbench({ text: 'The the' })
    const w = await mountApp(wb)
    const n = wb.tokens.value.length
    press('s')
    await until(() => wb.tokens.value.length === n + 1)
    expect(wb.lastPick.value?.source).toBe('sample')
    await until(() => wb.resultIsCurrent.value)
    const argmax = wb.prepared.value!.order[0]
    press('G') // case-insensitive (Caps Lock / Shift)
    await until(() => wb.tokens.value.length === n + 2)
    expect(wb.tokens.value.at(-1)).toBe(argmax)
    expect(wb.lastPick.value?.source).toBe('greedy')
    w.unmount()
  })

  it('is inactive while typing in a text field, a select or a dialog', async () => {
    const { wb } = await makeWorkbench({ text: 'The' })
    const w = await mountApp(wb)
    const n = wb.tokens.value.length
    const textarea = w.get('[data-testid="context-text"]').element
    press('s', {}, textarea)
    press('g', {}, w.get('[data-testid="max-tokens"]').element)
    press('s', {}, w.get('[data-testid="example-select"]').element)
    await settle()
    expect(wb.tokens.value.length).toBe(n)
    w.unmount()
  })

  it('holding a key repeats, but each token waits for its forward pass', async () => {
    const { wb, engine } = await makeWorkbench({ text: 'The' })
    engine.forwardDelayMs = 25
    const w = await mountApp(wb)
    const n = wb.tokens.value.length
    // OS key repeat: a keydown every 5 ms for ~250 ms.
    const t0 = Date.now()
    press('s')
    while (Date.now() - t0 < 250) {
      press('s', { repeat: true })
      await new Promise((r) => setTimeout(r, 5))
    }
    await until(() => wb.inferenceStatus.value === 'ready' || wb.inferenceStatus.value === 'idle')
    const added = wb.tokens.value.length - n
    // ~50 keydown events, but only as many tokens as forward passes could finish.
    expect(added).toBeGreaterThanOrEqual(2)
    expect(added).toBeLessThanOrEqual(12)
    // Every appended token was chosen from a distribution computed for exactly its context.
    const tree = wb.getTree()
    const appended = [...tree.nodes.values()].filter((x) => x.append?.source === 'sample')
    expect(appended).toHaveLength(added)
    for (const node of appended) expect(node.append!.originalProbability).not.toBeNull()
    w.unmount()
  })

  it('keeps Ctrl+Z / Ctrl+Y, Ctrl+Enter and Alt+S working', async () => {
    const { wb, engine } = await makeWorkbench({ text: 'The' })
    const w = await mountApp(wb)
    const start = [...wb.tokens.value]
    press('s', { altKey: true })
    await until(() => wb.tokens.value.length === start.length + 1)
    const after = [...wb.tokens.value]
    press('z', { ctrlKey: true })
    expect(wb.tokens.value).toEqual(start)
    press('y', { ctrlKey: true })
    expect(wb.tokens.value).toEqual(after)
    press('z', { ctrlKey: true })
    press('Z', { ctrlKey: true, shiftKey: true })
    expect(wb.tokens.value).toEqual(after)
    await settle()
    const calls = engine.forwardCalls.length
    // Ctrl+Enter forces a pass, even from inside the text editor
    press('Enter', { ctrlKey: true }, w.get('[data-testid="context-text"]').element)
    await settle()
    expect(engine.forwardCalls.length).toBe(calls + 1)
    // Ctrl+S is not the sampling shortcut
    const len = wb.tokens.value.length
    press('s', { ctrlKey: true })
    await settle()
    expect(wb.tokens.value.length).toBe(len)
    w.unmount()
  })

  it('? opens the shortcut cheat-sheet listing the current keys', async () => {
    const { wb } = await makeWorkbench({ text: 'The' })
    const w = await mountApp(wb)
    useShortcuts().setBinding('greedy', 'n')
    await settle()
    press('?', { shiftKey: true })
    await settle()
    const d = w.get('[data-testid="shortcuts-dialog"]')
    expect(d.attributes('open')).toBeDefined()
    expect(d.get('[data-testid="shortcut-row-sample"]').text()).toContain('S')
    expect(d.get('[data-testid="shortcut-row-greedy"]').text()).toContain('N')
    for (const k of ['Alt+S', 'Ctrl+Enter', 'Ctrl+Z', 'Ctrl+Y']) expect(d.text()).toContain(k)
    // Shortcuts are inactive while the dialog has focus
    const n = wb.tokens.value.length
    press('s', {}, d.element)
    await settle()
    expect(wb.tokens.value.length).toBe(n)
    w.unmount()
  })

  it('buttons show the current key hints', async () => {
    const { wb } = await makeWorkbench({ text: 'The' })
    const w = mountWith(GenerationControls, wb)
    expect(w.get('[data-testid="sample-one"]').text()).toBe('Sample 1 token S')
    expect(w.get('[data-testid="sample-one"]').attributes('aria-keyshortcuts')).toBe('S')
    expect(w.get('[data-testid="greedy-one"]').text()).toBe('Append most likely G')
    useShortcuts().setBinding('sample', 'x')
    await settle()
    expect(w.get('[data-testid="sample-one"]').text()).toBe('Sample 1 token X')
    w.unmount()
  })
})

describe('remapping shortcuts in Settings', () => {
  async function mountSettings(wb: Wb) {
    const w = mount(SettingsDialog, {
      props: { open: true, persistSession: false },
      attachTo: document.body,
      global: { provide: { [WORKBENCH_KEY as symbol]: wb } },
    })
    await settle()
    return w
  }

  it('captures a new key, persists it, and the new key works', async () => {
    const { wb } = await makeWorkbench({ text: 'The' })
    const s = await mountSettings(wb)
    const btn = s.get('[data-testid="remap-sample"]')
    await btn.trigger('click')
    expect(btn.text()).toBe('Press a key…')
    await btn.trigger('keydown', { key: 'k' })
    expect(s.get('[data-testid="shortcut-key-sample"]').text()).toBe('K')
    expect(JSON.parse(localStorage.getItem('lte:shortcuts')!)).toEqual({ sample: 'k', greedy: 'g' })
    s.unmount()
    const w = await mountApp(wb)
    const n = wb.tokens.value.length
    press('s')
    await settle()
    expect(wb.tokens.value.length).toBe(n) // the old key no longer samples
    press('k')
    await until(() => wb.tokens.value.length === n + 1)
    w.unmount()
  })

  it('detects conflicts and reserved keys, and resets to the defaults', async () => {
    const { wb } = await makeWorkbench({ loaded: false })
    const s = await mountSettings(wb)
    const greedy = s.get('[data-testid="remap-greedy"]')
    await greedy.trigger('click')
    await greedy.trigger('keydown', { key: 'S' })
    expect(s.get('[data-testid="shortcut-error"]').text()).toContain(
      'already used for “Sample one token”',
    )
    expect(s.get('[data-testid="shortcut-key-greedy"]').text()).toBe('G')
    await greedy.trigger('keydown', { key: '?' })
    expect(s.get('[data-testid="shortcut-error"]').text()).toContain('shortcut list')
    await greedy.trigger('keydown', { key: 'g', ctrlKey: true })
    expect(s.get('[data-testid="shortcut-error"]').text()).toContain('without Ctrl')
    await greedy.trigger('keydown', { key: 'Escape' })
    expect(greedy.text()).toBe('Change')
    expect(useShortcuts().capturing.value).toBe(false)

    await greedy.trigger('click')
    await greedy.trigger('keydown', { key: 'j' })
    expect(s.get('[data-testid="shortcut-key-greedy"]').text()).toBe('J')
    await s.get('[data-testid="reset-shortcuts"]').trigger('click')
    expect(s.get('[data-testid="shortcut-key-greedy"]').text()).toBe('G')
    expect(JSON.parse(localStorage.getItem('lte:shortcuts')!)).toEqual({ sample: 's', greedy: 'g' })
    s.unmount()
  })

  it('page shortcuts are paused while a key is being captured', async () => {
    const { wb } = await makeWorkbench({ text: 'The' })
    const w = await mountApp(wb)
    useShortcuts().capturing.value = true
    const n = wb.tokens.value.length
    press('s')
    await settle()
    expect(wb.tokens.value.length).toBe(n)
    useShortcuts().capturing.value = false
    w.unmount()
  })
})
