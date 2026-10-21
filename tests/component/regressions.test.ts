/**
 * Regression tests for bugs reported against the live site. Each was shown to fail before
 * its fix (see the PR description).
 */
import { describe, expect, it } from 'vitest'
import ContextWorkspace from '@/components/context/ContextWorkspace.vue'
import ProbabilityWorkspace from '@/components/probability/ProbabilityWorkspace.vue'
import { makeWorkbench, mountWith, settle } from './helpers'
import { FAKE_LOGITS_SIZE } from './fakeEngine'

describe('degenerate logits are reported instead of shown as a distribution', () => {
  // SmolLM2 q4f16 on WebGPU returned all-zero logits, which the app displayed as a uniform
  // distribution over 49,152 tokens.
  const cases: [string, () => Float32Array][] = [
    ['all zero', () => new Float32Array(FAKE_LOGITS_SIZE)],
    ['constant', () => new Float32Array(FAKE_LOGITS_SIZE).fill(3.25)],
    [
      'NaN',
      () => {
        const l = new Float32Array(FAKE_LOGITS_SIZE).map((_, i) => i / 10)
        l[4] = NaN
        return l
      },
    ],
    ['all NaN', () => new Float32Array(FAKE_LOGITS_SIZE).fill(NaN)],
  ]
  for (const [name, logits] of cases) {
    it(`${name} logits → inference error, no distribution`, async () => {
      const { wb, engine } = await makeWorkbench({ loaded: true })
      engine.logitsFn = logits
      const w = mountWith(ProbabilityWorkspace, wb)
      wb.setTokens([8, 4], 'test')
      await wb.runInference(true)
      await settle()
      expect(wb.inferenceStatus.value).toBe('error')
      expect(wb.inferenceError.value?.kind).toBe('degenerate-output')
      expect(wb.prepared.value).toBeNull()
      expect(w.text()).toMatch(/invalid next-token logits/i)
      w.unmount()
    })
  }

  it('healthy logits after a degenerate result are shown again', async () => {
    const { wb, engine } = await makeWorkbench({ loaded: true })
    engine.logitsFn = () => new Float32Array(FAKE_LOGITS_SIZE)
    wb.setTokens([8], 'test')
    await wb.runInference(true)
    expect(wb.inferenceStatus.value).toBe('error')
    engine.logitsFn = (ids) =>
      new Float32Array(FAKE_LOGITS_SIZE).map((_, i) => (i + ids.length) % 5)
    wb.setTokens([8, 4], 'test')
    await wb.runInference(true)
    expect(wb.inferenceStatus.value).toBe('ready')
    expect(wb.prepared.value).not.toBeNull()
  })
})

describe('Example select', () => {
  it('acts as a menu: returns to the placeholder so the same example can be chosen again', async () => {
    const { wb } = await makeWorkbench()
    const w = mountWith(ContextWorkspace, wb)
    const select = w.get('[data-testid="example-select"]')
    await select.setValue('minimal')
    await settle()
    expect(wb.tokens.value).toEqual([0])
    // In a real browser, re-selecting the option that is already selected fires no change
    // event; the select must therefore show the placeholder again after each choice.
    expect((select.element as HTMLSelectElement).value).toBe('')
    expect(w.get('[data-testid="loaded-example"]').text()).toContain('Minimal: document start')

    await w.get('[data-testid="clear-context"]').trigger('click')
    await settle()
    expect(wb.tokens.value).toEqual([])
    expect(w.find('[data-testid="loaded-example"]').exists()).toBe(false)

    await select.setValue('minimal')
    await settle()
    expect(wb.tokens.value).toEqual([0])
    expect((select.element as HTMLSelectElement).value).toBe('')
    w.unmount()
  })
})
