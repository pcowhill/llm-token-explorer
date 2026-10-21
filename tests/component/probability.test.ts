import { describe, expect, it } from 'vitest'
import ProbabilityWorkspace from '@/components/probability/ProbabilityWorkspace.vue'
import SamplingControls from '@/components/sampling/SamplingControls.vue'
import VocabularyExplorer from '@/components/probability/VocabularyExplorer.vue'
import { makeWorkbench, mountWith, settle } from './helpers'
import { FAKE_LOGITS_SIZE, fakeLogits } from './fakeEngine'
import { softmax } from '@/sampling/softmax'

describe('ProbabilityWorkspace', () => {
  it('shows logits, original and adjusted probabilities computed from the engine output', async () => {
    const { wb } = await makeWorkbench({ text: 'The the' })
    const w = mountWith(ProbabilityWorkspace, wb)
    await settle()
    expect(w.get('[data-testid="inference-status"]').text()).toContain('Ready')
    expect(w.get('[data-testid="logits-size"]').text()).toContain(String(FAKE_LOGITS_SIZE))
    expect(w.get('[data-testid="sum-original"]').text()).toBe('1.000000000')
    const expected = softmax(fakeLogits([8, 3]))
    const firstRow = w.get('[data-testid="candidate-table"] tbody tr')
    expect(firstRow.attributes('data-token-id')).toBe('4') // " cat" is the favourite after " the"
    expect(firstRow.text()).toContain(`${((expected.probs[4] as number) * 100).toFixed(1)}%`)
    expect(firstRow.text()).toContain('4.000') // raw logit
    expect(w.find('[data-testid="top-chart"]').exists()).toBe(true)
    w.unmount()
  })

  it('appending a candidate updates the context and runs the network again', async () => {
    const { wb, engine } = await makeWorkbench({ text: 'The' })
    const w = mountWith(ProbabilityWorkspace, wb)
    await settle()
    const before = engine.forwardCalls.length
    await w.get('[data-testid="pick-3"]').trigger('click')
    await settle()
    expect(wb.tokens.value).toEqual([8, 3])
    expect(engine.forwardCalls.length).toBe(before + 1)
    expect(engine.forwardCalls.at(-1)).toEqual([8, 3])
    w.unmount()
  })

  it('shows inference errors without crashing', async () => {
    const { wb, engine } = await makeWorkbench({ loaded: true })
    engine.failForward = 'device lost'
    wb.setTokens([8, 4], 'x')
    const w = mountWith(ProbabilityWorkspace, wb)
    await settle()
    expect(w.get('[data-testid="inference-error"]').text()).toContain('device lost')
    w.unmount()
  })

  it('changing sampling settings updates adjusted probabilities without re-running the network', async () => {
    const { wb, engine } = await makeWorkbench({ text: 'The the' })
    const prob = mountWith(ProbabilityWorkspace, wb)
    const ctrl = mountWith(SamplingControls, wb)
    await settle()
    const calls = engine.forwardCalls.length
    const logitsBefore = Array.from(wb.prepared.value!.logits)
    const originalBefore = Array.from(wb.prepared.value!.original.probs)
    const adjBefore = wb.adjusted.value!.probs[4] as number
    await ctrl.get('[data-testid="temperature-input"]').setValue('0.3')
    await ctrl.get('[data-testid="temperature-input"]').trigger('change')
    await settle()
    expect(wb.adjusted.value!.probs[4]).toBeGreaterThan(adjBefore)
    expect(Array.from(wb.prepared.value!.logits)).toEqual(logitsBefore)
    expect(Array.from(wb.prepared.value!.original.probs)).toEqual(originalBefore)
    expect(engine.forwardCalls.length).toBe(calls)
    // top-k filters others and the table marks them
    await ctrl.get('[data-testid="topk-input"]').setValue('1')
    await ctrl.get('[data-testid="topk-input"]').trigger('change')
    await settle()
    expect(prob.get('[data-testid="kept-count"]').text()).toContain('1 tokens')
    expect(prob.findAll('[data-testid="candidate-table"] tbody tr')[1]!.text()).toContain(
      'filtered',
    )
    expect(ctrl.get('[data-testid="pipeline-summary"]').text()).toContain('top-k 1')
    prob.unmount()
    ctrl.unmount()
  })

  it('greedy mode produces a one-hot adjusted distribution', async () => {
    const { wb } = await makeWorkbench({ text: 'The' })
    const ctrl = mountWith(SamplingControls, wb)
    await settle()
    await ctrl.get('[data-testid="greedy"]').setValue(true)
    expect(wb.adjusted.value!.keptCount).toBe(1)
    expect(wb.adjusted.value!.probs[3]).toBe(1)
    ctrl.unmount()
  })

  it('reports invalid sampling settings', async () => {
    const { wb } = await makeWorkbench({ text: 'The' })
    const ctrl = mountWith(SamplingControls, wb)
    await ctrl.get('[data-testid="topp-input"]').setValue('3')
    await ctrl.get('[data-testid="topp-input"]').trigger('change')
    expect(ctrl.text()).toContain('Top-p must be in (0, 1]')
    ctrl.unmount()
  })
})

describe('VocabularyExplorer', () => {
  it('searches, sorts, filters and paginates the full vocabulary, and appends any token', async () => {
    const { wb } = await makeWorkbench({ text: 'The' })
    const w = mountWith(VocabularyExplorer, wb, { props: { exact: false } })
    await settle()
    expect(w.get('[data-testid="vocab-page"]').text()).toContain('Page 1 of 1')
    expect(w.findAll('[data-testid="vocab-table"] tbody tr')).toHaveLength(19) // 17 entries + 2 unmapped logit rows
    await w.get('[data-testid="vocab-search"]').setValue('cat')
    expect(
      w.findAll('[data-testid="vocab-table"] tbody tr').map((r) => r.attributes('data-token-id')),
    ).toEqual(['4'])
    await w.get('[data-testid="vocab-search"]').setValue('')
    await w.get('[data-testid="vocab-filter"]').setValue('special')
    expect(w.findAll('[data-testid="vocab-table"] tbody tr')).toHaveLength(3)
    await w.get('[data-testid="vocab-filter"]').setValue('all')
    await w.get('[data-testid="vocab-sort"]').setValue('id')
    await w.get('[data-testid="vocab-direction"]').setValue('asc')
    expect(w.get('[data-testid="vocab-table"] tbody tr').attributes('data-token-id')).toBe('0')
    // pagination with a small page size
    await w.get('nav select').setValue(20)
    expect(w.get('[data-testid="vocab-next"]').attributes('disabled')).toBeDefined()
    // Append a low-probability token (not limited to top 20)
    await w.get('[data-testid="pick-14"]').trigger('click')
    expect(w.emitted('pick')?.[0]).toEqual([14])
    w.unmount()
  })
})
