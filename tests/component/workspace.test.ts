import { describe, expect, it } from 'vitest'
import ContextWorkspace from '@/components/context/ContextWorkspace.vue'
import { createWorkbench } from '@/stores/workbench'
import { makeWorkbench, mountWith, settle } from './helpers'
import { FakeEngine } from './fakeEngine'

const nodes = (wb: Awaited<ReturnType<typeof makeWorkbench>>['wb']) => wb.getTree().nodes.size
const notes = (wb: Awaited<ReturnType<typeof makeWorkbench>>['wb']) =>
  wb.notices.value.map((n) => n.message)

describe('loading a model resets the workspace', () => {
  it('catalog model: clears context and history, loads its default example, shows a note', async () => {
    const { wb } = await makeWorkbench({ text: 'The cat sat' })
    wb.pickToken(4)
    wb.pickToken(5)
    await settle()
    expect(nodes(wb)).toBeGreaterThan(3)
    wb.sampling.seed = 7
    await wb.loadCatalog('gpt2', { backend: 'wasm', dtype: 'fp16' })
    await settle()
    // Fresh tree: root + the default example only
    expect(nodes(wb)).toBe(2)
    expect(wb.canUndo.value).toBe(true)
    wb.undo()
    expect(wb.tokens.value).toEqual([])
    expect(wb.canUndo.value).toBe(false)
    wb.redo()
    expect(wb.loadedExample.value?.label).toBe('Text completion')
    expect(wb.lastPick.value).toBeNull()
    expect(notes(wb)).toContain('Context cleared for new model.')
    // The generator restarts from the current seed
    expect(wb.prng.value).toEqual((await import('@/sampling/prng')).seedToState(7))
  })

  it('reloading the same model also resets', async () => {
    const { wb } = await makeWorkbench({ text: 'The cat sat' })
    await wb.loadCatalog('smollm2-135m-instruct', { backend: 'wasm', dtype: 'q8' })
    await settle()
    expect(nodes(wb)).toBe(2)
    expect(wb.loadedExample.value?.label).toBe('Text completion')
  })

  it('Hugging Face repo and local models start from an empty context', async () => {
    const { wb } = await makeWorkbench({ text: 'The cat sat' })
    await wb.loadModel(
      { kind: 'hub', repo: 'a/b', revision: 'main', displayName: 'a/b' },
      'wasm',
      'q8',
    )
    await settle()
    expect(wb.tokens.value).toEqual([])
    expect(nodes(wb)).toBe(1)
    wb.setTokens([8, 4], 'typed')
    await wb.loadModel(
      {
        kind: 'local',
        displayName: 'tiny',
        files: [],
        subfolder: 'onnx',
        baseName: 'model',
        externalDataChunks: 0,
      },
      'wasm',
      'fp32',
    )
    await settle()
    expect(wb.tokens.value).toEqual([])
    expect(nodes(wb)).toBe(1)
  })

  it('a failed load keeps the current context', async () => {
    const { wb, engine } = await makeWorkbench({ text: 'The cat' })
    const before = [...wb.tokens.value]
    engine.failLoad = 'Failed to fetch'
    await wb.loadCatalog('gpt2', { backend: 'wasm', dtype: 'fp16' })
    expect(wb.tokens.value).toEqual(before)
  })

  it('keeps an imported experiment when its own model is loaded next', async () => {
    const { wb: source } = await makeWorkbench({ text: 'The cat sat' })
    source.pickToken(4)
    const json = source.exportExperiment()
    const tokens = [...source.tokens.value]

    const wb = createWorkbench(new FakeEngine(), { appVersion: 'test' })
    await wb.init()
    wb.importExperiment(json)
    const imported = wb.getTree().nodes.size
    await wb.loadCatalog('smollm2-135m-instruct', { backend: 'wasm', dtype: 'fp16' }) // other dtype, same model
    await settle()
    expect(wb.tokens.value).toEqual(tokens)
    expect(wb.getTree().nodes.size).toBe(imported)
    // …but a further model load starts fresh again
    await wb.loadCatalog('gpt2', { backend: 'wasm', dtype: 'fp16' })
    await settle()
    expect(wb.getTree().nodes.size).toBe(2)
  })

  it('an imported experiment is not kept for a different model', async () => {
    const { wb: source } = await makeWorkbench({ text: 'The cat sat' })
    const json = source.exportExperiment()
    const wb = createWorkbench(new FakeEngine(), { appVersion: 'test' })
    await wb.init()
    wb.importExperiment(json)
    await wb.loadCatalog('gpt2', { backend: 'wasm', dtype: 'fp16' })
    await settle()
    expect(wb.loadedExample.value?.label).toBe('Text completion')
    expect(wb.getTree().nodes.size).toBe(2)
  })

  it('the Example select shows its placeholder and the loaded default example after a load', async () => {
    const { wb } = await makeWorkbench()
    const w = mountWith(ContextWorkspace, wb)
    await w.get('[data-testid="example-select"]').setValue('special')
    await settle()
    expect(w.get('[data-testid="loaded-example"]').text()).toContain('Special-token demo')
    wb.pickToken(4)
    await settle()
    expect(w.get('[data-testid="loaded-example"]').text()).toContain('(edited since)')
    await wb.loadCatalog('gpt2', { backend: 'wasm', dtype: 'fp16' })
    await settle()
    expect((w.get('[data-testid="example-select"]').element as HTMLSelectElement).value).toBe('')
    expect(w.get('[data-testid="loaded-example"]').text()).toBe('Loaded: Text completion')
    w.unmount()
  })
})
