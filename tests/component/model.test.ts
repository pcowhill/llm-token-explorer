import { describe, expect, it } from 'vitest'
import ModelPanel from '@/components/model/ModelPanel.vue'
import LocalModelLoader from '@/components/model/LocalModelLoader.vue'
import AppHeader from '@/components/AppHeader.vue'
import { CATALOG } from '@/models/catalog'
import { makeWorkbench, mountWith, settle } from './helpers'

describe('ModelPanel', () => {
  it('lists every catalog model with verification status, license and size', async () => {
    const { wb } = await makeWorkbench({ loaded: false })
    const w = mountWith(ModelPanel, wb)
    for (const m of CATALOG) {
      const card = w.get(`[data-testid="model-card-${m.key}"]`)
      expect(card.text()).toContain(m.name)
      expect(card.text()).toContain(m.license.name)
      expect(card.text()).toMatch(/Verified|partial|documented|experimental/)
      expect(card.find(`a[href*="${m.revision}"]`).exists()).toBe(true)
    }
    w.unmount()
  })

  it('loads the pinned repo/revision/dtype and shows the loaded model', async () => {
    const { wb, engine } = await makeWorkbench({ loaded: false })
    const w = mountWith(ModelPanel, wb)
    await w.get('[data-testid="load-smollm2-135m-instruct"]').trigger('click')
    await settle()
    const req = engine.loadCalls[0]!
    expect(req.source).toMatchObject({
      kind: 'hub',
      repo: CATALOG[0]!.repo,
      revision: CATALOG[0]!.revision,
    })
    expect(req.dtype).toBe('fp16')
    expect(req.device).toBe('wasm')
    expect(w.get('[data-testid="loaded-model"]').text()).toContain('SmolLM2')
    w.unmount()
  })

  it('shows a useful error and allows retrying after a load failure', async () => {
    const { wb, engine } = await makeWorkbench({ loaded: false })
    engine.failLoad = 'Failed to fetch'
    const w = mountWith(ModelPanel, wb)
    await w.get('[data-testid="load-gpt2"]').trigger('click')
    await settle()
    expect(w.get('[data-testid="model-error"]').text()).toContain('Failed to fetch')
    expect(wb.modelStatus.value).toBe('error')
    engine.failLoad = null
    await w.get('[data-testid="load-gpt2"]').trigger('click')
    await settle()
    expect(wb.modelStatus.value).toBe('ready')
    w.unmount()
  })

  it('can load only the tokenizer (no inference)', async () => {
    const { wb, engine } = await makeWorkbench({ loaded: false })
    const w = mountWith(ModelPanel, wb)
    await w.get('[data-testid="load-tokenizer-gpt2"]').trigger('click')
    await settle()
    expect(engine.loadCalls[0]?.tokenizerOnly).toBe(true)
    expect(wb.canInfer.value).toBe(false)
    expect(wb.tokenizer.value).not.toBeNull()
    w.unmount()
  })

  it('header reflects loading state and backend', async () => {
    const { wb } = await makeWorkbench()
    const h = mountWith(AppHeader, wb, {
      props: { online: false, theme: 'system', updateAvailable: false, offlineReady: false },
    })
    expect(h.get('[data-testid="model-status"]').text()).toBe('Ready')
    expect(h.get('[data-testid="header-backend"]').text()).toContain('WASM')
    expect(h.get('[data-testid="online-status"]').text()).toContain('Offline')
    h.unmount()
  })
})

describe('LocalModelLoader', () => {
  const file = (name: string, content = '{}', rel?: string) => {
    const f = new File([content], name)
    if (rel) Object.defineProperty(f, 'webkitRelativePath', { value: rel })
    return f
  }

  it('validates a directory selection and loads it with the detected layout', async () => {
    const { wb, engine } = await makeWorkbench({ loaded: false })
    const w = mountWith(LocalModelLoader, wb, { props: { backend: 'wasm' } })
    const files = [
      file(
        'config.json',
        '{"model_type":"gpt2","vocab_size":1000,"architectures":["GPT2LMHeadModel"]}',
        'tiny/config.json',
      ),
      file('tokenizer.json', '{}', 'tiny/tokenizer.json'),
      file('tokenizer_config.json', '{}', 'tiny/tokenizer_config.json'),
      file('model.onnx', 'x'.repeat(3 * 1024 * 1024), 'tiny/onnx/model.onnx'),
    ]
    const input = w.get('[data-testid="local-dir-input"]')
    Object.defineProperty(input.element, 'files', { value: files })
    await input.trigger('change')
    await settle()
    expect(w.find('[data-testid="local-errors"]').exists()).toBe(false)
    await w.get('[data-testid="local-load"]').trigger('click')
    await settle()
    const src = engine.loadCalls[0]!.source
    expect(src.kind).toBe('local')
    if (src.kind === 'local') {
      expect(src.subfolder).toBe('onnx')
      expect(src.baseName).toBe('model')
      expect(src.files.map((f) => f.path).sort()).toEqual([
        'config.json',
        'onnx/model.onnx',
        'tokenizer.json',
        'tokenizer_config.json',
      ])
    }
    w.unmount()
  })

  it('reports missing files and blocks loading', async () => {
    const { wb } = await makeWorkbench({ loaded: false })
    const w = mountWith(LocalModelLoader, wb, { props: { backend: 'wasm' } })
    const input = w.get('[data-testid="local-files-input"]')
    Object.defineProperty(input.element, 'files', {
      value: [file('config.json', '{"model_type":"llama"}')],
    })
    await input.trigger('change')
    await settle()
    expect(w.get('[data-testid="local-errors"]').text()).toContain('tokenizer.json')
    expect(w.find('[data-testid="local-load"]').exists()).toBe(false)
    w.unmount()
  })
})
