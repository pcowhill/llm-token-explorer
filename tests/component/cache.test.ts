import { afterEach, describe, expect, it, vi } from 'vitest'
import ModelPanel from '@/components/model/ModelPanel.vue'
import { catalogModel, formatBytes } from '@/models/catalog'
import { makeWorkbench, mountWith, settle } from './helpers'
import { installFakeCaches, resolveUrl } from './fakeCaches'

const SMOL = catalogModel('smollm2-135m-instruct')!
const QWEN35 = catalogModel('qwen3.5-0.8b')!
const support = ['config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json']
const urls = (m: typeof SMOL, paths: string[]) =>
  paths.map((p) => resolveUrl(m.repo, m.revision, p))
const optionText = (w: ReturnType<typeof mountWith>, key: string, dtype: string) =>
  w.get(`[data-testid="dtype-${key}"] option[value="${dtype}"]`).text()

afterEach(() => vi.unstubAllGlobals())

describe('per-quantization cache state', () => {
  it('marks cached dtypes in the dropdown, the Load button and the Cached badge', async () => {
    installFakeCaches([
      ...urls(SMOL, support),
      ...urls(SMOL, ['onnx/model_fp16.onnx', 'onnx/model_quantized.onnx']),
    ])
    const { wb } = await makeWorkbench({ loaded: false })
    const w = mountWith(ModelPanel, wb)
    await settle(6)
    const card = w.get('[data-testid="model-card-smollm2-135m-instruct"]')
    expect(card.get('[data-testid="cached-badge"]').text()).toBe('Cached: fp16, q8')
    expect(optionText(w, SMOL.key, 'fp16')).toContain('✓ cached')
    expect(optionText(w, SMOL.key, 'q8')).toContain('✓ cached')
    const fp32 = SMOL.variants.find((v) => v.dtype === 'fp32')!
    expect(optionText(w, SMOL.key, 'fp32')).toContain(`↓ ${formatBytes(fp32.sizeBytes)}`)
    // WASM default is fp16, which is cached
    const load = card.get('[data-testid="load-smollm2-135m-instruct"]')
    expect(load.text()).toBe('Load model (cached)')
    await card.get(`[data-testid="dtype-${SMOL.key}"]`).setValue('fp32')
    expect(load.text()).toBe(`Download & load (${formatBytes(fp32.sizeBytes)})`)
    w.unmount()
  })

  it('counts the support files in the download size and lists dtypes in the Cache tab', async () => {
    installFakeCaches(urls(SMOL, ['onnx/model_quantized.onnx']))
    const { wb } = await makeWorkbench({ loaded: false })
    const w = mountWith(ModelPanel, wb)
    await settle(6)
    // Weights cached but tokenizer/config missing: still partial, the support files remain.
    expect(optionText(w, SMOL.key, 'q8')).toContain(
      `↓ ${formatBytes(SMOL.supportFilesBytes)} (partly cached)`,
    )
    expect(optionText(w, SMOL.key, 'fp16')).toContain(
      `↓ ${formatBytes(SMOL.variants[0]!.sizeBytes + SMOL.supportFilesBytes)}`,
    )
    expect(
      w
        .find('[data-testid="model-card-smollm2-135m-instruct"] [data-testid="cached-badge"]')
        .exists(),
    ).toBe(false)
    expect(w.get(`[data-testid="cache-dtypes-${SMOL.repo}"]`).text()).toBe('q8 (partial)')
    w.unmount()
  })

  it('treats a multi-file (external data) variant as cached only with every chunk', async () => {
    installFakeCaches([
      ...urls(QWEN35, support),
      ...urls(QWEN35, ['onnx/model.onnx', 'onnx/model.onnx_data']), // _data_1 missing
      ...urls(QWEN35, ['onnx/model_quantized.onnx', 'onnx/model_quantized.onnx_data']),
    ])
    const { wb } = await makeWorkbench({ loaded: false })
    wb.backendPref.value = 'webgpu'
    const w = mountWith(ModelPanel, wb)
    await settle(6)
    expect(optionText(w, QWEN35.key, 'fp32')).toContain('(partly cached)')
    expect(optionText(w, QWEN35.key, 'q8')).toContain('✓ cached')
    expect(
      w.get('[data-testid="model-card-qwen3.5-0.8b"] [data-testid="cached-badge"]').text(),
    ).toBe('Cached: q8')
    expect(w.get(`[data-testid="cache-dtypes-${QWEN35.repo}"]`).text()).toBe('fp32 (partial), q8')
    w.unmount()
  })

  it('shows "Cached: tokenizer" when only the tokenizer files are cached', async () => {
    installFakeCaches(urls(SMOL, support))
    const { wb } = await makeWorkbench({ loaded: false })
    const w = mountWith(ModelPanel, wb)
    await settle(6)
    expect(
      w.get('[data-testid="model-card-smollm2-135m-instruct"] [data-testid="cached-badge"]').text(),
    ).toBe('Cached: tokenizer')
    expect(w.get(`[data-testid="cache-dtypes-${SMOL.repo}"]`).text()).toBe('tokenizer/config only')
    w.unmount()
  })
})

describe('catalog cards', () => {
  it('lists the featured model first with its badge', async () => {
    const { wb } = await makeWorkbench({ loaded: false })
    const w = mountWith(ModelPanel, wb)
    const cards = w.findAll('[data-testid^="model-card-"]')
    expect(cards[0]!.attributes('data-testid')).toBe('model-card-qwen3.5-0.8b')
    expect(cards[0]!.get('[data-testid="featured-badge"]').text()).toBe('★ Patrick’s Choice')
    expect(w.findAll('[data-testid="featured-badge"]')).toHaveLength(1)
    w.unmount()
  })

  it('explains why Qwen3.5 cannot run on WASM and disables loading there', async () => {
    const { wb } = await makeWorkbench({ loaded: false })
    const w = mountWith(ModelPanel, wb)
    const card = w.get('[data-testid="model-card-qwen3.5-0.8b"]')
    expect(card.get('[data-testid="backend-note-qwen3.5-0.8b"]').text()).toContain(
      'GatherBlockQuantized',
    )
    const load = card.get('[data-testid="load-qwen3.5-0.8b"]')
    expect(load.attributes('disabled')).toBeDefined()
    expect(load.text()).toBe('Needs WebGPU')
    // The tokenizer alone still works on WASM
    expect(
      card.get('[data-testid="load-tokenizer-qwen3.5-0.8b"]').attributes('disabled'),
    ).toBeUndefined()
    w.unmount()
  })

  it('explains the variants that are not offered on WebGPU and never lists them as runnable', async () => {
    const { wb } = await makeWorkbench({ loaded: false })
    wb.backendPref.value = 'webgpu'
    const w = mountWith(ModelPanel, wb)
    await settle()
    const card = w.get('[data-testid="model-card-smollm2-135m-instruct"]')
    const excluded = card.get('[data-testid="excluded-smollm2-135m-instruct"]')
    expect(excluded.text()).toContain('Not offered on WebGPU: fp16, q4f16')
    expect(excluded.text()).toContain('exactly 0')
    const opt = (d: string) =>
      card.get(`[data-testid="dtype-smollm2-135m-instruct"] option[value="${d}"]`)
    expect(opt('fp16').attributes('disabled')).toBeDefined()
    expect(opt('q4f16').attributes('disabled')).toBeDefined()
    expect(opt('fp16').text()).toContain('WASM only')
    expect(
      (card.get('[data-testid="dtype-smollm2-135m-instruct"]').element as HTMLSelectElement).value,
    ).toBe('fp32')
    // f16 variants on WebGPU are labelled as not verified in CI
    expect(w.get('[data-testid="dtype-qwen3.5-0.8b"] option[value="fp16"]').text()).toContain(
      'not verified in CI',
    )
    w.unmount()
  })
})
