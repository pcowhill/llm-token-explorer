import { afterEach, describe, expect, it, vi } from 'vitest'
import HubRepoLoader from '@/components/model/HubRepoLoader.vue'
import { formatBytes } from '@/models/catalog'
import { makeWorkbench, mountWith, settle } from './helpers'
import { installFakeCaches, resolveUrl } from './fakeCaches'
import type { CachedModel } from '@/models/cacheStorage'

const REPO = 'onnx-community/Qwen3.5-0.8B-Text-ONNX'
const SHA = '25cdf33f386b487a25f3d743e86442517bdb77ff'
const FILES: [string, number][] = [
  ['config.json', 1989],
  ['generation_config.json', 120],
  ['tokenizer.json', 19_226_111],
  ['tokenizer_config.json', 10_027],
  ['onnx/model.onnx', 390_762],
  ['onnx/model.onnx_data', 2_080_548_864],
  ['onnx/model.onnx_data_1', 996_120_576],
  ['onnx/model_quantized.onnx', 496_549],
  ['onnx/model_quantized.onnx_data', 938_450_944],
  ['onnx/model_q4f16.onnx', 597_442],
  ['onnx/model_q4f16.onnx_data', 469_331_968],
]
const apiBody = (over: Record<string, unknown> = {}) => ({
  sha: SHA,
  gated: false,
  config: { model_type: 'qwen3_5_text', architectures: ['Qwen3_5ForCausalLM'] },
  siblings: FILES.map(([rfilename, size]) => ({ rfilename, size })),
  ...over,
})
function mockFetch(handler: (url: string) => Response | Promise<Response>) {
  const f = vi.fn(async (url: string) => handler(url))
  vi.stubGlobal('fetch', f)
  return f
}
const json = (status: number, body: unknown) =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })

async function mountLoader(cached: CachedModel[] = [], backend: 'wasm' | 'webgpu' = 'webgpu') {
  const { wb, engine } = await makeWorkbench({ loaded: false })
  wb.capabilities.value = {
    ...wb.capabilities.value!,
    webgpu: true,
    webgpuAdapter: { shaderF16: false },
  }
  const w = mountWith(HubRepoLoader, wb, { props: { backend, cached } })
  return { wb, engine, w }
}
async function enter(w: ReturnType<typeof mountWith>, repo: string, revision = 'main') {
  await w.get('[data-testid="hub-repo"]').setValue(repo)
  await w.get('[data-testid="hub-revision"]').setValue(revision)
  await new Promise((r) => setTimeout(r, 500)) // input debounce
  await settle()
}
const options = (w: ReturnType<typeof mountWith>) =>
  w.findAll('[data-testid="hub-dtype"] option').map((o) => o.text())

afterEach(() => vi.unstubAllGlobals())

describe('HubRepoLoader', () => {
  it('shows guidance (onnx-community link, compatibility, revision, dtype, memory) without the old warning', async () => {
    const { w } = await mountLoader()
    const guide = w.get('[data-testid="hub-guide"]')
    expect(guide.find('a[href="https://huggingface.co/onnx-community"]').exists()).toBe(true)
    for (const s of [
      'onnx/model*.onnx',
      'tokenizer.json',
      'config.json',
      'model_type',
      'commit SHA',
      'shader-f16',
      '32-bit',
      'Not curated',
    ])
      expect(guide.text()).toContain(s)
    expect(w.text()).not.toMatch(/Experimental and unverified/)
    w.unmount()
  })

  it('queries the Hub API and offers only the available dtypes with sizes', async () => {
    const f = mockFetch(() => json(200, apiBody()))
    const { w } = await mountLoader()
    await enter(w, REPO)
    expect(f).toHaveBeenCalledWith(
      `https://huggingface.co/api/models/${REPO}/revision/main?blobs=true`,
      expect.anything(),
    )
    const opts = options(w)
    expect(opts).toHaveLength(3)
    const supportBytes = 1989 + 120 + 19_226_111 + 10_027
    expect(opts[0]).toBe(
      `fp32 — ↓ ${formatBytes(390_762 + 2_080_548_864 + 996_120_576 + supportBytes)} · 3 files · recommended`,
    )
    expect(opts[1]).toContain('q8 — ↓')
    expect(opts[2]).toContain('q4f16')
    expect(opts[2]).toContain('needs shader-f16')
    const status = w.get('[data-testid="hub-status"]').text()
    expect(status).toContain('25cdf33')
    expect(status).toContain('qwen3_5_text')
    expect(status).toContain('recommended for WebGPU: fp32')
    w.unmount()
  })

  it('marks cached dtypes and loads with the detected external-data chunk count', async () => {
    mockFetch(() => json(200, apiBody()))
    const paths = [
      'config.json',
      'tokenizer.json',
      'tokenizer_config.json',
      'onnx/model_quantized.onnx',
      'onnx/model_quantized.onnx_data',
    ]
    const cached: CachedModel[] = [
      {
        repo: REPO,
        revision: 'main',
        totalBytes: 1,
        files: paths.map((p) => ({ path: p, url: resolveUrl(REPO, 'main', p), bytes: 1 })),
      },
    ]
    const { w, engine } = await mountLoader(cached)
    await enter(w, REPO)
    expect(options(w)[1]).toContain('q8 — ✓ cached')
    await w.get('[data-testid="hub-dtype"]').setValue('q8')
    expect(w.get('[data-testid="hub-load"]').text()).toBe('Load model (cached)')
    await w.get('[data-testid="hub-dtype"]').setValue('fp32')
    expect(w.get('[data-testid="hub-load"]').text()).toMatch(/^Download & load \(2\.9 GB\)$/)
    await w.get('[data-testid="hub-load"]').trigger('click')
    await settle()
    expect(engine.loadCalls[0]).toMatchObject({
      dtype: 'fp32',
      device: 'webgpu',
      source: { kind: 'hub', repo: REPO, revision: 'main', externalDataChunks: 2 },
    })
    w.unmount()
  })

  it('explains missing repos, bad revisions and gated repos and blocks loading', async () => {
    mockFetch((url) =>
      url.includes('/nope/')
        ? json(401, { error: 'Invalid username or password.' })
        : url.includes('/revision/badrev')
          ? json(404, { error: 'Invalid rev id: badrev' })
          : json(200, apiBody({ gated: 'manual' })),
    )
    const { w } = await mountLoader()
    await enter(w, 'nope/missing')
    expect(w.get('[data-testid="hub-status"]').text()).toContain('was not found')
    expect(w.get('[data-testid="hub-load"]').attributes('disabled')).toBeDefined()
    await enter(w, REPO, 'badrev')
    expect(w.get('[data-testid="hub-status"]').text()).toContain('Revision “badrev” does not exist')
    await enter(w, 'meta-llama/Gated-ONNX')
    expect(w.get('[data-testid="hub-status"]').text()).toContain('gated')
    expect(w.get('[data-testid="hub-load"]').attributes('disabled')).toBeDefined()
    w.unmount()
  })

  it('flags repositories that are not Transformers.js ONNX exports', async () => {
    mockFetch(() =>
      json(
        200,
        apiBody({ siblings: [{ rfilename: 'config.json' }, { rfilename: 'model.safetensors' }] }),
      ),
    )
    const { w } = await mountLoader()
    await enter(w, 'someone/pytorch-only')
    expect(w.get('[data-testid="hub-status"]').text()).toContain('No ONNX files')
    expect(w.get('[data-testid="hub-load"]').attributes('disabled')).toBeDefined()
    w.unmount()
  })

  it('offline: falls back to the quantizations already in the cache', async () => {
    installFakeCaches([])
    mockFetch(() => {
      throw new TypeError('Failed to fetch')
    })
    const onLine = vi.spyOn(navigator, 'onLine', 'get').mockReturnValue(false)
    const paths = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'onnx/model_q4.onnx']
    const cached: CachedModel[] = [
      {
        repo: 'o/r',
        revision: 'main',
        totalBytes: 1,
        files: paths.map((p) => ({ path: p, url: resolveUrl('o/r', 'main', p), bytes: 5 })),
      },
    ]
    const { w, engine } = await mountLoader(cached, 'wasm')
    await enter(w, 'o/r')
    expect(w.get('[data-testid="hub-status"]').text()).toContain('offline')
    expect(w.get('[data-testid="hub-status"]').text()).toContain('already in the browser cache')
    expect(options(w)).toEqual(['q4 — ✓ cached'])
    expect(w.get('[data-testid="hub-load"]').text()).toBe('Load model (cached)')
    await w.get('[data-testid="hub-load"]').trigger('click')
    await settle()
    expect(engine.loadCalls[0]).toMatchObject({ dtype: 'q4', source: { repo: 'o/r' } })
    onLine.mockRestore()
    w.unmount()
  })

  it('network error without cache: offers every dtype so loading can still be tried', async () => {
    mockFetch(() => {
      throw new TypeError('Failed to fetch')
    })
    const { w } = await mountLoader()
    await enter(w, 'o/r')
    expect(w.get('[data-testid="hub-status"]').text()).toContain(
      'Could not reach the Hugging Face API',
    )
    expect(options(w).map((o) => o.split(' ')[0])).toEqual([
      'fp32',
      'fp16',
      'q8',
      'int8',
      'uint8',
      'bnb4',
      'q4',
      'q4f16',
    ])
    expect(w.get('[data-testid="hub-load"]').attributes('disabled')).toBeUndefined()
    w.unmount()
  })

  it('ignores a stale response when the repo changes during the request', async () => {
    let release: (r: Response) => void = () => {}
    mockFetch((url) =>
      url.includes('/slow/') ? new Promise<Response>((r) => (release = r)) : json(200, apiBody()),
    )
    const { w } = await mountLoader()
    await enter(w, 'slow/repo')
    await enter(w, REPO)
    release(json(200, apiBody({ siblings: [] })))
    await settle()
    expect(options(w)).toHaveLength(3)
    w.unmount()
  })
})
