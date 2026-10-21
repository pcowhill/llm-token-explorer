import { describe, expect, it, vi } from 'vitest'
import {
  cachedVariants,
  parseVariantFile,
  recommendDtype,
  repoCacheStatus,
  variantsFromFiles,
} from '@/models/variants'
import { HubApiError, fetchRepoInfo, parseRepoInfo } from '@/models/hubApi'
import type { CachedModel } from '@/models/cacheStorage'

const QWEN35_FILES = [
  { path: 'config.json', size: 1989 },
  { path: 'generation_config.json', size: 120 },
  { path: 'tokenizer.json', size: 19_226_111 },
  { path: 'tokenizer_config.json', size: 10_027 },
  { path: 'model.safetensors', size: 1_504_827_608 },
  { path: 'onnx/model.onnx', size: 390_762 },
  { path: 'onnx/model.onnx_data', size: 2_080_548_864 },
  { path: 'onnx/model.onnx_data_1', size: 996_120_576 },
  { path: 'onnx/model_fp16.onnx', size: 504_653 },
  { path: 'onnx/model_fp16.onnx_data', size: 1_538_441_216 },
  { path: 'onnx/model_q4.onnx', size: 482_783 },
  { path: 'onnx/model_q4.onnx_data', size: 550_764_544 },
  { path: 'onnx/model_q4f16.onnx', size: 597_442 },
  { path: 'onnx/model_q4f16.onnx_data', size: 469_331_968 },
  { path: 'onnx/model_quantized.onnx', size: 496_549 },
  { path: 'onnx/model_quantized.onnx_data', size: 938_450_944 },
]

describe('variantsFromFiles', () => {
  it('parses graph and external-data file names', () => {
    expect(parseVariantFile('onnx/model.onnx')).toEqual({ dtype: 'fp32', chunk: -1 })
    expect(parseVariantFile('onnx/model.onnx_data')).toEqual({ dtype: 'fp32', chunk: 0 })
    expect(parseVariantFile('onnx/model.onnx_data_1')).toEqual({ dtype: 'fp32', chunk: 1 })
    expect(parseVariantFile('onnx/model_quantized.onnx')).toEqual({ dtype: 'q8', chunk: -1 })
    expect(parseVariantFile('onnx/model_q4f16.onnx_data')).toEqual({ dtype: 'q4f16', chunk: 0 })
    expect(parseVariantFile('onnx/decoder_model_merged.onnx')).toBeNull()
    expect(parseVariantFile('onnx/model_unknown.onnx')).toBeNull()
    expect(parseVariantFile('model.onnx')).toBeNull()
  })
  it('groups a repo listing into dtypes with sizes and data chunks (Qwen3.5)', () => {
    const v = variantsFromFiles(QWEN35_FILES)
    expect(v.map((x) => x.dtype)).toEqual(['fp32', 'fp16', 'q8', 'q4', 'q4f16'])
    expect(v[0]).toEqual({
      dtype: 'fp32',
      files: ['onnx/model.onnx', 'onnx/model.onnx_data', 'onnx/model.onnx_data_1'],
      sizeBytes: 3_077_060_202,
      externalDataChunks: 2,
    })
    expect(v.find((x) => x.dtype === 'q8')).toMatchObject({
      sizeBytes: 938_947_493,
      externalDataChunks: 1,
    })
  })
  it('ignores data chunks without their graph file', () => {
    expect(variantsFromFiles([{ path: 'onnx/model_q4.onnx_data', size: 5 }])).toEqual([])
  })
})

describe('repoCacheStatus', () => {
  const cached = (paths: string[]): CachedModel[] => [
    {
      repo: 'o/r',
      revision: 'abc',
      files: paths.map((p) => ({ url: `u/${p}`, path: p, bytes: 1 })),
      totalBytes: paths.length,
    },
  ]
  const variants = [
    {
      dtype: 'fp32' as const,
      files: ['onnx/model.onnx', 'onnx/model.onnx_data', 'onnx/model.onnx_data_1'],
    },
    { dtype: 'q8' as const, files: ['onnx/model_quantized.onnx'] },
  ]
  const support = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
  it('marks a dtype cached only when every chunk and the support files are cached', () => {
    const s = repoCacheStatus(
      cached([...support, 'onnx/model.onnx', 'onnx/model.onnx_data', 'onnx/model_quantized.onnx']),
      'o/r',
      'abc',
      variants,
    )
    expect(s.dtypes).toEqual({ fp32: 'partial', q8: 'cached' })
    expect(s.cachedDtypes).toEqual(['q8'])
    expect(s.supportCached).toBe(true)
  })
  it('weights without the tokenizer are only partial', () => {
    const s = repoCacheStatus(cached(['onnx/model_quantized.onnx']), 'o/r', 'abc', variants)
    expect(s.dtypes.q8).toBe('partial')
    expect(s.supportCached).toBe(false)
  })
  it('is per revision', () => {
    const s = repoCacheStatus(
      cached([...support, 'onnx/model_quantized.onnx']),
      'o/r',
      'zzz',
      variants,
    )
    expect(s.dtypes).toEqual({ fp32: 'none', q8: 'none' })
  })
  it('infers cached variants of arbitrary repos from the cached files', () => {
    const [entry] = cached([...support, 'onnx/model_q4.onnx', 'onnx/model_q4.onnx_data'])
    expect(cachedVariants(entry!)).toEqual([{ dtype: 'q4', complete: true }])
  })
})

describe('recommendDtype', () => {
  const v = variantsFromFiles(QWEN35_FILES)
  it('picks the most faithful variant that is likely to fit', () => {
    expect(recommendDtype(v, 'webgpu', true)?.dtype).toBe('fp32')
    expect(recommendDtype(v, 'wasm', false)?.dtype).toBe('q8')
    const small = variantsFromFiles([
      { path: 'onnx/model.onnx', size: 2_000_000_000 },
      { path: 'onnx/model_fp16.onnx', size: 300_000_000 },
      { path: 'onnx/model_q4f16.onnx', size: 100_000_000 },
    ])
    expect(recommendDtype(small, 'wasm', false)?.dtype).toBe('fp16')
    expect(recommendDtype(small, 'webgpu', false)?.dtype).toBe('fp32')
    expect(recommendDtype([small[2]!], 'webgpu', false)).toBeNull() // q4f16 needs shader-f16
  })
})

describe('Hub API', () => {
  const body = {
    sha: '25cdf33f386b487a25f3d743e86442517bdb77ff',
    gated: false as const,
    config: { model_type: 'qwen3_5_text', architectures: ['Qwen3_5ForCausalLM'] },
    siblings: QWEN35_FILES.map((f) => ({ rfilename: f.path, size: f.size })),
  }
  it('reports variants, the resolved SHA and no problems for a compatible repo', () => {
    const r = parseRepoInfo('onnx-community/Qwen3.5-0.8B-Text-ONNX', 'main', body)
    expect(r.sha).toBe(body.sha)
    expect(r.modelType).toBe('qwen3_5_text')
    expect(r.variants).toHaveLength(5)
    expect(r.errors).toEqual([])
    expect(r.warnings).toEqual([])
  })
  it('flags gated repos, missing files, non-ONNX repos and unknown architectures', () => {
    expect(parseRepoInfo('a/b', 'main', { ...body, gated: 'manual' }).errors[0]).toMatch(/gated/)
    const noTok = parseRepoInfo('a/b', 'main', {
      ...body,
      siblings: body.siblings.filter((s) => s.rfilename !== 'tokenizer.json'),
    })
    expect(noTok.errors).toContain('Missing tokenizer.json.')
    const pt = parseRepoInfo('a/b', 'main', {
      config: { model_type: 'llama' },
      siblings: [{ rfilename: 'config.json' }, { rfilename: 'model.safetensors' }],
    })
    expect(pt.errors.join(' ')).toMatch(/No ONNX files/)
    const bert = parseRepoInfo('a/b', 'main', {
      ...body,
      config: { model_type: 'bert', architectures: ['BertModel'] },
    })
    expect(bert.warnings).toHaveLength(2)
  })
  const respond = (status: number, json: unknown) =>
    vi.fn(async () => new Response(JSON.stringify(json), { status }))
  it('fetches the revision endpoint with blobs', async () => {
    const f = respond(200, body)
    const r = await fetchRepoInfo('o/r', 'main', { fetchFn: f as unknown as typeof fetch })
    expect(f).toHaveBeenCalledWith(
      'https://huggingface.co/api/models/o/r/revision/main?blobs=true',
      expect.anything(),
    )
    expect(r.variants).toHaveLength(5)
  })
  it('maps HTTP errors to clear messages', async () => {
    const run = (f: unknown) => fetchRepoInfo('o/r', 'bad', { fetchFn: f as typeof fetch })
    await expect(
      run(respond(401, { error: 'Invalid username or password.' })),
    ).rejects.toMatchObject({
      kind: 'not-found',
    })
    await expect(run(respond(404, { error: 'Invalid rev id: bad' }))).rejects.toMatchObject({
      kind: 'revision',
      message: expect.stringContaining('bad'),
    })
    await expect(run(respond(500, {}))).rejects.toBeInstanceOf(HubApiError)
    await expect(
      run(
        vi.fn(async () => {
          throw new TypeError('Failed to fetch')
        }),
      ),
    ).rejects.toMatchObject({ kind: 'network' })
  })
})
