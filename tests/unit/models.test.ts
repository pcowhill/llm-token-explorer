import { describe, expect, it } from 'vitest'
import { analyzeLocalFiles, checkConfigJson, parseOnnxFileName } from '@/models/localFiles'
import { CATALOG, defaultVariant, formatBytes } from '@/models/catalog'
import { parseCacheUrl } from '@/models/cacheStorage'
import { chooseDtype, resolveBackend } from '@/stores/workbench'
import type { Capabilities } from '@/types'

describe('parseOnnxFileName', () => {
  it('maps dtype suffixes', () => {
    expect(parseOnnxFileName('model_quantized.onnx')).toEqual({ baseName: 'model', dtype: 'q8' })
    expect(parseOnnxFileName('model_q4f16.onnx')).toEqual({ baseName: 'model', dtype: 'q4f16' })
    expect(parseOnnxFileName('model_q4.onnx')).toEqual({ baseName: 'model', dtype: 'q4' })
    expect(parseOnnxFileName('model.onnx')).toEqual({ baseName: 'model', dtype: 'fp32' })
    expect(parseOnnxFileName('decoder_model_merged_fp16.onnx')).toEqual({
      baseName: 'decoder_model_merged',
      dtype: 'fp16',
    })
    expect(parseOnnxFileName('config.json')).toBeNull()
  })
})

describe('analyzeLocalFiles', () => {
  const mk = (path: string, size = 100) => ({ path, size })
  it('accepts a standard directory selection and strips the root folder', () => {
    const r = analyzeLocalFiles([
      mk('my-model/config.json'),
      mk('my-model/tokenizer.json'),
      mk('my-model/tokenizer_config.json'),
      mk('my-model/generation_config.json'),
      mk('my-model/onnx/model_quantized.onnx', 130e6),
      mk('my-model/onnx/model_q4f16.onnx', 110e6),
      mk('my-model/model.safetensors'),
    ])
    expect(r.errors).toEqual([])
    expect(r.root).toBe('my-model/')
    expect(r.candidates.map((c) => c.dtype)).toEqual(['q4f16', 'q8'])
    expect(r.candidates[0]).toMatchObject({ subfolder: 'onnx', baseName: 'model' })
    expect(r.warnings.some((w) => w.includes('safetensors'))).toBe(true)
  })
  it('detects external data chunks', () => {
    const r = analyzeLocalFiles([
      mk('config.json'),
      mk('tokenizer.json'),
      mk('tokenizer_config.json'),
      mk('onnx/model_q4.onnx', 200_000),
      mk('onnx/model_q4.onnx_data', 300e6),
      mk('onnx/model_q4.onnx_data_1', 20e6),
    ])
    expect(r.candidates[0]).toMatchObject({ externalDataChunks: 2, missingExternalData: [] })
    expect(r.candidates[0]!.totalBytes).toBe(200_000 + 300e6 + 20e6)
  })
  it('warns when a tiny graph has no external data file', () => {
    const r = analyzeLocalFiles([
      mk('config.json'),
      mk('tokenizer.json'),
      mk('tokenizer_config.json'),
      mk('model.onnx', 1000),
    ])
    expect(r.candidates[0]?.subfolder).toBe('')
    expect(r.warnings.some((w) => w.includes('external data'))).toBe(true)
  })
  it('reports missing required files and missing weights', () => {
    const r = analyzeLocalFiles([mk('config.json')])
    expect(r.errors).toContain('Missing required file: tokenizer.json')
    expect(r.errors).toContain('Missing required file: tokenizer_config.json')
    expect(r.errors.some((e) => e.includes('No .onnx'))).toBe(true)
  })
})

describe('checkConfigJson', () => {
  it('flags unknown architectures and invalid JSON', () => {
    expect(checkConfigJson('{').errors).toHaveLength(1)
    expect(checkConfigJson('{"model_type":"llama","vocab_size":10}')).toMatchObject({
      modelType: 'llama',
      vocabSize: 10,
      errors: [],
      warnings: [],
    })
    expect(
      checkConfigJson('{"model_type":"bert","architectures":["BertModel"]}').warnings,
    ).toHaveLength(3) // unknown type, not causal, no vocab_size
  })
})

describe('catalog', () => {
  it('pins every model to a commit SHA and lists sources and licenses', () => {
    expect(CATALOG.length).toBeGreaterThanOrEqual(2)
    for (const m of CATALOG) {
      expect(m.revision).toMatch(/^[0-9a-f]{40}$/)
      expect(m.license.name).toBeTruthy()
      expect(m.variants.length).toBeGreaterThan(0)
      expect(m.examples.length).toBeGreaterThan(2)
      expect(defaultVariant(m, 'wasm')?.backends).toContain('wasm')
    }
  })
  it('formats byte sizes', () => {
    expect(formatBytes(512)).toBe('512 B')
    expect(formatBytes(135_700_000)).toBe('129 MB')
    expect(formatBytes(null)).toBe('—')
  })
})

describe('backend and dtype selection', () => {
  const caps = (o: Partial<Capabilities>): Capabilities => ({
    webgpu: false,
    webgpuAdapter: null,
    wasm: true,
    wasmSimd: true,
    crossOriginIsolated: false,
    hardwareConcurrency: 4,
    deviceMemoryGb: null,
    cacheStorage: true,
    opfs: true,
    serviceWorker: true,
    ...o,
  })
  it('auto picks WebGPU only when a non-fallback adapter exists', () => {
    expect(resolveBackend('auto', caps({}))).toBe('wasm')
    expect(
      resolveBackend('auto', caps({ webgpu: true, webgpuAdapter: { isFallbackAdapter: false } })),
    ).toBe('webgpu')
    expect(
      resolveBackend('auto', caps({ webgpu: true, webgpuAdapter: { isFallbackAdapter: true } })),
    ).toBe('wasm')
    expect(
      resolveBackend(
        'auto',
        caps({ webgpu: true, webgpuAdapter: { architecture: 'swiftshader' } }),
      ),
    ).toBe('wasm')
    expect(resolveBackend('webgpu', null)).toBe('webgpu')
  })
  it('avoids f16 variants when the GPU lacks shader-f16', () => {
    const m = CATALOG[0]!
    expect(
      chooseDtype(m, 'webgpu', caps({ webgpu: true, webgpuAdapter: { shaderF16: true } })),
    ).toBe('q4f16')
    expect(
      chooseDtype(m, 'webgpu', caps({ webgpu: true, webgpuAdapter: { shaderF16: false } })),
    ).toBe('fp32')
    expect(chooseDtype(m, 'wasm', caps({}))).toBe('fp16')
  })
})

describe('parseCacheUrl', () => {
  it('extracts repo, revision and path', () => {
    expect(
      parseCacheUrl(
        'https://huggingface.co/onnx-community/gpt2-ONNX/resolve/abc123/onnx/model_quantized.onnx',
      ),
    ).toEqual({
      repo: 'onnx-community/gpt2-ONNX',
      revision: 'abc123',
      path: 'onnx/model_quantized.onnx',
    })
    expect(parseCacheUrl('https://example.com/x')).toBeNull()
  })
})
