import type { Capabilities } from '@/types'

/** Minimal module using SIMD (i8x16.splat + i8x16.popcnt), from wasm-feature-detect. */
const SIMD_PROBE = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11,
])

interface GpuAdapterLike {
  info?: { vendor?: string; architecture?: string; description?: string }
  limits?: { maxBufferSize?: number; maxStorageBufferBindingSize?: number }
  features?: { has(name: string): boolean }
  isFallbackAdapter?: boolean
}

/**
 * Detect browser capabilities relevant to in-browser inference. Never throws.
 * GPU memory is NOT exposed by WebGPU; only buffer-size limits are reported.
 */
export async function detectCapabilities(): Promise<Capabilities> {
  const nav = (typeof navigator !== 'undefined' ? navigator : {}) as unknown as {
    gpu?: { requestAdapter(o?: unknown): Promise<GpuAdapterLike | null> }
    deviceMemory?: number
    hardwareConcurrency?: number
    storage?: StorageManager
  }
  let webgpuAdapter: Capabilities['webgpuAdapter'] = null
  let webgpuError: string | undefined
  if (nav.gpu) {
    try {
      const adapter = await nav.gpu.requestAdapter({ powerPreference: 'high-performance' })
      if (adapter) {
        webgpuAdapter = {
          vendor: adapter.info?.vendor || undefined,
          architecture: adapter.info?.architecture || undefined,
          description: adapter.info?.description || undefined,
          maxBufferSize: adapter.limits?.maxBufferSize,
          maxStorageBufferBindingSize: adapter.limits?.maxStorageBufferBindingSize,
          isFallbackAdapter: adapter.isFallbackAdapter,
          shaderF16: adapter.features?.has('shader-f16') ?? false,
        }
      } else webgpuError = 'navigator.gpu exists but no GPU adapter is available.'
    } catch (e) {
      webgpuError = `WebGPU adapter request failed: ${(e as Error).message}`
    }
  } else webgpuError = 'WebGPU (navigator.gpu) is not available in this browser/context.'

  const wasm = typeof WebAssembly === 'object' && typeof WebAssembly.instantiate === 'function'
  let wasmSimd = false
  if (wasm) {
    try {
      wasmSimd = WebAssembly.validate(SIMD_PROBE)
    } catch {
      wasmSimd = false
    }
  }
  let opfs = false
  try {
    opfs = typeof nav.storage?.getDirectory === 'function'
  } catch {
    opfs = false
  }
  return {
    webgpu: webgpuAdapter !== null,
    webgpuAdapter,
    ...(webgpuError ? { webgpuError } : {}),
    wasm,
    wasmSimd,
    crossOriginIsolated: typeof crossOriginIsolated !== 'undefined' ? crossOriginIsolated : false,
    hardwareConcurrency: typeof nav.hardwareConcurrency === 'number' ? nav.hardwareConcurrency : null,
    deviceMemoryGb: typeof nav.deviceMemory === 'number' ? nav.deviceMemory : null,
    cacheStorage: typeof caches !== 'undefined',
    opfs,
    serviceWorker: 'serviceWorker' in nav,
  }
}
