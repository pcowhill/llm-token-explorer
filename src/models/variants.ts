/**
 * Quantization variants as files: which dtypes a repository offers (from its file list),
 * and which of them are already in the browser cache.
 *
 * Transformers.js loads `onnx/model<suffix>.onnx` for a dtype (suffixes in
 * DTYPE_FILE_SUFFIX) plus any external-data chunks `onnx/model<suffix>.onnx_data`,
 * `…onnx_data_1`, … Every file is cached under its Hugging Face resolve URL, so a dtype is
 * "cached" exactly when its graph, all its data chunks and the support files are cached.
 */
import type { ModelDtype } from '@/types'
import { DTYPE_FILE_SUFFIX } from './catalog'
import type { CachedModel } from './cacheStorage'

/** Rough fidelity order (most faithful first), used to sort detected variants. */
export const DTYPE_ORDER: ModelDtype[] = [
  'fp32',
  'fp16',
  'q8',
  'int8',
  'uint8',
  'bnb4',
  'q4',
  'q4f16',
]

/** Support files a load needs; generation_config.json is optional. */
export const REQUIRED_SUPPORT_FILES = ['config.json', 'tokenizer.json', 'tokenizer_config.json']

export interface FileVariant {
  dtype: ModelDtype
  /** Graph file first, then data chunks in order. */
  files: string[]
  sizeBytes: number
  /** Number of external-data chunks (0 = weights inside the .onnx file). */
  externalDataChunks: number
}

const SUFFIX_TO_DTYPE = new Map(
  Object.entries(DTYPE_FILE_SUFFIX).map(([d, s]) => [s, d as ModelDtype]),
)
const ONNX_RE = /^onnx\/model(_[a-z0-9]+)?\.onnx(?:_data(?:_(\d+))?)?$/

/** dtype and chunk index of a model file path (graph = -1, `_data` = 0, `_data_N` = N). */
export function parseVariantFile(path: string): { dtype: ModelDtype; chunk: number } | null {
  const m = ONNX_RE.exec(path)
  if (!m) return null
  const dtype = SUFFIX_TO_DTYPE.get(m[1] ?? '')
  if (!dtype) return null
  const isData = path.includes('.onnx_data')
  return { dtype, chunk: isData ? Number(m[2] ?? 0) : -1 }
}

/** Group a repository file listing into the dtypes it can load, most faithful first. */
export function variantsFromFiles(files: { path: string; size: number | null }[]): FileVariant[] {
  const byDtype = new Map<ModelDtype, { path: string; size: number; chunk: number }[]>()
  for (const f of files) {
    const v = parseVariantFile(f.path)
    if (!v) continue
    if (!byDtype.has(v.dtype)) byDtype.set(v.dtype, [])
    byDtype.get(v.dtype)!.push({ path: f.path, size: f.size ?? 0, chunk: v.chunk })
  }
  const out: FileVariant[] = []
  for (const [dtype, list] of byDtype) {
    if (!list.some((f) => f.chunk === -1)) continue // data without a graph is unusable
    list.sort((a, b) => a.chunk - b.chunk)
    out.push({
      dtype,
      files: list.map((f) => f.path),
      sizeBytes: list.reduce((s, f) => s + f.size, 0),
      externalDataChunks: list.length - 1,
    })
  }
  return out.sort((a, b) => DTYPE_ORDER.indexOf(a.dtype) - DTYPE_ORDER.indexOf(b.dtype))
}

export type DtypeCacheState = 'cached' | 'partial' | 'none'

export interface RepoCacheStatus {
  /** Per dtype: every file cached, some files cached, or none. */
  dtypes: Partial<Record<ModelDtype, DtypeCacheState>>
  /** Per dtype: every weight file (graph + data chunks) cached, regardless of support files. */
  weightsCached: Partial<Record<ModelDtype, boolean>>
  /** config.json + tokenizer files cached (enough for "Tokenizer only"). */
  supportCached: boolean
  /** dtypes whose files are all cached (and the support files too), in the given order. */
  cachedDtypes: ModelDtype[]
}

/** Cache state of each variant of one repo@revision, from the cached file list. */
export function repoCacheStatus(
  cached: readonly CachedModel[],
  repo: string,
  revision: string,
  variants: readonly { dtype: ModelDtype; files: readonly string[] }[],
): RepoCacheStatus {
  const entry = cached.find((c) => c.repo === repo && c.revision === revision)
  const have = new Set(entry?.files.map((f) => f.path) ?? [])
  const supportCached = REQUIRED_SUPPORT_FILES.every((f) => have.has(f))
  const dtypes: RepoCacheStatus['dtypes'] = {}
  const weightsCached: RepoCacheStatus['weightsCached'] = {}
  const cachedDtypes: ModelDtype[] = []
  for (const v of variants) {
    const n = v.files.filter((f) => have.has(f)).length
    weightsCached[v.dtype] = v.files.length > 0 && n === v.files.length
    const state: DtypeCacheState =
      n === 0 ? 'none' : n === v.files.length && supportCached ? 'cached' : 'partial'
    dtypes[v.dtype] = state
    if (state === 'cached') cachedDtypes.push(v.dtype)
  }
  return { dtypes, weightsCached, supportCached, cachedDtypes }
}

/** Variants inferred from what is cached for a repo (used offline and in the Cache tab). */
export function cachedVariants(entry: CachedModel): { dtype: ModelDtype; complete: boolean }[] {
  const variants = variantsFromFiles(entry.files.map((f) => ({ path: f.path, size: f.bytes })))
  const status = repoCacheStatus([entry], entry.repo, entry.revision, variants)
  return variants.map((v) => ({ dtype: v.dtype, complete: status.dtypes[v.dtype] === 'cached' }))
}

const GB = 1024 ** 3

/**
 * A sensible dtype for an arbitrary repository (not curated): the most faithful variant that
 * is likely to fit the backend. WASM is a 32-bit heap (≈4 GB including runtime copies) and
 * expands fp16 weights to fp32 on the CPU; WebGPU needs shader-f16 for f16 variants.
 */
export function recommendDtype(
  variants: readonly FileVariant[],
  backend: 'wasm' | 'webgpu',
  shaderF16: boolean,
): FileVariant | null {
  const fits = (v: FileVariant) => {
    if (backend === 'wasm') {
      const effective = v.dtype === 'fp16' ? v.sizeBytes * 2 : v.sizeBytes
      return effective <= 1 * GB
    }
    if (v.dtype.includes('16') && !shaderF16) return false
    return v.sizeBytes <= 4 * GB
  }
  return variants.find(fits) ?? null
}
