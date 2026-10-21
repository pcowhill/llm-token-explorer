/**
 * Validation and layout detection for user-selected local model files.
 *
 * Supported layout (the standard Transformers.js / Optimum ONNX export):
 *
 *   <dir>/config.json                  required (model_type, vocab_size, …)
 *   <dir>/tokenizer.json               required (vocabulary, merges, added tokens)
 *   <dir>/tokenizer_config.json        required (special tokens, chat template)
 *   <dir>/generation_config.json       optional (stop tokens)
 *   <dir>/onnx/<base><suffix>.onnx     one or more ONNX weight files (or at <dir>/ root)
 *   <dir>/onnx/<base><suffix>.onnx_data[_N]   external weight chunks, if any
 *
 * where <suffix> is '', _fp16, _quantized, _int8, _uint8, _q4, _q4f16 or _bnb4.
 * Files never leave the browser: they are read directly from the File objects.
 */

import { DTYPE_FILE_SUFFIX } from './catalog'
import type { ModelDtype } from '@/types'

export interface LocalFileEntry {
  /** Path relative to the selection root, using '/' separators. */
  path: string
  size: number
}

export interface OnnxCandidate {
  /** Path of the .onnx file relative to the model root. */
  path: string
  subfolder: string
  /** Base file name without dtype suffix (e.g. "model", "decoder_model_merged"). */
  baseName: string
  dtype: ModelDtype
  /** Number of external-data chunks found next to it (<file>.onnx_data, _data_1, …). */
  externalDataChunks: number
  /** Total bytes of the .onnx plus its external data. */
  totalBytes: number
  missingExternalData: string[]
}

export interface LocalLayout {
  /** Directory prefix (relative to selection) that contains config.json. '' = root. */
  root: string
  files: Map<string, LocalFileEntry>
  candidates: OnnxCandidate[]
  errors: string[]
  warnings: string[]
}

export const REQUIRED_FILES = ['config.json', 'tokenizer.json', 'tokenizer_config.json'] as const
export const OPTIONAL_FILES = ['generation_config.json', 'special_tokens_map.json'] as const

const SUFFIX_TO_DTYPE: [string, ModelDtype][] = (
  Object.entries(DTYPE_FILE_SUFFIX) as [ModelDtype, string][]
)
  .filter(([, s]) => s !== '')
  .map(([d, s]) => [s, d] as [string, ModelDtype])
  .sort((a, b) => b[0].length - a[0].length)

export function normalizePath(p: string): string {
  return p.replace(/\\/g, '/').replace(/^\.\//, '').replace(/^\/+/, '')
}

export function parseOnnxFileName(
  fileName: string,
): { baseName: string; dtype: ModelDtype } | null {
  if (!fileName.endsWith('.onnx')) return null
  const stem = fileName.slice(0, -'.onnx'.length)
  for (const [suffix, dtype] of SUFFIX_TO_DTYPE) {
    if (stem.endsWith(suffix) && stem.length > suffix.length) {
      // q4f16 must win over a hypothetical "_f16" etc.; list is sorted by suffix length.
      return { baseName: stem.slice(0, -suffix.length), dtype }
    }
  }
  return { baseName: stem, dtype: 'fp32' }
}

export function analyzeLocalFiles(input: LocalFileEntry[]): LocalLayout {
  const errors: string[] = []
  const warnings: string[] = []
  const all = input.map((f) => ({ ...f, path: normalizePath(f.path) }))

  // Find the shallowest directory containing config.json.
  const configs = all
    .filter((f) => f.path === 'config.json' || f.path.endsWith('/config.json'))
    .sort((a, b) => a.path.split('/').length - b.path.split('/').length)
  const root = configs[0] ? configs[0].path.slice(0, -'config.json'.length) : ''

  const files = new Map<string, LocalFileEntry>()
  for (const f of all) {
    if (root && !f.path.startsWith(root)) continue
    const rel = f.path.slice(root.length)
    files.set(rel, { path: rel, size: f.size })
  }

  for (const req of REQUIRED_FILES) {
    if (!files.has(req)) errors.push(`Missing required file: ${req}`)
  }
  if (!files.has('generation_config.json'))
    warnings.push(
      'generation_config.json not found: stop tokens will fall back to the tokenizer’s EOS token.',
    )

  const candidates: OnnxCandidate[] = []
  for (const f of files.values()) {
    const slash = f.path.lastIndexOf('/')
    const subfolder = slash >= 0 ? f.path.slice(0, slash) : ''
    const name = slash >= 0 ? f.path.slice(slash + 1) : f.path
    const parsed = parseOnnxFileName(name)
    if (!parsed) continue
    if (subfolder.includes('/')) {
      warnings.push(
        `Ignoring ${f.path}: ONNX files must be in the model root or one subfolder (usually onnx/).`,
      )
      continue
    }
    let chunks = 0
    let total = f.size
    const prefix = subfolder ? `${subfolder}/` : ''
    while (files.has(`${prefix}${name}_data${chunks === 0 ? '' : `_${chunks}`}`)) {
      total += files.get(`${prefix}${name}_data${chunks === 0 ? '' : `_${chunks}`}`)!.size
      chunks++
    }
    const missing: string[] = []
    // A small .onnx (< 2 MB) with no data file is almost certainly a graph whose weights are external.
    if (chunks === 0 && f.size < 2 * 1024 * 1024) missing.push(`${prefix}${name}_data`)
    candidates.push({
      path: f.path,
      subfolder,
      baseName: parsed.baseName,
      dtype: parsed.dtype,
      externalDataChunks: chunks,
      totalBytes: total,
      missingExternalData: missing,
    })
  }
  candidates.sort((a, b) => a.totalBytes - b.totalBytes)
  if (candidates.length === 0)
    errors.push('No .onnx weight file found (expected e.g. onnx/model_quantized.onnx).')
  for (const c of candidates) {
    if (c.missingExternalData.length)
      warnings.push(
        `${c.path} is very small and has no external data file next to it (${c.missingExternalData.join(', ')}). If the weights are stored externally, select that file too.`,
      )
  }
  for (const f of files.keys()) {
    if (/\.(safetensors|bin|gguf|pt|pth|h5)$/i.test(f))
      warnings.push(
        `${f} is not an ONNX file and will be ignored (only ONNX weights can run in the browser).`,
      )
  }
  return { root, files, candidates, errors, warnings }
}

export interface LocalConfigCheck {
  modelType: string | null
  vocabSize: number | null
  errors: string[]
  warnings: string[]
}

/** Architectures known to load as AutoModelForCausalLM in Transformers.js decoder-only mode. */
export const KNOWN_CAUSAL_LM_TYPES = new Set([
  'gpt2',
  'gptj',
  'gpt_neo',
  'gpt_neox',
  'gpt_bigcode',
  'codegen',
  'llama',
  'mistral',
  'qwen2',
  'qwen3',
  'qwen3_moe',
  'qwen3_5_text',
  'phi',
  'phi3',
  'gemma',
  'gemma2',
  'gemma3_text',
  'olmo',
  'olmo2',
  'stablelm',
  'falcon',
  'bloom',
  'mpt',
  'opt',
  'cohere',
  'granite',
  'smollm3',
  'exaone',
  'helium',
  'glm',
  'ernie4_5',
  'lfm2',
  'arcee',
  'apertus',
  'afmoe',
  'cohere2',
  'mobilellm',
  'olmo3',
  'starcoder2',
  'trillion',
  'nanochat',
  'hunyuan_v1_dense',
  'falcon_h1',
])

export function checkConfigJson(text: string): LocalConfigCheck {
  const errors: string[] = []
  const warnings: string[] = []
  let cfg: Record<string, unknown>
  try {
    cfg = JSON.parse(text) as Record<string, unknown>
  } catch {
    return { modelType: null, vocabSize: null, errors: ['config.json is not valid JSON'], warnings }
  }
  const modelType = typeof cfg.model_type === 'string' ? cfg.model_type : null
  const textCfg = (cfg.text_config ?? {}) as Record<string, unknown>
  const vocab =
    typeof cfg.vocab_size === 'number'
      ? cfg.vocab_size
      : typeof textCfg.vocab_size === 'number'
        ? textCfg.vocab_size
        : null
  if (!modelType) errors.push('config.json has no model_type')
  else if (!KNOWN_CAUSAL_LM_TYPES.has(modelType))
    warnings.push(
      `model_type "${modelType}" is not in this app's list of tested decoder-only architectures; loading may fail.`,
    )
  const archs = Array.isArray(cfg.architectures) ? (cfg.architectures as string[]) : []
  if (archs.length && !archs.some((a) => /ForCausalLM|LMHeadModel/.test(a)))
    warnings.push(`architectures ${archs.join(', ')} does not look like a causal language model.`)
  if (vocab === null)
    warnings.push(
      'config.json has no vocab_size; the logits size will be taken from the model output.',
    )
  return { modelType, vocabSize: vocab, errors, warnings }
}
