#!/usr/bin/env node
/**
 * Download pinned model files used by the Playwright tests into tests/fixtures/models/
 * (git-ignored; cached in CI). Nothing here is committed to the repository.
 *
 *   node scripts/fetch-test-models.mjs                 # tiny fixture only (~0.9 MB)
 *   node scripts/fetch-test-models.mjs --smollm2       # + SmolLM2-135M fp16/q8 (~400 MB, CI smoke test)
 *   node scripts/fetch-test-models.mjs --qwen35        # + Qwen3.5-0.8B fp32 (~3.1 GB, CI WebGPU test)
 *   node scripts/fetch-test-models.mjs --model gpt2    # every variant of one catalog model
 *   node scripts/fetch-test-models.mjs --all           # every variant of every catalog model (~12 GB)
 *
 * Files are stored as tests/fixtures/models/hf/<owner>/<repo>/<revision>/<path> so the
 * real-inference tests can serve them in place of https://huggingface.co/<repo>/resolve/<rev>/<path>.
 * Behind a proxy with Node ≥ 22.21, run with NODE_USE_ENV_PROXY=1.
 */
import { createWriteStream, existsSync, mkdirSync, statSync, renameSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { Readable } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import { fileURLToPath } from 'node:url'

const root = join(dirname(fileURLToPath(import.meta.url)), '..', 'tests', 'fixtures', 'models')

export const TINY = {
  repo: 'hf-internal-testing/tiny-random-GPT2LMHeadModel',
  revision: 'af80da83ccf23171bfb7624542c8555a20e5951d',
  files: [
    'config.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'special_tokens_map.json',
    'onnx/model.onnx',
  ],
  dir: join(root, 'tiny-gpt2'),
}

const SUPPORT = ['config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json']

/**
 * Pinned files of every catalog model, per dtype (kept in sync with src/models/catalog.ts;
 * tests/unit/models.test.ts checks that). External-data chunks are listed explicitly.
 */
export const CATALOG_FILES = [
  {
    key: 'qwen3.5-0.8b',
    repo: 'onnx-community/Qwen3.5-0.8B-Text-ONNX',
    revision: '25cdf33f386b487a25f3d743e86442517bdb77ff',
    dtypes: ['fp32', 'fp16', 'q8', 'q4', 'q4f16'],
    files: {
      fp32: ['onnx/model.onnx', 'onnx/model.onnx_data', 'onnx/model.onnx_data_1'],
      fp16: ['onnx/model_fp16.onnx', 'onnx/model_fp16.onnx_data'],
      q8: ['onnx/model_quantized.onnx', 'onnx/model_quantized.onnx_data'],
      q4: ['onnx/model_q4.onnx', 'onnx/model_q4.onnx_data'],
      q4f16: ['onnx/model_q4f16.onnx', 'onnx/model_q4f16.onnx_data'],
    },
  },
  {
    key: 'smollm2-135m-instruct',
    repo: 'onnx-community/SmolLM2-135M-Instruct-ONNX',
    revision: 'b8a5c0f183b78c55955a5364f610c36668b5e681',
    dtypes: ['fp32', 'fp16', 'q8', 'q4f16'],
    files: {
      fp32: ['onnx/model.onnx'],
      fp16: ['onnx/model_fp16.onnx'],
      q8: ['onnx/model_quantized.onnx'],
      q4f16: ['onnx/model_q4f16.onnx'],
    },
  },
  {
    key: 'qwen2.5-0.5b-instruct',
    repo: 'onnx-community/Qwen2.5-0.5B-Instruct',
    revision: '22942cb7d7ba4cc81bb4673549ca4d4614469b5e',
    dtypes: ['fp32', 'fp16', 'bnb4', 'q8', 'q4', 'q4f16'],
    files: {
      fp32: ['onnx/model.onnx'],
      fp16: ['onnx/model_fp16.onnx'],
      bnb4: ['onnx/model_bnb4.onnx'],
      q8: ['onnx/model_quantized.onnx'],
      q4: ['onnx/model_q4.onnx'],
      q4f16: ['onnx/model_q4f16.onnx'],
    },
  },
  {
    key: 'gpt2',
    repo: 'onnx-community/gpt2-ONNX',
    revision: '1975fb9b4a2a92bd99f457407337ee25ab6cfde0',
    dtypes: ['fp32', 'fp16', 'q8'],
    files: {
      fp32: ['onnx/model.onnx'],
      fp16: ['onnx/model_fp16.onnx'],
      q8: ['onnx/model_quantized.onnx'],
    },
  },
]

async function download(url, dest) {
  if (existsSync(dest) && statSync(dest).size > 0) return false
  mkdirSync(dirname(dest), { recursive: true })
  for (let attempt = 1; attempt <= 4; attempt++) {
    try {
      const res = await fetch(url, { redirect: 'follow' })
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status} for ${url}`)
      const tmp = `${dest}.part`
      await pipeline(Readable.fromWeb(res.body), createWriteStream(tmp))
      renameSync(tmp, dest)
      return true
    } catch (e) {
      if (attempt === 4) throw e
      const wait = 2 ** attempt * 1000
      console.warn(`  retry ${attempt} in ${wait / 1000}s: ${e.message}`)
      await new Promise((r) => setTimeout(r, wait))
    }
  }
}

async function fetchFiles(repo, revision, files, dir) {
  for (const f of files) {
    const url = `https://huggingface.co/${repo}/resolve/${revision}/${f}`
    const dest = join(dir, f)
    const did = await download(url, dest)
    console.log(`${did ? 'downloaded' : 'cached    '} ${repo}/${f} (${statSync(dest).size} bytes)`)
  }
}

/** Download the support files plus the weights of the given dtypes of one catalog model. */
export async function fetchCatalogModel(key, dtypes) {
  const m = CATALOG_FILES.find((c) => c.key === key)
  if (!m) throw new Error(`Unknown catalog model "${key}"`)
  const files = [...SUPPORT, ...(dtypes ?? m.dtypes).flatMap((d) => m.files[d] ?? [])]
  await fetchFiles(m.repo, m.revision, files, join(root, 'hf', m.repo, m.revision))
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2)
  await fetchFiles(TINY.repo, TINY.revision, TINY.files, TINY.dir)
  if (args.includes('--smollm2')) await fetchCatalogModel('smollm2-135m-instruct', ['fp16', 'q8'])
  if (args.includes('--qwen35')) await fetchCatalogModel('qwen3.5-0.8b', ['fp32'])
  const i = args.indexOf('--model')
  if (i >= 0) await fetchCatalogModel(args[i + 1])
  if (args.includes('--all')) for (const m of CATALOG_FILES) await fetchCatalogModel(m.key)
}
