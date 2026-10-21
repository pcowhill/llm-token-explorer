#!/usr/bin/env node
/**
 * Download pinned model files used by the Playwright tests into tests/fixtures/models/
 * (git-ignored; cached in CI). Nothing here is committed to the repository.
 *
 *   node scripts/fetch-test-models.mjs            # tiny fixture only (~0.9 MB)
 *   node scripts/fetch-test-models.mjs --smollm2  # + SmolLM2-135M q8 (~140 MB) for the real smoke test
 *
 * Files are stored as tests/fixtures/models/hf/<owner>/<repo>/<revision>/<path> so the
 * real-inference test can serve them in place of https://huggingface.co/<repo>/resolve/<rev>/<path>.
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

export const SMOLLM2 = {
  repo: 'onnx-community/SmolLM2-135M-Instruct-ONNX',
  revision: 'b8a5c0f183b78c55955a5364f610c36668b5e681',
  files: [
    'config.json',
    'generation_config.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'onnx/model_fp16.onnx',
    'onnx/model_quantized.onnx',
  ],
}

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

async function fetchModel(m, dirFor) {
  for (const f of m.files) {
    const url = `https://huggingface.co/${m.repo}/resolve/${m.revision}/${f}`
    const dest = join(dirFor, f)
    const did = await download(url, dest)
    console.log(
      `${did ? 'downloaded' : 'cached    '} ${m.repo}/${f} (${statSync(dest).size} bytes)`,
    )
  }
}

await fetchModel(TINY, TINY.dir)
if (process.argv.includes('--smollm2')) {
  await fetchModel(SMOLLM2, join(root, 'hf', SMOLLM2.repo, SMOLLM2.revision))
}
