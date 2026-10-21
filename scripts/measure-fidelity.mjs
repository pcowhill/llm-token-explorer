#!/usr/bin/env node
/**
 * Quantization fidelity measurement (Node.js, onnxruntime-node CPU).
 *
 * Compares each variant's next-token distribution with a full-precision reference over
 * the fixed REFERENCE_PROMPTS below: mean KL(reference ‖ variant) in nats, top-1 agreement,
 * and a degenerate-logits check (all-equal / non-finite). The numbers back the `fidelity`
 * strings in src/models/catalog.ts and the table in docs/model-compatibility.md.
 *
 *   node scripts/fetch-test-models.mjs --all          # pinned files → tests/fixtures/models/hf
 *   node scripts/measure-fidelity.mjs <catalog-key> [refDtype] [dtype,dtype,...]
 *
 * Model files are read from tests/fixtures/models/hf/<repo>/<revision>/ (nothing is
 * downloaded here). Node's CPU kernels differ from browser WASM / WebGPU kernels, so this
 * measures the weights, not a particular browser backend; see the real-inference tests.
 */
import { existsSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { AutoModelForCausalLM, AutoTokenizer, Tensor, env } from '@huggingface/transformers'
import { CATALOG_FILES } from './fetch-test-models.mjs'

export const REFERENCE_PROMPTS = [
  { text: 'The capital of France is' },
  { text: 'Question: What is the boiling point of water at sea level?\nAnswer:' },
  {
    chat: [{ role: 'user', content: 'Give me one fun fact about octopuses.' }],
    // Models without a chat template get the same request as plain text.
    fallback: 'User: Give me one fun fact about octopuses.\nAssistant:',
  },
  { text: 'def fibonacci(n):\n    ' },
  { text: 'Once upon a time, in a small village,' },
]

const root = join(
  dirname(fileURLToPath(import.meta.url)),
  '..',
  'tests',
  'fixtures',
  'models',
  'hf',
)

function softmax(row) {
  let max = -Infinity
  for (const x of row) if (x > max) max = x
  const p = new Float64Array(row.length)
  let sum = 0
  for (let i = 0; i < row.length; i++) sum += p[i] = Math.exp(row[i] - max)
  for (let i = 0; i < row.length; i++) p[i] /= sum
  return p
}

function argmax(p) {
  let best = 0
  for (let i = 1; i < p.length; i++) if (p[i] > p[best]) best = i
  return best
}

async function main() {
  const [key, refDtype = 'fp32', dtypesArg] = process.argv.slice(2)
  const m = CATALOG_FILES.find((c) => c.key === key)
  if (!m)
    throw new Error(`Unknown model "${key}". Known: ${CATALOG_FILES.map((c) => c.key).join(', ')}`)
  const dir = join(root, m.repo, m.revision)
  if (!existsSync(join(dir, 'config.json')))
    throw new Error(`Missing ${dir}. Run: node scripts/fetch-test-models.mjs --model ${key}`)
  env.allowRemoteModels = false
  env.allowLocalModels = true
  env.localModelPath = '/'
  const tok = await AutoTokenizer.from_pretrained(dir)
  const prompts = REFERENCE_PROMPTS.map((p) => {
    if (p.chat && tok.chat_template) {
      const text = tok.apply_chat_template(p.chat, { tokenize: false, add_generation_prompt: true })
      return Array.from(tok.encode(text, { add_special_tokens: false }), Number)
    }
    return Array.from(tok.encode(p.text ?? p.fallback, { add_special_tokens: true }), Number)
  })

  async function run(dtype) {
    const model = await AutoModelForCausalLM.from_pretrained(dir, { dtype, device: 'cpu' })
    const rows = []
    for (const ids of prompts) {
      const input_ids = new Tensor('int64', BigInt64Array.from(ids, BigInt), [1, ids.length])
      const attention_mask = new Tensor('int64', new BigInt64Array(ids.length).fill(1n), [
        1,
        ids.length,
      ])
      const { logits } = await model({ input_ids, attention_mask })
      const f32 = logits.type === 'float32' ? logits : logits.to('float32')
      const V = logits.dims.at(-1)
      const n = logits.size / V
      rows.push(f32.data.slice((n - 1) * V, n * V))
    }
    await model.dispose()
    return rows
  }

  const ref = await run(refDtype)
  const dtypes = (dtypesArg ?? m.dtypes.filter((d) => d !== refDtype).join(',')).split(',')
  console.log(
    `${key} @ ${m.revision.slice(0, 7)} — reference ${refDtype}, ${prompts.length} prompts`,
  )
  for (const dtype of dtypes.filter(Boolean)) {
    let rows
    try {
      rows = await run(dtype)
    } catch (e) {
      console.log(`  ${dtype.padEnd(6)} FAILED: ${String(e.message).slice(0, 200)}`)
      continue
    }
    let kl = 0
    let agree = 0
    let degenerate = 0
    rows.forEach((row, i) => {
      let min = Infinity
      let max = -Infinity
      for (const x of row) {
        if (x < min) min = x
        if (x > max) max = x
      }
      if (!row.every(Number.isFinite) || max - min < 1e-6) {
        degenerate++
        return
      }
      const p = softmax(ref[i])
      const q = softmax(row)
      for (let j = 0; j < p.length; j++)
        if (p[j] > 0) kl += p[j] * Math.log(p[j] / Math.max(q[j], 1e-300))
      if (argmax(p) === argmax(q)) agree++
    })
    const n = rows.length - degenerate
    const meanKl = n ? kl / n : NaN
    console.log(
      `  ${dtype.padEnd(6)} mean KL ${meanKl < 0.0001 ? '< 0.0001' : meanKl.toFixed(4)} nats, top-1 ${agree}/${rows.length}${degenerate ? `, DEGENERATE on ${degenerate}` : ''}`,
    )
  }
}

await main()
