/**
 * REAL MODEL INFERENCE SMOKE TEST (mandatory).
 *
 * Loads the real SmolLM2-135M-Instruct q8 ONNX weights + tokenizer through the app's
 * catalog button, runs forward passes in the inference Web Worker (ONNX Runtime Web),
 * and checks the complete next-token logits vector and the derived distributions.
 *
 * Network: files are requested from https://huggingface.co/<repo>/resolve/<pinned sha>/…
 * When tests/fixtures/models/hf/ holds the identical pinned files (populated by
 * `node scripts/fetch-test-models.mjs --smollm2`, cached in CI) they are served from disk
 * to avoid re-downloading 140 MB per run; otherwise the real network is used.
 */
import { expect, test } from '@playwright/test'
import { serveHfFromDisk, state, waitForDistribution } from './helpers'

const REPO = 'onnx-community/SmolLM2-135M-Instruct-ONNX'
const REV = 'b8a5c0f183b78c55955a5364f610c36668b5e681'
const VOCAB = 49152

async function runSmokeTest(page: import('@playwright/test').Page, backend: 'wasm' | 'webgpu') {
  await page.getByTestId('backend-select').selectOption(backend)
  // fp16 is the exact default; the SwiftShader WebGPU adapter in CI lacks shader-f16, so use q8 there.
  const dtype = backend === 'webgpu' ? 'q8' : 'fp16'
  await page.getByLabel('Quantization for SmolLM2 135M Instruct').selectOption(dtype)
  await page.getByTestId('load-smollm2-135m-instruct').click()
  // 1–2. Real weights + tokenizer
  await expect(page.getByTestId('model-status')).toHaveText('Ready', { timeout: 600_000 })
  const info = (await state(page)).info!
  expect(info.backend).toBe(backend)
  expect(info.dtype).toBe(dtype)
  expect(info.identity).toContain(`${REPO}@${REV}`)
  expect(info.logitsSize).toBe(VOCAB)
  expect(info.tokenizerSize).toBe(VOCAB)

  // 3. Tokenize a real prompt (the catalog loads the "completion" example automatically)
  await expect.poll(async () => (await state(page)).tokens).toEqual([504, 3575, 282, 4649, 314]) // "The capital of France is"

  // 4–5. Forward pass → logits for the final position
  await waitForDistribution(page, 300_000)
  const s1 = await state(page)
  // 6. Complete vocabulary dimension
  expect(s1.outputDims).toEqual([1, 5, VOCAB])
  expect(s1.outputDtype).toBe('float32')
  expect(s1.logits).toHaveLength(VOCAB)
  expect(s1.logitsFinite).toBe(true)
  // 7–9. Softmax: finite, non-negative, sums to 1
  expect(s1.probsFinite).toBe(true)
  expect(s1.probsNonNegative).toBe(true)
  expect(Math.abs(s1.probSum - 1)).toBeLessThan(1e-6)
  // Real-model sanity: confident and semantically plausible (not random weights)
  expect(s1.original[0]!).toBeGreaterThan(0.2)
  if (dtype === 'fp16') {
    // fp16 reproduces the fp32 reference distribution (" Paris" ≈ 46.9%, measured in Node.js with fp32).
    expect(s1.top[0]).toBe(7042)
    expect(s1.original[0]!).toBeGreaterThan(0.44)
    expect(s1.original[0]!).toBeLessThan(0.5)
  }
  const topText = await page.evaluate((ids) => {
    const w = (window as unknown as { __lte: { tokenizer: { value: { decoded: string[] } } } })
      .__lte
    return ids.map((i) => w.tokenizer.value.decoded[i])
  }, s1.top)
  expect(topText.join('|')).toMatch(/ (located|in|Paris|the|a|situated|called)/)

  // 10–11. Select (greedy) a valid next token and append its ID
  await page.getByTestId('greedy-one').click()
  await expect.poll(async () => (await state(page)).tokens.length).toBe(6)
  const appended = (await state(page)).tokens[5]!
  expect(appended).toBe(s1.top[0])

  // 12–13. Second forward pass → another valid distribution
  await waitForDistribution(page, 300_000)
  const s2 = await state(page)
  expect(s2.outputDims).toEqual([1, 6, VOCAB])
  expect(s2.logitsFinite).toBe(true)
  expect(Math.abs(s2.probSum - 1)).toBeLessThan(1e-6)
  expect(s2.logits).not.toEqual(s1.logits)

  // Sampling then a third pass, and the chat template uses real special-token IDs
  await page.getByTestId('sample-one').click()
  await expect.poll(async () => (await state(page)).tokens.length).toBe(7)
  await waitForDistribution(page, 300_000)
  await page.getByTestId('example-select').selectOption('chat')
  await expect.poll(async () => (await state(page)).tokens[0]).toBe(1) // <|im_start|>
  await waitForDistribution(page, 300_000)
  const s3 = await state(page)
  expect(s3.logits).toHaveLength(VOCAB)
  expect(Math.abs(s3.probSum - 1)).toBeLessThan(1e-6)
  return { s1, s2, s3, topText }
}

test('real SmolLM2-135M inference on the WASM backend', async ({ page, context }, testInfo) => {
  const served = await serveHfFromDisk(context)
  await page.goto('./?debug=1')
  const r = await runSmokeTest(page, 'wasm')
  testInfo.annotations.push({
    type: 'real-inference',
    description: `wasm q8: top tokens ${JSON.stringify(r.topText)} p=${r.s1.original.map((p) => p.toFixed(4))}; files served from disk cache: ${served()}`,
  })
  await page.screenshot({ path: testInfo.outputPath('real-inference-wasm.png'), fullPage: true })
})

test('real SmolLM2-135M inference on the WebGPU backend (when an adapter exists)', async ({
  page,
  context,
}, testInfo) => {
  await serveHfFromDisk(context)
  await page.goto('./?debug=1')
  await expect(page.getByTestId('capabilities')).not.toContainText('Detecting', { timeout: 30_000 })
  const hasGpu = await page.getByTestId('capabilities').textContent()
  test.skip(
    !/WebGPU:\s*available/.test(hasGpu ?? ''),
    `No WebGPU adapter in this environment: ${hasGpu}`,
  )
  const r = await runSmokeTest(page, 'webgpu')
  testInfo.annotations.push({
    type: 'real-inference',
    description: `webgpu q8 (${hasGpu?.trim()}): top tokens ${JSON.stringify(r.topText)}`,
  })
})

test('a cached model loads and runs fully offline (service worker + Cache Storage)', async ({
  page,
  context,
}) => {
  // q8 (136 MB) also fits the in-memory cache of incognito-like test contexts.
  await serveHfFromDisk(context)
  await page.goto('./?debug=1')
  await page.evaluate(async () => (await navigator.serviceWorker.ready).scope)
  const load = async () => {
    await page.getByTestId('backend-select').selectOption('wasm')
    await page.getByLabel('Quantization for SmolLM2 135M Instruct').selectOption('q8')
    await page.getByTestId('load-smollm2-135m-instruct').click()
    // Fail fast with the real error instead of waiting for the full timeout.
    await expect(page.getByTestId('model-status')).toHaveText(/Ready|Load failed/, {
      timeout: 600_000,
    })
    const err = page.getByTestId('model-error')
    if (await err.isVisible()) throw new Error(`Model load failed: ${await err.textContent()}`)
  }
  await load()
  expect((await state(page)).info?.dtype).toBe('q8')

  await context.setOffline(true)
  const hfRequests: string[] = []
  context.on('request', (r) => {
    if (r.url().includes('huggingface.co')) hfRequests.push(r.url())
  })
  await page.reload()
  await expect(page.getByTestId('online-status')).toHaveText(/Offline/)
  await load()
  const info = await page.evaluate(
    () =>
      (window as unknown as { __lte: { modelInfo: { value: { fromCache: boolean } } } }).__lte
        .modelInfo.value,
  )
  expect(info.fromCache).toBe(true)
  await page.getByTestId('context-text').fill('The capital of France is')
  await waitForDistribution(page, 300_000)
  const s = await state(page)
  expect(s.logits).toHaveLength(VOCAB)
  expect(Math.abs(s.probSum - 1)).toBeLessThan(1e-6)
  expect(hfRequests).toEqual([])
  await context.setOffline(false)
})

// --------------------------------------------------------------------- Qwen3.5 (featured)
const QWEN35 = {
  repo: 'onnx-community/Qwen3.5-0.8B-Text-ONNX',
  rev: '25cdf33f386b487a25f3d743e86442517bdb77ff',
  logits: 248320,
  tokenizer: 248077,
  // Default example: chat template, thinking off — "Pick a random number between 1 and 10. Reply with just the number."
  chatIds: [
    248045, 846, 198, 35728, 264, 4055, 1324, 1881, 220, 16, 321, 220, 16, 15, 13, 17308, 440, 1066,
    279, 1324, 13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
  ],
}

async function decodedTop(page: import('@playwright/test').Page, ids: number[]) {
  return page.evaluate((ids) => {
    const w = (window as unknown as { __lte: { tokenizer: { value: { decoded: string[] } } } })
      .__lte
    return ids.map((i) => w.tokenizer.value.decoded[i])
  }, ids)
}

test('real Qwen3.5-0.8B inference: default variant (fp32) on WebGPU, starter examples', async ({
  page,
  context,
}, testInfo) => {
  test.setTimeout(1_800_000)
  const served = await serveHfFromDisk(context)
  await page.goto('./?debug=1')
  await expect(page.getByTestId('capabilities')).not.toContainText('Detecting', { timeout: 30_000 })
  const caps = await page.getByTestId('capabilities').textContent()
  test.skip(!/WebGPU:\s*available/.test(caps ?? ''), `No WebGPU adapter: ${caps}`)
  await page.getByTestId('backend-select').selectOption('webgpu')
  // The default is the most faithful variant that runs on WebGPU: fp32.
  await expect(page.getByTestId('dtype-qwen3.5-0.8b')).toHaveValue('fp32')
  await page.getByTestId('load-qwen3.5-0.8b').click()
  await page.getByRole('button', { name: 'Load anyway' }).click() // large-download confirmation
  await expect(page.getByTestId('model-status')).toHaveText(/Ready|Load failed/, {
    timeout: 1_200_000,
  })
  const err = page.getByTestId('model-error')
  if (await err.isVisible()) throw new Error(`Model load failed: ${await err.textContent()}`)
  const info = (await state(page)).info!
  expect(info.identity).toContain(`${QWEN35.repo}@${QWEN35.rev}#fp32`)
  expect(info.backend).toBe('webgpu')
  expect(info.logitsSize).toBe(QWEN35.logits)
  expect(info.tokenizerSize).toBe(QWEN35.tokenizer)
  const stop = await page.evaluate(
    () =>
      (window as unknown as { __lte: { stopTokenIds: { value: number[] } } }).__lte.stopTokenIds
        .value,
  )
  expect(stop).toEqual(expect.arrayContaining([248044, 248046])) // <|endoftext|>, <|im_end|>

  // Default starter example: chat template with thinking off
  await expect.poll(async () => (await state(page)).tokens).toEqual(QWEN35.chatIds)
  await expect(page.getByTestId('loaded-example')).toContainText('thinking off')
  await waitForDistribution(page, 600_000)
  const s1 = await state(page)
  expect(s1.outputDims).toEqual([1, QWEN35.chatIds.length, QWEN35.logits])
  expect(s1.outputDtype).toBe('float32')
  expect(s1.logitsFinite).toBe(true)
  expect(Math.abs(s1.probSum - 1)).toBeLessThan(1e-6)
  const top1 = await decodedTop(page, s1.top)
  // fp32 reference (Node.js): "6" 30.1 %, "7" 21.8 %, "5" 18.0 %, "4" 12.6 %
  expect(top1[0]).toBe('6')
  expect(s1.original[0]!).toBeGreaterThan(0.27)
  expect(s1.original[0]!).toBeLessThan(0.33)
  expect(top1.slice(0, 4).sort()).toEqual(['4', '5', '6', '7'])

  // Append the answer greedily and run another pass
  await page.getByTestId('greedy-one').click()
  await expect.poll(async () => (await state(page)).tokens.length).toBe(QWEN35.chatIds.length + 1)
  await waitForDistribution(page, 600_000)
  const s2 = await state(page)
  expect(s2.logitsFinite).toBe(true)
  expect(Math.abs(s2.probSum - 1)).toBeLessThan(1e-6)

  // Thinking on: the context ends with an open <think> block and the model starts reasoning
  await page.getByTestId('example-select').selectOption('thinking')
  await expect.poll(async () => (await state(page)).tokens.slice(-2)).toEqual([248068, 198])
  await waitForDistribution(page, 600_000)
  const s3 = await state(page)
  const top3 = await decodedTop(page, s3.top)
  expect(top3[0]).toBe('Thinking')
  expect(s3.original[0]!).toBeGreaterThan(0.85)
  testInfo.annotations.push({
    type: 'real-inference',
    description: `qwen3.5 fp32 webgpu (${caps?.trim()}): chat top ${JSON.stringify(top1.slice(0, 4))} p=${s1.original.slice(0, 4).map((p) => p.toFixed(4))}; thinking top ${JSON.stringify(top3[0])} p=${s3.original[0]!.toFixed(4)}; files from disk: ${served()}`,
  })
})
