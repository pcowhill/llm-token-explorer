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
import { expect, test, type BrowserContext } from '@playwright/test'
import { existsSync } from 'node:fs'
import { join } from 'node:path'
import { HF_CACHE_DIR, state, waitForDistribution } from './helpers'

const REPO = 'onnx-community/SmolLM2-135M-Instruct-ONNX'
const REV = 'b8a5c0f183b78c55955a5364f610c36668b5e681'
const VOCAB = 49152

async function serveHfFromDisk(context: BrowserContext) {
  let served = 0
  await context.route('https://huggingface.co/**', async (route) => {
    const url = new URL(route.request().url())
    const m = /^\/(.+?)\/resolve\/([^/]+)\/(.+)$/.exec(url.pathname)
    const file = m ? join(HF_CACHE_DIR, m[1]!, decodeURIComponent(m[2]!), m[3]!) : null
    if (
      !process.env.NO_HF_DISK &&
      file &&
      existsSync(file) &&
      route.request().method() === 'GET' &&
      !route.request().headers()['range']
    ) {
      served++
      // Redirect to the local static server (route.fulfill cannot stream 100+ MB bodies).
      await route.fulfill({
        status: 302,
        headers: {
          location: `http://localhost:4174/${m![1]}/${m![2]}/${m![3]}`,
          'access-control-allow-origin': '*',
        },
      })
    } else {
      await route.continue()
    }
  })
  return () => served
}

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
