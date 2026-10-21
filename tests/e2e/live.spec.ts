/**
 * Live deployment smoke test (run with LIVE_ONLY=1 LIVE_URL=<pages url>).
 * Verifies the deployed site under its repository subpath: assets, worker, WASM,
 * manifest, service worker, basic interactions and one real inference pass.
 */
import { expect, test } from '@playwright/test'
import { state, waitForDistribution } from './helpers'

test('deployed site loads with correct subpath assets and no runtime errors', async ({
  page,
  baseURL,
}) => {
  const errors: string[] = []
  const failed: string[] = []
  page.on('pageerror', (e) => errors.push(e.message))
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()))
  page.on('response', (r) => {
    if (r.url().startsWith(new URL(baseURL!).origin) && r.status() >= 400)
      failed.push(`${r.status()} ${r.url()}`)
  })
  const res = await page.goto('./')
  expect(res?.status()).toBe(200)
  await expect(page.getByRole('heading', { name: 'LLM Token Explorer' })).toBeVisible()
  // Not a blank page: catalog and workbench rendered
  await expect(page.getByTestId('model-card-smollm2-135m-instruct')).toBeVisible()
  await expect(page.getByTestId('context-workspace')).toBeVisible()
  // All script / style URLs live under the repository subpath
  const assets = await page.evaluate(() => [
    ...Array.from(document.querySelectorAll('script[src]'), (s) => (s as HTMLScriptElement).src),
    ...Array.from(
      document.querySelectorAll('link[rel="stylesheet"],link[rel="manifest"],link[rel="icon"]'),
      (l) => (l as HTMLLinkElement).href,
    ),
  ])
  expect(assets.length).toBeGreaterThan(2)
  for (const a of assets) {
    expect(a).toContain('/llm-token-explorer/')
    expect((await page.request.get(a)).status()).toBe(200)
  }
  // ONNX Runtime WASM is self-hosted under the subpath
  const wasm = await page.request.head(
    new URL('ort/ort-wasm-simd-threaded.asyncify.wasm', baseURL).href,
  )
  expect(wasm.status()).toBe(200)
  // Service worker + manifest
  const scope = await page.evaluate(async () => (await navigator.serviceWorker.ready).scope)
  expect(scope).toMatch(/\/llm-token-explorer\/$/)
  // Featured model listed first; f16 note; Hub guidance; shortcut cheat-sheet
  await expect(page.locator('[data-testid^="model-card-"]').first()).toHaveAttribute(
    'data-testid',
    'model-card-qwen3.5-0.8b',
  )
  await expect(page.getByTestId('featured-badge')).toHaveText('★ Patrick’s Choice')
  await page.getByRole('tab', { name: 'Hugging Face repo' }).click()
  await expect(page.getByTestId('hub-guide')).toContainText('onnx-community')
  await page.getByRole('tab', { name: 'Recommended' }).click()
  // Basic interactions
  await page.getByRole('button', { name: 'Learn / Help' }).click()
  await expect(page.getByTestId('learn-panel')).toBeVisible()
  await page.keyboard.press('Escape')
  await page.keyboard.press('?')
  await expect(page.getByTestId('shortcuts-dialog')).toBeVisible()
  await page.keyboard.press('Escape')
  expect(failed).toEqual([])
  expect(errors).toEqual([])
})

test('deployed site performs real SmolLM2 inference (WASM, fp16)', async ({ page }) => {
  await page.goto('./?debug=1')
  await page.getByTestId('backend-select').selectOption('wasm')
  await page.getByLabel('Quantization for SmolLM2 135M Instruct').selectOption('fp16')
  await page.getByTestId('load-smollm2-135m-instruct').click()
  await expect(page.getByTestId('model-status')).toHaveText('Ready', { timeout: 600_000 })
  await waitForDistribution(page, 300_000)
  const s = await state(page)
  expect(s.logits).toHaveLength(49152)
  expect(s.logitsFinite).toBe(true)
  expect(Math.abs(s.probSum - 1)).toBeLessThan(1e-6)
  expect(s.top[0]).toBe(7042) // " Paris"
  await expect(page.getByTestId('loaded-example')).toContainText('Text completion')
  // Sampling shortcut (S), outside text fields
  await page.locator('body').click({ position: { x: 5, y: 5 } })
  await page.keyboard.press('s')
  await expect.poll(async () => (await state(page)).tokens.length).toBe(6)
  await waitForDistribution(page, 300_000)
  expect(Math.abs((await state(page)).probSum - 1)).toBeLessThan(1e-6)
})
