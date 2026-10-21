/**
 * Full workflow e2e test (production build, real Web Worker, real ONNX Runtime WASM).
 * The model is a tiny RANDOM-WEIGHT GPT-2 test fixture loaded through the local-files
 * path, so the network genuinely executes but its predictions are meaningless.
 * Real-model verification lives in real-inference.spec.ts.
 */
import { expect, test } from '@playwright/test'
import { readFileSync } from 'node:fs'
import { FIXTURE_DIR, requireFixture, setNumber, state, waitForDistribution } from './helpers'

test.beforeAll(requireFixture)

test('complete exploration workflow with exact token IDs, sampling, branching and experiments', async ({
  page,
}, testInfo) => {
  const errors: string[] = []
  page.on('pageerror', (e) => errors.push(e.message))

  // 1. Open the application
  await page.goto('./?debug=1')
  await expect(page.getByRole('heading', { name: 'LLM Token Explorer' })).toBeVisible()

  // 2. Load the controlled fixture model from local files (directory selection)
  await page.getByTestId('backend-select').selectOption('wasm')
  await page.getByRole('tab', { name: 'Local files' }).click()
  await page.getByTestId('local-dir-input').setInputFiles(FIXTURE_DIR)
  await expect(page.getByTestId('local-candidate')).toContainText('onnx/model.onnx')
  await page.getByTestId('local-load').click()
  await expect(page.getByTestId('model-status')).toHaveText('Ready', { timeout: 60_000 })
  const info = (await state(page)).info!
  expect(info.logitsSize).toBe(1024)
  expect(info.backend).toBe('wasm')

  // 3. Tokenize a sample context
  await page.getByTestId('context-text').fill('hello world')
  await expect.poll(async () => (await state(page)).tokens.length).toBeGreaterThan(1)
  await waitForDistribution(page)
  const textTokens = (await state(page)).tokens

  // 4. Inspect token boundaries
  const chips = page.getByTestId('token-strip').locator('button')
  await expect(chips).toHaveCount(textTokens.length)
  await chips.nth(1).click()
  await expect(page.getByTestId('inspector-id')).toHaveText(String(textTokens[1]))

  // 5. Insert a special token by its real ID (<|endoftext|>)
  await page.getByRole('tab', { name: 'Special tokens' }).click()
  await page.getByTestId('special-search').fill('endoftext')
  const row = page.locator('[data-testid^="special-row-"]').first()
  const specialId = Number((await row.getAttribute('data-testid'))!.replace('special-row-', ''))
  await row.getByRole('button', { name: /Append/ }).click()
  await expect.poll(async () => (await state(page)).tokens).toEqual([...textTokens, specialId])
  await expect(chips.last()).toHaveClass(/special/)

  // 6. Calculate next-token probabilities
  await page.getByTestId('calculate').click()
  await waitForDistribution(page)
  const s1 = await state(page)
  expect(s1.logits).toHaveLength(1024)
  expect(s1.logitsFinite).toBe(true)
  expect(Math.abs(s1.probSum - 1)).toBeLessThan(1e-6)
  await expect(page.getByTestId('sum-original')).toHaveText(/^1\.0000000|^0\.9999999/)

  // 7. Change temperature (sharpen strongly so the top token's share must rise)
  const forwardsBefore = s1.forwardCount
  await setNumber(page, 'temperature-input', '0.05')
  // 8. Adjusted probabilities change appropriately
  await expect
    .poll(async () => (await state(page)).adjustedTop[0])
    .toBeGreaterThan(s1.original[0]! + 1e-6)
  const s2 = await state(page)
  expect(Math.abs(s2.adjustedSum - 1)).toBeLessThan(1e-9)
  // 9. Original logits remain unchanged and the network was not re-run
  expect(s2.logits).toEqual(s1.logits)
  expect(s2.original).toEqual(s1.original)
  expect(s2.forwardCount).toBe(forwardsBefore)

  // 10–11. Select a token manually (not the top one) and append it
  const manual = s1.top[3]!
  await page.getByTestId('candidate-table').getByTestId(`pick-${manual}`).click()
  await expect.poll(async () => (await state(page)).tokens.at(-1)).toBe(manual)
  await waitForDistribution(page)
  const afterManual = (await state(page)).tokens

  // 12. Sample a token
  await page.getByTestId('sample-one').click()
  await expect.poll(async () => (await state(page)).tokens.length).toBe(afterManual.length + 1)
  await expect(page.getByTestId('last-pick')).toContainText('sample')
  await waitForDistribution(page)

  // 13. Undo a generation step
  await page.getByTestId('undo').click()
  await expect.poll(async () => (await state(page)).tokens).toEqual(afterManual)

  // 14. Create an alternate branch by choosing a different token
  await waitForDistribution(page)
  const alt = (await state(page)).top.find((id) => id !== manual)!
  await page.getByTestId('tab-vocab').click()
  await page.getByTestId('vocab-search').fill(`#${alt}`)
  await page.getByTestId('vocab-field').selectOption('id')
  await page.getByTestId('vocab-table').getByTestId(`pick-${alt}`).click()
  await expect.poll(async () => (await state(page)).tokens).toEqual([...afterManual, alt])
  await expect(page.getByTestId('history-panel')).toContainText('Branch 2 of 2')
  const finalTokens = (await state(page)).tokens
  const nodes = (await state(page)).nodes

  // 15. Export the experiment
  const downloadPromise = page.waitForEvent('download')
  await page.getByTestId('export').click()
  const download = await downloadPromise
  const exportPath = testInfo.outputPath('experiment.json')
  await download.saveAs(exportPath)
  const exported = JSON.parse(readFileSync(exportPath, 'utf8'))
  expect(exported.format).toBe('llm-token-explorer/experiment')
  expect(exported.sampling.temperature).toBe(0.05)

  // 16. Import into a fresh session (no model loaded; persistence cleared)
  await page.evaluate(() => localStorage.clear())
  await page.goto('./?debug=1')
  await expect(page.getByTestId('token-strip-empty')).toBeVisible()
  await page.getByTestId('import-input').setInputFiles(exportPath)

  // 17. Verify restored token IDs and settings
  await expect.poll(async () => (await state(page)).tokens).toEqual(finalTokens)
  const restored = await state(page)
  expect(restored.nodes).toBe(nodes)
  expect(restored.sampling.temperature).toBe(0.05)
  await expect(page.getByTestId('temperature-input')).toHaveValue('0.05')
  await expect(page.getByTestId('token-strip').locator('button')).toHaveCount(finalTokens.length)

  expect(errors).toEqual([])
})

test('automatic generation runs, reports progress and respects the maximum', async ({ page }) => {
  await page.goto('./?debug=1')
  await page.getByTestId('backend-select').selectOption('wasm')
  await page.getByRole('tab', { name: 'Local files' }).click()
  await page.getByTestId('local-dir-input').setInputFiles(FIXTURE_DIR)
  await page.getByTestId('local-load').click()
  await expect(page.getByTestId('model-status')).toHaveText('Ready', { timeout: 60_000 })
  await page.getByTestId('context-text').fill('hello')
  await waitForDistribution(page)
  const start = (await state(page)).tokens.length
  await page.getByTestId('stop-at-eos').uncheck()
  await page.getByTestId('max-tokens').fill('6')
  await page.getByTestId('auto-start').click()
  await expect(page.getByTestId('auto-stop-reason')).toContainText('maximum of 6', {
    timeout: 60_000,
  })
  expect((await state(page)).tokens.length).toBe(start + 6)
  await expect(page.getByTestId('auto-count')).toHaveText('6 / 6')
  await expect(page.getByTestId('auto-tps')).toContainText('tok/s')
})

test('useful errors for an incomplete local model directory', async ({ page }) => {
  await page.goto('./')
  await page.getByTestId('backend-select').selectOption('wasm')
  await page.getByRole('tab', { name: 'Local files' }).click()
  await page.getByTestId('local-files-input').setInputFiles(`${FIXTURE_DIR}/config.json`)
  await expect(page.getByTestId('local-errors')).toContainText('tokenizer.json')
  await expect(page.getByTestId('local-errors')).toContainText('No .onnx')
})
