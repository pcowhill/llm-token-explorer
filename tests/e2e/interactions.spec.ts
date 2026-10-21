/**
 * E2E tests for interactive features (production build, real worker, tiny random-weight
 * GPT-2 fixture through the local-files path): the Example menu, the workspace reset on
 * model load, sampling shortcuts (incl. key repeat and remapping) and Hub dtype detection.
 */
import { expect, test, type Page } from '@playwright/test'
import { FIXTURE_DIR, requireFixture, state, waitForDistribution } from './helpers'

test.beforeAll(requireFixture)

async function loadFixture(page: Page) {
  await page.getByTestId('backend-select').selectOption('wasm')
  await page.getByRole('tab', { name: 'Local files' }).click()
  await page.getByTestId('local-dir-input').setInputFiles(FIXTURE_DIR)
  await page.getByTestId('local-load').click()
  await expect(page.getByTestId('model-status')).toHaveText('Ready', { timeout: 60_000 })
}

test('Example menu loads the same example again after Clear (regression)', async ({ page }) => {
  await page.goto('./?debug=1')
  await loadFixture(page)
  const select = page.getByTestId('example-select')
  await select.selectOption({ label: 'Minimal: start token' })
  await expect.poll(async () => (await state(page)).tokens.length).toBe(1)
  // The menu returns to its placeholder, so choosing the same entry fires again.
  await expect(select).toHaveValue('')
  await expect(page.getByTestId('loaded-example')).toHaveText(/Loaded: Minimal: start token/)
  await page.getByTestId('clear-context').click()
  await expect.poll(async () => (await state(page)).tokens.length).toBe(0)
  await expect(page.getByTestId('loaded-example')).toBeHidden()
  await select.selectOption({ label: 'Minimal: start token' })
  await expect.poll(async () => (await state(page)).tokens.length).toBe(1)
  await expect(select).toHaveValue('')
})

test('loading a model clears the context and starts a fresh history', async ({ page }) => {
  await page.goto('./?debug=1')
  await loadFixture(page)
  await page.getByTestId('context-text').fill('hello world')
  await waitForDistribution(page)
  const nodesBefore = (await state(page)).nodes
  await page.getByTestId('sample-one').click()
  await expect.poll(async () => (await state(page)).nodes).toBe(nodesBefore + 1)
  await page.getByTestId('toggle-model-sources').click()
  await page.getByTestId('local-load').click()
  await expect(page.getByTestId('model-status')).toHaveText('Ready', { timeout: 60_000 })
  await expect(page.getByText('Context cleared for new model.')).toBeVisible()
  const s = await state(page)
  expect(s.tokens).toEqual([])
  expect(s.nodes).toBe(1)
  await expect(page.getByTestId('example-select')).toHaveValue('')
})

test('S / G shortcuts sample and append greedily; holding repeats one token per forward pass', async ({
  page,
}) => {
  await page.goto('./?debug=1')
  await loadFixture(page)
  await page.getByTestId('context-text').fill('hello')
  await waitForDistribution(page)
  await expect(page.getByTestId('sample-one')).toContainText('S')
  await page.locator('body').click({ position: { x: 5, y: 5 } }) // leave the text field
  const n = (await state(page)).tokens.length
  await page.keyboard.press('s')
  await expect.poll(async () => (await state(page)).tokens.length).toBe(n + 1)
  await waitForDistribution(page)
  const top = (await state(page)).top[0]
  await page.keyboard.press('g')
  await expect.poll(async () => (await state(page)).tokens.at(-1)).toBe(top)
  await waitForDistribution(page)

  // Hold S: repeated keydown events (repeat=true) for a while.
  const before = (await state(page)).tokens.length
  for (let i = 0; i < 40; i++) {
    await page.keyboard.down('s')
    await page.waitForTimeout(15)
  }
  await page.keyboard.up('s')
  await waitForDistribution(page)
  const s = await state(page)
  const added = s.tokens.length - before
  expect(added).toBeGreaterThanOrEqual(1)
  expect(added).toBeLessThanOrEqual(40)
  // Each sampled token came from the distribution of its own context (no blind appends).
  const nodes = await page.evaluate(() => {
    type N = { append?: { source: string; originalProbability: number | null } }
    const w = (window as unknown as { __lte: { getTree: () => { nodes: Map<string, N> } } }).__lte
    return [...w.getTree().nodes.values()]
      .filter((x) => x.append?.source === 'sample')
      .map((x) => x.append!.originalProbability)
  })
  expect(nodes.every((p) => p !== null)).toBe(true)

  // Shortcuts are inactive in text fields
  await page.getByTestId('toggle-model-sources').click()
  await page.getByRole('tab', { name: 'Hugging Face repo' }).click()
  await page.getByTestId('hub-repo').fill('')
  const len = (await state(page)).tokens.length
  await page.getByTestId('hub-repo').press('s')
  await page.getByTestId('hub-repo').press('g')
  await expect(page.getByTestId('hub-repo')).toHaveValue('sg')
  expect((await state(page)).tokens.length).toBe(len)
})

test('shortcut cheat-sheet and remapping work from the keyboard and persist', async ({ page }) => {
  await page.goto('./?debug=1')
  await loadFixture(page)
  await page.getByTestId('context-text').fill('hello')
  await waitForDistribution(page)
  await page.locator('body').click({ position: { x: 5, y: 5 } })
  await page.keyboard.press('?')
  const sheet = page.getByTestId('shortcuts-dialog')
  await expect(sheet).toBeVisible()
  await expect(sheet.getByRole('heading', { name: 'Keyboard shortcuts' })).toBeVisible()
  await expect(sheet).toContainText('Alt+S')
  await page.keyboard.press('Escape')
  await expect(sheet).toBeHidden()

  await page.getByRole('button', { name: 'Settings' }).click()
  const remap = page.getByTestId('remap-sample')
  await remap.focus()
  await page.keyboard.press('Enter')
  await expect(remap).toHaveText('Press a key…')
  await page.keyboard.press('g') // conflict with greedy
  await expect(page.getByTestId('shortcut-error')).toContainText('already used')
  await page.keyboard.press('k')
  await expect(page.getByTestId('shortcut-key-sample')).toHaveText('K')
  await page.keyboard.press('Escape') // closes Settings
  await expect(page.getByTestId('sample-one')).toContainText('K')

  const n = (await state(page)).tokens.length
  await page.keyboard.press('k')
  await expect.poll(async () => (await state(page)).tokens.length).toBe(n + 1)

  await page.reload()
  await expect(page.getByTestId('sample-one')).toContainText('K')
  await page.getByRole('button', { name: 'Settings' }).click()
  await page.getByTestId('reset-shortcuts').click()
  await expect(page.getByTestId('shortcut-key-sample')).toHaveText('S')
})

test('Hugging Face repo tab lists the quantizations the Hub API reports', async ({ page }) => {
  await page.route('https://huggingface.co/api/models/**', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      headers: { 'access-control-allow-origin': '*' },
      body: JSON.stringify({
        sha: '1975fb9b4a2a92bd99f457407337ee25ab6cfde0',
        gated: false,
        config: { model_type: 'gpt2', architectures: ['GPT2LMHeadModel'] },
        siblings: [
          { rfilename: 'config.json', size: 900 },
          { rfilename: 'tokenizer.json', size: 2_100_000 },
          { rfilename: 'tokenizer_config.json', size: 300 },
          { rfilename: 'onnx/model.onnx', size: 498_126_358 },
          { rfilename: 'onnx/model_fp16.onnx', size: 249_252_545 },
          { rfilename: 'onnx/model_quantized.onnx', size: 279_633_549 },
        ],
      }),
    }),
  )
  await page.goto('./')
  await page.getByTestId('backend-select').selectOption('wasm')
  await page.getByRole('tab', { name: 'Hugging Face repo' }).click()
  await expect(page.getByTestId('hub-guide')).toContainText('onnx-community')
  await page.getByTestId('hub-repo').fill('onnx-community/gpt2-ONNX')
  await expect(page.getByTestId('hub-status')).toContainText('3 quantizations found')
  const options = page.getByTestId('hub-dtype').locator('option')
  await expect(options).toHaveCount(3)
  await expect(options.nth(0)).toContainText('fp32 — ↓')
  await expect(page.getByTestId('hub-status')).toContainText('recommended for WASM: fp32')
  await expect(page.getByTestId('hub-dtype')).toHaveValue('fp32')
  await expect(page.getByTestId('hub-load')).toContainText('Download & load')
})
