#!/usr/bin/env node
/**
 * Capture documentation screenshots of the real running app (docs/screenshots/).
 * Requires `npx vite preview --port 4173` (production build) to be running.
 * Model files are fetched from Hugging Face (or from tests/fixtures/models/hf via
 * scripts/serve-test-models.mjs on :4174 when available).
 */
import { chromium } from '@playwright/test'
import { existsSync, mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const BASE = process.env.APP_URL ?? 'http://localhost:4173/llm-token-explorer/'
const OUT = 'docs/screenshots'
const HF = join('tests', 'fixtures', 'models', 'hf')

// A persistent (non-incognito) profile, so downloaded models really land in Cache Storage
// and the per-quantization cache marks appear.
const profile = mkdtempSync(join(tmpdir(), 'lte-shots-'))
const context = await chromium.launchPersistentContext(profile, {
  viewport: { width: 1600, height: 1000 },
  deviceScaleFactor: 1,
})
await context.route('https://huggingface.co/**', async (route) => {
  const m = /^\/(.+?)\/resolve\/([^/]+)\/(.+)$/.exec(new URL(route.request().url()).pathname)
  if (m && existsSync(join(HF, m[1], m[2], m[3])))
    return route.fulfill({
      status: 302,
      headers: {
        location: `http://localhost:4174/${m[1]}/${m[2]}/${m[3]}`,
        'access-control-allow-origin': '*',
      },
    })
  return route.continue()
})
const page = await context.newPage()
const shot = async (name, locator) => {
  if (locator) await locator.screenshot({ path: `${OUT}/${name}.png` })
  else await page.screenshot({ path: `${OUT}/${name}.png` })
  console.log('saved', name)
}
await page.goto(`${BASE}?debug=1`)
await page.getByLabel('Color theme').selectOption('light')
await page.getByTestId('backend-select').selectOption('wasm')
await shot('01-model-selection', page.getByTestId('model-panel'))
await page.getByTestId('load-smollm2-135m-instruct').click()
await page.getByTestId('model-status').filter({ hasText: 'Ready' }).waitFor({ timeout: 600000 })
await page.getByTestId('inference-status').filter({ hasText: 'Ready' }).waitFor({ timeout: 300000 })
await page.getByTestId('context-text').fill('The capital of France is')
await page.waitForTimeout(1500)
await page.getByTestId('inference-status').filter({ hasText: 'Ready' }).waitFor()
await page.getByTestId('token-strip').locator('[data-index="1"]').click()
await shot('02-tokenized-context', page.getByTestId('context-workspace'))
await page.getByTestId('temperature-input').fill('0.7')
await page.getByTestId('temperature-input').press('Tab')
await page.getByTestId('topk-input').fill('8')
await page.getByTestId('topk-input').press('Tab')
await page.waitForTimeout(300)
await shot('03-probability-distribution', page.getByTestId('probability-workspace'))
await shot('04-sampling-controls', page.getByTestId('sampling-controls'))
await page.getByTestId('sample-one').click()
await page.getByTestId('inference-status').filter({ hasText: 'Ready' }).waitFor()
await page.getByTestId('undo').click()
await page.getByTestId('inference-status').filter({ hasText: 'Ready' }).waitFor()
await page.getByTestId('candidate-table').getByTestId('pick-260').click()
await page.getByTestId('inference-status').filter({ hasText: 'Ready' }).waitFor()
await page.getByTestId('greedy-one').click()
await page.getByTestId('inference-status').filter({ hasText: 'Ready' }).waitFor()
await shot('06-generation-history', page.getByTestId('history-panel'))
await page.getByTestId('tab-vocab').click()
await page.getByTestId('vocab-search').fill('Par')
await page.waitForTimeout(300)
await shot('05-vocabulary-explorer', page.getByTestId('probability-workspace'))
await page.getByTestId('diagnostics').locator('summary').click()
await page.waitForTimeout(300)
await shot('07-diagnostics', page.getByTestId('diagnostics'))
await page.getByTestId('tab-vocab').click()
await page.getByRole('tab', { name: 'Top candidates' }).click()
await page.screenshot({ path: `${OUT}/00-overview.png`, fullPage: false })
console.log('saved 00-overview')
await page.getByLabel('Color theme').selectOption('dark')
await page.screenshot({ path: `${OUT}/08-dark-theme.png` })
console.log('saved 08-dark-theme')
await page.getByLabel('Color theme').selectOption('light')

// Per-quantization cache state: the loaded SmolLM2 card and the Cache tab
await page.getByTestId('toggle-model-sources').click()
await page.getByTestId('cached-badge').first().waitFor()
await shot('09-cache-per-quantization', page.getByTestId('model-card-smollm2-135m-instruct'))
await page.getByRole('tab', { name: 'Cache' }).click()
await page.getByTestId('cache-table').waitFor()
await shot('10-cache-tab', page.locator('#model-tabpanel-cache'))

// Hugging Face repo tab: guidance and dtypes detected from the Hub API (real network)
await page.getByRole('tab', { name: 'Hugging Face repo' }).click()
await page.getByTestId('hub-repo').fill('onnx-community/Qwen3.5-0.8B-Text-ONNX')
await page.getByTestId('hub-status').filter({ hasText: 'quantizations found' }).waitFor()
await shot('11-hugging-face-repo', page.getByTestId('hub-loader'))

// Keyboard shortcut cheat-sheet (press "?")
await page.locator('body').click({ position: { x: 5, y: 5 } })
await page.keyboard.press('?')
await page.getByTestId('shortcuts-dialog').waitFor()
await shot('12-keyboard-shortcuts', page.getByTestId('shortcuts-dialog'))
await context.close()
rmSync(profile, { recursive: true, force: true })
