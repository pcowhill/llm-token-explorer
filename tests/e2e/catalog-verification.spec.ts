/**
 * Opt-in REAL inference verification of every catalog model (large downloads).
 * Run with: VERIFY_ALL_MODELS=1 npx playwright test --project=real-inference catalog
 * Results back the "verified" statuses in src/models/catalog.ts and docs/model-compatibility.md.
 */
import { expect, test } from '@playwright/test'
import { CATALOG } from '../../src/models/catalog'
import { state, waitForDistribution } from './helpers'

for (const model of CATALOG) {
  test(`verify ${model.key} (WASM, default quantization): full-vocabulary logits`, async ({
    page,
  }, testInfo) => {
    test.skip(
      !process.env.VERIFY_ALL_MODELS,
      'Set VERIFY_ALL_MODELS=1 to download and verify every catalog model',
    )
    test.setTimeout(1_800_000)
    await page.goto('./?debug=1')
    await page.getByTestId('backend-select').selectOption('wasm')
    await page.getByTestId(`load-${model.key}`).click()
    const confirm = page.getByRole('button', { name: 'Load anyway' })
    if (await confirm.isVisible().catch(() => false)) await confirm.click()
    await expect(page.getByTestId('model-status')).toHaveText('Ready', { timeout: 1_500_000 })
    await waitForDistribution(page, 600_000)
    const s = await state(page)
    expect(s.info!.logitsSize).toBe(model.logitsSize)
    expect(s.info!.tokenizerSize).toBe(model.tokenizerSize)
    expect(s.logits).toHaveLength(model.logitsSize)
    expect(s.outputDims?.at(-1)).toBe(model.logitsSize)
    expect(s.logitsFinite).toBe(true)
    expect(Math.abs(s.probSum - 1)).toBeLessThan(1e-6)
    await page.getByTestId('greedy-one').click()
    await waitForDistribution(page, 600_000)
    const s2 = await state(page)
    expect(s2.logits).toHaveLength(model.logitsSize)
    expect(Math.abs(s2.probSum - 1)).toBeLessThan(1e-6)
    const top = await page.evaluate((ids) => {
      const w = (window as unknown as { __lte: { tokenizer: { value: { decoded: string[] } } } })
        .__lte
      return ids.map((i) => w.tokenizer.value.decoded[i])
    }, s.top)
    testInfo.annotations.push({
      type: 'verified',
      description: `${model.repo}@${model.revision.slice(0, 7)} wasm: logits ${s.outputDims?.join('x')}, top ${JSON.stringify(top)} p=${s.original.map((p) => p.toFixed(3))}`,
    })
    console.log(
      `VERIFIED ${model.key} ${s.info!.dtype}: dims=${JSON.stringify(s.outputDims)} top=${JSON.stringify(top)} p=${s.original.map((p) => p.toFixed(3))}`,
    )
  })
}
