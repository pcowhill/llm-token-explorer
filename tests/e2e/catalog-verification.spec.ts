/**
 * Opt-in REAL inference verification of every catalog model, at every quantization, on
 * every backend it is offered for (large downloads: `node scripts/fetch-test-models.mjs --all`
 * puts them on disk first).
 *
 *   VERIFY_ALL_MODELS=1 npx playwright test --project=real-inference catalog
 *   VERIFY_ALL_MODELS=1 VERIFY_MODELS=gpt2,smollm2-135m-instruct npx playwright test --project=real-inference catalog
 *
 * Each case loads the variant through the model card, runs the default example, checks the
 * full-vocabulary logits (finite, not degenerate, plausible: far from uniform), appends a
 * token and runs a second pass. f16 variants on WebGPU need shader-f16; on an adapter
 * without it (CI's SwiftShader) they are skipped and reported as not verified.
 * Results back the statuses in src/models/catalog.ts and docs/model-compatibility.md.
 */
import { expect, test } from '@playwright/test'
import { CATALOG } from '../../src/models/catalog'
import { serveHfFromDisk, state, waitForDistribution } from './helpers'

const only = process.env.VERIFY_MODELS?.split(',').filter(Boolean)

for (const model of CATALOG) {
  for (const variant of model.variants) {
    for (const backend of variant.backends) {
      test(`verify ${model.key} ${variant.dtype} on ${backend}`, async ({
        page,
        context,
      }, testInfo) => {
        test.skip(
          !process.env.VERIFY_ALL_MODELS,
          'Set VERIFY_ALL_MODELS=1 to download and verify every catalog model',
        )
        test.skip(!!only && !only.includes(model.key), 'Not selected by VERIFY_MODELS')
        test.setTimeout(2_400_000)
        await serveHfFromDisk(context)
        await page.goto('./?debug=1')
        await expect(page.getByTestId('capabilities')).not.toContainText('Detecting', {
          timeout: 30_000,
        })
        if (backend === 'webgpu') {
          const caps = (await page.getByTestId('capabilities').textContent()) ?? ''
          test.skip(!/WebGPU:\s*available/.test(caps), `No WebGPU adapter: ${caps}`)
          test.skip(
            variant.dtype.includes('16') && /shader-f16:\s*no/.test(caps),
            `NOT VERIFIED: ${variant.dtype} needs shader-f16, which this adapter lacks (${caps.trim()})`,
          )
        }
        await page.getByTestId('backend-select').selectOption(backend)
        await page.getByTestId(`dtype-${model.key}`).selectOption(variant.dtype)
        await page.getByTestId(`load-${model.key}`).click()
        const confirm = page.getByRole('button', { name: 'Load anyway' })
        if (await confirm.isVisible().catch(() => false)) await confirm.click()
        await expect(page.getByTestId('model-status')).toHaveText(/Ready|Load failed/, {
          timeout: 1_800_000,
        })
        const err = page.getByTestId('model-error')
        if (await err.isVisible()) throw new Error(`Load failed: ${await err.textContent()}`)
        await waitForDistribution(page, 900_000)
        const s = await state(page)
        expect(s.info!.dtype).toBe(variant.dtype)
        expect(s.info!.backend).toBe(backend)
        expect(s.info!.logitsSize).toBe(model.logitsSize)
        expect(s.info!.tokenizerSize).toBe(model.tokenizerSize)
        expect(s.logits).toHaveLength(model.logitsSize)
        expect(s.outputDims?.at(-1)).toBe(model.logitsSize)
        expect(s.logitsFinite).toBe(true)
        expect(Math.abs(s.probSum - 1)).toBeLessThan(1e-6)
        // Not degenerate: the logits vary, and the top token is far above uniform (1/V).
        let min = Infinity
        let max = -Infinity
        for (const x of s.logits!) {
          if (x < min) min = x
          if (x > max) max = x
        }
        expect(max - min).toBeGreaterThan(5)
        expect(s.original[0]!).toBeGreaterThan(200 / model.logitsSize)
        await page.getByTestId('greedy-one').click()
        await waitForDistribution(page, 900_000)
        const s2 = await state(page)
        expect(s2.logits).toHaveLength(model.logitsSize)
        expect(s2.logitsFinite).toBe(true)
        expect(Math.abs(s2.probSum - 1)).toBeLessThan(1e-6)
        const top = await page.evaluate((ids) => {
          const w = (
            window as unknown as { __lte: { tokenizer: { value: { decoded: string[] } } } }
          ).__lte
          return ids.map((i) => w.tokenizer.value.decoded[i])
        }, s.top)
        const ms = (await page.evaluate(
          () =>
            (window as unknown as { __lte: { lastForwardMs: { value: number } } }).__lte
              .lastForwardMs.value,
        )) as number
        const line = `${model.key} ${variant.dtype} ${backend}: dims=${JSON.stringify(s.outputDims)} top=${JSON.stringify(top)} p=${s.original.map((p) => p.toFixed(3))} pass=${Math.round(ms)}ms`
        testInfo.annotations.push({ type: 'verified', description: line })
        console.log(`VERIFIED ${line}`)
      })
    }
  }
}
