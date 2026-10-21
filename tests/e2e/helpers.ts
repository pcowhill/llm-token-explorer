import { expect, type BrowserContext, type Page } from '@playwright/test'
import { existsSync } from 'node:fs'
import { join } from 'node:path'

export const FIXTURE_DIR = join(process.cwd(), 'tests', 'fixtures', 'models', 'tiny-gpt2')
export const HF_CACHE_DIR = join(process.cwd(), 'tests', 'fixtures', 'models', 'hf')

export function requireFixture() {
  if (!existsSync(join(FIXTURE_DIR, 'onnx', 'model.onnx')))
    throw new Error('Test fixture missing. Run: node scripts/fetch-test-models.mjs')
}

/** Read-only snapshot of workbench state via the ?debug=1 hook. */
export async function state(page: Page) {
  return page.evaluate(() => {
    type W = {
      tokens: { value: number[] }
      prepared: {
        value: {
          logits: Float32Array
          original: { probs: Float64Array }
          fullOrder: Uint32Array
        } | null
      }
      adjusted: { value: { probs: Float64Array; keptCount: number } | null }
      forwardCount: { value: number }
      sampling: {
        temperature: number
        topK: number
        topP: number
        minP: number
        seed: number
        greedy: boolean
      }
      modelInfo: {
        value: {
          logitsSize: number
          tokenizerSize: number
          backend: string
          dtype: string
          identity: string
        } | null
      }
      result: { value: { outputDims: number[]; outputDtype: string; inferenceMs: number } | null }
      resultIsCurrent: { value: boolean }
      getTree: () => { nodes: Map<string, unknown>; currentId: string }
    }
    const w = (window as unknown as { __lte: W }).__lte
    const p = w.prepared.value
    let sum = 0
    let finite = true
    let nonneg = true
    if (p) {
      for (const x of p.original.probs) {
        sum += x
        if (!Number.isFinite(x)) finite = false
        if (x < 0) nonneg = false
      }
    }
    let adjSum = 0
    if (w.adjusted.value) for (const x of w.adjusted.value.probs) adjSum += x
    return {
      tokens: [...w.tokens.value],
      logits: p ? Array.from(p.logits) : null,
      logitsFinite: p ? Array.from(p.logits).every(Number.isFinite) : false,
      top: p ? Array.from(p.fullOrder.subarray(0, 5)) : [],
      original: p
        ? Array.from(p.fullOrder.subarray(0, 5), (i) => p.original.probs[i] as number)
        : [],
      adjustedTop:
        p && w.adjusted.value
          ? Array.from(p.fullOrder.subarray(0, 5), (i) => w.adjusted.value!.probs[i] as number)
          : [],
      probSum: sum,
      probsFinite: finite,
      probsNonNegative: nonneg,
      adjustedSum: adjSum,
      kept: w.adjusted.value?.keptCount ?? null,
      forwardCount: w.forwardCount.value,
      sampling: { ...w.sampling },
      info: w.modelInfo.value ? { ...w.modelInfo.value } : null,
      outputDims: w.result.value?.outputDims ?? null,
      outputDtype: w.result.value?.outputDtype ?? null,
      current: w.resultIsCurrent.value,
      nodes: w.getTree().nodes.size,
    }
  })
}

export async function waitForDistribution(page: Page, timeout = 120_000) {
  await expect(page.getByTestId('inference-status')).toHaveText(/^Ready/, { timeout })
  await expect.poll(async () => (await state(page)).current, { timeout }).toBe(true)
}

export async function setNumber(page: Page, testId: string, value: string) {
  const input = page.getByTestId(testId)
  await input.fill(value)
  await input.press('Tab')
}

/**
 * Serve pinned https://huggingface.co/<repo>/resolve/<rev>/<path> requests from
 * tests/fixtures/models/hf/ when the file is there (see scripts/fetch-test-models.mjs), by
 * redirecting to scripts/serve-test-models.mjs on :4174 (route.fulfill cannot stream
 * 100 MB+ bodies). Everything else goes to the real network. Returns a served-file counter.
 */
export async function serveHfFromDisk(context: BrowserContext) {
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
