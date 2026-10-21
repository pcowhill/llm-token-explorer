import { expect, test } from '@playwright/test'

test('manifest, service worker and offline reload of the app shell', async ({
  page,
  context,
  baseURL,
}) => {
  await page.goto('./')
  const manifestHref = await page.locator('link[rel="manifest"]').getAttribute('href')
  expect(manifestHref).toBeTruthy()
  const manifest = await (await page.request.get(new URL(manifestHref!, baseURL).href)).json()
  expect(manifest.name).toBe('LLM Token Explorer')
  expect(manifest.scope).toBe('/llm-token-explorer/')
  expect(manifest.start_url).toBe('/llm-token-explorer/')
  for (const icon of manifest.icons) {
    const r = await page.request.get(new URL(icon.src, new URL(manifestHref!, baseURL)).href)
    expect(r.status()).toBe(200)
  }

  // Service worker registered under the repository subpath and controlling the page
  const scope = await page.evaluate(async () => {
    const reg = await navigator.serviceWorker.ready
    return reg.scope
  })
  expect(scope).toMatch(/\/llm-token-explorer\/$/)
  await expect(page.getByText('Offline-ready')).toBeVisible({ timeout: 20_000 })

  // Reload offline: the app shell must come from the service worker cache
  await page.reload()
  await context.setOffline(true)
  await page.reload()
  await expect(page.getByRole('heading', { name: 'LLM Token Explorer' })).toBeVisible()
  await expect(page.getByTestId('online-status')).toHaveText(/Offline/)
  await expect(page.getByTestId('model-card-smollm2-135m-instruct')).toBeVisible()
  await context.setOffline(false)
})
