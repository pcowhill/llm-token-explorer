import { expect, test } from '@playwright/test'
import AxeBuilder from '@axe-core/playwright'

test('no serious accessibility violations on the initial page', async ({ page }) => {
  await page.goto('./')
  await expect(page.getByRole('heading', { name: 'LLM Token Explorer' })).toBeVisible()
  const results = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa']).analyze()
  const serious = results.violations.filter(
    (v) => v.impact === 'serious' || v.impact === 'critical',
  )
  expect(
    serious.map(
      (v) =>
        `${v.id}: ${v.nodes
          .slice(0, 3)
          .map((n) => n.target.join(' '))
          .join(' | ')}`,
    ),
  ).toEqual([])
})

test('dark theme also has no serious contrast violations', async ({ page }) => {
  await page.goto('./')
  await page.getByLabel('Color theme').selectOption('dark')
  const results = await new AxeBuilder({ page }).withTags(['wcag2aa']).analyze()
  const serious = results.violations.filter(
    (v) => v.impact === 'serious' || v.impact === 'critical',
  )
  expect(
    serious.map(
      (v) =>
        `${v.id}: ${v.nodes
          .slice(0, 4)
          .map((n) => n.target.join(' ') + ' ' + (n.any[0]?.message ?? ''))
          .join(' | ')}`,
    ),
  ).toEqual([])
})

test('core controls are keyboard reachable and the help dialog opens and closes', async ({
  page,
}) => {
  await page.goto('./')
  await page.keyboard.press('Tab')
  await expect(page.getByText('Skip to main content')).toBeFocused()
  await page.getByRole('button', { name: 'Learn / Help' }).focus()
  await page.keyboard.press('Enter')
  await expect(page.getByTestId('learn-panel')).toBeVisible()
  await expect(page.getByTestId('learn-panel')).toContainText('P(token_i) = exp(logit_i)')
  await page.keyboard.press('Escape')
  await expect(page.getByTestId('learn-panel')).toBeHidden()
})

async function seriousViolations(page: import('@playwright/test').Page) {
  const results = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa']).analyze()
  return results.violations
    .filter((v) => v.impact === 'serious' || v.impact === 'critical')
    .map(
      (v) =>
        `${v.id}: ${v.nodes
          .slice(0, 3)
          .map((n) => n.target.join(' '))
          .join(' | ')}`,
    )
}

test('shortcut cheat-sheet: opens with "?", is accessible, closes with Escape', async ({
  page,
}) => {
  await page.goto('./')
  await expect(page.getByRole('heading', { name: 'LLM Token Explorer' })).toBeVisible()
  await page.keyboard.press('?')
  const dialog = page.getByRole('dialog', { name: 'Keyboard shortcuts' })
  await expect(dialog).toBeVisible()
  expect(await seriousViolations(page)).toEqual([])
  await page.keyboard.press('Escape')
  await expect(dialog).toBeHidden()
})

test('Settings shortcut remapping is keyboard-operable and accessible (light and dark)', async ({
  page,
}) => {
  await page.goto('./')
  await page.getByRole('button', { name: 'Settings' }).focus()
  await page.keyboard.press('Enter')
  const settings = page.getByRole('dialog', { name: 'Settings' })
  await expect(settings).toBeVisible()
  await expect(settings.getByRole('group', { name: 'Keyboard shortcuts' })).toBeVisible()
  // Tab to the first "Change" button, activate it with Space, press a new key.
  const change = settings.getByTestId('remap-greedy')
  for (let i = 0; i < 20 && !(await change.evaluate((el) => el === document.activeElement)); i++)
    await page.keyboard.press('Tab')
  await expect(change).toBeFocused()
  await page.keyboard.press(' ')
  await expect(change).toHaveAttribute('aria-pressed', 'true')
  await page.keyboard.press('h')
  await expect(settings.getByTestId('shortcut-key-greedy')).toHaveText('H')
  await expect(settings.getByTestId('shortcut-status')).toContainText('H')
  expect(await seriousViolations(page)).toEqual([])
  await page.keyboard.press('Escape')
  await page.getByLabel('Color theme').selectOption('dark')
  await page.getByRole('button', { name: 'Settings' }).click()
  expect(await seriousViolations(page)).toEqual([])
})

test('Hugging Face repo guidance and the featured model card are accessible', async ({ page }) => {
  await page.goto('./')
  await expect(page.getByTestId('featured-badge')).toHaveText('★ Patrick’s Choice')
  await page.getByRole('tab', { name: 'Hugging Face repo' }).click()
  await expect(page.getByTestId('hub-guide')).toBeVisible()
  expect(await seriousViolations(page)).toEqual([])
  await page.getByLabel('Color theme').selectOption('dark')
  await page.getByRole('tab', { name: 'Recommended' }).click()
  expect(await seriousViolations(page)).toEqual([])
})
