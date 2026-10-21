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
