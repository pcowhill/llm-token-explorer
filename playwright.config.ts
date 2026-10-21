import { defineConfig, devices } from '@playwright/test'

/**
 * Playwright projects (run separately, see package.json scripts):
 *
 * - e2e:            full user workflow against the production build, using a tiny
 *                   random-weight GPT-2 fixture loaded through the LOCAL FILES path.
 *                   The ONNX network really executes (no mocks) but its outputs are
 *                   meaningless; it verifies the UI/data flow, not model quality.
 * - real-inference: mandatory smoke test with the real SmolLM2-135M-Instruct q8 weights
 *                   loaded from the catalog (WASM; WebGPU too when an adapter exists).
 * - live:           smoke test against the deployed GitHub Pages site (LIVE_URL).
 */
const PORT = 4173
const BASE = `http://localhost:${PORT}/llm-token-explorer/`
const chromium = process.env.CHROMIUM_PATH
  ? {
      launchOptions: {
        executablePath: process.env.CHROMIUM_PATH,
        args: ['--enable-unsafe-webgpu'],
      },
    }
  : { launchOptions: { args: ['--enable-unsafe-webgpu'] } }

export default defineConfig({
  testDir: 'tests/e2e',
  fullyParallel: false,
  workers: 1,
  forbidOnly: !!process.env.CI,
  retries: 0,
  reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : 'list',
  use: {
    baseURL: BASE,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    ...devices['Desktop Chrome'],
    viewport: { width: 1600, height: 1000 },
    ...chromium,
  },
  projects: [
    {
      name: 'e2e',
      testMatch: /workflow\.spec\.ts|interactions\.spec\.ts|pwa\.spec\.ts|a11y\.spec\.ts/,
      timeout: 120_000,
    },
    {
      name: 'real-inference',
      testMatch: /real-inference\.spec\.ts|catalog-verification\.spec\.ts/,
      timeout: 900_000,
    },
    {
      name: 'live',
      testMatch: /live\.spec\.ts/,
      timeout: 900_000,
      use: { baseURL: process.env.LIVE_URL ?? 'https://pcowhill.github.io/llm-token-explorer/' },
    },
  ],
  webServer: process.env.LIVE_ONLY
    ? undefined
    : [
        {
          command: `npx vite build && npx vite preview --port ${PORT} --strictPort`,
          url: BASE,
          reuseExistingServer: !process.env.CI,
          timeout: 240_000,
          stdout: 'ignore',
          stderr: 'pipe',
        },
        {
          // Serves cached pinned Hugging Face files for the real-inference test (see script).
          command: 'node scripts/serve-test-models.mjs',
          port: 4174,
          reuseExistingServer: !process.env.CI,
          stdout: 'ignore',
        },
      ],
})
