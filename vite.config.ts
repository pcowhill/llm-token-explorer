import { fileURLToPath, URL } from 'node:url'
import { readFileSync, createReadStream, statSync, existsSync } from 'node:fs'
import { createRequire } from 'node:module'
import path from 'node:path'
import { defineConfig, type Plugin } from 'vite'
import vue from '@vitejs/plugin-vue'
import { VitePWA } from 'vite-plugin-pwa'

const require = createRequire(import.meta.url)

/**
 * GitHub Pages serves this project from https://<owner>.github.io/llm-token-explorer/.
 * Every asset, worker, WASM file, manifest and service-worker scope is resolved
 * relative to this base. Override with BASE_PATH=/ for root hosting.
 */
const BASE = process.env.BASE_PATH ?? '/llm-token-explorer/'

const pkg = JSON.parse(readFileSync(new URL('./package.json', import.meta.url), 'utf8')) as {
  version: string
}
/** Resolve a package root without relying on "./package.json" being in its exports map. */
function packageRoot(req: NodeJS.Require, name: string): string {
  let dir = path.dirname(req.resolve(name))
  while (!existsSync(path.join(dir, 'package.json')) || readPkg(dir).name !== name) {
    const parent = path.dirname(dir)
    if (parent === dir) throw new Error(`Cannot find package root of ${name}`)
    dir = parent
  }
  return dir
}
function readPkg(dir: string): { name: string; version: string } {
  return JSON.parse(readFileSync(path.join(dir, 'package.json'), 'utf8')) as { name: string; version: string }
}
const transformersDir = packageRoot(require, '@huggingface/transformers')
// Use the exact onnxruntime-web that Transformers.js itself resolves.
const ortDir = packageRoot(createRequire(path.join(transformersDir, 'package.json')), 'onnxruntime-web')
const ortPkg = readPkg(ortDir)
const transformersPkg = readPkg(transformersDir)

/**
 * ONNX Runtime Web files that Transformers.js would otherwise download from
 * cdn.jsdelivr.net at runtime. Self-hosting them keeps the app working offline,
 * avoids a third-party request, and pins the exact runtime version we tested.
 */
export const ORT_FILES = [
  'ort-wasm-simd-threaded.asyncify.mjs',
  'ort-wasm-simd-threaded.asyncify.wasm',
] as const

function selfHostOrtRuntime(): Plugin {
  return {
    name: 'self-host-onnxruntime-web',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = req.url?.split('?')[0] ?? ''
        const prefix = `${BASE}ort/`
        if (!url.startsWith(prefix)) return next()
        const file = url.slice(prefix.length)
        if (!(ORT_FILES as readonly string[]).includes(file)) return next()
        const full = path.join(ortDir, 'dist', file)
        res.setHeader('Content-Type', file.endsWith('.wasm') ? 'application/wasm' : 'text/javascript')
        res.setHeader('Content-Length', String(statSync(full).size))
        createReadStream(full).pipe(res)
      })
    },
    generateBundle() {
      for (const file of ORT_FILES) {
        this.emitFile({
          type: 'asset',
          fileName: `ort/${file}`,
          source: readFileSync(path.join(ortDir, 'dist', file)),
        })
      }
    },
  }
}

export default defineConfig({
  base: BASE,
  define: {
    __APP_VERSION__: JSON.stringify(pkg.version),
    __TRANSFORMERS_VERSION__: JSON.stringify(transformersPkg.version),
    __ORT_VERSION__: JSON.stringify(ortPkg.version),
  },
  plugins: [
    vue(),
    selfHostOrtRuntime(),
    VitePWA({
      registerType: 'prompt',
      injectRegister: false,
      includeAssets: ['favicon.svg', 'icons/*.png'],
      manifest: {
        name: 'LLM Token Explorer',
        short_name: 'Token Explorer',
        description:
          'Explore tokenization, next-token logits, probabilities and sampling with real in-browser LLM inference.',
        theme_color: '#1f2937',
        background_color: '#0f172a',
        display: 'standalone',
        start_url: BASE,
        scope: BASE,
        icons: [
          { src: 'icons/icon-192.png', sizes: '192x192', type: 'image/png' },
          { src: 'icons/icon-512.png', sizes: '512x512', type: 'image/png' },
          { src: 'icons/icon-512.png', sizes: '512x512', type: 'image/png', purpose: 'maskable' },
        ],
      },
      workbox: {
        // App shell: HTML, JS (incl. the inference worker), CSS, icons, manifest.
        globPatterns: ['**/*.{js,mjs,css,html,svg,png,webmanifest}'],
        // The inference worker bundles Transformers.js + ONNX Runtime JS glue.
        maximumFileSizeToCacheInBytes: 8 * 1024 * 1024,
        // The 27 MB ORT WASM binary is NOT precached; it is runtime-cached on first use
        // (and Transformers.js also caches it in Cache Storage) so that the first visit
        // does not silently download it unless a model is actually loaded.
        globIgnores: ['**/ort/*.wasm'],
        navigateFallback: `${BASE}index.html`,
        cleanupOutdatedCaches: true,
        runtimeCaching: [
          {
            urlPattern: ({ url }) => url.pathname.startsWith(`${BASE}ort/`),
            handler: 'CacheFirst',
            options: {
              cacheName: 'ort-runtime',
              expiration: { maxEntries: 8 },
            },
          },
        ],
        // Model weights are cached explicitly by the app (Cache Storage, managed in the
        // UI) and are never intercepted or precached by the service worker.
      },
      devOptions: { enabled: false },
    }),
  ],
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  worker: {
    format: 'es',
  },
  build: {
    target: 'es2022',
    sourcemap: true,
    chunkSizeWarningLimit: 4000,
  },
  server: { port: 5173 },
  preview: { port: 4173 },
})
