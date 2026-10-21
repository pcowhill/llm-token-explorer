#!/usr/bin/env node
/**
 * Minimal static file server (CORS-enabled) for cached Hugging Face test files in
 * tests/fixtures/models/hf/. Used only by the Playwright real-inference test, which
 * redirects pinned https://huggingface.co/... requests here when the identical files are
 * cached, so CI does not re-download the weights on every run.
 */
import { createServer } from 'node:http'
import { createReadStream, existsSync, statSync } from 'node:fs'
import { join, normalize, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = join(
  dirname(fileURLToPath(import.meta.url)),
  '..',
  'tests',
  'fixtures',
  'models',
  'hf',
)
const port = Number(process.env.PORT ?? 4174)

createServer((req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*')
  res.setHeader('Access-Control-Expose-Headers', 'Content-Length')
  if (req.method === 'OPTIONS') return res.writeHead(204).end()
  const rel = normalize(decodeURIComponent(new URL(req.url ?? '/', 'http://x').pathname)).replace(
    /^([/\\])+/,
    '',
  )
  const file = join(root, rel)
  if (!file.startsWith(root) || !existsSync(file) || !statSync(file).isFile())
    return res.writeHead(404).end()
  res.writeHead(200, {
    'Content-Type': 'application/octet-stream',
    'Content-Length': statSync(file).size,
  })
  if (req.method === 'HEAD') return res.end()
  createReadStream(file).pipe(res)
}).listen(port, () => console.log(`serving ${root} on http://localhost:${port}/`))
