/** In-memory stand-in for window.caches (Cache Storage), for component tests only. */
import { vi } from 'vitest'

export const HF = 'https://huggingface.co'

export function resolveUrl(repo: string, revision: string, path: string) {
  return `${HF}/${repo}/resolve/${revision}/${path}`
}

export function installFakeCaches(urls: string[]) {
  const store = new Map<string, number>(urls.map((u) => [u, 1000]))
  const key = (r: string | { url: string }) => (typeof r === 'string' ? r : r.url)
  const cache = {
    keys: async () => [...store.keys()].map((url) => ({ url })),
    match: async (r: string | { url: string }) =>
      store.has(key(r))
        ? new Response('x', { headers: { 'content-length': String(store.get(key(r))) } })
        : undefined,
    delete: async (r: string | { url: string }) => store.delete(key(r)),
  }
  vi.stubGlobal('caches', {
    open: async () => cache,
    delete: async () => {
      store.clear()
      return true
    },
  })
  return store
}
