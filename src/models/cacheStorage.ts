/**
 * Model cache management (main thread).
 *
 * Downloaded Hugging Face files are stored by Transformers.js in Cache Storage
 * (cache name "transformers-cache"), keyed by their resolve URL:
 *   https://huggingface.co/<owner>/<repo>/resolve/<revision>/<path>
 * This module lists, sizes and deletes those entries and reports storage quota.
 * Local (user-selected) model files are never written to any cache.
 */

export const MODEL_CACHE_NAME = 'transformers-cache'
const RESOLVE_RE = /^https:\/\/huggingface\.co\/(.+?)\/resolve\/([^/]+)\/(.+)$/

export interface CachedFile {
  url: string
  path: string
  bytes: number | null
}

export interface CachedModel {
  repo: string
  revision: string
  files: CachedFile[]
  totalBytes: number
}

export interface StorageInfo {
  usage: number | null
  quota: number | null
  persisted: boolean | null
  supported: boolean
}

export function cacheSupported(): boolean {
  return typeof caches !== 'undefined'
}

export function parseCacheUrl(url: string): { repo: string; revision: string; path: string } | null {
  const m = RESOLVE_RE.exec(url)
  if (!m) return null
  return { repo: m[1] as string, revision: decodeURIComponent(m[2] as string), path: m[3] as string }
}

async function entrySize(cache: Cache, req: Request): Promise<number | null> {
  const res = await cache.match(req)
  if (!res) return null
  const len = res.headers.get('content-length')
  if (len && /^\d+$/.test(len)) return Number(len)
  try {
    return (await res.blob()).size
  } catch {
    return null
  }
}

export async function listCachedModels(): Promise<CachedModel[]> {
  if (!cacheSupported()) return []
  const cache = await caches.open(MODEL_CACHE_NAME)
  const keys = await cache.keys()
  const groups = new Map<string, CachedModel>()
  for (const req of keys) {
    const parsed = parseCacheUrl(req.url)
    if (!parsed) continue
    const k = `${parsed.repo}@${parsed.revision}`
    if (!groups.has(k)) groups.set(k, { repo: parsed.repo, revision: parsed.revision, files: [], totalBytes: 0 })
    const g = groups.get(k)!
    const bytes = await entrySize(cache, req)
    g.files.push({ url: req.url, path: parsed.path, bytes })
    g.totalBytes += bytes ?? 0
  }
  return [...groups.values()].sort((a, b) => a.repo.localeCompare(b.repo))
}

export async function isFileCached(repo: string, revision: string, path: string): Promise<boolean> {
  if (!cacheSupported()) return false
  const cache = await caches.open(MODEL_CACHE_NAME)
  const res = await cache.match(`https://huggingface.co/${repo}/resolve/${encodeURIComponent(revision)}/${path}`)
  return !!res
}

/** Delete every cached file of a repo (optionally only one revision). Returns the number of files removed. */
export async function deleteCachedModel(repo: string, revision?: string): Promise<number> {
  if (!cacheSupported()) return 0
  const cache = await caches.open(MODEL_CACHE_NAME)
  let n = 0
  for (const req of await cache.keys()) {
    const parsed = parseCacheUrl(req.url)
    if (!parsed || parsed.repo !== repo) continue
    if (revision && parsed.revision !== revision) continue
    if (await cache.delete(req)) n++
  }
  return n
}

export async function deleteAllCachedModels(): Promise<boolean> {
  if (!cacheSupported()) return false
  return caches.delete(MODEL_CACHE_NAME)
}

export async function getStorageInfo(): Promise<StorageInfo> {
  const storage = typeof navigator !== 'undefined' ? navigator.storage : undefined
  if (!storage?.estimate) return { usage: null, quota: null, persisted: null, supported: false }
  try {
    const est = await storage.estimate()
    const persisted = storage.persisted ? await storage.persisted() : null
    return { usage: est.usage ?? null, quota: est.quota ?? null, persisted, supported: true }
  } catch {
    return { usage: null, quota: null, persisted: null, supported: false }
  }
}

/** Ask the browser not to evict our storage under pressure. */
export async function requestPersistentStorage(): Promise<boolean | null> {
  const storage = typeof navigator !== 'undefined' ? navigator.storage : undefined
  if (!storage?.persist) return null
  try {
    return await storage.persist()
  } catch {
    return null
  }
}
