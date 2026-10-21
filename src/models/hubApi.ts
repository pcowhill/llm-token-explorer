/**
 * Hugging Face Hub API (main thread): what a repository offers before anything is
 * downloaded. One request, `GET /api/models/<repo>/revision/<rev>?blobs=true`, returns the
 * commit SHA, gating, `config.model_type` and every file with its size (CORS-enabled).
 */
import { KNOWN_CAUSAL_LM_TYPES } from './localFiles'
import { REQUIRED_SUPPORT_FILES, variantsFromFiles, type FileVariant } from './variants'

export const HUB_API = 'https://huggingface.co/api/models'
export const REPO_ID_RE = /^[\w.-]+\/[\w.-]+$/

export interface HubRepoInfo {
  repo: string
  /** Revision as requested (branch, tag or SHA). */
  revision: string
  /** Commit SHA the revision resolves to. */
  sha: string | null
  modelType: string | null
  files: { path: string; size: number | null }[]
  /** Loadable quantizations, most faithful first. */
  variants: FileVariant[]
  /** Problems that make the repo unloadable here. */
  errors: string[]
  /** Things that may go wrong but do not block loading. */
  warnings: string[]
}

export type HubErrorKind = 'offline' | 'not-found' | 'revision' | 'gated' | 'network'

export class HubApiError extends Error {
  constructor(
    message: string,
    public readonly kind: HubErrorKind,
  ) {
    super(message)
    this.name = 'HubApiError'
  }
}

interface ApiModelInfo {
  sha?: string
  gated?: false | 'auto' | 'manual'
  private?: boolean
  config?: { model_type?: string; architectures?: string[] }
  siblings?: { rfilename: string; size?: number }[]
}

/** Interpret the API response (pure; unit-tested). */
export function parseRepoInfo(repo: string, revision: string, body: ApiModelInfo): HubRepoInfo {
  const files = (body.siblings ?? []).map((s) => ({ path: s.rfilename, size: s.size ?? null }))
  const paths = new Set(files.map((f) => f.path))
  const variants = variantsFromFiles(files)
  const errors: string[] = []
  const warnings: string[] = []
  if (body.gated)
    errors.push(
      'This repository is gated: its files need a Hugging Face login and accepted terms, which a browser download without a token cannot provide.',
    )
  for (const f of REQUIRED_SUPPORT_FILES) if (!paths.has(f)) errors.push(`Missing ${f}.`)
  if (!variants.length) {
    const other = files.filter((f) => f.path.endsWith('.onnx'))
    errors.push(
      other.length
        ? `No onnx/model*.onnx files (found ${other
            .slice(0, 3)
            .map((f) => f.path)
            .join(
              ', ',
            )}); this app loads the onnx/model<suffix>.onnx layout used by onnx-community exports.`
        : 'No ONNX files: this is not a Transformers.js export. Look for an ONNX version (e.g. under onnx-community).',
    )
  }
  const modelType = body.config?.model_type ?? null
  if (!modelType) warnings.push('config.json has no model_type.')
  else if (!KNOWN_CAUSAL_LM_TYPES.has(modelType))
    warnings.push(
      `model_type "${modelType}" is not in this app's list of decoder-only architectures known to Transformers.js; loading may fail.`,
    )
  const archs = body.config?.architectures ?? []
  if (archs.length && !archs.some((a) => /ForCausalLM|LMHeadModel/.test(a)))
    warnings.push(`Architecture ${archs.join(', ')} is not a causal language model.`)
  return { repo, revision, sha: body.sha ?? null, modelType, files, variants, errors, warnings }
}

export async function fetchRepoInfo(
  repo: string,
  revision: string,
  opts: { signal?: AbortSignal; fetchFn?: typeof fetch } = {},
): Promise<HubRepoInfo> {
  const fetchFn = opts.fetchFn ?? fetch
  const url = `${HUB_API}/${repo}/revision/${encodeURIComponent(revision)}?blobs=true`
  let res: Response
  try {
    res = await fetchFn(url, { signal: opts.signal, headers: { accept: 'application/json' } })
  } catch (e) {
    if ((e as Error).name === 'AbortError') throw e
    if (typeof navigator !== 'undefined' && navigator.onLine === false)
      throw new HubApiError('You are offline, so the available files cannot be checked.', 'offline')
    throw new HubApiError(
      `Could not reach the Hugging Face API (${(e as Error).message}).`,
      'network',
    )
  }
  if (res.status === 401 || res.status === 403)
    // The Hub answers 401 for repositories that do not exist and for private ones.
    throw res.status === 403
      ? new HubApiError(`Access to ${repo} is restricted (gated or private).`, 'gated')
      : new HubApiError(`Repository ${repo} was not found (or it is private).`, 'not-found')
  if (res.status === 404) {
    const msg = await res
      .json()
      .then((b: { error?: string }) => b.error ?? '')
      .catch(() => '')
    throw new HubApiError(
      /rev/i.test(msg)
        ? `Revision “${revision}” does not exist in ${repo}.`
        : `Repository ${repo} was not found.`,
      /rev/i.test(msg) ? 'revision' : 'not-found',
    )
  }
  if (!res.ok) throw new HubApiError(`The Hugging Face API answered HTTP ${res.status}.`, 'network')
  return parseRepoInfo(repo, revision, (await res.json()) as ApiModelInfo)
}
