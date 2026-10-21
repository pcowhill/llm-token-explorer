/** Display formatting for probabilities, logits and timings. */

/** Human-friendly probability: percentages ≥ 0.01 %, otherwise scientific notation. */
export function formatProb(p: number | null | undefined): string {
  if (p === null || p === undefined || Number.isNaN(p)) return '—'
  if (p === 0) return '0'
  if (p === 1) return '100%'
  if (p >= 0.0001) return `${(p * 100).toFixed(p >= 0.1 ? 1 : 2)}%`
  return p.toExponential(2)
}

/** Full-precision representation (shortest round-trip double). */
export function exactNumber(x: number | null | undefined): string {
  if (x === null || x === undefined) return '—'
  if (Number.isNaN(x)) return 'NaN'
  if (!Number.isFinite(x)) return x > 0 ? '+Infinity' : '-Infinity'
  return String(x)
}

export function formatLogit(x: number | null | undefined): string {
  if (x === null || x === undefined || Number.isNaN(x)) return '—'
  if (!Number.isFinite(x)) return x > 0 ? '+∞' : '−∞'
  return x.toFixed(3)
}

/** Natural-log probability. */
export function formatLogProb(x: number | null | undefined): string {
  if (x === null || x === undefined || Number.isNaN(x)) return '—'
  if (x === -Infinity) return '−∞'
  return x.toFixed(3)
}

export function formatMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return '—'
  if (ms < 1) return `${ms.toFixed(2)} ms`
  if (ms < 1000) return `${ms.toFixed(0)} ms`
  return `${(ms / 1000).toFixed(2)} s`
}

export function formatCount(n: number | null | undefined): string {
  if (n === null || n === undefined) return '—'
  return n.toLocaleString('en-US')
}

export function shortSha(sha: string | undefined): string {
  return sha ? sha.slice(0, 7) : ''
}
