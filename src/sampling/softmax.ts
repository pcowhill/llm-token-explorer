/**
 * Numerically stable softmax / log-softmax over large vocabularies.
 *
 * P(token_i) = exp(logit_i) / Σ_j exp(logit_j)
 *
 * Implementation: subtract the maximum finite logit before exponentiating
 * (log-sum-exp trick) and accumulate in float64. This never overflows and keeps
 * relative precision for probabilities far below float32's smallest normal.
 *
 * Invalid inputs:
 * - NaN logits are treated as -Infinity (probability exactly 0) and counted.
 * - -Infinity logits get probability 0 (this is how masking is expressed).
 * - +Infinity logits: the mass is split evenly among the +Infinity entries
 *   (the mathematical limit), and they are counted as invalid.
 * - If every logit is -Infinity/NaN the distribution is undefined → `ok: false`.
 */

export interface SoftmaxResult {
  ok: boolean
  probs: Float64Array
  /** Natural-log probabilities; -Infinity where probability is 0. */
  logProbs: Float64Array
  /** log Σ exp(x_j) of the (sanitized) inputs. */
  logSumExp: number
  maxIndex: number
  nanCount: number
  posInfCount: number
  error?: string
}

export function softmax(logits: ArrayLike<number>, scale = 1): SoftmaxResult {
  const n = logits.length
  const probs = new Float64Array(n)
  const logProbs = new Float64Array(n)
  let nanCount = 0
  let posInfCount = 0
  let max = -Infinity
  let maxIndex = -1

  for (let i = 0; i < n; i++) {
    const x = logits[i] as number
    if (Number.isNaN(x)) {
      nanCount++
      continue
    }
    if (x === Infinity) {
      posInfCount++
      if (max !== Infinity) {
        max = Infinity
        maxIndex = i
      }
      continue
    }
    const v = x * scale
    if (v > max) {
      max = v
      maxIndex = i
    }
  }

  if (n === 0 || maxIndex === -1 || max === -Infinity) {
    logProbs.fill(-Infinity)
    return {
      ok: false,
      probs,
      logProbs,
      logSumExp: -Infinity,
      maxIndex: -1,
      nanCount,
      posInfCount,
      error: n === 0 ? 'Empty logits vector' : 'All logits are -Infinity or NaN',
    }
  }

  if (max === Infinity) {
    const p = 1 / posInfCount
    for (let i = 0; i < n; i++) {
      if (logits[i] === Infinity) {
        probs[i] = p
        logProbs[i] = Math.log(p)
      } else {
        logProbs[i] = -Infinity
      }
    }
    return { ok: true, probs, logProbs, logSumExp: Infinity, maxIndex, nanCount, posInfCount }
  }

  let sum = 0
  for (let i = 0; i < n; i++) {
    const x = logits[i] as number
    if (Number.isNaN(x) || x === -Infinity) continue
    const e = Math.exp(x * scale - max)
    probs[i] = e
    sum += e
  }
  const logSum = Math.log(sum)
  const lse = max + logSum
  const inv = 1 / sum
  for (let i = 0; i < n; i++) {
    const x = logits[i] as number
    if (Number.isNaN(x) || x === -Infinity) {
      logProbs[i] = -Infinity
      probs[i] = 0
      continue
    }
    logProbs[i] = x * scale - lse
    probs[i] = (probs[i] as number) * inv
  }
  return { ok: true, probs, logProbs, logSumExp: lse, maxIndex, nanCount, posInfCount }
}

/** Kahan-compensated sum, used to verify normalization in tests and diagnostics. */
export function stableSum(values: ArrayLike<number>): number {
  let sum = 0
  let c = 0
  for (let i = 0; i < values.length; i++) {
    const y = (values[i] as number) - c
    const t = sum + y
    c = t - sum - y
    sum = t
  }
  return sum
}

/** Indices sorted by value descending; ties broken by lower index (deterministic). NaN last. */
export function argsortDescending(values: ArrayLike<number>): Uint32Array {
  const n = values.length
  const idx = new Uint32Array(n)
  for (let i = 0; i < n; i++) idx[i] = i
  const key = (i: number) => {
    const v = values[i] as number
    return Number.isNaN(v) ? -Infinity : v
  }
  idx.sort((a, b) => {
    const d = key(b) - key(a)
    if (d > 0 || d < 0) return d
    // Equal (or both ±Infinity): stable by index
    return a - b
  })
  return idx
}
