import type { SamplingSettings } from '@/types'
import { argsortDescending, softmax, type SoftmaxResult } from './softmax'

/**
 * The sampling pipeline (see docs/sampling.md).
 *
 * Inputs: the raw logits z (never modified) and the sampling settings.
 *
 *  0. Eligibility mask: output IDs that have no tokenizer entry (padding rows of the
 *     logits matrix, e.g. Qwen2.5 IDs 151665–151935) and NaN logits are ineligible.
 *  1. Greedy path: if `greedy` is on or temperature == 0, the adjusted distribution is
 *     one-hot on argmax(z) over eligible IDs (ties → lowest ID). No division by zero.
 *  2. Temperature: p_i = softmax(z / T)_i over eligible IDs.
 *  3. Top-k (k > 0): keep the k most probable tokens.
 *  4. Top-p (p < 1): sort descending; keep the smallest prefix whose cumulative
 *     probability (renormalized after step 3) reaches p. The boundary token that
 *     crosses the threshold is kept, so at least one token always survives.
 *  5. Min-p (m > 0): keep tokens with p_i >= m · max_j p_j (renormalized after step 4).
 *  6. Renormalize the survivors so they sum to 1.
 *
 * This is the same order Hugging Face `transformers` applies its logits warpers
 * (temperature → top-k → top-p → min-p). Because temperature is a monotonic transform,
 * the ranking of tokens never changes, so a single descending sort of the raw logits
 * is reused for every setting change.
 */

export interface PreparedLogits {
  logits: Float32Array
  /** Softmax of the raw logits (T = 1) over ALL output IDs: the model's distribution. */
  original: SoftmaxResult
  /** Eligible output IDs sorted by raw logit, descending (ties: lower ID first). */
  order: Uint32Array
  /** ALL output IDs sorted by raw logit, descending (NaN last). */
  fullOrder: Uint32Array
  /** eligible[i] = 1 when ID i can be sampled. */
  eligible: Uint8Array
  eligibleCount: number
}

export interface AdjustedDistribution {
  /** Final sampling probabilities (0 for filtered tokens); sums to 1. */
  probs: Float64Array
  /** Number of tokens with non-zero adjusted probability. */
  keptCount: number
  /** Survivors in descending probability order. */
  kept: Uint32Array
  greedy: boolean
  /** Diagnostics about what each stage removed. */
  stages: {
    afterTemperature: number
    afterTopK: number
    afterTopP: number
    afterMinP: number
  }
  /** Effective temperature used (1 when greedy). */
  temperature: number
}

export interface PrepareOptions {
  /**
   * Number of output IDs that map to a tokenizer entry. IDs >= mappedCount, and IDs
   * listed in `unmappedIds`, are excluded from sampling (but still shown).
   */
  mappedCount?: number
  unmappedIds?: Iterable<number>
}

export function prepareLogits(logits: Float32Array, options: PrepareOptions = {}): PreparedLogits {
  const n = logits.length
  const original = softmax(logits)
  const eligible = new Uint8Array(n)
  const mapped = Math.min(options.mappedCount ?? n, n)
  for (let i = 0; i < mapped; i++) {
    const x = logits[i] as number
    eligible[i] = Number.isNaN(x) || x === -Infinity ? 0 : 1
  }
  if (options.unmappedIds) for (const id of options.unmappedIds) if (id < n) eligible[id] = 0
  const fullOrder = argsortDescending(logits)
  let eligibleCount = 0
  for (let i = 0; i < n; i++) eligibleCount += eligible[i] as number
  const order = new Uint32Array(eligibleCount)
  let k = 0
  for (let i = 0; i < n; i++) {
    const id = fullOrder[i] as number
    if (eligible[id]) order[k++] = id
  }
  return { logits, original, order, fullOrder, eligible, eligibleCount }
}

export function validateSamplingSettings(s: SamplingSettings): string[] {
  const errors: string[] = []
  if (!Number.isFinite(s.temperature) || s.temperature < 0)
    errors.push('Temperature must be a finite number ≥ 0.')
  if (!Number.isInteger(s.topK) || s.topK < 0) errors.push('Top-k must be an integer ≥ 0.')
  if (!Number.isFinite(s.topP) || s.topP <= 0 || s.topP > 1) errors.push('Top-p must be in (0, 1].')
  if (!Number.isFinite(s.minP) || s.minP < 0 || s.minP > 1) errors.push('Min-p must be in [0, 1].')
  if (!Number.isInteger(s.seed) || s.seed < 0 || s.seed > 0xffffffff)
    errors.push('Seed must be an integer between 0 and 4294967295.')
  return errors
}

export function isGreedy(s: SamplingSettings): boolean {
  return s.greedy || s.temperature === 0
}

export function adjustDistribution(
  prepared: PreparedLogits,
  settings: SamplingSettings,
): AdjustedDistribution {
  const { logits, order } = prepared
  const n = logits.length
  const probs = new Float64Array(n)
  const m = order.length
  if (m === 0) {
    return {
      probs,
      keptCount: 0,
      kept: new Uint32Array(0),
      greedy: isGreedy(settings),
      stages: { afterTemperature: 0, afterTopK: 0, afterTopP: 0, afterMinP: 0 },
      temperature: settings.temperature,
    }
  }

  if (isGreedy(settings)) {
    const top = order[0] as number
    probs[top] = 1
    return {
      probs,
      keptCount: 1,
      kept: Uint32Array.of(top),
      greedy: true,
      stages: { afterTemperature: 1, afterTopK: 1, afterTopP: 1, afterMinP: 1 },
      temperature: 0,
    }
  }

  const T = settings.temperature
  // Step 2: temperature-scaled softmax over eligible IDs, in descending order.
  // order[0] has the max logit, so subtracting it keeps exp() in (0, 1].
  const maxScaled = (logits[order[0] as number] as number) / T
  const sorted = new Float64Array(m)
  let total = 0
  for (let r = 0; r < m; r++) {
    const e = Math.exp((logits[order[r] as number] as number) / T - maxScaled)
    sorted[r] = e
    total += e
  }
  for (let r = 0; r < m; r++) sorted[r] = (sorted[r] as number) / total
  let keep = m
  const afterTemperature = countPositive(sorted, keep)

  // Step 3: top-k
  if (settings.topK > 0 && settings.topK < keep) keep = settings.topK
  const afterTopK = keep

  // Step 4: top-p over the renormalized top-k survivors
  if (settings.topP < 1) {
    let mass = 0
    for (let r = 0; r < keep; r++) mass += sorted[r] as number
    const threshold = settings.topP * mass
    let cum = 0
    let cut = keep
    for (let r = 0; r < keep; r++) {
      cum += sorted[r] as number
      // Keep the boundary token whose inclusion first reaches the threshold.
      // A tiny relative epsilon guards against float round-off at exact boundaries.
      if (cum >= threshold * (1 - 1e-12)) {
        cut = r + 1
        break
      }
    }
    keep = Math.max(1, cut)
  }
  const afterTopP = keep

  // Step 5: min-p relative to the most probable surviving token (always rank 0).
  if (settings.minP > 0) {
    const cutoff = settings.minP * (sorted[0] as number)
    let cut = 1
    for (let r = 1; r < keep; r++) {
      if ((sorted[r] as number) >= cutoff) cut = r + 1
      else break
    }
    keep = cut
  }
  const afterMinP = keep

  // Step 6: renormalize survivors. Drop survivors whose probability underflowed to 0.
  let keptMass = 0
  let positive = 0
  for (let r = 0; r < keep; r++) {
    const p = sorted[r] as number
    if (p > 0) {
      keptMass += p
      positive++
    }
  }
  const kept = new Uint32Array(positive)
  let w = 0
  for (let r = 0; r < keep; r++) {
    const p = sorted[r] as number
    if (p > 0) {
      const id = order[r] as number
      probs[id] = p / keptMass
      kept[w++] = id
    }
  }
  return {
    probs,
    keptCount: positive,
    kept,
    greedy: false,
    stages: { afterTemperature, afterTopK, afterTopP, afterMinP },
    temperature: T,
  }
}

function countPositive(arr: Float64Array, upto: number): number {
  let c = 0
  for (let i = 0; i < upto; i++) if ((arr[i] as number) > 0) c++
  return c
}

/**
 * Draw one token from an adjusted distribution using a uniform u ∈ [0, 1).
 * Uses inverse-CDF over the survivors in descending-probability order, so the
 * outcome is a deterministic function of (distribution, u).
 */
export function sampleFromAdjusted(adjusted: AdjustedDistribution, u: number): number {
  if (adjusted.keptCount === 0) throw new Error('Cannot sample: no eligible tokens')
  if (!(u >= 0 && u < 1)) throw new Error(`Uniform draw out of range: ${u}`)
  let cum = 0
  for (let r = 0; r < adjusted.kept.length; r++) {
    const id = adjusted.kept[r] as number
    cum += adjusted.probs[id] as number
    if (u < cum) return id
  }
  // Floating-point slack: return the last survivor.
  return adjusted.kept[adjusted.kept.length - 1] as number
}
