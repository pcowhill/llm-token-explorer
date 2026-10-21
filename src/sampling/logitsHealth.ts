/**
 * Sanity check for a model's raw next-token logits.
 *
 * A broken variant/backend combination can return a vector that softmax would happily
 * turn into a "distribution": all-zero or constant logits become a perfectly uniform
 * distribution (this is what SmolLM2 q4f16 produced on WebGPU, see
 * docs/model-compatibility.md#fp16-overflow-on-webgpu). Such output must never be shown as
 * the model's prediction.
 */

export type LogitsProblem =
  | { kind: 'empty'; message: string }
  | { kind: 'nan'; count: number; message: string }
  | { kind: 'infinite'; count: number; message: string }
  | { kind: 'constant'; value: number; message: string }

/** Returns null for a usable logits vector, otherwise what is wrong with it. */
export function checkLogits(logits: ArrayLike<number>): LogitsProblem | null {
  const n = logits.length
  if (n === 0) return { kind: 'empty', message: 'The model returned an empty logits vector.' }
  let nan = 0
  let posInf = 0
  let min = Infinity
  let max = -Infinity
  for (let i = 0; i < n; i++) {
    const x = logits[i] as number
    if (Number.isNaN(x)) nan++
    else if (x === Infinity) posInf++
    else if (x !== -Infinity) {
      if (x < min) min = x
      if (x > max) max = x
    }
  }
  if (nan)
    return {
      kind: 'nan',
      count: nan,
      message: `${nan.toLocaleString('en-US')} of ${n.toLocaleString('en-US')} logits are NaN.`,
    }
  if (posInf)
    return {
      kind: 'infinite',
      count: posInf,
      message: `${posInf.toLocaleString('en-US')} logits are +Infinity.`,
    }
  // -Infinity is how masking is expressed; it is fine as long as the rest is informative.
  if (min === Infinity) return { kind: 'infinite', count: n, message: 'Every logit is -Infinity.' }
  if (max - min === 0)
    return {
      kind: 'constant',
      value: min,
      message: `Every finite logit has the same value (${min}), which would be a meaningless uniform distribution.`,
    }
  return null
}
