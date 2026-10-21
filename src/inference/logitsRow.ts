/**
 * Extract the final position's logits row from a `[batch, seq, V]` output as float32.
 *
 * fp16 exports (e.g. Qwen3.5 fp16/q4f16) return float16 logits. ONNX Runtime Web hands
 * them over either as a Float16Array (where the browser supports it) or as a Uint16Array
 * of raw IEEE-754 half-precision bits; both are handled here explicitly, and only the last
 * row is converted (not the whole `seq × V` tensor).
 */
import { EngineError } from './engine'

export type LogitsData = Float32Array | Float64Array | Uint16Array | ArrayLike<number>

/** Decode one IEEE-754 binary16 bit pattern. */
export function halfToFloat(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1
  const exp = (bits >> 10) & 0x1f
  const frac = bits & 0x3ff
  if (exp === 0) return sign * frac * 2 ** -24 // zero / subnormal
  if (exp === 0x1f) return frac ? NaN : sign * Infinity
  return sign * (1 + frac / 1024) * 2 ** (exp - 15)
}

export function lastLogitsRow(
  dims: readonly number[],
  type: string,
  data: LogitsData,
): Float32Array {
  const V = dims[dims.length - 1] ?? 0
  if (!V)
    throw new EngineError(
      'The logits output has no vocabulary dimension',
      'unsupported-architecture',
    )
  const rows = Math.floor(data.length / V)
  const start = (rows - 1) * V
  if (type === 'float32' && data instanceof Float32Array) return data.slice(start, start + V)
  if (type === 'float16') {
    const out = new Float32Array(V)
    if (data instanceof Uint16Array) {
      // Raw half-precision bits (runtimes without Float16Array).
      for (let i = 0; i < V; i++) out[i] = halfToFloat(data[start + i] as number)
    } else {
      // Float16Array: elements are already numbers.
      for (let i = 0; i < V; i++) out[i] = data[start + i] as number
    }
    return out
  }
  if (type === 'float32' || type === 'float64') {
    const out = new Float32Array(V)
    for (let i = 0; i < V; i++) out[i] = data[start + i] as number
    return out
  }
  throw new EngineError(`Unsupported logits data type: ${type}`, 'unsupported-architecture')
}
