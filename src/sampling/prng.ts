/**
 * Reproducible pseudorandom number generator for sampling.
 *
 * Algorithm: SFC32 ("Small Fast Counting", Chris Doty-Humphrey, from PractRand),
 * a 128-bit-state generator with a 32-bit output. The 32-bit user seed is expanded
 * into the 4-word state with SplitMix32, then 15 outputs are discarded to decorrelate
 * nearby seeds.
 *
 * Uniform doubles in [0, 1) use 53 random bits built from two 32-bit outputs, so every
 * draw is exactly representable and can select tokens with probabilities down to ~1e-16.
 *
 * Limitations: NOT cryptographically secure. Reproducibility is guaranteed only for the
 * same implementation, the same logits, the same settings and the same PRNG state.
 * Different inference backends / precisions produce slightly different logits, which
 * can change the sampled token even with an identical seed.
 */

export type PrngState = readonly [number, number, number, number]

export class Sfc32 {
  private a: number
  private b: number
  private c: number
  private d: number

  constructor(state: PrngState) {
    this.a = state[0] >>> 0
    this.b = state[1] >>> 0
    this.c = state[2] >>> 0
    this.d = state[3] >>> 0
  }

  static fromSeed(seed: number): Sfc32 {
    return new Sfc32(seedToState(seed))
  }

  /** Next 32-bit unsigned integer. */
  nextUint32(): number {
    const t = (((this.a + this.b) | 0) + this.d) | 0
    this.d = (this.d + 1) | 0
    this.a = this.b ^ (this.b >>> 9)
    this.b = (this.c + (this.c << 3)) | 0
    this.c = (this.c << 21) | (this.c >>> 11)
    this.c = (this.c + t) | 0
    return t >>> 0
  }

  /** Uniform double in [0, 1) with 53 bits of randomness. */
  nextFloat(): number {
    const hi = this.nextUint32() >>> 5 // 27 bits
    const lo = this.nextUint32() >>> 6 // 26 bits
    return (hi * 67108864 + lo) / 9007199254740992
  }

  getState(): PrngState {
    return [this.a >>> 0, this.b >>> 0, this.c >>> 0, this.d >>> 0]
  }
}

export function seedToState(seed: number): PrngState {
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) {
    throw new RangeError(`Seed must be an integer in [0, 2^32-1], got ${seed}`)
  }
  let s = seed >>> 0
  const splitmix = () => {
    s = (s + 0x9e3779b9) | 0
    let z = s
    z = Math.imul(z ^ (z >>> 16), 0x85ebca6b)
    z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35)
    return (z ^ (z >>> 16)) >>> 0
  }
  const gen = new Sfc32([splitmix(), splitmix(), splitmix(), splitmix()])
  for (let i = 0; i < 15; i++) gen.nextUint32()
  return gen.getState()
}

export function isPrngState(value: unknown): value is PrngState {
  return (
    Array.isArray(value) &&
    value.length === 4 &&
    value.every((v) => Number.isInteger(v) && v >= 0 && v <= 0xffffffff)
  )
}
