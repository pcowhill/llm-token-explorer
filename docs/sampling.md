# Sampling pipeline

Implemented in `src/sampling/pipeline.ts`, tested in `tests/unit/pipeline.test.ts`.
Sampling settings never modify the logits, the original distribution or the model.

## Order of operations

Given raw logits `z` (length V) and settings `{T, k, p, m, greedy, seed}`:

0. **Eligibility.** IDs without a tokenizer entry (e.g. Qwen padding rows) and NaN logits
   are ineligible. Eligible IDs are sorted once by logit, descending (ties → lower ID).
   Because temperature is monotonic, this order is reused for every setting change.
1. **Greedy path.** If `greedy` is on **or** `T = 0`: the adjusted distribution is one-hot on
   the highest-logit eligible ID. No division by zero ever happens.
2. **Temperature.** `qᵢ = softmax(z / T)ᵢ` over eligible IDs (stable softmax, float64).
3. **Top-k** (`k > 0`): keep the `k` most probable tokens.
4. **Top-p** (`p < 1`): renormalize the survivors, walk them in descending order and keep the
   smallest prefix whose cumulative probability reaches `p`. The token that crosses the
   threshold is kept, so at least one token always survives.
5. **Min-p** (`m > 0`): keep tokens with `qᵢ ≥ m · max_j q_j` (relative to the most probable
   surviving token, which is always kept).
6. **Renormalize** the survivors to sum to 1.

This matches the order of Hugging Face `transformers` logits warpers
(temperature → top-k → top-p → min-p). Repetition penalties and other processors are not
applied; the app always shows the raw model distribution first.

### Edge cases

- Aggressive filtering can never remove every token: top-p keeps the boundary token and
  min-p always keeps the maximum.
- Probabilities that underflow to exactly 0 at very low temperature are dropped from the
  survivor set; very low `T` therefore converges to greedy without NaN.
- Invalid settings (negative temperature, top-p outside (0, 1], non-integer k, …) are
  reported and sampling is disabled until fixed.

## Drawing a token

A uniform `u ∈ [0, 1)` is drawn and the token is the first survivor (in descending
probability order) whose cumulative adjusted probability exceeds `u` (inverse CDF). The
outcome is a deterministic function of `(adjusted distribution, u)`.

## Random seed and reproducibility

- Generator: **SFC32** (Chris Doty-Humphrey's Small Fast Counting generator from PractRand),
  128-bit state. The 32-bit seed is expanded with SplitMix32 and 15 outputs are discarded.
- Each uniform uses 53 random bits (two 32-bit outputs), so every double in [0, 1) at
  2⁻⁵³ resolution is reachable.
- The generator state is stored in every history node (and in exported experiments), so
  navigating back and sampling again reproduces the same draw.
- Not cryptographically secure. Reproducibility holds for the same implementation, logits,
  settings and generator state. **Different backends (WASM vs. WebGPU) or precisions (fp16
  vs. q8) produce different logits and can therefore change the sampled token.**

## Example

Probabilities `[0.5, 0.25, 0.125, 0.0625, 0.0625]`:

| Settings    | Survivors                    | Adjusted                                                  |
| ----------- | ---------------------------- | --------------------------------------------------------- |
| T = 1       | all 5                        | unchanged                                                 |
| T = 0.5     | all 5                        | `p²` renormalized → `[0.744, 0.186, 0.047, 0.012, 0.012]` |
| top-k = 2   | 2                            | `[0.667, 0.333]`                                          |
| top-p = 0.8 | 3 (0.5 + 0.25 + 0.125 ≥ 0.8) | `[0.571, 0.286, 0.143]`                                   |
| min-p = 0.3 | 2 (threshold 0.15)           | `[0.667, 0.333]`                                          |
| greedy      | 1                            | `[1]`                                                     |
