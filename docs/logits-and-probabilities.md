# Logits, probabilities and log-probabilities

| Quantity                      | What it is                                                                                                     | Where it comes from                                               |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| **Logit** `zᵢ`                | Unnormalized real-valued score for vocabulary entry _i_ at the next position. Can be negative; no fixed scale. | Output tensor `logits` of the ONNX model, last sequence position. |
| **Original probability** `pᵢ` | `softmax(z)ᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)` over **all** output IDs. The model's own next-token distribution.         | Computed by this app (`src/sampling/softmax.ts`).                 |
| **Log-probability** `log pᵢ`  | `zᵢ − logsumexp(z)` (natural log). Stays exact when `pᵢ` underflows to 0.                                      | Computed by this app.                                             |
| **Adjusted probability** `qᵢ` | Distribution actually sampled from after temperature, top-k, top-p, min-p and renormalization.                 | `src/sampling/pipeline.ts` — see [sampling.md](sampling.md).      |
| Generated token ID            | One integer chosen from `q` (or picked manually).                                                              | Token selection, not the network.                                 |

Not shown (and never confused with logits): attention weights, hidden states, embeddings.

## Numerical stability

```
m   = max_j z_j                 (ignoring NaN)
pᵢ  = exp(zᵢ − m) / Σⱼ exp(zⱼ − m)
log pᵢ = zᵢ − (m + log Σⱼ exp(zⱼ − m))
```

- Subtracting `m` makes every exponent ≤ 0, so `exp` never overflows; the result is
  mathematically identical.
- Sums and probabilities are computed in **float64** even though logits arrive as float32.
- `NaN` logits are excluded (probability 0) and counted; `−∞` means "masked" (probability
  0); `+∞` logits share the probability mass equally. If no logit is finite the
  distribution is reported as invalid rather than producing NaN.
- The UI shows probabilities ≥ 0.01 % as percentages and smaller ones in scientific
  notation; the _exact values_ toggle (and tooltips) show the full float64 value.

## Worked example

Logits `[2.0, 1.0, 0.1]` → `exp` `[7.389, 2.718, 1.105]`, sum `11.212` → probabilities
`[0.659, 0.242, 0.099]`.

## Which position?

A causal LM returns logits for every input position (`[1, seq, vocab]`). The logits at
position _t_ predict token _t+1_; the app uses the **last** row. With the optional KV cache
only the new tokens are fed, and the last row is still the next-token prediction.

## Vocabulary size vs. logits size

The number of logits (`config.vocab_size`, the embedding/LM-head rows) can differ from
the number of tokenizer entries:

- Qwen2.5: 151,936 logits, 151,665 tokenizer entries — IDs 151,665–151,935 are padding
  rows with no text. They appear in the original distribution (with tiny probability) but
  are **excluded from sampling**.
- Gemma 3 (not in the catalog) has 262,145 tokenizer entries but 262,144 logits — the last
  tokenizer ID cannot be predicted or fed to the model, and the app rejects it as input.
