<script setup lang="ts">
import { ref, watch, nextTick } from 'vue'
import SamplingPlayground from './SamplingPlayground.vue'

const props = defineProps<{ open: boolean }>()
const emit = defineEmits<{ close: [] }>()
const dialog = ref<HTMLDialogElement | null>(null)
watch(
  () => props.open,
  async (o) => {
    await nextTick()
    if (!dialog.value) return
    if (o && !dialog.value.open) {
      if (typeof dialog.value.showModal === 'function') dialog.value.showModal()
      else dialog.value.setAttribute('open', '')
    }
    if (!o && dialog.value.open) {
      if (typeof dialog.value.close === 'function') dialog.value.close()
      else dialog.value.removeAttribute('open')
    }
  },
)
</script>

<template>
  <dialog
    ref="dialog"
    class="learn card"
    aria-labelledby="learn-title"
    data-testid="learn-panel"
    @close="emit('close')"
    @cancel="emit('close')"
  >
    <div class="row head">
      <h2 id="learn-title">How language models pick the next token</h2>
      <span class="spacer" />
      <button type="button" class="btn btn-sm" autofocus @click="emit('close')">Close</button>
    </div>
    <div class="content">
      <nav aria-label="Topics" class="toc small">
        <a href="#l-tokens">Tokens</a> · <a href="#l-special">Special tokens</a> ·
        <a href="#l-logits">Logits &amp; softmax</a> · <a href="#l-temperature">Temperature</a> ·
        <a href="#l-filters">Top-k / top-p / min-p</a> · <a href="#l-seed">Seeds &amp; greedy</a> ·
        <a href="#l-vary">Why outputs vary</a> · <a href="#l-quant">Quantization</a> ·
        <a href="#l-browser">Browser inference</a> · <a href="#l-context">Context length</a> ·
        <a href="#l-privacy">Privacy</a> · <a href="#l-keys">Keyboard</a>
      </nav>

      <section id="l-tokens">
        <h3>What is a token?</h3>
        <p>
          A model reads and writes <strong>tokens</strong>: integer IDs into a fixed vocabulary
          (49,152 entries for SmolLM2, ~151k for Qwen2.5, 50,257 for GPT-2). A
          <strong>tokenizer</strong> converts text to IDs, typically with byte-pair encoding (BPE):
          frequent character sequences become single tokens, rare ones are split into pieces.
        </p>
        <p>
          Token boundaries rarely match words. A leading space is usually part of the next word’s
          token (shown here as <code>␣</code>), punctuation and numbers split in surprising ways,
          and a single emoji or accented character may be several byte-level tokens that only form a
          valid character together (shown as <code>�</code> individually).
        </p>
        <p>
          Different models have different vocabularies, so the same text becomes different IDs — and
          the same ID means different text in different models.
        </p>
      </section>

      <section id="l-special">
        <h3>Special tokens</h3>
        <p>
          Special tokens are control symbols such as beginning/end of sequence, end of turn, or chat
          role delimiters (<code>&lt;|im_start|&gt;</code>, <code>&lt;|im_end|&gt;</code>). They
          have their own IDs. Typing their spelling in text <em>may</em> or may not produce the
          special ID depending on the tokenizer; inserting by ID (as this app does) is exact.
        </p>
        <p>
          Chat models are trained on a specific <strong>chat template</strong> built from these
          tokens. Using the wrong format degrades answers.
        </p>
      </section>

      <section id="l-logits">
        <h3>Logits and softmax</h3>
        <p>
          For every position the network outputs a vector of <strong>logits</strong>: one
          real-valued score per vocabulary entry. Logits are not probabilities — they can be
          negative and do not sum to one. Only the logits of the <em>last</em> position predict the
          next token.
        </p>
        <p><strong>Softmax</strong> converts logits to probabilities:</p>
        <p class="formula">P(token_i) = exp(logit_i) / Σ_j exp(logit_j)</p>
        <p>
          Worked example with logits [2.0, 1.0, 0.1]: exp → [7.389, 2.718, 1.105], sum 11.212 →
          probabilities [0.659, 0.242, 0.099].
        </p>
        <p>
          The implementation subtracts the maximum logit first (<code>exp(z_i − max)</code>), which
          gives identical results without overflow, and works in 64-bit floats so probabilities as
          small as 1e-300 stay meaningful. The “log p” column shows natural-log probabilities, which
          remain exact even when p underflows.
        </p>
      </section>

      <section id="l-temperature">
        <h3>Temperature</h3>
        <p class="formula">P_T(token_i) = softmax(logit / T)_i</p>
        <p>
          Same logits with T = 0.5 → [4, 2, 0.2] → [0.864, 0.117, 0.019]: the favourite becomes more
          dominant. T = 2 → [1, 0.5, 0.05] → [0.502, 0.304, 0.194]: flatter and more random. T = 0
          is treated as greedy decoding instead of dividing by zero.
        </p>
      </section>

      <section id="l-filters">
        <h3>Top-k, top-p and min-p</h3>
        <ul>
          <li><strong>Top-k</strong> keeps only the k most probable tokens.</li>
          <li>
            <strong>Top-p</strong> (nucleus) keeps the smallest set of most-probable tokens whose
            probabilities add up to at least p (including the token that crosses p).
          </li>
          <li>
            <strong>Min-p</strong> keeps tokens with probability ≥ min_p × (probability of the most
            likely token), so it adapts to the model’s confidence.
          </li>
        </ul>
        <p>
          This app applies <strong>temperature → top-k → top-p → min-p</strong>, renormalizing so
          the survivors sum to 1. At least one token always survives. Ineligible IDs (logit rows
          with no tokenizer entry) are never sampled.
        </p>
      </section>

      <section id="l-seed">
        <h3>Random seeds and greedy decoding</h3>
        <p>
          Sampling draws u ∈ [0, 1) from a seeded pseudorandom generator (SFC32) and walks the
          cumulative adjusted distribution until it exceeds u. The same seed, settings and logits
          reproduce the same token. <strong>Greedy decoding</strong> skips randomness and always
          takes the highest-scoring token.
        </p>
      </section>

      <section id="l-vary">
        <h3>Why outputs vary</h3>
        <ul>
          <li>Sampling is random by design (unless greedy).</li>
          <li>
            WebGPU and WASM, fp32 and quantized weights, and even batching (KV cache) produce
            slightly different logits, which can flip close decisions.
          </li>
          <li>
            Sampling settings change only the <em>selection</em>: they never modify the model or its
            weights.
          </li>
        </ul>
      </section>

      <section id="l-quant">
        <h3>Quantization</h3>
        <p>
          Quantization stores weights with fewer bits (8-bit “q8”, 4-bit “q4”) to cut download size
          and memory by 4–8× at a small cost in accuracy. q8 here uses
          <em>dynamic</em> quantization: activations are also quantized on the fly, with scales
          computed from the current batch of tokens.
        </p>
      </section>

      <section id="l-browser">
        <h3>Why browser inference is slower than native</h3>
        <p>
          Native runtimes (llama.cpp, Ollama) use hand-tuned CPU/GPU kernels, all CPU cores and
          direct GPU memory. In the browser, WASM runs single-threaded unless the page is
          cross-origin isolated (GitHub Pages cannot send the required headers), WebGPU adds
          abstraction overhead, and memory per tab is limited.
        </p>
      </section>

      <section id="l-context">
        <h3>Context length</h3>
        <p>
          Each model was trained with a maximum sequence length (1,024 tokens for GPT-2, 8,192 for
          SmolLM2, 32,768 for Qwen2.5). Beyond that, position information is out of distribution and
          quality collapses; this app refuses to run past the limit. Every forward pass here
          re-reads the whole context, so cost grows with its length.
        </p>
      </section>

      <section id="l-privacy">
        <h3>Privacy and network requests</h3>
        <p>
          Everything runs in your browser. Your text and token sequences never leave the device. The
          only network requests are: the app’s own static files, and model files from
          <code>huggingface.co</code> (and its CDN) when you choose to load a model. Locally
          selected model files are read directly and never uploaded. There is no analytics or
          tracking.
        </p>
      </section>

      <section id="l-keys">
        <h3>Keyboard</h3>
        <ul>
          <li>
            Token chips: <kbd>←</kbd>/<kbd>→</kbd> move, <kbd>Home</kbd>/<kbd>End</kbd>,
            <kbd>Delete</kbd> removes, <kbd>Esc</kbd> deselects.
          </li>
          <li>
            <kbd>Ctrl</kbd>+<kbd>Z</kbd> undo, <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>Z</kbd> redo
            (outside text fields).
          </li>
          <li>
            <kbd>Ctrl</kbd>+<kbd>Enter</kbd> calculate · <kbd>Alt</kbd>+<kbd>S</kbd> sample one
            token.
          </li>
        </ul>
      </section>

      <SamplingPlayground />
    </div>
  </dialog>
</template>

<style scoped>
.learn {
  width: min(56rem, 96vw);
  max-height: 90vh;
  padding: 0;
  color: var(--text);
  background: var(--surface);
}
.learn::backdrop {
  background: rgb(0 0 0 / 45%);
}
.head {
  padding: 0.7rem 1rem;
  border-bottom: 1px solid var(--border);
  position: sticky;
  top: 0;
  background: var(--surface);
}
.content {
  padding: 0.5rem 1rem 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.9rem;
}
.toc {
  line-height: 1.8;
}
section h3 {
  margin-bottom: 0.25rem;
}
</style>
