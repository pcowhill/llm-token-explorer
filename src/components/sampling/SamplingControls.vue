<script setup lang="ts">
import { computed } from 'vue'
import { useWorkbench } from '@/composables/useWorkbench'
import InfoTip from '@/components/common/InfoTip.vue'
import { DEFAULT_SAMPLING } from '@/types'
import { formatMs } from '@/utils/format'

const wb = useWorkbench()
const s = wb.sampling
const greedy = computed(() => s.greedy || s.temperature === 0)
const prngHex = computed(() => wb.prng.value.map((x) => x.toString(16).padStart(8, '0')).join(' '))
const stages = computed(() => wb.adjusted.value?.stages ?? null)

function preset(name: 'neutral' | 'precise' | 'creative' | 'model') {
  if (name === 'neutral')
    Object.assign(s, { temperature: 1, topK: 0, topP: 1, minP: 0, greedy: false })
  if (name === 'precise')
    Object.assign(s, { temperature: 0.3, topK: 0, topP: 0.9, minP: 0, greedy: false })
  if (name === 'creative')
    Object.assign(s, { temperature: 1.3, topK: 0, topP: 1, minP: 0.05, greedy: false })
  if (name === 'model' && wb.suggestedSampling.value) {
    const g = wb.suggestedSampling.value
    Object.assign(s, {
      temperature: g.temperature ?? 1,
      topK: g.topK ?? 0,
      topP: g.topP ?? 1,
      minP: 0,
      greedy: false,
    })
  }
}
function randomSeed() {
  const a = new Uint32Array(1)
  crypto.getRandomValues(a)
  wb.resetSeed(a[0] as number)
}
function num(v: string, fallback: number) {
  const n = Number(v)
  return Number.isFinite(n) ? n : fallback
}
</script>

<template>
  <section class="panel card" aria-labelledby="sampling-title" data-testid="sampling-controls">
    <div class="row">
      <h2 id="sampling-title">Sampling</h2>
      <InfoTip label="Sampling pipeline">
        <p>
          Sampling settings change only how the next token is <em>chosen</em> from the model’s
          output. They do not modify the logits or the trained weights.
        </p>
        <p>
          Order: <strong>temperature → top-k → top-p → min-p → renormalize</strong> (same order as
          Hugging Face transformers).
        </p>
      </InfoTip>
      <span class="spacer" />
      <button
        type="button"
        class="btn btn-sm"
        data-testid="reset-sampling"
        @click="Object.assign(s, { ...DEFAULT_SAMPLING, seed: s.seed })"
      >
        Reset
      </button>
    </div>
    <div class="row small presets">
      <span class="muted">Presets:</span>
      <button type="button" class="btn btn-sm" @click="preset('neutral')">Neutral</button>
      <button type="button" class="btn btn-sm" @click="preset('precise')">Focused</button>
      <button type="button" class="btn btn-sm" @click="preset('creative')">Creative</button>
      <button
        v-if="wb.suggestedSampling.value"
        type="button"
        class="btn btn-sm"
        title="From the model's generation_config.json"
        @click="preset('model')"
      >
        Model default
      </button>
    </div>

    <div class="control">
      <div class="row">
        <label for="temp">Temperature</label>
        <InfoTip label="Temperature">
          <p>Divides every logit by T before softmax: <code>softmax(z / T)</code>.</p>
          <p>
            T &lt; 1 sharpens the distribution (more predictable), T &gt; 1 flattens it (more
            random). T = 1 leaves it unchanged.
          </p>
          <p>
            T = 0 would divide by zero, so it is handled explicitly as
            <strong>greedy decoding</strong> (always pick the highest logit).
          </p>
        </InfoTip>
        <span class="spacer" />
        <input
          id="temp-num"
          class="num"
          type="number"
          min="0"
          max="5"
          step="0.01"
          :value="s.temperature"
          :disabled="s.greedy"
          aria-label="Temperature value"
          data-testid="temperature-input"
          @change="s.temperature = num(($event.target as HTMLInputElement).value, s.temperature)"
        />
      </div>
      <input
        id="temp"
        v-model.number="s.temperature"
        type="range"
        min="0"
        max="2"
        step="0.01"
        :disabled="s.greedy"
        data-testid="temperature"
      />
    </div>

    <div class="control">
      <div class="row">
        <label for="topk">Top-k</label>
        <InfoTip label="Top-k">
          <p>Keep only the k most probable tokens, then renormalize. 0 disables it.</p>
        </InfoTip>
        <span class="spacer" />
        <input
          class="num"
          type="number"
          min="0"
          step="1"
          :value="s.topK"
          aria-label="Top-k value"
          data-testid="topk-input"
          @change="
            s.topK = Math.max(0, Math.round(num(($event.target as HTMLInputElement).value, s.topK)))
          "
        />
      </div>
      <input
        id="topk"
        v-model.number="s.topK"
        type="range"
        min="0"
        max="200"
        step="1"
        data-testid="topk"
      />
      <span class="small faint">{{ s.topK === 0 ? 'off' : `keep ${s.topK}` }}</span>
    </div>

    <div class="control">
      <div class="row">
        <label for="topp">Top-p (nucleus)</label>
        <InfoTip label="Top-p (nucleus sampling)">
          <p>
            Sort tokens by probability and keep the smallest set whose cumulative probability
            reaches p. The token that crosses the threshold is kept, so at least one token always
            remains. 1 disables it.
          </p>
        </InfoTip>
        <span class="spacer" />
        <input
          class="num"
          type="number"
          min="0.01"
          max="1"
          step="0.01"
          :value="s.topP"
          aria-label="Top-p value"
          data-testid="topp-input"
          @change="s.topP = num(($event.target as HTMLInputElement).value, s.topP)"
        />
      </div>
      <input
        id="topp"
        v-model.number="s.topP"
        type="range"
        min="0.01"
        max="1"
        step="0.01"
        data-testid="topp"
      />
    </div>

    <div class="control">
      <div class="row">
        <label for="minp">Min-p</label>
        <InfoTip label="Min-p">
          <p>
            Keep tokens whose probability is at least <code>min_p × p_max</code>, where p_max is the
            most likely token’s probability (after the previous steps). It adapts to how confident
            the model is. 0 disables it.
          </p>
        </InfoTip>
        <span class="spacer" />
        <input
          class="num"
          type="number"
          min="0"
          max="1"
          step="0.01"
          :value="s.minP"
          aria-label="Min-p value"
          data-testid="minp-input"
          @change="s.minP = num(($event.target as HTMLInputElement).value, s.minP)"
        />
      </div>
      <input
        id="minp"
        v-model.number="s.minP"
        type="range"
        min="0"
        max="1"
        step="0.01"
        data-testid="minp"
      />
    </div>

    <div class="row">
      <label class="row nowrap-label">
        <input v-model="s.greedy" type="checkbox" data-testid="greedy" />
        Greedy decoding
      </label>
      <InfoTip label="Greedy decoding">
        <p>
          Skips randomness entirely: the adjusted distribution puts 100% on the token with the
          highest logit (ties → lowest ID). Deterministic, but often repetitive.
        </p>
      </InfoTip>
      <span class="small faint">(always choose the most probable token)</span>
    </div>

    <div class="control">
      <div class="row">
        <label for="seed">Random seed</label>
        <InfoTip label="Random seed">
          <p>
            Sampling draws a uniform random number u ∈ [0, 1) from a seeded SFC32 generator and
            picks the token where the cumulative adjusted probability first exceeds u.
          </p>
          <p>
            The same logits, settings and generator state reproduce the same choice. Different
            backends/precisions produce slightly different logits, so results can still differ
            across devices.
          </p>
        </InfoTip>
        <span class="spacer" />
        <input
          id="seed"
          class="num seed"
          type="number"
          min="0"
          max="4294967295"
          step="1"
          :value="s.seed"
          data-testid="seed"
          @change="
            wb.resetSeed(
              Math.max(
                0,
                Math.min(
                  4294967295,
                  Math.round(num(($event.target as HTMLInputElement).value, s.seed)),
                ),
              ),
            )
          "
        />
      </div>
      <div class="row small">
        <button type="button" class="btn btn-sm" @click="randomSeed">Random seed</button>
        <button type="button" class="btn btn-sm" data-testid="reset-prng" @click="wb.resetSeed()">
          Restart generator from seed
        </button>
      </div>
      <p class="small faint">
        Generator state: <code data-testid="prng-state">{{ prngHex }}</code>
      </p>
    </div>

    <ul v-if="wb.samplingErrors.value.length" class="alert alert-error small">
      <li v-for="e in wb.samplingErrors.value" :key="e">{{ e }}</li>
    </ul>
    <p v-if="greedy" class="small badge badge-accent">
      Greedy mode: adjusted distribution is one-hot.
    </p>
    <p v-if="stages" class="small muted pipeline" data-testid="pipeline-summary">
      Eligible tokens: {{ stages.afterTemperature.toLocaleString() }} → top-k
      {{ stages.afterTopK.toLocaleString() }} → top-p {{ stages.afterTopP.toLocaleString() }} →
      min-p {{ stages.afterMinP.toLocaleString() }}
      <span v-if="wb.samplingMs.value !== null" class="faint"
        >({{ formatMs(wb.samplingMs.value) }}, no network re-run)</span
      >
    </p>
  </section>
</template>

<style scoped>
.panel {
  padding: 0.8rem;
  display: flex;
  flex-direction: column;
  gap: 0.55rem;
}
.control {
  display: flex;
  flex-direction: column;
  gap: 0.1rem;
}
.control label {
  font-weight: 600;
}
.num {
  width: 6rem;
  text-align: right;
}
.num.seed {
  width: 8rem;
}
.pipeline {
  margin: 0;
}
</style>
