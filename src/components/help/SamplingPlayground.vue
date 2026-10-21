<script setup lang="ts">
/**
 * Educational playground: the REAL sampling pipeline applied to a small, hand-editable
 * toy logits vector. Clearly labelled as not model output.
 */
import { computed, reactive } from 'vue'
import { adjustDistribution, prepareLogits } from '@/sampling/pipeline'
import { formatProb } from '@/utils/format'
import { DEFAULT_SAMPLING, type SamplingSettings } from '@/types'

const toy = reactive([
  { label: ' Paris', logit: 6.0 },
  { label: ' the', logit: 4.2 },
  { label: ' a', logit: 3.6 },
  { label: ' located', logit: 3.1 },
  { label: ' known', logit: 2.0 },
  { label: ' Lyon', logit: 0.5 },
  { label: ' banana', logit: -2.0 },
])
const s = reactive<SamplingSettings>({ ...DEFAULT_SAMPLING })
const prepared = computed(() => prepareLogits(Float32Array.from(toy.map((t) => t.logit))))
const adjusted = computed(() => adjustDistribution(prepared.value, { ...s }))
const maxP = computed(() =>
  Math.max(...Array.from(prepared.value.original.probs), ...Array.from(adjusted.value.probs)),
)
</script>

<template>
  <section class="stack" aria-labelledby="pg-title">
    <h3 id="pg-title">
      Sampling playground
      <span class="badge badge-warning">Toy example — hand-written logits, not model output</span>
    </h3>
    <p class="small muted">
      Edit the logits and move the sliders to see how the same pipeline used by the workbench
      reshapes a distribution.
    </p>
    <div class="row small controls">
      <label
        >T {{ s.temperature.toFixed(2)
        }}<input v-model.number="s.temperature" type="range" min="0" max="3" step="0.05"
      /></label>
      <label
        >top-k {{ s.topK || 'off'
        }}<input v-model.number="s.topK" type="range" min="0" max="7" step="1"
      /></label>
      <label
        >top-p {{ s.topP.toFixed(2)
        }}<input v-model.number="s.topP" type="range" min="0.05" max="1" step="0.05"
      /></label>
      <label
        >min-p {{ s.minP.toFixed(2)
        }}<input v-model.number="s.minP" type="range" min="0" max="1" step="0.05"
      /></label>
    </div>
    <table class="data">
      <thead>
        <tr>
          <th>Token</th>
          <th class="num">Logit</th>
          <th>Original</th>
          <th>Adjusted</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(t, i) in toy" :key="i">
          <td>
            <code>{{ t.label.replace(/ /g, '␣') }}</code>
          </td>
          <td class="num">
            <input
              v-model.number="t.logit"
              type="number"
              step="0.1"
              class="logit"
              :aria-label="`Logit for ${t.label}`"
            />
          </td>
          <td>
            <span
              class="bar o"
              :style="{ width: (100 * (prepared.original.probs[i] ?? 0)) / maxP + '%' }"
            />
            <span class="small">{{ formatProb(prepared.original.probs[i]) }}</span>
          </td>
          <td>
            <span class="bar a" :style="{ width: (100 * (adjusted.probs[i] ?? 0)) / maxP + '%' }" />
            <span class="small">{{
              (adjusted.probs[i] ?? 0) > 0 ? formatProb(adjusted.probs[i]) : '✗ filtered'
            }}</span>
          </td>
        </tr>
      </tbody>
    </table>
  </section>
</template>

<style scoped>
.controls label {
  display: flex;
  flex-direction: column;
  min-width: 8rem;
}
.logit {
  width: 5rem;
  text-align: right;
}
.bar {
  display: inline-block;
  height: 0.7rem;
  max-width: 8rem;
  margin-right: 0.4rem;
  vertical-align: middle;
  border-radius: 2px;
}
.bar.o {
  background: var(--original);
}
.bar.a {
  background: var(--adjusted);
}
td:nth-child(3),
td:nth-child(4) {
  width: 12rem;
}
</style>
