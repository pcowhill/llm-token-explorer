<script setup lang="ts">
import { computed, ref } from 'vue'
import { useWorkbench } from '@/composables/useWorkbench'
import TopChart, { type ChartMode } from './TopChart.vue'
import CandidateTable from './CandidateTable.vue'
import VocabularyExplorer from './VocabularyExplorer.vue'
import InfoTip from '@/components/common/InfoTip.vue'
import { stableSum } from '@/sampling/softmax'
import { exactNumber, formatMs, formatProb } from '@/utils/format'

const wb = useWorkbench()
const tab = ref<'top' | 'vocab'>('top')
const mode = ref<ChartMode>('compare')
const topN = ref(20)
const exact = ref(false)

const p = computed(() => (wb.resultIsCurrent.value ? wb.prepared.value : null))
const specialSet = computed(() => new Set(wb.tokenizer.value?.specialIds ?? []))
const topIds = computed(() =>
  p.value ? Array.from(p.value.fullOrder.subarray(0, topN.value)) : [],
)
const rank = computed(() => {
  const m = new Map<number, number>()
  topIds.value.forEach((id, i) => m.set(id, i + 1))
  return m
})

function entropyBits(probs: Float64Array): number {
  let h = 0
  for (let i = 0; i < probs.length; i++) {
    const x = probs[i] as number
    if (x > 0) h -= x * Math.log2(x)
  }
  return h
}
const stats = computed(() => {
  const pr = p.value
  if (!pr) return null
  const adj = wb.adjusted.value
  const top = pr.fullOrder[0] as number
  let topMass = 0
  for (const id of topIds.value) topMass += pr.original.probs[id] ?? 0
  return {
    vocab: pr.logits.length,
    sumOriginal: stableSum(pr.original.probs),
    sumAdjusted: adj ? stableSum(adj.probs) : null,
    hOriginal: entropyBits(pr.original.probs),
    hAdjusted: adj ? entropyBits(adj.probs) : null,
    kept: adj?.keptCount ?? null,
    top,
    topP: pr.original.probs[top] ?? 0,
    topMass,
    nan: pr.original.nanCount,
    ineligible: pr.logits.length - pr.eligibleCount,
  }
})
const statusText = computed(() => {
  switch (wb.inferenceStatus.value) {
    case 'pending':
      return 'Waiting for edits to settle…'
    case 'running':
      return 'Running the neural network…'
    case 'error':
      return 'Inference failed'
    case 'ready':
      return wb.result.value?.fromResultCache
        ? 'Ready (reused cached logits for this exact context)'
        : 'Ready'
    default:
      return wb.canInfer.value
        ? wb.tokens.value.length
          ? 'Not calculated yet'
          : 'Context is empty'
        : 'Load a model to run inference'
  }
})
function pick(id: number) {
  wb.pickToken(id)
}
</script>

<template>
  <section class="panel card" aria-labelledby="prob-title" data-testid="probability-workspace">
    <div class="row">
      <h2 id="prob-title">Next-token distribution</h2>
      <InfoTip label="Logits vs. probabilities">
        <p>
          The network outputs one <strong>logit</strong> (an unnormalized score) per vocabulary
          entry for the next position.
        </p>
        <p>
          <strong>Softmax</strong> turns logits into probabilities:
          <code>P(i) = exp(zᵢ) / Σⱼ exp(zⱼ)</code>. This is the
          <strong>original</strong> distribution.
        </p>
        <p>
          Sampling settings then produce the <strong>adjusted</strong> distribution that tokens are
          actually drawn from. They never change the logits or the model.
        </p>
      </InfoTip>
      <span class="spacer" />
      <span
        class="badge"
        :class="{
          'badge-warning':
            wb.inferenceStatus.value === 'running' || wb.inferenceStatus.value === 'pending',
          'badge-success': wb.inferenceStatus.value === 'ready',
          'badge-danger': wb.inferenceStatus.value === 'error',
        }"
        role="status"
        data-testid="inference-status"
      >
        {{ statusText }}
      </span>
      <label class="row small"
        ><input v-model="wb.generation.autoInfer" type="checkbox" data-testid="auto-infer" />
        auto-calculate</label
      >
      <button
        type="button"
        class="btn btn-primary btn-sm"
        :disabled="
          !wb.canInfer.value || !wb.tokens.value.length || wb.inferenceStatus.value === 'running'
        "
        data-testid="calculate"
        @click="wb.runInference(true)"
      >
        Calculate
      </button>
    </div>
    <div
      v-if="wb.inferenceError.value && wb.inferenceStatus.value === 'error'"
      class="alert alert-error small"
      role="alert"
      data-testid="inference-error"
    >
      {{ wb.inferenceError.value.message
      }}<span v-if="wb.inferenceError.value.hint"> — {{ wb.inferenceError.value.hint }}</span>
    </div>
    <p
      v-if="!wb.resultIsCurrent.value && wb.result.value && wb.inferenceStatus.value !== 'running'"
      class="small faint"
    >
      The shown distribution is out of date for the edited context.
    </p>

    <dl v-if="stats" class="stats small" data-testid="dist-stats">
      <div>
        <dt>Logits vector</dt>
        <dd data-testid="logits-size">{{ stats.vocab.toLocaleString() }} values</dd>
      </div>
      <div>
        <dt>Most likely</dt>
        <dd>ID {{ stats.top }} · {{ formatProb(stats.topP) }}</dd>
      </div>
      <div>
        <dt>Top-{{ topN }} mass</dt>
        <dd>{{ formatProb(stats.topMass) }}</dd>
      </div>
      <div>
        <dt>Σ original p</dt>
        <dd :title="exactNumber(stats.sumOriginal)" data-testid="sum-original">
          {{ stats.sumOriginal.toFixed(9) }}
        </dd>
      </div>
      <div v-if="stats.sumAdjusted !== null">
        <dt>Σ adjusted p</dt>
        <dd :title="exactNumber(stats.sumAdjusted)">{{ stats.sumAdjusted.toFixed(9) }}</dd>
      </div>
      <div>
        <dt>Entropy</dt>
        <dd>{{ stats.hOriginal.toFixed(3) }} → {{ stats.hAdjusted?.toFixed(3) ?? '—' }} bits</dd>
      </div>
      <div>
        <dt>Eligible after sampling</dt>
        <dd data-testid="kept-count">{{ stats.kept?.toLocaleString() ?? '—' }} tokens</dd>
      </div>
      <div v-if="wb.result.value">
        <dt>Forward pass</dt>
        <dd>
          {{ formatMs(wb.result.value.inferenceMs)
          }}{{ wb.result.value.usedKvCache ? ' (KV cache)' : '' }}
        </dd>
      </div>
      <div v-if="stats.ineligible">
        <dt>Never sampled</dt>
        <dd :title="'IDs without a tokenizer entry or with NaN logits'">
          {{ stats.ineligible.toLocaleString() }} IDs
        </dd>
      </div>
    </dl>

    <div class="tabs" role="tablist" aria-label="Distribution views">
      <button
        id="prob-tab-top"
        type="button"
        role="tab"
        class="tab"
        :aria-selected="tab === 'top'"
        aria-controls="prob-panel-top"
        @click="tab = 'top'"
      >
        Top candidates
      </button>
      <button
        id="prob-tab-vocab"
        type="button"
        role="tab"
        class="tab"
        :aria-selected="tab === 'vocab'"
        aria-controls="prob-panel-vocab"
        data-testid="tab-vocab"
        @click="tab = 'vocab'"
      >
        Full vocabulary
      </button>
    </div>

    <div
      v-show="tab === 'top'"
      id="prob-panel-top"
      role="tabpanel"
      aria-labelledby="prob-tab-top"
      class="stack"
    >
      <div class="row small">
        <fieldset class="row seg">
          <legend class="sr-only">Displayed distribution</legend>
          <label
            v-for="m in ['compare', 'original', 'adjusted', 'logits'] as const"
            :key="m"
            class="row"
          >
            <input
              v-model="mode"
              type="radio"
              name="chart-mode"
              :value="m"
              :data-testid="`mode-${m}`"
            />
            {{
              {
                compare: 'Compare',
                original: 'Original p',
                adjusted: 'Adjusted p',
                logits: 'Raw logits',
              }[m]
            }}
          </label>
        </fieldset>
        <span class="spacer" />
        <label
          >Show top
          <select v-model.number="topN">
            <option :value="10">10</option>
            <option :value="20">20</option>
            <option :value="50">50</option>
            <option :value="100">100</option>
          </select></label
        >
        <label class="row"
          ><input v-model="exact" type="checkbox" data-testid="exact-values" /> exact values</label
        >
      </div>
      <template v-if="p && wb.tokenizer.value">
        <div class="split">
          <TopChart
            :ids="topIds.slice(0, Math.min(topN, 30))"
            :original="p.original.probs"
            :adjusted="wb.adjusted.value?.probs ?? null"
            :logits="p.logits"
            :decoded="wb.tokenizer.value.decoded"
            :special="specialSet"
            :mode="mode"
            @pick="pick"
          />
          <CandidateTable
            :ids="topIds"
            :rank-of="(id) => rank.get(id) ?? -1"
            :original="p.original.probs"
            :log-probs="p.original.logProbs"
            :adjusted="wb.adjusted.value?.probs ?? null"
            :logits="p.logits"
            :decoded="wb.tokenizer.value.decoded"
            :raw="wb.tokenizer.value.raw"
            :special="specialSet"
            :exact="exact"
            @pick="pick"
          />
        </div>
      </template>
      <p v-else class="small faint empty">
        {{
          wb.canInfer.value
            ? 'No distribution for this context yet.'
            : 'Load a model to compute real next-token logits. Nothing here is simulated.'
        }}
      </p>
    </div>
    <div
      v-show="tab === 'vocab'"
      id="prob-panel-vocab"
      role="tabpanel"
      aria-labelledby="prob-tab-vocab"
    >
      <VocabularyExplorer :exact="exact" @pick="pick" />
    </div>
  </section>
</template>

<style scoped>
.panel {
  padding: 0.8rem;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}
.stats {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(9.5rem, 1fr));
  gap: 0.3rem 0.8rem;
  margin: 0;
  padding: 0.5rem;
  background: var(--surface-2);
  border-radius: var(--radius-sm);
}
.stats dt {
  color: var(--text-faint);
  font-size: 0.72rem;
}
.stats dd {
  margin: 0;
  font-family: var(--mono);
  font-size: 0.82rem;
}
.seg {
  border: none;
  padding: 0;
  margin: 0;
}
.panel {
  container-type: inline-size;
}
.split {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 0.75rem;
}
@container (min-width: 1000px) {
  .split {
    grid-template-columns: minmax(0, 1fr) minmax(0, 1.3fr);
  }
}
.empty {
  padding: 1rem;
  text-align: center;
}
</style>
