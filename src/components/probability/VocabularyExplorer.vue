<script setup lang="ts">
/**
 * Full vocabulary browser with search, sort, filter and pagination. Only the current
 * page is rendered; the underlying probabilities always cover the whole vocabulary.
 */
import { computed, reactive, ref, watch } from 'vue'
import { useWorkbench } from '@/composables/useWorkbench'
import { DEFAULT_VOCAB_QUERY, paginate, queryVocab, type VocabQuery } from '@/tokenization/vocab'
import CandidateTable from './CandidateTable.vue'

const props = defineProps<{ exact: boolean }>()
const emit = defineEmits<{ pick: [number] }>()
const wb = useWorkbench()
const q = reactive<VocabQuery>({ ...DEFAULT_VOCAB_QUERY })
const page = ref(0)
const pageSize = ref(20)
const jump = ref('')

const size = computed(() => {
  const t = wb.tokenizer.value?.raw.length ?? 0
  const l = wb.prepared.value?.logits.length ?? wb.modelInfo.value?.logitsSize ?? 0
  return Math.max(t, l)
})
const specialSet = computed(() => new Set(wb.tokenizer.value?.specialIds ?? []))
const source = computed(() => {
  const tok = wb.tokenizer.value
  if (!tok) return null
  return {
    size: size.value,
    raw: tok.raw,
    decoded: tok.decoded,
    isSpecial: (id: number) => specialSet.value.has(id),
  }
})
const dist = computed(() => {
  const p = wb.prepared.value
  if (!p || !wb.resultIsCurrent.value) return null
  return {
    logits: p.logits,
    original: p.original.probs,
    adjusted: wb.adjusted.value?.probs ?? new Float64Array(p.logits.length),
    logitOrder: p.fullOrder,
  }
})
const result = computed(() =>
  source.value
    ? queryVocab(source.value, dist.value, { ...q })
    : { ids: new Uint32Array(0), total: 0 },
)
const current = computed(() => paginate(result.value.ids, page.value, pageSize.value))
watch(
  () => [q.text, q.field, q.filter, q.sort, q.direction, q.caseSensitive, pageSize.value],
  () => (page.value = 0),
)
const rankMap = computed(() => {
  const m = new Map<number, number>()
  const order = wb.prepared.value?.fullOrder
  if (order) for (let i = 0; i < order.length; i++) m.set(order[i] as number, i + 1)
  return m
})
const empty64 = new Float64Array(0)
const emptyLogits = computed(() => new Float32Array(size.value).fill(NaN))
function goPage() {
  const n = Number(jump.value)
  if (Number.isInteger(n) && n >= 1) page.value = n - 1
}
</script>

<template>
  <section class="stack" aria-label="Vocabulary explorer" data-testid="vocab-explorer">
    <p v-if="!source" class="small faint">Load a model or tokenizer to browse its vocabulary.</p>
    <template v-else>
      <p class="small muted">
        {{ size.toLocaleString() }} IDs ({{
          wb.tokenizer.value!.raw.length.toLocaleString()
        }}
        tokenizer entries, {{ (wb.modelInfo.value?.logitsSize ?? 0).toLocaleString() }} logits).
        Probabilities are computed over the complete vocabulary; only the visible page is rendered.
        <template v-if="!dist">
          No distribution for the current context yet — run inference to see
          probabilities.</template
        >
      </p>
      <div class="row small controls">
        <label>
          <span class="sr-only">Search vocabulary</span>
          <input
            v-model="q.text"
            type="search"
            placeholder="Search text, raw token or ID (\n, \t, \s escapes)"
            aria-label="Search vocabulary"
            data-testid="vocab-search"
          />
        </label>
        <label>
          in
          <select v-model="q.field" data-testid="vocab-field">
            <option value="any">text, raw &amp; ID</option>
            <option value="decoded">decoded text</option>
            <option value="raw">raw token</option>
            <option value="id">token ID</option>
          </select>
        </label>
        <label class="row"
          ><input v-model="q.caseSensitive" type="checkbox" /> case-sensitive</label
        >
        <label>
          Filter
          <select v-model="q.filter" data-testid="vocab-filter">
            <option value="all">all tokens</option>
            <option value="special">special only</option>
            <option value="regular">non-special</option>
            <option value="eligible" :disabled="!dist">eligible for sampling</option>
            <option value="filtered-out" :disabled="!dist">filtered out</option>
          </select>
        </label>
        <label>
          Sort
          <select v-model="q.sort" data-testid="vocab-sort">
            <option value="probability">probability</option>
            <option value="adjusted">adjusted probability</option>
            <option value="logit">logit</option>
            <option value="id">token ID</option>
            <option value="alphabetical">alphabetical</option>
          </select>
        </label>
        <label>
          <select v-model="q.direction" aria-label="Sort direction" data-testid="vocab-direction">
            <option value="desc">descending</option>
            <option value="asc">ascending</option>
          </select>
        </label>
      </div>
      <CandidateTable
        testid="vocab-table"
        :ids="current.ids"
        :rank-of="(id) => rankMap.get(id) ?? -1"
        :original="dist?.original ?? empty64"
        :log-probs="wb.prepared.value && dist ? wb.prepared.value.original.logProbs : empty64"
        :adjusted="dist ? dist.adjusted : null"
        :logits="dist?.logits ?? emptyLogits"
        :decoded="wb.tokenizer.value!.decoded"
        :raw="wb.tokenizer.value!.raw"
        :special="specialSet"
        :exact="props.exact"
        @pick="emit('pick', $event)"
      />
      <nav class="row small" aria-label="Vocabulary pages">
        <button type="button" class="btn btn-sm" :disabled="current.page === 0" @click="page = 0">
          « First
        </button>
        <button
          type="button"
          class="btn btn-sm"
          :disabled="current.page === 0"
          data-testid="vocab-prev"
          @click="page = current.page - 1"
        >
          ‹ Prev
        </button>
        <span data-testid="vocab-page"
          >Page {{ (current.page + 1).toLocaleString() }} of
          {{ current.pageCount.toLocaleString() }}</span
        >
        <button
          type="button"
          class="btn btn-sm"
          :disabled="current.page >= current.pageCount - 1"
          data-testid="vocab-next"
          @click="page = current.page + 1"
        >
          Next ›
        </button>
        <button
          type="button"
          class="btn btn-sm"
          :disabled="current.page >= current.pageCount - 1"
          @click="page = current.pageCount - 1"
        >
          Last »
        </button>
        <span class="faint"
          >{{ result.total.toLocaleString() }} match(es) · showing
          {{ current.start + (current.ids.length ? 1 : 0) }}–{{ current.end }}</span
        >
        <span class="spacer" />
        <form class="row" @submit.prevent="goPage">
          <label
            >Go to page <input v-model="jump" type="text" inputmode="numeric" size="5"
          /></label>
        </form>
        <label>
          Rows
          <select v-model.number="pageSize">
            <option :value="20">20</option>
            <option :value="50">50</option>
            <option :value="100">100</option>
            <option :value="250">250</option>
          </select>
        </label>
      </nav>
    </template>
  </section>
</template>

<style scoped>
.controls input[type='search'] {
  width: 18rem;
  max-width: 100%;
}
</style>
