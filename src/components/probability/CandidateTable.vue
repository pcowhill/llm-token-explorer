<script setup lang="ts">
/** Sortable table of candidate tokens with logits, original and adjusted probabilities. */
import { computed, ref } from 'vue'
import TokenText from '@/components/common/TokenText.vue'
import { exactNumber, formatLogProb, formatLogit, formatProb } from '@/utils/format'

type Key = 'rank' | 'id' | 'token' | 'logit' | 'logprob' | 'original' | 'adjusted'

const props = defineProps<{
  ids: number[]
  /** 1-based rank under the original distribution, by id. */
  rankOf: (id: number) => number
  original: Float64Array
  logProbs: Float64Array
  adjusted: Float64Array | null
  logits: Float32Array
  decoded: readonly string[]
  raw: readonly string[]
  special: Set<number>
  exact: boolean
  testid?: string
}>()
const emit = defineEmits<{ pick: [number] }>()
const sortKey = ref<Key>('rank')
const asc = ref(true)

const rows = computed(() => {
  const r = props.ids.map((id) => ({
    id,
    rank: props.rankOf(id),
    token: props.decoded[id] ?? '',
    raw: props.raw[id] ?? '',
    logit: props.logits[id] ?? NaN,
    logprob: props.logProbs[id] ?? -Infinity,
    original: props.original[id] ?? 0,
    adjusted: props.adjusted?.[id] ?? 0,
  }))
  const k = sortKey.value
  const dir = asc.value ? 1 : -1
  r.sort((a, b) => {
    const va = a[k]
    const vb = b[k]
    if (typeof va === 'string' && typeof vb === 'string')
      return va < vb ? -dir : va > vb ? dir : a.id - b.id
    const d = (va as number) - (vb as number)
    return d === 0 || Number.isNaN(d) ? a.id - b.id : d * dir
  })
  return r
})

function sortBy(k: Key) {
  if (sortKey.value === k) asc.value = !asc.value
  else {
    sortKey.value = k
    asc.value = k === 'rank' || k === 'id' || k === 'token'
  }
}
const ariaSort = (k: Key) =>
  sortKey.value === k ? (asc.value ? 'ascending' : 'descending') : 'none'
const fmtP = (p: number) => (props.exact ? exactNumber(p) : formatProb(p))
const fmtL = (x: number) => (props.exact ? exactNumber(x) : formatLogit(x))
const cols: [Key, string, boolean][] = [
  ['rank', 'Rank', true],
  ['id', 'ID', true],
  ['token', 'Token', false],
  ['logit', 'Raw logit', true],
  ['logprob', 'log p', true],
  ['original', 'Original p', true],
  ['adjusted', 'Adjusted p', true],
]
</script>

<template>
  <div class="table-wrap">
    <table class="data" :data-testid="props.testid ?? 'candidate-table'">
      <caption class="sr-only">
        Next-token candidates with raw logits, original and adjusted probabilities
      </caption>
      <thead>
        <tr>
          <th
            v-for="[k, label, num] in cols"
            :key="k"
            :class="{ num }"
            :aria-sort="ariaSort(k)"
            scope="col"
          >
            <button type="button" class="sort" @click="sortBy(k)">
              {{ label
              }}<span aria-hidden="true">{{ sortKey === k ? (asc ? ' ▲' : ' ▼') : '' }}</span>
            </button>
          </th>
          <th scope="col">Eligible</th>
          <th scope="col"><span class="sr-only">Actions</span></th>
        </tr>
      </thead>
      <tbody>
        <tr
          v-for="r in rows"
          :key="r.id"
          :class="{ ineligible: props.adjusted && r.adjusted === 0 }"
          :data-token-id="r.id"
        >
          <td class="num">{{ r.rank > 0 ? r.rank : '—' }}</td>
          <td class="num">{{ r.id }}</td>
          <td class="tok" :title="r.raw ? `raw: ${r.raw}` : 'no tokenizer entry'">
            <TokenText :text="r.token" :special="props.special.has(r.id)" />
          </td>
          <td class="num" :title="exactNumber(r.logit)">{{ fmtL(r.logit) }}</td>
          <td class="num" :title="exactNumber(r.logprob)">
            {{ props.exact ? exactNumber(r.logprob) : formatLogProb(r.logprob) }}
          </td>
          <td class="num" :title="exactNumber(r.original)">{{ fmtP(r.original) }}</td>
          <td class="num" :title="exactNumber(r.adjusted)">
            {{ props.adjusted ? fmtP(r.adjusted) : '—' }}
          </td>
          <td>
            <span v-if="!props.adjusted" class="faint">—</span>
            <span v-else-if="r.adjusted > 0" class="badge badge-success">✓ kept</span>
            <span v-else class="badge">✗ filtered</span>
          </td>
          <td>
            <button
              type="button"
              class="btn btn-sm"
              :aria-label="`Append token ${r.id}`"
              :data-testid="`pick-${r.id}`"
              @click="emit('pick', r.id)"
            >
              Append
            </button>
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<style scoped>
.table-wrap {
  overflow: auto;
  max-height: 32rem;
}
.sort {
  background: none;
  border: none;
  padding: 0;
  font-weight: 600;
  color: inherit;
}
.tok {
  max-width: 14rem;
  overflow: hidden;
  text-overflow: ellipsis;
}
</style>
