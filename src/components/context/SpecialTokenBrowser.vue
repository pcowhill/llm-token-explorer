<script setup lang="ts">
/**
 * Searchable list of the loaded tokenizer's REAL special tokens (from tokenizer metadata).
 * Inserting a token inserts its exact ID — never its spelling as text.
 */
import { computed, ref } from 'vue'
import { useWorkbench } from '@/composables/useWorkbench'
import { escapeLiteral } from '@/tokenization/display'

const props = defineProps<{ selected: number | null }>()
const wb = useWorkbench()
const q = ref('')
const all = computed(() => wb.tokenizer.value?.specialTokens ?? [])
const filtered = computed(() => {
  const s = q.value.trim().toLowerCase()
  if (!s) return all.value
  return all.value.filter(
    (t) =>
      t.content.toLowerCase().includes(s) ||
      String(t.id) === s ||
      t.roles.some((r) => r.includes(s)) ||
      t.purpose.toLowerCase().includes(s),
  )
})
const common = computed(() =>
  all.value
    .filter((t) => t.roles.some((r) => r === 'bos' || r === 'eos' || r === 'stop' || r === 'chat'))
    .slice(0, 8),
)
function insert(id: number, atSelection: boolean) {
  const n = wb.tokens.value.length
  const pos = atSelection && props.selected !== null ? props.selected + 1 : n
  try {
    wb.insertToken(pos, id)
  } catch (e) {
    wb.notify('error', (e as Error).message)
  }
}
</script>

<template>
  <section class="stack" aria-labelledby="special-title" data-testid="special-browser">
    <div class="row">
      <h3 id="special-title">Special tokens</h3>
      <span class="badge">{{ all.length }}</span>
    </div>
    <template v-if="!wb.tokenizer.value">
      <p class="small faint">Load a model or tokenizer to browse its special tokens.</p>
    </template>
    <template v-else-if="all.length === 0">
      <p class="small">This tokenizer defines no special tokens.</p>
    </template>
    <template v-else>
      <div v-if="common.length" class="row small" aria-label="Common special tokens">
        <span class="muted">Quick insert:</span>
        <button
          v-for="t in common"
          :key="t.id"
          type="button"
          class="btn btn-sm special-btn"
          :data-testid="`quick-special-${t.id}`"
          :title="`${t.purpose} (ID ${t.id})`"
          @click="insert(t.id, false)"
        >
          {{ t.content }} <span class="faint">{{ t.id }}</span>
        </button>
      </div>
      <label class="small">
        <span class="sr-only">Search special tokens</span>
        <input
          v-model="q"
          type="search"
          placeholder="Search name, ID, role…"
          aria-label="Search special tokens"
          data-testid="special-search"
        />
      </label>
      <div class="scroll">
        <table class="data">
          <thead>
            <tr>
              <th class="num">ID</th>
              <th>Token</th>
              <th>Role</th>
              <th>Purpose</th>
              <th><span class="sr-only">Actions</span></th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="t in filtered" :key="t.id" :data-testid="`special-row-${t.id}`">
              <td class="num">{{ t.id }}</td>
              <td>
                <code class="special">{{ escapeLiteral(t.content) }}</code>
              </td>
              <td>
                <span
                  v-for="r in t.roles"
                  :key="r"
                  class="badge"
                  :class="{ 'badge-warning': r === 'stop' }"
                  >{{ r }}</span
                >
              </td>
              <td class="small">{{ t.purpose }}</td>
              <td class="actions">
                <button
                  type="button"
                  class="btn btn-sm"
                  :aria-label="`Append ${t.content}`"
                  @click="insert(t.id, false)"
                >
                  Append
                </button>
                <button
                  v-if="props.selected !== null"
                  type="button"
                  class="btn btn-sm"
                  :aria-label="`Insert ${t.content} after selection`"
                  @click="insert(t.id, true)"
                >
                  After sel.
                </button>
              </td>
            </tr>
          </tbody>
        </table>
        <p v-if="!filtered.length" class="small faint">No matches.</p>
      </div>
    </template>
  </section>
</template>

<style scoped>
.scroll {
  max-height: 16rem;
  overflow: auto;
}
.special {
  color: var(--special);
}
.special-btn {
  font-family: var(--mono);
}
.actions {
  white-space: nowrap;
}
</style>
