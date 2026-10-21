<script setup lang="ts">
/** Insert an exact token ID at a chosen position. */
import { computed, ref, watch } from 'vue'
import { useWorkbench } from '@/composables/useWorkbench'
import { parseTokenId } from '@/tokenization/sequence'
import { escapeLiteral } from '@/tokenization/display'

const props = defineProps<{ selected: number | null; insertAt: number | null }>()
const wb = useWorkbench()
const idText = ref('')
const where = ref<'end' | 'before' | 'after'>('end')
watch(
  () => props.insertAt,
  (v) => {
    if (v !== null) where.value = 'before'
  },
)
const parsed = computed(() => parseTokenId(idText.value))
const limit = computed(() => wb.modelInfo.value?.logitsSize ?? wb.tokenizer.value?.raw.length ?? 0)
const preview = computed(() => {
  const id = parsed.value
  const tok = wb.tokenizer.value
  if (id === null) return idText.value.trim() ? 'Enter a non-negative integer ID.' : ''
  if (limit.value && id >= limit.value) return `Out of range (valid: 0–${limit.value - 1}).`
  if (!tok) return ''
  const raw = tok.raw[id]
  if (!raw)
    return 'This ID has a model embedding/logit but no tokenizer entry (decodes to nothing).'
  return `${escapeLiteral(tok.decoded[id] ?? '')}${tok.specialIds.includes(id) ? ' (special)' : ''}`
})
const valid = computed(() => parsed.value !== null && (!limit.value || parsed.value < limit.value))
const position = computed(() => {
  const n = wb.tokens.value.length
  const ref = props.insertAt ?? props.selected
  if (where.value === 'end' || ref === null) return n
  return where.value === 'before' ? ref : ref + 1
})
function insert() {
  if (!valid.value || parsed.value === null) return
  try {
    wb.insertToken(position.value, parsed.value)
    idText.value = ''
  } catch (e) {
    wb.notify('error', (e as Error).message)
  }
}
</script>

<template>
  <form class="row insert" aria-label="Insert token by ID" @submit.prevent="insert">
    <label class="small">
      Insert token ID
      <input
        v-model="idText"
        type="text"
        inputmode="numeric"
        size="8"
        placeholder="e.g. 0"
        data-testid="insert-id"
        :aria-invalid="!!idText && !valid"
      />
    </label>
    <label class="small">
      at
      <select v-model="where" data-testid="insert-where">
        <option value="end">end of context</option>
        <option value="before" :disabled="props.selected === null && props.insertAt === null">
          before selected
        </option>
        <option value="after" :disabled="props.selected === null && props.insertAt === null">
          after selected
        </option>
      </select>
    </label>
    <button
      type="submit"
      class="btn btn-sm"
      :disabled="!valid || !wb.tokenizer.value"
      data-testid="insert-submit"
    >
      Insert at {{ position }}
    </button>
    <span class="small muted" aria-live="polite" data-testid="insert-preview">{{ preview }}</span>
  </form>
</template>
