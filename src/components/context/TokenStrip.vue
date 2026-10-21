<script setup lang="ts">
/**
 * The authoritative token sequence as individually distinguishable chips.
 * Keyboard: ←/→ move, Home/End jump, Delete/Backspace remove, Enter/Space select.
 */
import { computed, nextTick, ref, watch } from 'vue'
import { useWorkbench } from '@/composables/useWorkbench'
import { escapeVisible } from '@/tokenization/display'

const props = defineProps<{ selected: number | null; showIds: boolean }>()
const emit = defineEmits<{ select: [number | null] }>()
const wb = useWorkbench()
const list = ref<HTMLElement | null>(null)
const MAX_RENDER = 4000

const specialSet = computed(() => new Set(wb.tokenizer.value?.specialIds ?? []))
const chips = computed(() => {
  const tok = wb.tokenizer.value
  const ids = wb.tokens.value
  const start = Math.max(0, ids.length - MAX_RENDER)
  return ids.slice(start).map((id, k) => {
    const i = start + k
    const raw = tok?.raw[id]
    const decoded = tok?.decoded[id] ?? ''
    const unknown = !tok || raw === undefined || raw === ''
    return {
      i,
      id,
      text: unknown ? (tok ? '∅' : `#${id}`) : escapeVisible(decoded) || '∅',
      special: specialSet.value.has(id),
      unknown,
      invalid: !!wb.modelInfo.value && id >= wb.modelInfo.value.logitsSize,
      stop: wb.stopTokenIds.value.includes(id),
    }
  })
})
const truncated = computed(() => wb.tokens.value.length > MAX_RENDER)

function focusChip(i: number) {
  void nextTick(() => {
    const el = list.value?.querySelector<HTMLElement>(`[data-index="${i}"]`)
    el?.focus()
  })
}

function onKey(e: KeyboardEvent, i: number) {
  const n = wb.tokens.value.length
  let next: number | null = null
  if (e.key === 'ArrowRight') next = Math.min(n - 1, i + 1)
  else if (e.key === 'ArrowLeft') next = Math.max(0, i - 1)
  else if (e.key === 'Home') next = 0
  else if (e.key === 'End') next = n - 1
  else if (e.key === 'Delete' || e.key === 'Backspace') {
    e.preventDefault()
    wb.removeToken(i)
    const sel = Math.min(i, wb.tokens.value.length - 1)
    emit('select', sel >= 0 ? sel : null)
    if (sel >= 0) focusChip(sel)
    return
  } else if (e.key === 'Escape') {
    emit('select', null)
    return
  }
  if (next !== null) {
    e.preventDefault()
    emit('select', next)
    focusChip(next)
  }
}

watch(
  () => wb.tokens.value.length,
  () => {
    if (props.selected !== null && props.selected >= wb.tokens.value.length) emit('select', null)
  },
)
</script>

<template>
  <div class="strip-wrap">
    <p v-if="truncated" class="small faint">
      Showing the last {{ MAX_RENDER.toLocaleString() }} of
      {{ wb.tokens.value.length.toLocaleString() }} tokens.
    </p>
    <ol
      v-if="chips.length"
      ref="list"
      class="strip"
      aria-label="Context tokens (authoritative token-ID sequence)"
      data-testid="token-strip"
    >
      <li v-for="c in chips" :key="c.i">
        <button
          type="button"
          class="chip"
          :class="{
            alt: c.i % 2 === 1,
            special: c.special,
            unknown: c.unknown,
            invalid: c.invalid,
            selected: props.selected === c.i,
          }"
          :data-index="c.i"
          :data-token-id="c.id"
          :tabindex="props.selected === c.i || (props.selected === null && c.i === 0) ? 0 : -1"
          :aria-pressed="props.selected === c.i"
          :aria-label="`Position ${c.i}, token ID ${c.id}${c.special ? ', special token' : ''}: ${c.text}`"
          :title="`#${c.i} · ID ${c.id}`"
          @click="emit('select', props.selected === c.i ? null : c.i)"
          @keydown="onKey($event, c.i)"
        >
          <span class="txt">{{ c.text }}</span>
          <span v-if="props.showIds" class="id">{{ c.id }}</span>
          <span v-if="c.stop" class="sr-only">(stop token)</span>
        </button>
      </li>
    </ol>
    <p v-else class="empty small faint" data-testid="token-strip-empty">
      The context is empty. Type text, load an example, or insert a token (e.g. a document-start
      token) to get a next-token prediction.
    </p>
  </div>
</template>

<style scoped>
.strip {
  list-style: none;
  margin: 0;
  padding: 0.5rem;
  display: flex;
  flex-wrap: wrap;
  gap: 3px;
  background: var(--surface-2);
  border-radius: var(--radius-sm);
  max-height: 22rem;
  overflow-y: auto;
}
.chip {
  display: inline-flex;
  flex-direction: column;
  align-items: center;
  padding: 1px 4px;
  border-radius: 4px;
  border: 1px solid var(--tok-border-a);
  background: var(--tok-a);
  font-family: var(--mono);
  font-size: 0.85rem;
  line-height: 1.25;
  min-width: 1.2rem;
}
.chip.alt {
  background: var(--tok-b);
  border-color: var(--tok-border-b);
}
.chip.special {
  background: var(--special-soft);
  border: 1px dashed var(--special);
  color: var(--special);
  font-weight: 600;
}
.chip.unknown {
  border-style: dotted;
  color: var(--text-faint);
}
.chip.invalid {
  background: var(--danger-soft);
  border-color: var(--danger);
  color: var(--danger);
}
.chip.selected {
  outline: 2px solid var(--accent);
  outline-offset: 1px;
}
.txt {
  white-space: pre;
}
.id {
  font-size: 0.65rem;
  color: var(--text-faint);
}
.empty {
  padding: 0.6rem;
  background: var(--surface-2);
  border-radius: var(--radius-sm);
  margin: 0;
}
</style>
