<script setup lang="ts">
/** Click/keyboard-toggled help popover. Works without hover; Escape closes it. */
import { ref, useId, onBeforeUnmount } from 'vue'

const props = defineProps<{ label: string }>()
const open = ref(false)
const id = useId()
const root = ref<HTMLElement | null>(null)

function onDoc(e: MouseEvent) {
  if (root.value && !root.value.contains(e.target as Node)) open.value = false
}
function toggle() {
  open.value = !open.value
  if (open.value) document.addEventListener('mousedown', onDoc)
  else document.removeEventListener('mousedown', onDoc)
}
function close() {
  open.value = false
  document.removeEventListener('mousedown', onDoc)
}
onBeforeUnmount(() => document.removeEventListener('mousedown', onDoc))
</script>

<template>
  <span ref="root" class="infotip" @keydown.esc="close">
    <button
      type="button"
      class="infotip-btn"
      :aria-expanded="open"
      :aria-controls="id"
      :aria-label="`Help: ${props.label}`"
      @click="toggle"
    >
      ?
    </button>
    <span v-if="open" :id="id" role="note" class="infotip-pop card">
      <strong class="infotip-title">{{ props.label }}</strong>
      <slot />
    </span>
  </span>
</template>

<style scoped>
.infotip {
  position: relative;
  display: inline-block;
  vertical-align: middle;
}
.infotip-btn {
  width: 1.25rem;
  height: 1.25rem;
  border-radius: 50%;
  border: 1px solid var(--border-strong);
  background: var(--surface-2);
  color: var(--text-muted);
  font-size: 0.72rem;
  font-weight: 700;
  line-height: 1;
  padding: 0;
}
.infotip-pop {
  position: absolute;
  z-index: 50;
  top: 1.6rem;
  left: -0.5rem;
  width: min(22rem, 80vw);
  padding: 0.6rem 0.75rem;
  font-size: 0.84rem;
  font-weight: 400;
  color: var(--text);
  display: block;
  text-align: left;
  white-space: normal;
}
.infotip-title {
  display: block;
  margin-bottom: 0.25rem;
}
</style>
