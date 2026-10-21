<script setup lang="ts">
import { useWorkbench } from '@/composables/useWorkbench'
const wb = useWorkbench()
</script>

<template>
  <div class="notices" aria-live="polite" data-testid="notices">
    <div
      v-for="n in wb.notices.value"
      :key="n.id"
      class="notice alert"
      :class="`alert-${n.level === 'success' ? 'success' : n.level === 'info' ? 'info' : n.level}`"
      :role="n.level === 'error' ? 'alert' : 'status'"
    >
      <span>{{ n.message }}</span>
      <button
        type="button"
        class="btn btn-sm btn-ghost"
        aria-label="Dismiss notification"
        @click="wb.dismiss(n.id)"
      >
        ✕
      </button>
    </div>
  </div>
</template>

<style scoped>
.notices {
  position: fixed;
  right: 1rem;
  bottom: 1rem;
  z-index: 60;
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
  width: min(26rem, calc(100vw - 2rem));
}
.notice {
  display: flex;
  gap: 0.5rem;
  align-items: flex-start;
  justify-content: space-between;
  box-shadow: var(--shadow);
}
</style>
