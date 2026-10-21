<script setup lang="ts">
import { computed } from 'vue'
import { useWorkbench } from '@/composables/useWorkbench'
import type { ThemePref } from '@/composables/usePreferences'

const props = defineProps<{
  online: boolean
  theme: ThemePref
  updateAvailable: boolean
  offlineReady: boolean
}>()
const emit = defineEmits<{
  'update:theme': [ThemePref]
  help: []
  settings: []
  reload: []
}>()
const wb = useWorkbench()

const statusText = computed(() => {
  switch (wb.modelStatus.value) {
    case 'loading':
      return `Loading ${wb.loadingLabel.value}…`
    case 'ready':
      return wb.modelInfo.value?.tokenizerOnly ? 'Tokenizer only' : 'Ready'
    case 'error':
      return 'Load failed'
    default:
      return 'No model loaded'
  }
})
const statusClass = computed(
  () =>
    ({
      loading: 'badge-warning',
      ready: wb.modelInfo.value?.tokenizerOnly ? 'badge-accent' : 'badge-success',
      error: 'badge-danger',
      idle: '',
    })[wb.modelStatus.value],
)
</script>

<template>
  <header class="app-header">
    <div class="brand">
      <svg class="logo" viewBox="0 0 32 32" aria-hidden="true">
        <rect x="2" y="9" width="8" height="14" rx="2" class="l1" />
        <rect x="12" y="9" width="8" height="14" rx="2" class="l2" />
        <rect x="22" y="9" width="8" height="14" rx="2" class="l1" />
      </svg>
      <h1>LLM Token Explorer</h1>
    </div>
    <div class="status" role="status" aria-live="polite">
      <span class="model-name" data-testid="header-model">{{
        wb.modelInfo.value?.displayName ?? '—'
      }}</span>
      <span class="badge" :class="statusClass" data-testid="model-status">{{ statusText }}</span>
      <span
        v-if="wb.modelInfo.value"
        class="badge"
        data-testid="header-backend"
        :title="`Inference backend: ${wb.modelInfo.value.backend}`"
      >
        {{ wb.modelInfo.value.backend === 'webgpu' ? 'WebGPU' : 'WASM / CPU' }} ·
        {{ wb.modelInfo.value.dtype }}
      </span>
      <span
        class="badge"
        :class="props.online ? 'badge-success' : 'badge-warning'"
        data-testid="online-status"
      >
        <span aria-hidden="true">{{ props.online ? '●' : '○' }}</span
        >{{ props.online ? 'Online' : 'Offline' }}
      </span>
      <span v-if="props.offlineReady" class="badge" title="The app shell is cached for offline use"
        >Offline-ready</span
      >
    </div>
    <div class="actions">
      <button
        v-if="props.updateAvailable"
        type="button"
        class="btn btn-sm btn-primary"
        @click="emit('reload')"
      >
        Update available — reload
      </button>
      <label class="theme">
        <span class="sr-only">Theme</span>
        <select
          :value="props.theme"
          aria-label="Color theme"
          @change="emit('update:theme', ($event.target as HTMLSelectElement).value as ThemePref)"
        >
          <option value="system">System theme</option>
          <option value="light">Light</option>
          <option value="dark">Dark</option>
        </select>
      </label>
      <button type="button" class="btn btn-sm" @click="emit('settings')">Settings</button>
      <button type="button" class="btn btn-sm" @click="emit('help')">Learn / Help</button>
    </div>
  </header>
</template>

<style scoped>
.app-header {
  display: flex;
  align-items: center;
  gap: 1rem;
  flex-wrap: wrap;
  padding: 0.6rem 1rem;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  position: sticky;
  top: 0;
  z-index: 20;
}
.brand {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.brand h1 {
  font-size: 1.1rem;
  letter-spacing: 0.01em;
}
.logo {
  width: 26px;
  height: 26px;
}
.l1 {
  fill: var(--accent);
}
.l2 {
  fill: var(--adjusted);
}
.status {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  flex-wrap: wrap;
  flex: 1;
}
.model-name {
  font-weight: 600;
  font-size: 0.9rem;
}
.actions {
  display: flex;
  gap: 0.4rem;
  align-items: center;
}
.theme select {
  font-size: 0.85rem;
  padding: 0.2em 0.4em;
}
</style>
