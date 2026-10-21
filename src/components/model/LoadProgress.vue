<script setup lang="ts">
import { computed } from 'vue'
import { useWorkbench } from '@/composables/useWorkbench'
import { formatBytes } from '@/models/catalog'

const wb = useWorkbench()
const files = computed(() =>
  Object.entries(wb.fileProgress).sort((a, b) => b[1].total - a[1].total),
)
const totals = computed(() => {
  let loaded = 0
  let total = 0
  for (const [, f] of files.value) {
    loaded += f.done && f.total === 0 ? 0 : f.loaded
    total += f.total
  }
  return { loaded, total }
})
const pct = computed(() =>
  totals.value.total ? Math.min(100, (100 * totals.value.loaded) / totals.value.total) : 0,
)
</script>

<template>
  <div class="progress card" data-testid="load-progress" aria-live="polite">
    <div class="row">
      <strong>Loading {{ wb.loadingTarget.value?.label ?? wb.loadingLabel.value }}</strong>
      <span class="badge"
        >{{ wb.loadingTarget.value?.backend === 'webgpu' ? 'WebGPU' : 'WASM' }} ·
        {{ wb.loadingTarget.value?.dtype }}</span
      >
      <span class="spacer" />
      <button type="button" class="btn btn-sm btn-danger" @click="wb.cancelLoad()">Cancel</button>
    </div>
    <p class="small muted" data-testid="load-stage">{{ wb.loadingStage.value || 'Preparing…' }}</p>
    <div
      class="bar"
      role="progressbar"
      aria-label="Total download progress"
      :aria-valuenow="Math.round(pct)"
      aria-valuemin="0"
      aria-valuemax="100"
    >
      <div class="fill" :style="{ width: pct + '%' }" />
    </div>
    <p class="small">
      {{ formatBytes(totals.loaded) }} of {{ formatBytes(totals.total) }} ({{ pct.toFixed(0) }}%)
      <span v-if="wb.loadingTarget.value?.expectedBytes" class="faint">
        · weights ≈ {{ formatBytes(wb.loadingTarget.value.expectedBytes) }}</span
      >
    </p>
    <ul class="files small">
      <li v-for="[name, f] in files" :key="name">
        <code>{{ name }}</code>
        <span class="faint">
          {{
            f.done
              ? '✓ done'
              : f.total
                ? `${formatBytes(f.loaded)} / ${formatBytes(f.total)}`
                : 'waiting…'
          }}
        </span>
      </li>
    </ul>
  </div>
</template>

<style scoped>
.progress {
  padding: 0.7rem;
  margin-top: 0.5rem;
}
.bar {
  height: 8px;
  border-radius: 4px;
  background: var(--surface-3);
  overflow: hidden;
}
.fill {
  height: 100%;
  background: var(--accent);
  transition: width 0.2s;
}
.files {
  list-style: none;
  padding: 0;
  margin: 0.3rem 0 0;
  display: grid;
  gap: 0.15rem;
}
.files li {
  display: flex;
  justify-content: space-between;
  gap: 0.5rem;
}
</style>
