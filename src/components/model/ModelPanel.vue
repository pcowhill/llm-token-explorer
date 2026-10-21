<script setup lang="ts">
import { computed, ref, shallowRef, watch } from 'vue'
import { useWorkbench } from '@/composables/useWorkbench'
import { catalogInDisplayOrder, formatBytes } from '@/models/catalog'
import { resolveBackend } from '@/stores/workbench'
import type { CachedModel } from '@/models/cacheStorage'
import ModelCard from './ModelCard.vue'
import LoadProgress from './LoadProgress.vue'
import LocalModelLoader from './LocalModelLoader.vue'
import HubRepoLoader from './HubRepoLoader.vue'
import CacheManager from './CacheManager.vue'
import InfoTip from '@/components/common/InfoTip.vue'
import { formatMs, shortSha } from '@/utils/format'

const wb = useWorkbench()
const tab = ref<'catalog' | 'local' | 'hub' | 'cache'>('catalog')
/** The source browser is collapsed once a model is ready, to keep the workbench prominent. */
const showSources = ref(true)
watch(
  () => wb.modelStatus.value,
  (s, prev) => {
    if (s === 'ready' && prev === 'loading') showSources.value = false
    if (s === 'error') showSources.value = true
  },
)
const cached = shallowRef<CachedModel[]>([])
const backend = computed(() => resolveBackend(wb.backendPref.value, wb.capabilities.value))
const caps = computed(() => wb.capabilities.value)
</script>

<template>
  <section class="panel card" aria-labelledby="model-panel-title" data-testid="model-panel">
    <div class="row head">
      <h2 id="model-panel-title">Model</h2>
      <span class="spacer" />
      <label class="row small">
        Inference backend
        <select
          v-model="wb.backendPref.value"
          data-testid="backend-select"
          aria-describedby="backend-help"
        >
          <option value="auto">
            Automatic ({{ resolveBackend('auto', caps) === 'webgpu' ? 'WebGPU' : 'WASM' }})
          </option>
          <option value="webgpu" :disabled="caps !== null && !caps.webgpu">
            WebGPU{{ caps && !caps.webgpu ? ' (unavailable)' : '' }}
          </option>
          <option value="wasm">WASM / CPU</option>
        </select>
      </label>
      <InfoTip label="Inference backends">
        <p>
          <strong>WebGPU</strong> runs the network on your GPU: usually much faster for larger
          models, but needs a supported browser/driver. f16 variants (fp16, q4f16) additionally need
          the <code>shader-f16</code> GPU feature.
        </p>
        <p>
          <strong>WASM / CPU</strong> runs on the CPU via WebAssembly SIMD. Works almost everywhere;
          slower. 8-bit (q8) weights are the best fit.
        </p>
        <p>
          Different backends and precisions produce slightly different logits for the same input.
        </p>
      </InfoTip>
    </div>
    <p id="backend-help" class="small muted caps" data-testid="capabilities">
      <template v-if="caps">
        WebGPU:
        <strong>{{ caps.webgpu ? 'available' : 'unavailable' }}</strong>
        <template v-if="caps.webgpuAdapter">
          ({{
            [caps.webgpuAdapter.vendor, caps.webgpuAdapter.architecture]
              .filter(Boolean)
              .join(' / ') || 'unknown adapter'
          }}{{ caps.webgpuAdapter.isFallbackAdapter ? ', software fallback' : '' }}, shader-f16:
          {{ caps.webgpuAdapter.shaderF16 ? 'yes' : 'no' }})
        </template>
        <template v-else-if="caps.webgpuError"> — {{ caps.webgpuError }}</template>
        · WASM SIMD: <strong>{{ caps.wasmSimd ? 'yes' : 'no' }}</strong> · threads:
        {{ caps.crossOriginIsolated ? 'multi' : 'single (no cross-origin isolation)' }}
      </template>
      <template v-else>Detecting browser capabilities…</template>
    </p>

    <div
      v-if="wb.modelInfo.value && wb.modelStatus.value === 'ready'"
      class="loaded alert alert-success small"
      data-testid="loaded-model"
    >
      <strong>{{ wb.modelInfo.value.displayName }}</strong>
      <span v-if="wb.modelInfo.value.repo">
        ·
        <code>{{ wb.modelInfo.value.repo }}@{{ shortSha(wb.modelInfo.value.revision) }}</code></span
      >
      ·
      {{
        wb.modelInfo.value.tokenizerOnly
          ? 'tokenizer only (no weights)'
          : `${wb.modelInfo.value.backend} · ${wb.modelInfo.value.dtype}`
      }}
      · loaded in {{ formatMs(wb.modelInfo.value.loadTimeMs) }}
      <span v-if="wb.modelInfo.value.downloadBytes">
        · {{ formatBytes(wb.modelInfo.value.downloadBytes) }}
        {{ wb.modelInfo.value.fromCache ? 'from cache' : 'transferred' }}</span
      >
      <button type="button" class="btn btn-sm" @click="wb.unloadModel()">Unload</button>
    </div>
    <div
      v-if="wb.modelError.value && wb.modelStatus.value !== 'loading'"
      class="alert alert-error small"
      role="alert"
      data-testid="model-error"
    >
      <strong>Model loading failed:</strong> {{ wb.modelError.value.message }}
      <p v-if="wb.modelError.value.hint">{{ wb.modelError.value.hint }}</p>
      <p>You can choose another model, quantization or backend and try again.</p>
    </div>
    <LoadProgress v-if="wb.modelStatus.value === 'loading'" />

    <div class="row">
      <button
        type="button"
        class="btn btn-sm"
        :aria-expanded="showSources"
        aria-controls="model-sources"
        data-testid="toggle-model-sources"
        @click="showSources = !showSources"
      >
        {{ showSources ? 'Hide model sources' : 'Change model / manage cache…' }}
      </button>
    </div>
    <div v-show="showSources" id="model-sources">
      <div class="tabs" role="tablist" aria-label="Model sources">
        <button
          v-for="t in [
            ['catalog', 'Recommended'],
            ['local', 'Local files'],
            ['hub', 'Hugging Face repo'],
            ['cache', 'Cache'],
          ] as const"
          :id="`model-tab-${t[0]}`"
          :key="t[0]"
          type="button"
          role="tab"
          class="tab"
          :aria-selected="tab === t[0]"
          :aria-controls="`model-tabpanel-${t[0]}`"
          @click="tab = t[0]"
        >
          {{ t[1] }}
        </button>
      </div>
      <div
        v-show="tab === 'catalog'"
        id="model-tabpanel-catalog"
        role="tabpanel"
        aria-labelledby="model-tab-catalog"
        class="cards"
      >
        <ModelCard
          v-for="m in catalogInDisplayOrder()"
          :key="m.key"
          :model="m"
          :backend="backend"
          :cached="cached"
        />
      </div>
      <div
        v-show="tab === 'local'"
        id="model-tabpanel-local"
        role="tabpanel"
        aria-labelledby="model-tab-local"
      >
        <LocalModelLoader :backend="backend" />
      </div>
      <div
        v-show="tab === 'hub'"
        id="model-tabpanel-hub"
        role="tabpanel"
        aria-labelledby="model-tab-hub"
        class="stack"
      >
        <HubRepoLoader :backend="backend" :cached="cached" />
      </div>
      <div
        v-show="tab === 'cache'"
        id="model-tabpanel-cache"
        role="tabpanel"
        aria-labelledby="model-tab-cache"
      >
        <CacheManager @changed="cached = $event" />
      </div>
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
.cards {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(min(100%, 30rem), 1fr));
  gap: 0.6rem;
  padding-top: 0.5rem;
}
.caps {
  margin: 0;
}
.loaded .btn {
  margin-left: 0.5rem;
}
[role='tabpanel'] {
  padding-top: 0.5rem;
}
</style>
