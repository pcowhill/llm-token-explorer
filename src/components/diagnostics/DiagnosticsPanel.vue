<script setup lang="ts">
import { computed, onMounted, ref, shallowRef, watch } from 'vue'
import { useWorkbench } from '@/composables/useWorkbench'
import { catalogModel, formatBytes } from '@/models/catalog'
import { getStorageInfo, type StorageInfo } from '@/models/cacheStorage'
import { formatCount, formatMs } from '@/utils/format'
import { pathTo } from '@/history/tree'
import InfoTip from '@/components/common/InfoTip.vue'

const wb = useWorkbench()
const open = ref(false)
const storage = shallowRef<StorageInfo | null>(null)
const heap = ref<{ used: number; limit: number } | null>(null)

async function refresh() {
  storage.value = await getStorageInfo()
  const mem = (
    performance as unknown as { memory?: { usedJSHeapSize: number; jsHeapSizeLimit: number } }
  ).memory
  heap.value = mem ? { used: mem.usedJSHeapSize, limit: mem.jsHeapSizeLimit } : null
}
onMounted(refresh)
watch(open, (o) => o && refresh())

const info = computed(() => wb.modelInfo.value)
const cat = computed(() =>
  info.value?.catalogKey ? catalogModel(info.value.catalogKey) : undefined,
)
const variant = computed(() => cat.value?.variants.find((v) => v.dtype === info.value?.dtype))
const gpu = computed(() => wb.capabilities.value?.webgpuAdapter ?? null)
const generatedInPath = computed(() => {
  void wb.historyVersion.value
  const t = wb.getTree()
  return pathTo(t, t.currentId).filter((n) => n.kind === 'append').length
})
type Row = [label: string, value: string, kind: 'measured' | 'estimate' | 'config' | 'unavailable']
const rows = computed<Row[]>(() => {
  const i = info.value
  const caps = wb.capabilities.value
  const r = wb.result.value
  return [
    ['Model', i?.displayName ?? 'none', 'config'],
    ['Model type / architecture', i?.modelType ?? '—', 'config'],
    ['Precision / quantization', i?.dtype ?? '—', 'config'],
    ['Inference backend', i ? (i.backend === 'webgpu' ? 'WebGPU' : 'WASM (CPU)') : '—', 'config'],
    [
      'WASM threads',
      caps
        ? caps.crossOriginIsolated
          ? 'multi-threaded'
          : '1 (page is not cross-origin isolated)'
        : '—',
      'config',
    ],
    [
      'Download size (weights, catalog)',
      variant.value ? formatBytes(variant.value.sizeBytes) : '—',
      'config',
    ],
    [
      'Bytes transferred/read during load',
      i?.downloadBytes ? formatBytes(i.downloadBytes) : '—',
      'measured',
    ],
    [
      'Model cache status',
      i
        ? i.source === 'local'
          ? 'local files (not cached)'
          : i.fromCache === null
            ? 'unknown'
            : i.fromCache
              ? 'loaded from browser cache'
              : 'downloaded (now cached)'
        : '—',
      'measured',
    ],
    ['Model loading time', formatMs(i?.loadTimeMs), 'measured'],
    ['Tokenization time (last)', formatMs(wb.tokenizeMs.value), 'measured'],
    ['Neural network inference (last forward pass)', formatMs(r?.inferenceMs), 'measured'],
    [
      'Tokens fed in last forward pass',
      r ? formatCount(r.tokensProcessed) + (r.usedKvCache ? ' (KV cache reused)' : '') : '—',
      'measured',
    ],
    ['Sampling calculation time', formatMs(wb.samplingMs.value), 'measured'],
    ['Forward passes this session', formatCount(wb.forwardCount.value), 'measured'],
    ['Logits output tensor', r ? `[${r.outputDims.join(', ')}] ${r.outputDtype}` : '—', 'measured'],
    ['Context tokens', formatCount(wb.tokens.value.length), 'measured'],
    ['Model context length limit', i?.contextLength ? formatCount(i.contextLength) : '—', 'config'],
    ['Appended tokens on current path', formatCount(generatedInPath.value), 'measured'],
    [
      'Automatic generation speed',
      wb.autoTokensPerSecond.value ? `${wb.autoTokensPerSecond.value.toFixed(2)} tokens/s` : '—',
      'measured',
    ],
    [
      'Vocabulary (tokenizer / logits)',
      i ? `${formatCount(i.tokenizerSize)} / ${formatCount(i.logitsSize)}` : '—',
      'config',
    ],
    [
      'Estimated model memory',
      cat.value && i ? formatBytes(cat.value.memoryEstimateBytes[i.backend]) : '—',
      'estimate',
    ],
    [
      'GPU adapter',
      gpu.value
        ? [gpu.value.vendor, gpu.value.architecture, gpu.value.description]
            .filter(Boolean)
            .join(' / ') || 'available (no details exposed)'
        : (caps?.webgpuError ?? '—'),
      gpu.value ? 'measured' : 'unavailable',
    ],
    [
      'GPU max buffer size',
      gpu.value?.maxBufferSize ? formatBytes(gpu.value.maxBufferSize) : '—',
      gpu.value ? 'measured' : 'unavailable',
    ],
    ['GPU memory', 'Not exposed by WebGPU', 'unavailable'],
    [
      'Device memory (navigator.deviceMemory)',
      caps?.deviceMemoryGb ? `≈ ${caps.deviceMemoryGb} GB (coarse)` : 'not exposed',
      caps?.deviceMemoryGb ? 'estimate' : 'unavailable',
    ],
    [
      'JS heap (main thread)',
      heap.value
        ? `${formatBytes(heap.value.used)} of ${formatBytes(heap.value.limit)}`
        : 'not exposed',
      heap.value ? 'measured' : 'unavailable',
    ],
    [
      'Browser storage used',
      storage.value?.supported
        ? `${formatBytes(storage.value.usage)} of ≈ ${formatBytes(storage.value.quota)}`
        : 'not exposed',
      storage.value?.supported ? 'estimate' : 'unavailable',
    ],
    [
      'Library versions',
      `Transformers.js ${__TRANSFORMERS_VERSION__} · ONNX Runtime Web ${__ORT_VERSION__} · app ${__APP_VERSION__}`,
      'config',
    ],
  ]
})
</script>

<template>
  <details
    class="panel card section"
    data-testid="diagnostics"
    @toggle="open = ($event.target as HTMLDetailsElement).open"
  >
    <summary class="row">
      <h2>Performance &amp; diagnostics</h2>
      <span class="faint small">(click to expand)</span>
    </summary>
    <div class="body">
      <div class="row small">
        <label class="row">
          <input v-model="wb.generation.useKvCache" type="checkbox" data-testid="kv-cache" />
          Reuse KV cache when the context only grows (faster generation)
        </label>
        <InfoTip label="KV cache and exactness">
          <p>
            With the KV cache, only new tokens are processed and earlier keys/values are reused.
            With fp32 weights the logits match a full pass to ~1e-5.
          </p>
          <p>
            With <strong>dynamically quantized (q8)</strong> models, activations are quantized with
            scales computed over the tokens processed together, so incremental passes produce
            noticeably different logits than a full pass. It is therefore
            <strong>off by default</strong> so the distribution for a context never depends on how
            you reached it.
          </p>
        </InfoTip>
        <label class="row">
          Debounce
          <input
            v-model.number="wb.generation.debounceMs"
            type="number"
            min="0"
            max="5000"
            step="50"
            class="num"
          />
          ms
        </label>
        <button type="button" class="btn btn-sm" @click="refresh">Refresh</button>
      </div>
      <table class="data" aria-label="Diagnostics">
        <thead>
          <tr>
            <th>Metric</th>
            <th>Value</th>
            <th>Kind</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="[label, value, kind] in rows" :key="label">
            <td>{{ label }}</td>
            <td class="mono small">{{ value }}</td>
            <td>
              <span
                class="badge"
                :class="{
                  'badge-success': kind === 'measured',
                  'badge-warning': kind === 'estimate',
                  'badge-accent': kind === 'config',
                }"
                >{{ kind }}</span
              >
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </details>
</template>

<style scoped>
.panel {
  padding: 0.8rem;
}
.body {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  margin-top: 0.5rem;
}
.num {
  width: 5rem;
}
</style>
