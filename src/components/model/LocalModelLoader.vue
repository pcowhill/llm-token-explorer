<script setup lang="ts">
/**
 * Load a model from local files / a local directory. Files are read in the browser
 * (passed to the inference worker as File objects) and are never uploaded anywhere.
 */
import { computed, ref, shallowRef } from 'vue'
import { useWorkbench } from '@/composables/useWorkbench'
import {
  analyzeLocalFiles,
  checkConfigJson,
  normalizePath,
  type LocalConfigCheck,
  type LocalLayout,
} from '@/models/localFiles'
import { formatBytes } from '@/models/catalog'
import type { Backend } from '@/types'

const props = defineProps<{ backend: Backend }>()
const wb = useWorkbench()
const files = shallowRef<{ path: string; file: File }[]>([])
const layout = shallowRef<LocalLayout | null>(null)
const configCheck = shallowRef<LocalConfigCheck | null>(null)
const choice = ref(0)
const dragOver = ref(false)
const name = ref('')

async function accept(list: FileList | File[] | null) {
  if (!list || list.length === 0) return
  const arr = Array.from(list).map((f) => ({
    path: normalizePath((f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name),
    file: f,
  }))
  files.value = arr
  const l = analyzeLocalFiles(arr.map((f) => ({ path: f.path, size: f.file.size })))
  layout.value = l
  choice.value = 0
  const cfg = arr.find((f) => f.path === `${l.root}config.json`)
  configCheck.value = cfg ? checkConfigJson(await cfg.file.text()) : null
  const first = arr[0]?.path ?? ''
  name.value = l.root
    ? l.root.replace(/\/$/, '').split('/').pop()!
    : first.includes('/')
      ? first.split('/')[0]!
      : 'Local model'
}

const errors = computed(() => [
  ...(layout.value?.errors ?? []),
  ...(configCheck.value?.errors ?? []),
])
const warnings = computed(() => [
  ...(layout.value?.warnings ?? []),
  ...(configCheck.value?.warnings ?? []),
])
const candidate = computed(() => layout.value?.candidates[choice.value] ?? null)
const canLoad = computed(
  () => !!candidate.value && errors.value.length === 0 && wb.modelStatus.value !== 'loading',
)

function load() {
  const l = layout.value
  const c = candidate.value
  if (!l || !c) return
  const rel = files.value
    .filter((f) => f.path.startsWith(l.root))
    .map((f) => ({ path: f.path.slice(l.root.length), file: f.file }))
  void wb.loadModel(
    {
      kind: 'local',
      displayName: `${name.value} (local)`,
      files: rel,
      subfolder: c.subfolder,
      baseName: c.baseName,
      externalDataChunks: c.externalDataChunks,
    },
    props.backend,
    c.dtype,
  )
}

function onDrop(e: DragEvent) {
  dragOver.value = false
  void accept(e.dataTransfer?.files ?? null)
}
</script>

<template>
  <section class="local stack" aria-labelledby="local-title">
    <h3 id="local-title">Load local model files</h3>
    <p class="small muted">
      Select a Transformers.js-compatible ONNX export: <code>config.json</code>,
      <code>tokenizer.json</code>, <code>tokenizer_config.json</code>, optional
      <code>generation_config.json</code>, and <code>onnx/*.onnx</code> (plus any
      <code>*.onnx_data</code> files). Files stay on this device — nothing is uploaded.
    </p>
    <div
      class="drop"
      :class="{ over: dragOver }"
      @dragover.prevent="dragOver = true"
      @dragleave="dragOver = false"
      @drop.prevent="onDrop"
    >
      <label class="btn">
        Choose model folder…
        <input
          type="file"
          class="sr-only"
          webkitdirectory
          directory
          multiple
          data-testid="local-dir-input"
          @change="accept(($event.target as HTMLInputElement).files)"
        />
      </label>
      <label class="btn">
        Choose files…
        <input
          type="file"
          class="sr-only"
          multiple
          accept=".json,.onnx,.onnx_data,.txt,.model"
          data-testid="local-files-input"
          @change="accept(($event.target as HTMLInputElement).files)"
        />
      </label>
      <span class="small faint">or drop files here</span>
    </div>

    <template v-if="layout">
      <p class="small">
        {{ files.length }} file(s) selected<span v-if="layout.root">
          · model root <code>{{ layout.root }}</code></span
        >
        <span v-if="configCheck?.modelType">
          · <code>model_type: {{ configCheck.modelType }}</code></span
        >
      </p>
      <ul v-if="errors.length" class="alert alert-error small" data-testid="local-errors">
        <li v-for="e in errors" :key="e">{{ e }}</li>
      </ul>
      <ul v-if="warnings.length" class="alert alert-warning small">
        <li v-for="w in warnings" :key="w">{{ w }}</li>
      </ul>
      <div v-if="layout.candidates.length" class="row small">
        <label>
          Weights file
          <select v-model.number="choice" data-testid="local-candidate">
            <option v-for="(c, i) in layout.candidates" :key="c.path" :value="i">
              {{ c.path }} — {{ c.dtype }}, {{ formatBytes(c.totalBytes)
              }}{{ c.externalDataChunks ? `, ${c.externalDataChunks} data file(s)` : '' }}
            </option>
          </select>
        </label>
        <label>
          Display name
          <input v-model="name" type="text" />
        </label>
        <button
          type="button"
          class="btn btn-primary"
          :disabled="!canLoad"
          data-testid="local-load"
          @click="load"
        >
          Load local model
        </button>
      </div>
    </template>
  </section>
</template>

<style scoped>
.drop {
  display: flex;
  gap: 0.5rem;
  align-items: center;
  flex-wrap: wrap;
  padding: 0.7rem;
  border: 2px dashed var(--border-strong);
  border-radius: var(--radius);
}
.drop.over {
  border-color: var(--accent);
  background: var(--accent-soft);
}
.drop label.btn:focus-within {
  outline: 2px solid var(--focus);
}
ul.alert {
  margin: 0;
  padding-left: 1.6rem;
}
</style>
