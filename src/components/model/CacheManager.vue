<script setup lang="ts">
import { onMounted, ref, shallowRef, watch } from 'vue'
import {
  cacheSupported,
  deleteAllCachedModels,
  deleteCachedModel,
  getStorageInfo,
  listCachedModels,
  requestPersistentStorage,
  type CachedModel,
  type StorageInfo,
} from '@/models/cacheStorage'
import { CATALOG, formatBytes } from '@/models/catalog'
import { useWorkbench } from '@/composables/useWorkbench'

const emit = defineEmits<{ changed: [CachedModel[]] }>()
const wb = useWorkbench()
const models = shallowRef<CachedModel[]>([])
const storage = shallowRef<StorageInfo | null>(null)
const busy = ref(false)
const supported = cacheSupported()

async function refresh() {
  if (!supported) return
  busy.value = true
  try {
    models.value = await listCachedModels()
    storage.value = await getStorageInfo()
    emit('changed', models.value)
  } finally {
    busy.value = false
  }
}

async function remove(m: CachedModel) {
  const n = await deleteCachedModel(m.repo, m.revision)
  wb.notify('success', `Removed ${n} cached file(s) of ${m.repo}.`)
  await refresh()
}

async function removeAll() {
  await deleteAllCachedModels()
  wb.notify('success', 'Removed all cached model files.')
  await refresh()
}

async function persist() {
  const r = await requestPersistentStorage()
  wb.notify(
    r ? 'success' : 'info',
    r
      ? 'Storage marked as persistent.'
      : 'The browser did not grant persistent storage (it may still keep the cache).',
  )
  await refresh()
}

const label = (repo: string) => CATALOG.find((c) => c.repo === repo)?.name ?? repo

onMounted(refresh)
watch(
  () => wb.modelStatus.value,
  (s) => s === 'ready' && refresh(),
)
defineExpose({ refresh })
</script>

<template>
  <section class="stack" aria-labelledby="cache-title">
    <div class="row">
      <h3 id="cache-title">Model cache</h3>
      <span class="spacer" />
      <button type="button" class="btn btn-sm" :disabled="busy || !supported" @click="refresh">
        Refresh
      </button>
    </div>
    <p v-if="!supported" class="alert alert-warning small">
      Cache Storage is unavailable (private browsing or an insecure context). Models will be
      re-downloaded on every visit.
    </p>
    <template v-else>
      <p class="small muted">
        Downloaded files are stored in this browser’s Cache Storage so later visits (and offline
        use) skip the download.
      </p>
      <p class="small" data-testid="storage-usage">
        <template v-if="storage?.supported">
          Storage used: <strong>{{ formatBytes(storage.usage) }}</strong> of ≈
          {{ formatBytes(storage.quota) }} quota (browser estimate) · persistent:
          {{ storage.persisted === null ? 'unknown' : storage.persisted ? 'yes' : 'no' }}
        </template>
        <template v-else>Storage usage is not exposed by this browser.</template>
      </p>
      <table v-if="models.length" class="data" data-testid="cache-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Revision</th>
            <th class="num">Files</th>
            <th class="num">Size</th>
            <th><span class="sr-only">Actions</span></th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="m in models" :key="m.repo + m.revision">
            <td>{{ label(m.repo) }}</td>
            <td>
              <code>{{ m.revision.slice(0, 7) }}</code>
            </td>
            <td class="num">{{ m.files.length }}</td>
            <td class="num">{{ formatBytes(m.totalBytes) }}</td>
            <td>
              <button type="button" class="btn btn-sm btn-danger" @click="remove(m)">Remove</button>
            </td>
          </tr>
        </tbody>
      </table>
      <p v-else class="small faint">No cached models.</p>
      <div class="row">
        <button
          type="button"
          class="btn btn-sm btn-danger"
          :disabled="!models.length"
          @click="removeAll"
        >
          Remove all cached models
        </button>
        <button type="button" class="btn btn-sm" @click="persist">
          Request persistent storage
        </button>
      </div>
    </template>
  </section>
</template>
