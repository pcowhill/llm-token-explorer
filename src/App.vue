<script setup lang="ts">
import { onBeforeUnmount, onMounted, provide, ref, watch } from 'vue'
import type { Workbench } from '@/stores/workbench'
import { WORKBENCH_KEY } from '@/composables/useWorkbench'
import { readJson, removeKey, useOnline, useTheme, writeJson } from '@/composables/usePreferences'
import AppHeader from '@/components/AppHeader.vue'
import ModelPanel from '@/components/model/ModelPanel.vue'
import ContextWorkspace from '@/components/context/ContextWorkspace.vue'
import ProbabilityWorkspace from '@/components/probability/ProbabilityWorkspace.vue'
import SamplingControls from '@/components/sampling/SamplingControls.vue'
import GenerationControls from '@/components/sampling/GenerationControls.vue'
import HistoryPanel from '@/components/history/HistoryPanel.vue'
import DiagnosticsPanel from '@/components/diagnostics/DiagnosticsPanel.vue'
import LearnPanel from '@/components/help/LearnPanel.vue'
import SettingsDialog from '@/components/SettingsDialog.vue'
import NoticeArea from '@/components/NoticeArea.vue'
import { catalogModel } from '@/models/catalog'
import type { Backend, BackendPreference, ModelDtype } from '@/types'

const props = defineProps<{
  workbench: Workbench
  pwa?: {
    updateAvailable: { value: boolean }
    offlineReady: { value: boolean }
    update: () => void
  }
}>()
const wb = props.workbench
provide(WORKBENCH_KEY, wb)

const { theme } = useTheme()
const online = useOnline()
const helpOpen = ref(false)
const settingsOpen = ref(false)
const persistSession = ref(readJson<boolean>('persist', true))
const updateAvailable = ref(false)
const offlineReady = ref(false)
const lastModel = ref(
  readJson<{ catalogKey: string; dtype: ModelDtype; backend: Backend } | null>('lastModel', null),
)

// ------------------------------------------------------------ persistence
const SESSION_KEY = 'session'
function restore() {
  const pref = readJson<BackendPreference>('backend', 'auto')
  wb.backendPref.value = pref
  if (!persistSession.value) return
  const saved = readJson<string | null>(SESSION_KEY, null)
  if (!saved) return
  try {
    wb.importExperiment(saved)
    wb.importReport.value = null
  } catch {
    removeKey(SESSION_KEY)
  }
}
let saveTimer: ReturnType<typeof setTimeout> | null = null
function scheduleSave() {
  if (!persistSession.value) return
  if (saveTimer) clearTimeout(saveTimer)
  saveTimer = setTimeout(() => writeJson(SESSION_KEY, wb.exportExperiment()), 800)
}
watch([wb.historyVersion, () => ({ ...wb.sampling }), () => ({ ...wb.generation })], scheduleSave, {
  deep: true,
})
watch(wb.backendPref, (b) => writeJson('backend', b))
watch(persistSession, (p) => {
  writeJson('persist', p)
  if (!p) removeKey(SESSION_KEY)
  else scheduleSave()
})
watch(
  () => wb.modelInfo.value,
  (i) => {
    if (i?.catalogKey && !i.tokenizerOnly) {
      lastModel.value = {
        catalogKey: i.catalogKey,
        dtype: i.dtype as ModelDtype,
        backend: i.backend,
      }
      writeJson('lastModel', lastModel.value)
    }
  },
)
function clearSaved() {
  removeKey(SESSION_KEY)
  removeKey('lastModel')
  lastModel.value = null
  wb.notify('success', 'Saved session data cleared.')
}
function reloadLast() {
  const m = lastModel.value
  if (m) void wb.loadCatalog(m.catalogKey, { dtype: m.dtype, backend: m.backend })
}

// ------------------------------------------------------------- keyboard
function onKey(e: KeyboardEvent) {
  const t = e.target as HTMLElement | null
  const editing =
    !!t &&
    (t.tagName === 'INPUT' ||
      t.tagName === 'TEXTAREA' ||
      t.tagName === 'SELECT' ||
      t.isContentEditable)
  const mod = e.ctrlKey || e.metaKey
  if (mod && e.key === 'Enter') {
    e.preventDefault()
    void wb.runInference(true)
    return
  }
  if (editing) return
  if (mod && !e.shiftKey && e.key.toLowerCase() === 'z') {
    e.preventDefault()
    wb.undo()
  } else if (mod && ((e.shiftKey && e.key.toLowerCase() === 'z') || e.key.toLowerCase() === 'y')) {
    e.preventDefault()
    wb.redo()
  } else if (e.altKey && e.key.toLowerCase() === 's') {
    e.preventDefault()
    void wb.sampleOne()
  }
}

onMounted(() => {
  restore()
  void wb.init()
  window.addEventListener('keydown', onKey)
  if (props.pwa) {
    watch(
      () => props.pwa!.updateAvailable.value,
      (v) => (updateAvailable.value = v),
      { immediate: true },
    )
    watch(
      () => props.pwa!.offlineReady.value,
      (v) => (offlineReady.value = v),
      { immediate: true },
    )
  }
})
onBeforeUnmount(() => window.removeEventListener('keydown', onKey))
</script>

<template>
  <a href="#main" class="visually-hidden-focusable skip">Skip to main content</a>
  <AppHeader
    v-model:theme="theme"
    :online="online"
    :update-available="updateAvailable"
    :offline-ready="offlineReady"
    @help="helpOpen = true"
    @settings="settingsOpen = true"
    @reload="props.pwa?.update()"
  />
  <main id="main">
    <div v-if="!online" class="alert alert-warning small banner" role="status">
      You are offline. The app works offline; models already in the browser cache (or local files)
      can still be loaded. New downloads are unavailable.
    </div>
    <div
      v-if="lastModel && wb.modelStatus.value === 'idle'"
      class="alert alert-info small banner row"
      data-testid="resume-banner"
    >
      <span
        >Last session used
        <strong>{{ catalogModel(lastModel.catalogKey)?.name ?? lastModel.catalogKey }}</strong> ({{
          lastModel.dtype
        }}, {{ lastModel.backend }}).</span
      >
      <button type="button" class="btn btn-sm btn-primary" @click="reloadLast">
        Load it again
      </button>
    </div>
    <div class="layout">
      <div class="area-model"><ModelPanel /></div>
      <div class="area-context"><ContextWorkspace /></div>
      <div class="area-prob"><ProbabilityWorkspace /></div>
      <div class="area-sampling"><SamplingControls /></div>
      <div class="area-gen"><GenerationControls /></div>
      <div class="area-history"><HistoryPanel /></div>
      <div class="area-diag"><DiagnosticsPanel /></div>
    </div>
  </main>
  <footer class="footer small muted">
    All inference runs locally in your browser. Prompts and local files never leave this device. No
    analytics. · Built with Vue, Transformers.js and ONNX Runtime Web. ·
    <a
      href="https://github.com/pcowhill/llm-token-explorer"
      target="_blank"
      rel="noopener noreferrer"
      >Source code</a
    >
  </footer>
  <NoticeArea />
  <LearnPanel :open="helpOpen" @close="helpOpen = false" />
  <SettingsDialog
    v-model:persist-session="persistSession"
    :open="settingsOpen"
    @close="settingsOpen = false"
    @clear-saved="clearSaved"
  />
</template>

<style scoped>
.skip {
  position: absolute;
  left: 1rem;
  top: 0.5rem;
  z-index: 100;
  background: var(--surface);
  padding: 0.3rem 0.6rem;
}
.layout {
  display: grid;
  gap: 0.8rem;
  padding: 0.8rem;
  grid-template-columns: minmax(0, 1fr);
  grid-template-areas:
    'model'
    'context'
    'prob'
    'sampling'
    'gen'
    'history'
    'diag';
  max-width: 1900px;
  margin: 0 auto;
}
.banner {
  max-width: 1900px;
  margin: 0.8rem auto 0;
  width: calc(100% - 1.6rem);
}
.area-model {
  grid-area: model;
}
.area-context {
  grid-area: context;
}
.area-prob {
  grid-area: prob;
}
.area-sampling {
  grid-area: sampling;
}
.area-gen {
  grid-area: gen;
}
.area-history {
  grid-area: history;
}
.area-diag {
  grid-area: diag;
}
@media (min-width: 900px) {
  .layout {
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    grid-template-areas:
      'model model'
      'context prob'
      'sampling prob'
      'gen prob'
      'history history'
      'diag diag';
  }
}
@media (min-width: 1400px) {
  .layout {
    grid-template-columns: minmax(0, 1fr) minmax(0, 1.6fr) minmax(19rem, 0.55fr);
    grid-template-areas:
      'model model model'
      'context prob sampling'
      'history prob gen'
      'history prob gen'
      'diag diag diag';
  }
}
.footer {
  text-align: center;
  padding: 1rem;
}
</style>
