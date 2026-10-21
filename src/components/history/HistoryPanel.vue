<script setup lang="ts">
import { computed, ref } from 'vue'
import { useWorkbench } from '@/composables/useWorkbench'
import { flatten, siblings } from '@/history/tree'
import { ExperimentValidationError } from '@/history/experiment'
import { escapeVisible } from '@/tokenization/display'
import { formatProb } from '@/utils/format'

const wb = useWorkbench()
const importMessages = ref<string[]>([])
const importError = ref<string | null>(null)
const fileInput = ref<HTMLInputElement | null>(null)

const rows = computed(() => {
  void wb.historyVersion.value
  return flatten(wb.getTree())
})
const branchInfo = computed(() => {
  void wb.historyVersion.value
  const t = wb.getTree()
  const sibs = siblings(t, t.currentId)
  return { index: sibs.findIndex((n) => n.id === t.currentId) + 1, count: sibs.length }
})
function lastTokenText(tokens: number[]) {
  const id = tokens[tokens.length - 1]
  if (id === undefined) return '∅'
  const d = wb.tokenizer.value?.decoded[id]
  return d === undefined ? `#${id}` : escapeVisible(d) || '∅'
}

function exportFile() {
  const json = wb.exportExperiment()
  const blob = new Blob([json], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `llm-token-experiment-${new Date().toISOString().replace(/[:.]/g, '-')}.json`
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
  wb.notify('success', 'Experiment exported (token IDs, settings and history — no model weights).')
}

async function importFile(e: Event) {
  const f = (e.target as HTMLInputElement).files?.[0]
  if (!f) return
  importError.value = null
  importMessages.value = []
  try {
    const report = wb.importExperiment(await f.text())
    importMessages.value = report.messages
    wb.notify('success', `Imported experiment with ${wb.getTree().nodes.size} history node(s).`)
  } catch (err) {
    importError.value =
      err instanceof ExperimentValidationError
        ? err.message
        : `Import failed: ${(err as Error).message}`
  } finally {
    if (fileInput.value) fileInput.value.value = ''
  }
}
</script>

<template>
  <section class="panel card" aria-labelledby="hist-title" data-testid="history-panel">
    <div class="row">
      <h2 id="hist-title">History &amp; branches</h2>
      <span class="spacer" />
      <button
        type="button"
        class="btn btn-sm"
        :disabled="!wb.canUndo.value"
        title="Undo (Ctrl+Z)"
        data-testid="undo"
        @click="wb.undo()"
      >
        ↶ Undo
      </button>
      <button
        type="button"
        class="btn btn-sm"
        :disabled="!wb.canRedo.value"
        title="Redo (Ctrl+Shift+Z)"
        data-testid="redo"
        @click="wb.redo()"
      >
        ↷ Redo
      </button>
    </div>
    <div class="row small">
      <span>Branch {{ branchInfo.index }} of {{ branchInfo.count }} at this step</span>
      <button
        type="button"
        class="btn btn-sm"
        :disabled="branchInfo.count < 2"
        aria-label="Previous branch"
        data-testid="branch-prev"
        @click="wb.switchBranch(-1)"
      >
        ◀
      </button>
      <button
        type="button"
        class="btn btn-sm"
        :disabled="branchInfo.count < 2"
        aria-label="Next branch"
        data-testid="branch-next"
        @click="wb.switchBranch(1)"
      >
        ▶
      </button>
      <span class="faint">Undo, then choose a different token to create an alternate branch.</span>
    </div>
    <ol class="tree" aria-label="History tree" data-testid="history-tree">
      <li v-for="r in rows" :key="r.node.id" :style="{ paddingLeft: r.depth * 0.9 + 'rem' }">
        <button
          type="button"
          class="node"
          :class="{ current: r.isCurrent, path: r.onCurrentPath }"
          :aria-current="r.isCurrent ? 'step' : undefined"
          :data-testid="`history-node-${r.node.id}`"
          @click="wb.goToNode(r.node.id)"
        >
          <span v-if="r.branchCount > 1" class="faint"
            >{{ r.branchIndex + 1 }}/{{ r.branchCount }}</span
          >
          <span class="lbl">{{ r.node.label }}</span>
          <code class="tok">{{ lastTokenText(r.node.tokens) }}</code>
          <span v-if="r.node.append?.adjustedProbability != null" class="faint"
            >p={{ formatProb(r.node.append.adjustedProbability) }}</span
          >
          <span class="faint">{{ r.node.tokens.length }} tok</span>
        </button>
      </li>
    </ol>
    <div class="row">
      <button type="button" class="btn btn-sm" data-testid="export" @click="exportFile">
        Export experiment (JSON)
      </button>
      <label class="btn btn-sm">
        Import experiment…
        <input
          ref="fileInput"
          type="file"
          accept="application/json,.json"
          class="sr-only"
          data-testid="import-input"
          @change="importFile"
        />
      </label>
    </div>
    <p
      v-if="importError"
      class="alert alert-error small"
      role="alert"
      data-testid="import-error"
      style="white-space: pre-wrap"
    >
      {{ importError }}
    </p>
    <ul
      v-if="importMessages.length"
      class="alert alert-warning small"
      data-testid="import-messages"
    >
      <li v-for="m in importMessages" :key="m">{{ m }}</li>
    </ul>
  </section>
</template>

<style scoped>
.panel {
  padding: 0.8rem;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}
.tree {
  list-style: none;
  margin: 0;
  padding: 0.3rem;
  max-height: 16rem;
  overflow: auto;
  background: var(--surface-2);
  border-radius: var(--radius-sm);
}
.node {
  display: flex;
  gap: 0.5rem;
  align-items: baseline;
  width: 100%;
  text-align: left;
  border: 1px solid transparent;
  background: transparent;
  border-radius: 4px;
  padding: 0.1rem 0.35rem;
  font-size: 0.82rem;
}
.node:hover {
  background: var(--surface-3);
}
.node.path {
  font-weight: 600;
}
.node.current {
  border-color: var(--accent);
  background: var(--accent-soft);
}
.lbl {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.tok {
  white-space: pre;
}
label.btn:focus-within {
  outline: 2px solid var(--focus);
}
ul.alert {
  margin: 0;
  padding-left: 1.6rem;
}
</style>
