<script setup lang="ts">
import { nextTick, ref, watch } from 'vue'
import { useWorkbench } from '@/composables/useWorkbench'
import {
  SHORTCUT_ACTIONS,
  keyLabel,
  useShortcuts,
  type ShortcutAction,
} from '@/composables/useShortcuts'

const props = defineProps<{ open: boolean; persistSession: boolean }>()
const emit = defineEmits<{
  close: []
  'update:persistSession': [boolean]
  clearSaved: []
  shortcuts: []
}>()
const wb = useWorkbench()
const dialog = ref<HTMLDialogElement | null>(null)

// ------------------------------------------------------ shortcut remapping
const shortcuts = useShortcuts()
const waitingFor = ref<ShortcutAction | null>(null)
const keyError = ref('')
const keyStatus = ref('')
function startCapture(action: ShortcutAction) {
  waitingFor.value = action
  shortcuts.capturing.value = true
  keyError.value = ''
  keyStatus.value = 'Press the new key (Esc cancels).'
}
function stopCapture() {
  waitingFor.value = null
  shortcuts.capturing.value = false
}
function onCaptureKey(e: KeyboardEvent, action: ShortcutAction) {
  if (waitingFor.value !== action) return
  if (e.key === 'Tab') {
    stopCapture()
    keyStatus.value = ''
    return
  }
  e.preventDefault()
  // Keep Escape from closing the dialog while a key is being captured.
  e.stopPropagation()
  if (e.key === 'Escape') {
    stopCapture()
    keyStatus.value = 'Unchanged.'
    return
  }
  if (['Shift', 'Control', 'Alt', 'Meta', 'CapsLock', 'AltGraph'].includes(e.key)) return
  if (e.ctrlKey || e.metaKey || e.altKey) {
    keyError.value = 'Use a single key without Ctrl, Alt or ⌘.'
    return
  }
  const err = shortcuts.setBinding(action, e.key)
  if (err) {
    keyError.value = err
    return
  }
  stopCapture()
  keyError.value = ''
  keyStatus.value = `${SHORTCUT_ACTIONS.find((a) => a.id === action)!.label}: ${keyLabel(e.key)}.`
}
function onCaptureBlur(action: ShortcutAction) {
  if (waitingFor.value === action) stopCapture()
}
function resetKeys() {
  shortcuts.resetBindings()
  stopCapture()
  keyError.value = ''
  keyStatus.value = 'Shortcuts reset to S and G.'
}
watch(
  () => props.open,
  async (o) => {
    await nextTick()
    if (!dialog.value) return
    if (o && !dialog.value.open) {
      if (typeof dialog.value.showModal === 'function') dialog.value.showModal()
      else dialog.value.setAttribute('open', '')
    }
    if (!o) stopCapture()
    if (!o && dialog.value.open) {
      if (typeof dialog.value.close === 'function') dialog.value.close()
      else dialog.value.removeAttribute('open')
    }
  },
)
</script>

<template>
  <dialog
    ref="dialog"
    class="settings card"
    aria-labelledby="settings-title"
    @close="emit('close')"
    @cancel="emit('close')"
  >
    <div class="row head">
      <h2 id="settings-title">Settings</h2>
      <span class="spacer" />
      <button type="button" class="btn btn-sm" @click="emit('close')">Close</button>
    </div>
    <div class="stack body">
      <label class="row"
        ><input v-model="wb.generation.autoInfer" type="checkbox" /> Automatically run inference
        after edits</label
      >
      <label class="row"
        >Debounce interval
        <input
          v-model.number="wb.generation.debounceMs"
          type="number"
          min="0"
          max="5000"
          step="50"
          class="num"
        />
        ms</label
      >
      <label class="row"
        ><input v-model="wb.generation.useKvCache" type="checkbox" /> Reuse the KV cache when the
        context grows (faster; see Diagnostics for the exactness caveat)</label
      >
      <label class="row">
        Default backend
        <select v-model="wb.backendPref.value">
          <option value="auto">Automatic</option>
          <option value="webgpu">WebGPU</option>
          <option value="wasm">WASM / CPU</option>
        </select>
      </label>
      <label class="row">
        <input
          type="checkbox"
          :checked="props.persistSession"
          @change="emit('update:persistSession', ($event.target as HTMLInputElement).checked)"
        />
        Remember my context, history and settings on this device (localStorage)
      </label>
      <div class="row">
        <button type="button" class="btn btn-sm btn-danger" @click="emit('clearSaved')">
          Clear saved session data
        </button>
      </div>
      <p class="small muted">
        Saved data never includes model weights and never leaves this browser.
      </p>
      <fieldset class="keys" data-testid="shortcut-settings">
        <legend>Keyboard shortcuts</legend>
        <p class="small muted">
          Single keys, inactive while typing in a text field. Hold a key to repeat.
        </p>
        <div v-for="a in SHORTCUT_ACTIONS" :key="a.id" class="row small">
          <span :id="`key-label-${a.id}`" class="key-label">{{ a.label }}</span>
          <kbd :data-testid="`shortcut-key-${a.id}`">{{
            keyLabel(shortcuts.bindings.value[a.id])
          }}</kbd>
          <button
            type="button"
            class="btn btn-sm"
            :aria-describedby="`key-label-${a.id}`"
            :aria-pressed="waitingFor === a.id"
            :data-testid="`remap-${a.id}`"
            @click="startCapture(a.id)"
            @keydown="onCaptureKey($event, a.id)"
            @blur="onCaptureBlur(a.id)"
          >
            {{ waitingFor === a.id ? 'Press a key…' : 'Change' }}
          </button>
        </div>
        <p
          v-if="keyError"
          class="alert alert-error small"
          role="alert"
          data-testid="shortcut-error"
        >
          {{ keyError }}
        </p>
        <p v-else class="small faint" aria-live="polite" data-testid="shortcut-status">
          {{ keyStatus }}
        </p>
        <div class="row">
          <button type="button" class="btn btn-sm" data-testid="reset-shortcuts" @click="resetKeys">
            Reset to defaults
          </button>
          <button type="button" class="btn btn-sm" @click="emit('shortcuts')">
            Show all shortcuts (?)
          </button>
        </div>
      </fieldset>
    </div>
  </dialog>
</template>

<style scoped>
.settings {
  width: min(34rem, 94vw);
  padding: 0;
  color: var(--text);
  background: var(--surface);
}
.settings::backdrop {
  background: rgb(0 0 0 / 45%);
}
.head {
  padding: 0.7rem 1rem;
  border-bottom: 1px solid var(--border);
}
.body {
  padding: 0.8rem 1rem 1rem;
}
.num {
  width: 5.5rem;
}
.keys {
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 0.5rem 0.7rem;
}
.keys legend {
  font-weight: 600;
  font-size: 0.88rem;
}
.key-label {
  min-width: 17rem;
}
</style>
