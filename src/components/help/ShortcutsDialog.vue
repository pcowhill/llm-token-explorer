<script setup lang="ts">
import { nextTick, ref, watch } from 'vue'
import { SHORTCUT_ACTIONS, keyLabel, useShortcuts } from '@/composables/useShortcuts'

const props = defineProps<{ open: boolean }>()
const emit = defineEmits<{ close: []; settings: [] }>()
const { bindings } = useShortcuts()
const dialog = ref<HTMLDialogElement | null>(null)
watch(
  () => props.open,
  async (o) => {
    await nextTick()
    const d = dialog.value
    if (!d) return
    if (o && !d.open) {
      if (typeof d.showModal === 'function') d.showModal()
      else d.setAttribute('open', '')
    }
    if (!o && d.open) {
      if (typeof d.close === 'function') d.close()
      else d.removeAttribute('open')
    }
  },
)
function openSettings() {
  emit('close')
  emit('settings')
}
</script>

<template>
  <dialog
    ref="dialog"
    class="shortcuts card"
    aria-labelledby="shortcuts-title"
    data-testid="shortcuts-dialog"
    @close="emit('close')"
    @cancel="emit('close')"
  >
    <div class="row head">
      <h2 id="shortcuts-title">Keyboard shortcuts</h2>
      <span class="spacer" />
      <button type="button" class="btn btn-sm" @click="emit('close')">Close</button>
    </div>
    <div class="body">
      <table class="data">
        <caption class="small muted">
          Single-key shortcuts work anywhere except while typing in a text field. Hold a key to
          repeat: each new token waits for its forward pass.
        </caption>
        <thead>
          <tr>
            <th scope="col">Keys</th>
            <th scope="col">Action</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="a in SHORTCUT_ACTIONS" :key="a.id" :data-testid="`shortcut-row-${a.id}`">
            <td>
              <kbd>{{ keyLabel(bindings[a.id]) }}</kbd>
            </td>
            <td>{{ a.label }} (hold to repeat)</td>
          </tr>
          <tr>
            <td><kbd>Alt</kbd>+<kbd>S</kbd></td>
            <td>Sample one token</td>
          </tr>
          <tr>
            <td><kbd>Ctrl</kbd>+<kbd>Enter</kbd></td>
            <td>Calculate the next-token distribution (also in text fields)</td>
          </tr>
          <tr>
            <td><kbd>Ctrl</kbd>+<kbd>Z</kbd></td>
            <td>Undo</td>
          </tr>
          <tr>
            <td><kbd>Ctrl</kbd>+<kbd>Y</kbd> or <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>Z</kbd></td>
            <td>Redo</td>
          </tr>
          <tr>
            <td><kbd>←</kbd> <kbd>→</kbd> <kbd>Home</kbd> <kbd>End</kbd> <kbd>Delete</kbd></td>
            <td>Move between / remove token chips (when a chip has focus)</td>
          </tr>
          <tr>
            <td><kbd>?</kbd></td>
            <td>Show this list</td>
          </tr>
          <tr>
            <td><kbd>Esc</kbd></td>
            <td>Close dialogs</td>
          </tr>
        </tbody>
      </table>
      <p class="small">
        On macOS use <kbd>⌘</kbd> instead of <kbd>Ctrl</kbd>.
        <button
          type="button"
          class="btn btn-sm"
          data-testid="shortcuts-remap"
          @click="openSettings"
        >
          Change keys in Settings…
        </button>
      </p>
    </div>
  </dialog>
</template>

<style scoped>
.shortcuts {
  width: min(34rem, 94vw);
  padding: 0;
  color: var(--text);
  background: var(--surface);
}
.shortcuts::backdrop {
  background: rgb(0 0 0 / 45%);
}
.head {
  padding: 0.7rem 1rem;
  border-bottom: 1px solid var(--border);
}
.body {
  padding: 0.6rem 1rem 1rem;
}
caption {
  text-align: left;
  padding-bottom: 0.4rem;
}
</style>
