<script setup lang="ts">
import { nextTick, ref, watch } from 'vue'
import { useWorkbench } from '@/composables/useWorkbench'

const props = defineProps<{ open: boolean; persistSession: boolean }>()
const emit = defineEmits<{ close: []; 'update:persistSession': [boolean]; clearSaved: [] }>()
const wb = useWorkbench()
const dialog = ref<HTMLDialogElement | null>(null)
watch(
  () => props.open,
  async (o) => {
    await nextTick()
    if (!dialog.value) return
    if (o && !dialog.value.open) {
      if (typeof dialog.value.showModal === 'function') dialog.value.showModal()
      else dialog.value.setAttribute('open', '')
    }
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
</style>
