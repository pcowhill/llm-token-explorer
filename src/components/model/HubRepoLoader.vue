<script setup lang="ts">
import { computed, onBeforeUnmount, ref, shallowRef, watch } from 'vue'
import type { Backend, ModelDtype } from '@/types'
import { useWorkbench } from '@/composables/useWorkbench'
import { CATALOG, DTYPE_DESCRIPTIONS, formatBytes, notVerifiedInCi } from '@/models/catalog'
import type { CachedModel } from '@/models/cacheStorage'
import { HubApiError, REPO_ID_RE, fetchRepoInfo, type HubRepoInfo } from '@/models/hubApi'
import {
  DTYPE_ORDER,
  recommendDtype,
  repoCacheStatus,
  variantsFromFiles,
  type FileVariant,
} from '@/models/variants'

const props = defineProps<{ backend: Backend; cached: CachedModel[] }>()
const wb = useWorkbench()

const repo = ref('')
const revision = ref('main')
const dtype = ref<ModelDtype | ''>('')
const userChoseDtype = ref(false)
const info = shallowRef<HubRepoInfo | null>(null)
const error = shallowRef<HubApiError | null>(null)
const checking = ref(false)
let controller: AbortController | null = null
let timer: ReturnType<typeof setTimeout> | null = null

const repoId = computed(() => repo.value.trim())
const rev = computed(() => revision.value.trim() || 'main')
const validId = computed(() => REPO_ID_RE.test(repoId.value))

async function check() {
  controller?.abort()
  info.value = null
  error.value = null
  if (!validId.value) return
  const c = new AbortController()
  controller = c
  checking.value = true
  try {
    const r = await fetchRepoInfo(repoId.value, rev.value, { signal: c.signal })
    if (controller !== c) return
    info.value = r
  } catch (e) {
    if (controller !== c || (e as Error).name === 'AbortError') return
    error.value = e instanceof HubApiError ? e : new HubApiError((e as Error).message, 'network')
  } finally {
    if (controller === c) checking.value = false
  }
}
watch([repoId, rev], () => {
  if (timer) clearTimeout(timer)
  userChoseDtype.value = false
  timer = setTimeout(() => void check(), 450)
})
onBeforeUnmount(() => {
  if (timer) clearTimeout(timer)
  controller?.abort()
})

/** Variants to offer: from the API; offline/on errors, whatever is cached for this repo. */
const cachedEntry = computed(() =>
  props.cached.find((c) => c.repo === repoId.value && c.revision === rev.value),
)
/** Only connectivity problems fall back; a missing, gated or mistyped repo cannot be loaded. */
const fallback = computed(
  () => !info.value && (error.value?.kind === 'offline' || error.value?.kind === 'network'),
)
const variants = computed<FileVariant[]>(() => {
  if (info.value) return info.value.variants
  if (!fallback.value) return []
  if (cachedEntry.value) {
    const fromCache = variantsFromFiles(
      cachedEntry.value.files.map((f) => ({ path: f.path, size: f.bytes })),
    )
    if (fromCache.length) return fromCache
  }
  // Unknown files: offer every dtype so loading can still be attempted.
  return DTYPE_ORDER.map((d) => ({ dtype: d, files: [], sizeBytes: 0, externalDataChunks: 0 }))
})
const cacheStatus = computed(() =>
  repoCacheStatus(props.cached, repoId.value, rev.value, variants.value),
)
const recommended = computed(() =>
  info.value
    ? recommendDtype(
        variants.value,
        props.backend,
        !!wb.capabilities.value?.webgpuAdapter?.shaderF16,
      )
    : null,
)
watch(
  [variants, recommended],
  () => {
    if (userChoseDtype.value && variants.value.some((v) => v.dtype === dtype.value)) return
    dtype.value = recommended.value?.dtype ?? variants.value[0]?.dtype ?? ''
  },
  { immediate: true },
)
const selected = computed(() => variants.value.find((v) => v.dtype === dtype.value) ?? null)
const selectedCached = computed(
  () => !!selected.value && cacheStatus.value.dtypes[selected.value.dtype] === 'cached',
)

const SUPPORT = ['config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json']
const supportBytes = computed(() =>
  (info.value?.files ?? [])
    .filter((f) => SUPPORT.includes(f.path))
    .reduce((s, f) => s + (f.size ?? 0), 0),
)
/** Bytes still to download (0 when unknown, e.g. offline). */
function missingBytes(v: FileVariant): number {
  if (!v.sizeBytes) return 0
  return (
    (cacheStatus.value.weightsCached[v.dtype] ? 0 : v.sizeBytes) +
    (cacheStatus.value.supportCached ? 0 : supportBytes.value)
  )
}
function optionLabel(v: FileVariant): string {
  const state = cacheStatus.value.dtypes[v.dtype]
  const mark =
    state === 'cached'
      ? '✓ cached'
      : v.sizeBytes
        ? `↓ ${formatBytes(missingBytes(v))}${state === 'partial' ? ' (partly cached)' : ''}`
        : 'size unknown'
  const notes: string[] = []
  if (v.externalDataChunks) notes.push(`${v.files.length} files`)
  if (recommended.value?.dtype === v.dtype) notes.push('recommended')
  if (notVerifiedInCi(v.dtype, props.backend)) notes.push('needs shader-f16')
  return `${v.dtype} — ${mark}${notes.length ? ` · ${notes.join(' · ')}` : ''}`
}

const blocked = computed(() => !!info.value?.errors.length)
/** A curated catalog model typed here: point to its card, which knows what really runs. */
const catalogMatch = computed(() => CATALOG.find((m) => m.repo === repoId.value) ?? null)
const loadLabel = computed(() => {
  if (selectedCached.value) return 'Load model (cached)'
  if (selected.value?.sizeBytes)
    return `Download & load (${formatBytes(missingBytes(selected.value))})`
  return 'Load model'
})

function load() {
  if (!validId.value) {
    wb.notify('error', 'Enter a repository id like "owner/name".')
    return
  }
  const v = selected.value
  if (!v) return
  void wb.loadModel(
    {
      kind: 'hub',
      repo: repoId.value,
      revision: rev.value,
      displayName: repoId.value,
      ...(v.sizeBytes ? { expectedBytes: v.sizeBytes } : {}),
      ...(v.externalDataChunks ? { externalDataChunks: v.externalDataChunks } : {}),
    },
    props.backend,
    v.dtype,
  )
}

function onDtypeChange() {
  userChoseDtype.value = true
}
</script>

<template>
  <section class="stack" aria-labelledby="hub-title" data-testid="hub-loader">
    <h3 id="hub-title">Load any Hugging Face ONNX repository</h3>
    <div class="alert alert-info small hub-guide" data-testid="hub-guide">
      <p>
        <strong>Where to find models.</strong> The
        <a href="https://huggingface.co/onnx-community" target="_blank" rel="noopener noreferrer"
          >onnx-community</a
        >
        organization publishes ready-to-use Transformers.js exports of many small language models —
        start there.
      </p>
      <p>
        <strong>What makes a repo compatible.</strong> ONNX weights at
        <code>onnx/model*.onnx</code> (one file per quantization, e.g. <code>model_q4.onnx</code>,
        optionally with <code>.onnx_data</code> chunks), plus <code>tokenizer.json</code>,
        <code>tokenizer_config.json</code> and <code>config.json</code> with a decoder-only
        <code>model_type</code> that Transformers.js supports (llama, qwen2, qwen3, gemma3_text,
        phi3, gpt2, …). The available files are checked as you type.
      </p>
      <p>
        <strong>Revision.</strong> <code>main</code> is the latest version. For reproducible results
        paste a commit SHA from the repo’s <em>Files and versions → History</em> page; branch and
        tag names work too.
      </p>
      <p>
        <strong>Which dtype.</strong> On <strong>WebGPU</strong>: fp32 or q8 for accuracy, q4 for
        size; fp16 and q4f16 need the <code>shader-f16</code> GPU feature. On <strong>WASM</strong>:
        q8, q4 or bnb4, or fp16 for small models (WASM expands fp16 to fp32 in memory). The most
        accurate variant that fits is preselected.
      </p>
      <p>
        <strong>Size and memory.</strong> WASM runs in a 32-bit memory space: plan for well under ~1
        GB of weights. WebGPU is limited by GPU memory and buffer limits; multi-GB models also need
        that much free disk for the browser cache. Large downloads can take minutes.
      </p>
      <p class="faint">
        Not curated: this app has not tested arbitrary repositories. If the probabilities look
        wrong, try another dtype or backend.
      </p>
    </div>
    <div class="row small">
      <label
        >Repository
        <input
          v-model="repo"
          type="text"
          placeholder="owner/name"
          spellcheck="false"
          data-testid="hub-repo"
      /></label>
      <label
        >Revision
        <input
          v-model="revision"
          type="text"
          size="12"
          spellcheck="false"
          data-testid="hub-revision"
      /></label>
      <label>
        dtype
        <select
          v-model="dtype"
          :disabled="!variants.length"
          data-testid="hub-dtype"
          @change="onDtypeChange"
        >
          <option v-if="!variants.length" value="" disabled>
            {{ checking ? 'checking…' : 'enter a repository' }}
          </option>
          <option v-for="v in variants" :key="v.dtype" :value="v.dtype">
            {{ optionLabel(v) }}
          </option>
        </select>
      </label>
      <button
        type="button"
        class="btn btn-primary"
        :disabled="wb.modelStatus.value === 'loading' || !selected || blocked || !validId"
        data-testid="hub-load"
        @click="load"
      >
        {{ loadLabel }}
      </button>
    </div>
    <p v-if="dtype" class="small faint">{{ DTYPE_DESCRIPTIONS[dtype] }}</p>
    <p v-if="catalogMatch" class="small alert alert-info" data-testid="hub-catalog-match">
      This is <strong>{{ catalogMatch.name }}</strong> from the Recommended tab, where its variants
      are verified per backend (pinned to <code>{{ catalogMatch.revision.slice(0, 7) }}</code
      >).
      <template v-if="catalogMatch.backendNotes?.[props.backend]">
        {{ catalogMatch.backendNotes[props.backend] }}</template
      >
    </p>
    <p v-if="repoId && !validId" class="small alert alert-warning" data-testid="hub-status">
      Enter a repository id like <code>onnx-community/SmolLM2-135M-Instruct-ONNX</code>.
    </p>
    <p v-else-if="checking" class="small faint" role="status" data-testid="hub-status">
      Checking the files of {{ repoId }}…
    </p>
    <div
      v-else-if="error"
      class="small alert"
      :class="
        error.kind === 'offline' || error.kind === 'network' ? 'alert-warning' : 'alert-error'
      "
      role="status"
      data-testid="hub-status"
    >
      {{ error.message }}
      <template v-if="error.kind === 'offline' || error.kind === 'network'">
        {{
          cachedEntry
            ? 'Showing the quantizations already in the browser cache.'
            : 'Showing every dtype; loading works only if the files exist.'
        }}
      </template>
    </div>
    <div v-else-if="info" class="small" role="status" data-testid="hub-status">
      <p>
        <strong>{{ info.repo }}</strong>
        <template v-if="info.sha">
          @ <code>{{ info.sha.slice(0, 7) }}</code></template
        >
        <template v-if="info.modelType">
          · model_type <code>{{ info.modelType }}</code></template
        >
        · {{ info.variants.length }} quantization{{
          info.variants.length === 1 ? '' : 's'
        }}
        found<template v-if="recommended">
          · recommended for {{ props.backend === 'webgpu' ? 'WebGPU' : 'WASM' }}:
          <strong>{{ recommended.dtype }}</strong></template
        >
      </p>
      <p v-for="e in info.errors" :key="e" class="alert alert-error">{{ e }}</p>
      <p v-for="w in info.warnings" :key="w" class="alert alert-warning">{{ w }}</p>
      <p
        v-if="info.variants.length && !recommended && !info.errors.length"
        class="alert alert-warning"
      >
        No quantization is likely to fit {{ props.backend === 'webgpu' ? 'WebGPU' : 'WASM' }}; it
        may run out of memory.
      </p>
    </div>
  </section>
</template>

<style scoped>
.hub-guide p {
  margin: 0.35em 0;
}
</style>
