<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { Backend, CatalogModel, ModelDtype, ModelVariant } from '@/types'
import {
  DTYPE_DESCRIPTIONS,
  SUPPORT_FILES,
  fileUrl,
  formatBytes,
  notVerifiedInCi,
  repoUrl,
  supportsBackend,
} from '@/models/catalog'
import type { CachedModel } from '@/models/cacheStorage'
import { repoCacheStatus } from '@/models/variants'
import { useWorkbench } from '@/composables/useWorkbench'
import { chooseDtype } from '@/stores/workbench'

const props = defineProps<{ model: CatalogModel; backend: Backend; cached: CachedModel[] }>()
const wb = useWorkbench()
const dtype = ref<ModelDtype>(chooseDtype(props.model, props.backend, wb.capabilities.value))
/** Set once the user picks a quantization; late capability detection must not override it. */
const userChoseDtype = ref(false)
watch(
  () => wb.capabilities.value,
  () => {
    if (!userChoseDtype.value)
      dtype.value = chooseDtype(props.model, props.backend, wb.capabilities.value)
  },
)
watch(
  () => props.backend,
  () => {
    // A backend switch can make the chosen variant unsupported: re-pick the default.
    userChoseDtype.value = false
    dtype.value = chooseDtype(props.model, props.backend, wb.capabilities.value)
  },
)
const variant = computed(
  () => props.model.variants.find((v) => v.dtype === dtype.value) ?? props.model.variants[0]!,
)
const backendName = (b: Backend) => (b === 'webgpu' ? 'WebGPU' : 'WASM')
const runsHere = computed(() => supportsBackend(props.model, props.backend))
const supported = computed(() => variant.value.backends.includes(props.backend))
const f16Problem = computed(
  () =>
    props.backend === 'webgpu' &&
    dtype.value.includes('16') &&
    wb.capabilities.value?.webgpuAdapter?.shaderF16 === false,
)

// ------------------------------------------------------------- cache state
const cacheStatus = computed(() =>
  repoCacheStatus(props.cached, props.model.repo, props.model.revision, props.model.variants),
)
const variantCached = (v: ModelVariant) => cacheStatus.value.dtypes[v.dtype] === 'cached'
/** Bytes still to download for a variant (weights + support files not yet cached). */
const missingBytes = (v: ModelVariant) =>
  (cacheStatus.value.weightsCached[v.dtype] ? 0 : v.sizeBytes) +
  (cacheStatus.value.supportCached ? 0 : props.model.supportFilesBytes)
const selectedCached = computed(() => variantCached(variant.value))
const cachedBadge = computed(() => {
  const d = cacheStatus.value.cachedDtypes
  if (d.length) return `Cached: ${d.join(', ')}`
  return cacheStatus.value.supportCached ? 'Cached: tokenizer' : ''
})
function optionLabel(v: ModelVariant): string {
  const state = cacheStatus.value.dtypes[v.dtype]
  const mark =
    state === 'cached'
      ? '✓ cached'
      : `↓ ${formatBytes(missingBytes(v))}${state === 'partial' ? ' (partly cached)' : ''}`
  const notes: string[] = []
  if (!v.backends.includes(props.backend))
    notes.push(`${v.backends.map(backendName).join('/')} only`)
  else if (notVerifiedInCi(v.dtype, props.backend)) notes.push('not verified in CI')
  return `${v.dtype} — ${mark}${notes.length ? ` · ${notes.join(' · ')}` : ''}`
}

const memEstimate = computed(
  () => variant.value.memoryEstimateBytes ?? props.model.memoryEstimateBytes[props.backend],
)
const resourceWarning = computed(() => {
  const caps = wb.capabilities.value
  const warnings: string[] = []
  if (caps?.deviceMemoryGb && memEstimate.value > caps.deviceMemoryGb * 1024 ** 3 * 0.5)
    warnings.push(
      `Estimated memory (${formatBytes(memEstimate.value)}) is more than half of this device's reported RAM (~${caps.deviceMemoryGb} GB).`,
    )
  const maxBuf = caps?.webgpuAdapter?.maxBufferSize
  if (props.backend === 'webgpu' && maxBuf && variant.value.sizeBytes > maxBuf * 4)
    warnings.push(
      'The weights are large relative to this GPU adapter’s buffer limits; loading may fail.',
    )
  if (!selectedCached.value && variant.value.sizeBytes > 400e6)
    warnings.push(`This is a large download (${formatBytes(missingBytes(variant.value))}).`)
  return warnings
})
const excludedHere = computed(() =>
  (props.model.excludedVariants ?? []).filter((e) => e.backends.includes(props.backend)),
)

const confirming = ref(false)
const isActive = computed(
  () => wb.activeCatalogKey.value === props.model.key && wb.modelStatus.value === 'ready',
)
const busy = computed(() => wb.modelStatus.value === 'loading')
const loadLabel = computed(() => {
  if (!runsHere.value) return `Needs ${props.backend === 'wasm' ? 'WebGPU' : 'WASM'}`
  const info = wb.modelInfo.value
  if (isActive.value && info?.dtype === dtype.value && info.backend === props.backend)
    return 'Reload (cached)'
  return selectedCached.value
    ? 'Load model (cached)'
    : `Download & load (${formatBytes(missingBytes(variant.value))})`
})
const canSwitchToGpu = computed(
  () => props.backend === 'wasm' && !runsHere.value && !!wb.capabilities.value?.webgpu,
)

const verificationClass = computed(
  () =>
    ({
      verified: 'badge-success',
      partial: 'badge-accent',
      documented: 'badge-warning',
      experimental: 'badge-danger',
    })[props.model.verification.status],
)

function onDtypeChange() {
  userChoseDtype.value = true
}

function load(tokenizerOnly = false) {
  if (!tokenizerOnly && resourceWarning.value.length && !confirming.value) {
    confirming.value = true
    return
  }
  confirming.value = false
  void wb.loadCatalog(props.model.key, {
    dtype: dtype.value,
    backend: props.backend,
    tokenizerOnly,
  })
}

function switchToGpu() {
  wb.backendPref.value = 'webgpu'
}
</script>

<template>
  <article
    class="model-card card"
    :class="{ active: isActive, featured: !!props.model.featured }"
    :data-testid="`model-card-${props.model.key}`"
  >
    <header class="row">
      <h3>{{ props.model.name }}</h3>
      <span v-if="props.model.featured" class="badge badge-gold" data-testid="featured-badge">{{
        props.model.featured
      }}</span>
      <span class="badge" :class="verificationClass" :title="props.model.verification.summary">
        {{
          props.model.verification.status === 'verified'
            ? '✓ Verified (real inference)'
            : props.model.verification.status
        }}
      </span>
      <span class="badge">{{
        props.model.modelType === 'base' ? 'Base model' : 'Instruction-tuned'
      }}</span>
      <span v-if="cachedBadge" class="badge badge-accent" data-testid="cached-badge">{{
        cachedBadge
      }}</span>
      <span v-if="isActive" class="badge badge-success">Loaded</span>
    </header>
    <p class="small muted">{{ props.model.description }}</p>
    <dl class="facts small">
      <div>
        <dt>Publisher</dt>
        <dd>{{ props.model.publisher }}</dd>
      </div>
      <div>
        <dt>Parameters</dt>
        <dd>{{ props.model.parameters }}</dd>
      </div>
      <div>
        <dt>Architecture</dt>
        <dd>{{ props.model.architecture }}</dd>
      </div>
      <div>
        <dt>Context length</dt>
        <dd>{{ props.model.contextLength.toLocaleString() }} tokens</dd>
      </div>
      <div>
        <dt>Vocabulary</dt>
        <dd>
          {{ props.model.tokenizerSize.toLocaleString() }} tokenizer entries /
          {{ props.model.logitsSize.toLocaleString() }} logits
        </dd>
      </div>
      <div>
        <dt>License</dt>
        <dd>
          <a :href="props.model.license.url" target="_blank" rel="noopener noreferrer">{{
            props.model.license.name
          }}</a>
        </dd>
      </div>
      <div>
        <dt>Download</dt>
        <dd>
          ≈ {{ formatBytes(variant.sizeBytes) }} weights
          <span class="faint"
            >(+ {{ formatBytes(props.model.supportFilesBytes) }} tokenizer/config)</span
          >
        </dd>
      </div>
      <div>
        <dt>Memory (estimate)</dt>
        <dd>≈ {{ formatBytes(memEstimate) }} on {{ backendName(props.backend) }}</dd>
      </div>
    </dl>
    <p
      v-if="!runsHere"
      class="alert alert-info small"
      :data-testid="`backend-note-${props.model.key}`"
    >
      {{
        props.model.backendNotes?.[props.backend] ??
        `${props.model.name} does not run on ${backendName(props.backend)}.`
      }}
      <button
        v-if="canSwitchToGpu"
        type="button"
        class="btn btn-sm"
        data-testid="switch-to-webgpu"
        @click="switchToGpu"
      >
        Switch to WebGPU
      </button>
    </p>
    <div class="row small">
      <label>
        Quantization
        <select
          v-model="dtype"
          :aria-label="`Quantization for ${props.model.name}`"
          :data-testid="`dtype-${props.model.key}`"
          @change="onDtypeChange"
        >
          <option
            v-for="v in props.model.variants"
            :key="v.dtype"
            :value="v.dtype"
            :disabled="runsHere && !v.backends.includes(props.backend)"
          >
            {{ optionLabel(v) }}
          </option>
        </select>
      </label>
      <span class="faint">{{ DTYPE_DESCRIPTIONS[dtype] }}</span>
    </div>
    <p
      v-if="variant.fidelity || variant.note"
      class="small muted fidelity"
      data-testid="variant-fidelity"
    >
      <strong>Fidelity:</strong> {{ variant.fidelity ?? 'not measured' }}
      <template v-if="variant.note"> {{ variant.note }}</template>
    </p>
    <details
      v-if="excludedHere.length"
      class="small excluded"
      :data-testid="`excluded-${props.model.key}`"
    >
      <summary>
        Not offered on {{ backendName(props.backend) }}:
        {{ excludedHere.map((e) => e.dtype).join(', ') }} — why?
      </summary>
      <ul>
        <li v-for="e in excludedHere" :key="e.dtype">
          <strong>{{ e.dtype }}:</strong> {{ e.reason }}
        </li>
      </ul>
    </details>
    <p v-if="runsHere && !supported" class="alert alert-warning small">
      {{ dtype }} is not offered on {{ backendName(props.backend) }}; choose another quantization or
      backend.
    </p>
    <p v-if="f16Problem" class="alert alert-warning small">
      This GPU adapter does not support <code>shader-f16</code>, which {{ dtype }} requires. Choose
      a variant without “16”, or use WASM.
    </p>
    <div v-if="confirming" class="alert alert-warning small" role="alert">
      <p v-for="w in resourceWarning" :key="w">{{ w }}</p>
      <div class="row">
        <button type="button" class="btn btn-sm btn-primary" @click="load()">Load anyway</button>
        <button type="button" class="btn btn-sm" @click="confirming = false">Cancel</button>
      </div>
    </div>
    <div class="row">
      <button
        type="button"
        class="btn btn-primary"
        :disabled="busy || !runsHere"
        :data-testid="`load-${props.model.key}`"
        @click="load()"
      >
        {{ loadLabel }}
      </button>
      <button
        type="button"
        class="btn"
        :disabled="busy"
        :data-testid="`load-tokenizer-${props.model.key}`"
        @click="load(true)"
      >
        Tokenizer only
      </button>
    </div>
    <details class="small">
      <summary>Sources, files, verification &amp; limitations</summary>
      <ul class="links">
        <li>
          ONNX repository (pinned):
          <a
            :href="repoUrl(props.model.repo, props.model.revision)"
            target="_blank"
            rel="noopener noreferrer"
            >{{ props.model.repo }}</a
          >
          <code class="faint">@{{ props.model.revision.slice(0, 7) }}</code>
        </li>
        <li>
          Original model card:
          <a :href="repoUrl(props.model.upstreamRepo)" target="_blank" rel="noopener noreferrer">{{
            props.model.upstreamRepo
          }}</a>
        </li>
        <li>
          Required files:
          <span v-for="f in [...SUPPORT_FILES, ...variant.files]" :key="f">
            <a
              :href="fileUrl(props.model.repo, props.model.revision, f)"
              target="_blank"
              rel="noopener noreferrer"
              ><code>{{ f }}</code></a
            >&nbsp;
          </span>
        </li>
      </ul>
      <p><strong>Prompt format.</strong> {{ props.model.promptFormat }}</p>
      <p>
        <strong>Verification ({{ props.model.verification.date }}).</strong>
        {{ props.model.verification.summary }}
      </p>
      <p><strong>Limitations</strong></p>
      <ul>
        <li v-for="l in props.model.limitations" :key="l">{{ l }}</li>
      </ul>
    </details>
  </article>
</template>

<style scoped>
.model-card {
  padding: 0.75rem;
  display: flex;
  flex-direction: column;
  gap: 0.45rem;
}
.model-card.featured {
  border-color: color-mix(in srgb, var(--gold) 70%, var(--border));
  box-shadow: 0 0 0 1px color-mix(in srgb, var(--gold) 45%, transparent);
}
.model-card.active {
  border-color: var(--accent);
}
.facts {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(12rem, 1fr));
  gap: 0.2rem 0.8rem;
  margin: 0;
}
.facts dt {
  color: var(--text-faint);
  font-size: 0.75rem;
}
.facts dd {
  margin: 0;
}
.links {
  padding-left: 1.1rem;
}
.excluded ul {
  padding-left: 1.1rem;
  margin: 0.3rem 0 0;
}
select {
  max-width: 100%;
}
</style>
