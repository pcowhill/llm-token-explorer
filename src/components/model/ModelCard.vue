<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { Backend, CatalogModel, ModelDtype } from '@/types'
import { DTYPE_DESCRIPTIONS, SUPPORT_FILES, fileUrl, formatBytes, repoUrl } from '@/models/catalog'
import { useWorkbench } from '@/composables/useWorkbench'
import { chooseDtype } from '@/stores/workbench'

const props = defineProps<{ model: CatalogModel; backend: Backend; cached: boolean }>()
const wb = useWorkbench()
const dtype = ref<ModelDtype>(chooseDtype(props.model, props.backend, wb.capabilities.value))
watch(
  () => [props.backend, wb.capabilities.value] as const,
  () => (dtype.value = chooseDtype(props.model, props.backend, wb.capabilities.value)),
)
const variant = computed(
  () => props.model.variants.find((v) => v.dtype === dtype.value) ?? props.model.variants[0]!,
)
const supported = computed(() => variant.value.backends.includes(props.backend))
const f16Problem = computed(
  () =>
    props.backend === 'webgpu' &&
    dtype.value.includes('16') &&
    wb.capabilities.value?.webgpuAdapter?.shaderF16 === false,
)
const memEstimate = computed(() => props.model.memoryEstimateBytes[props.backend])
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
  if (variant.value.sizeBytes > 400e6)
    warnings.push(`This is a large download (${formatBytes(variant.value.sizeBytes)}).`)
  return warnings
})
const confirming = ref(false)
const isActive = computed(
  () => wb.activeCatalogKey.value === props.model.key && wb.modelStatus.value === 'ready',
)
const busy = computed(() => wb.modelStatus.value === 'loading')

const verificationClass = computed(
  () =>
    ({
      verified: 'badge-success',
      partial: 'badge-accent',
      documented: 'badge-warning',
      experimental: 'badge-danger',
    })[props.model.verification.status],
)

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
</script>

<template>
  <article
    class="model-card card"
    :class="{ active: isActive }"
    :data-testid="`model-card-${props.model.key}`"
  >
    <header class="row">
      <h3>{{ props.model.name }}</h3>
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
      <span v-if="props.cached" class="badge badge-accent">Cached</span>
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
        <dd>
          ≈ {{ formatBytes(memEstimate) }} on {{ props.backend === 'webgpu' ? 'WebGPU' : 'WASM' }}
        </dd>
      </div>
    </dl>
    <div class="row small">
      <label>
        Quantization
        <select v-model="dtype" :aria-label="`Quantization for ${props.model.name}`">
          <option v-for="v in props.model.variants" :key="v.dtype" :value="v.dtype">
            {{ v.dtype }} — {{ formatBytes(v.sizeBytes) }} ({{ v.backends.join(', ') }})
          </option>
        </select>
      </label>
      <span class="faint">{{ DTYPE_DESCRIPTIONS[dtype] }}</span>
    </div>
    <p v-if="!supported" class="alert alert-warning small">
      {{ dtype }} is not recommended on {{ props.backend }}; choose another quantization or backend.
    </p>
    <p v-if="f16Problem" class="alert alert-warning small">
      This GPU adapter does not support <code>shader-f16</code>, which {{ dtype }} requires. Choose
      q8 or fp32, or use WASM.
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
        :disabled="busy"
        :data-testid="`load-${props.model.key}`"
        @click="load()"
      >
        {{ isActive ? 'Reload' : 'Load model' }}
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
select {
  max-width: 100%;
}
</style>
