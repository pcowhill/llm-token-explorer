<script setup lang="ts">
import { computed } from 'vue'
import { useWorkbench } from '@/composables/useWorkbench'
import {
  CATEGORY_LABELS,
  categorizeToken,
  describeCodePoints,
  escapeLiteral,
  formatBytes,
  isIncompleteUtf8,
  tokenBytes,
} from '@/tokenization/display'
import { pathTo } from '@/history/tree'
import { formatLogit, formatProb } from '@/utils/format'

const props = defineProps<{ index: number }>()
const emit = defineEmits<{ close: []; insertBefore: [number] }>()
const wb = useWorkbench()

const id = computed(() => wb.tokens.value[props.index])
const info = computed(() => {
  const tok = wb.tokenizer.value
  const tid = id.value
  if (tid === undefined) return null
  const raw = tok?.raw[tid] ?? ''
  const decoded = tok?.decoded[tid] ?? ''
  const special = !!tok?.specialIds.includes(tid)
  const added = !!tok?.addedIds.includes(tid)
  const scheme = tok?.scheme ?? 'unknown'
  const bytes = special ? null : tokenBytes(raw, scheme)
  const st = tok?.specialTokens.find((s) => s.id === tid)
  return {
    raw,
    decoded,
    special,
    added,
    bytes,
    partial: bytes ? isIncompleteUtf8(bytes) : false,
    category: tok ? categorizeToken({ raw, decoded, special, added, scheme }) : null,
    codepoints: describeCodePoints(decoded),
    specialInfo: st,
    scheme,
    outOfRange: !!wb.modelInfo.value && tid >= wb.modelInfo.value.logitsSize,
  }
})

function removeThis() {
  wb.removeToken(props.index)
  emit('close')
}

/** If this token was appended by sampling / picking, show the probabilities recorded then. */
const appendRecord = computed(() => {
  void wb.historyVersion.value
  const tree = wb.getTree()
  const path = pathTo(tree, tree.currentId)
  const node = path.find((n) => n.kind === 'append' && n.tokens.length === props.index + 1)
  return node?.append && node.append.tokenId === id.value ? node.append : null
})
</script>

<template>
  <aside
    v-if="info && id !== undefined"
    class="inspector card"
    aria-labelledby="inspector-title"
    data-testid="token-inspector"
  >
    <div class="row">
      <h3 id="inspector-title">Token at position {{ props.index }}</h3>
      <span class="spacer" />
      <button
        type="button"
        class="btn btn-sm"
        aria-label="Close token inspector"
        @click="emit('close')"
      >
        ✕
      </button>
    </div>
    <dl class="grid">
      <dt>Token ID</dt>
      <dd>
        <code data-testid="inspector-id">{{ id }}</code>
      </dd>
      <dt>Position</dt>
      <dd>{{ props.index }} of {{ wb.tokens.value.length }} (0-based)</dd>
      <dt>Decoded text</dt>
      <dd>
        <code data-testid="inspector-decoded">{{ escapeLiteral(info.decoded) }}</code>
      </dd>
      <dt>Raw token</dt>
      <dd>
        <code>{{ info.raw ? escapeLiteral(info.raw) : '(no tokenizer entry)' }}</code>
        <span v-if="info.scheme === 'byte-level-bpe' && !info.special" class="small faint">
          byte-level BPE: Ġ = space, Ċ = newline</span
        >
        <span v-else-if="info.scheme === 'sentencepiece' && !info.special" class="small faint">
          SentencePiece: ▁ = space</span
        >
      </dd>
      <dt>Special token</dt>
      <dd>
        <span v-if="info.special" class="badge badge-special">special</span>
        <span v-else>no</span>
        <span v-if="info.specialInfo" class="small">
          · {{ info.specialInfo.roles.join(', ') }} — {{ info.specialInfo.purpose }}</span
        >
      </dd>
      <dt>Category</dt>
      <dd>{{ info.category ? CATEGORY_LABELS[info.category] : '—' }}</dd>
      <template v-if="info.bytes">
        <dt>UTF-8 bytes</dt>
        <dd>
          <code>{{ formatBytes(info.bytes) }}</code>
          <span v-if="info.partial" class="badge badge-warning"
            >incomplete UTF-8 — combines with neighbouring tokens</span
          >
        </dd>
      </template>
      <template v-if="appendRecord">
        <dt>When appended</dt>
        <dd class="small">
          {{ appendRecord.source }} · original p =
          {{ formatProb(appendRecord.originalProbability) }} · adjusted p =
          {{ formatProb(appendRecord.adjustedProbability) }} · logit
          {{ formatLogit(appendRecord.logit) }}
        </dd>
      </template>
    </dl>
    <p v-if="info.outOfRange" class="alert alert-error small">
      This ID is outside the loaded model's input range.
    </p>
    <details v-if="info.codepoints.length" class="small">
      <summary>Characters ({{ info.codepoints.length }})</summary>
      <table class="data">
        <thead>
          <tr>
            <th>Shown</th>
            <th>Code point</th>
            <th>Category</th>
            <th>Name</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(c, k) in info.codepoints" :key="k">
            <td>
              <code>{{ c.visible }}</code>
            </td>
            <td>
              <code>{{ c.codePoint }}</code>
            </td>
            <td>{{ c.category }}</td>
            <td>{{ c.name }}</td>
          </tr>
        </tbody>
      </table>
    </details>
    <div class="row">
      <button type="button" class="btn btn-sm" @click="emit('insertBefore', props.index)">
        Insert before…
      </button>
      <button
        type="button"
        class="btn btn-sm btn-danger"
        data-testid="inspector-remove"
        @click="removeThis"
      >
        Remove token
      </button>
    </div>
  </aside>
</template>

<style scoped>
.inspector {
  padding: 0.7rem;
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}
.grid {
  display: grid;
  grid-template-columns: max-content 1fr;
  gap: 0.25rem 0.8rem;
  margin: 0;
  font-size: 0.88rem;
}
.grid dt {
  color: var(--text-muted);
}
.grid dd {
  margin: 0;
  word-break: break-all;
}
</style>
