<script setup lang="ts">
import { computed, ref } from 'vue'
import { useWorkbench } from '@/composables/useWorkbench'
import { catalogModel } from '@/models/catalog'
import type { PromptExample } from '@/types'
import TokenStrip from './TokenStrip.vue'
import TokenInspector from './TokenInspector.vue'
import TokenInsertBar from './TokenInsertBar.vue'
import SpecialTokenBrowser from './SpecialTokenBrowser.vue'
import InfoTip from '@/components/common/InfoTip.vue'
import { formatMs } from '@/utils/format'

const wb = useWorkbench()
const selected = ref<number | null>(null)
const insertAt = ref<number | null>(null)
const showIds = ref(true)
const tab = ref<'text' | 'chat' | 'special'>('text')

const examples = computed<PromptExample[]>(() => {
  const key = wb.activeCatalogKey.value
  const cat = key ? catalogModel(key) : undefined
  if (cat) return cat.examples
  const tok = wb.tokenizer.value
  if (!tok) return []
  const out: PromptExample[] = []
  const start = tok.bosTokenId ?? tok.eosTokenIds[0]
  if (start !== undefined && start !== null)
    out.push({
      id: 'minimal',
      label: 'Minimal: start token',
      description: 'A single BOS/EOS token.',
      kind: 'parts',
      parts: [{ tokenId: start }],
    })
  out.push({
    id: 'completion',
    label: 'Text completion',
    description: 'Plain text.',
    kind: 'text',
    text: 'The capital of France is',
  })
  out.push({
    id: 'qa',
    label: 'Question & answer',
    description: 'Plain-text Q/A.',
    kind: 'text',
    text: 'Q: What is the largest ocean?\nA:',
  })
  if (tok.hasChatTemplate)
    out.push({
      id: 'chat',
      label: 'Chat (chat template)',
      description: 'Rendered with the tokenizer’s chat template.',
      kind: 'chat',
      messages: [{ role: 'user', content: 'Hello! Who are you?' }],
      addGenerationPrompt: true,
    })
  return out
})
const exampleId = ref('')
function onExample() {
  const ex = examples.value.find((e) => e.id === exampleId.value)
  if (ex) void wb.loadExample(ex)
  selected.value = null
}

const ctxLen = computed(() => wb.modelInfo.value?.contextLength ?? null)
const ctxPct = computed(() =>
  ctxLen.value ? Math.min(100, (100 * wb.tokens.value.length) / ctxLen.value) : 0,
)

// Chat builder
const sys = ref('')
const user = ref('Write a haiku about tokens.')
const genPrompt = ref(true)
function buildChat() {
  const msgs = [
    ...(sys.value.trim() ? [{ role: 'system', content: sys.value }] : []),
    { role: 'user', content: user.value },
  ]
  void wb.applyChat(msgs, genPrompt.value)
}

function clearAll() {
  wb.clearContext()
  selected.value = null
}

function onSelect(i: number | null) {
  selected.value = i
  insertAt.value = null
}

function onInsertBefore(i: number) {
  insertAt.value = i
  selected.value = i
}
</script>

<template>
  <section class="panel card" aria-labelledby="ctx-title" data-testid="context-workspace">
    <div class="row">
      <h2 id="ctx-title">Context</h2>
      <InfoTip label="Tokens and context">
        <p>
          A <strong>token</strong> is an integer ID in the model’s vocabulary. The model never sees
          characters — only this sequence of IDs.
        </p>
        <p>
          Token boundaries often don’t match words: a leading space is usually part of the token
          (“␣the”), rare words split into pieces, and some tokens are single bytes of a multi-byte
          character.
        </p>
        <p>
          The <strong>token sequence below is authoritative</strong>. The text box is an editing
          convenience.
        </p>
      </InfoTip>
      <span class="spacer" />
      <label v-if="examples.length" class="small row">
        Example
        <select v-model="exampleId" data-testid="example-select" @change="onExample">
          <option value="" disabled>Load an example…</option>
          <option v-for="e in examples" :key="e.id" :value="e.id" :title="e.description">
            {{ e.label }}
          </option>
        </select>
      </label>
      <button
        type="button"
        class="btn btn-sm"
        :disabled="!wb.tokens.value.length"
        data-testid="clear-context"
        @click="clearAll"
      >
        Clear
      </button>
    </div>

    <div class="tabs" role="tablist" aria-label="Context editing modes">
      <button
        id="ctx-tab-text"
        type="button"
        role="tab"
        class="tab"
        :aria-selected="tab === 'text'"
        aria-controls="ctx-panel-text"
        @click="tab = 'text'"
      >
        Text
      </button>
      <button
        id="ctx-tab-chat"
        type="button"
        role="tab"
        class="tab"
        :aria-selected="tab === 'chat'"
        aria-controls="ctx-panel-chat"
        :disabled="!wb.tokenizer.value?.hasChatTemplate"
        @click="tab = 'chat'"
      >
        Chat template
      </button>
      <button
        id="ctx-tab-special"
        type="button"
        role="tab"
        class="tab"
        :aria-selected="tab === 'special'"
        aria-controls="ctx-panel-special"
        @click="tab = 'special'"
      >
        Special tokens
      </button>
    </div>

    <div
      v-show="tab === 'text'"
      id="ctx-panel-text"
      role="tabpanel"
      aria-labelledby="ctx-tab-text"
      class="stack"
    >
      <div
        v-if="wb.textLocked.value && wb.textSource.value === 'tokens'"
        class="alert alert-warning small"
        data-testid="roundtrip-warning"
      >
        The token sequence was edited at the token level and
        <strong
          >re-tokenizing its decoded text would change {{ wb.roundTripDiff.value }} token(s)</strong
        >
        (e.g. special tokens, or a non-canonical split). The text below is a read-only decoded
        preview.
        <button type="button" class="btn btn-sm" data-testid="unlock-text" @click="wb.unlockText()">
          Edit as text anyway (re-tokenizes everything)
        </button>
      </div>
      <label for="ctx-text" class="small muted">
        Text
        <span class="faint"
          >— editing re-tokenizes the <em>entire</em> text and replaces the token sequence</span
        >
      </label>
      <textarea
        id="ctx-text"
        rows="5"
        :value="wb.textDraft.value"
        :readonly="!wb.tokenizer.value || (wb.textLocked.value && wb.textSource.value === 'tokens')"
        :placeholder="
          wb.tokenizer.value
            ? 'Type or paste text…'
            : 'Load a model (or just its tokenizer) to tokenize text.'
        "
        data-testid="context-text"
        spellcheck="false"
        @input="wb.setText(($event.target as HTMLTextAreaElement).value)"
      />
      <div class="row small">
        <label class="row">
          <input v-model="wb.addSpecialTokens.value" type="checkbox" data-testid="add-special" />
          Add special tokens when tokenizing text
        </label>
        <span class="faint" data-testid="adds-special-note">
          <template v-if="wb.tokenizer.value">
            ({{
              wb.tokenizer.value.addsSpecialTokensByDefault
                ? 'this tokenizer adds e.g. a BOS token'
                : 'this tokenizer adds none by default'
            }})
          </template>
        </span>
        <span class="spacer" />
        <span v-if="wb.tokenizing.value" class="faint">tokenizing…</span>
        <span
          v-else-if="wb.roundTrips.value === true"
          class="badge badge-success"
          title="Re-tokenizing the text reproduces exactly these token IDs"
          >text ↔ tokens in sync</span
        >
        <span v-else-if="wb.roundTrips.value === false" class="badge badge-warning"
          >text is a decoded preview</span
        >
        <span v-if="wb.tokenizeMs.value !== null" class="faint"
          >tokenized in {{ formatMs(wb.tokenizeMs.value) }}</span
        >
      </div>
    </div>

    <div
      v-show="tab === 'chat'"
      id="ctx-panel-chat"
      role="tabpanel"
      aria-labelledby="ctx-tab-chat"
      class="stack"
    >
      <p class="small muted">
        Builds the context with the tokenizer’s real Jinja chat template (role delimiters and
        special tokens exactly as the model expects).
      </p>
      <label class="small">System message (optional)<textarea v-model="sys" rows="2" /></label>
      <label class="small"
        >User message<textarea v-model="user" rows="2" data-testid="chat-user" />
      </label>
      <div class="row">
        <label class="row small"
          ><input v-model="genPrompt" type="checkbox" /> Add generation prompt (assistant
          prefix)</label
        >
        <button
          type="button"
          class="btn btn-primary btn-sm"
          :disabled="!wb.tokenizer.value?.hasChatTemplate"
          data-testid="apply-chat"
          @click="buildChat"
        >
          Apply chat template
        </button>
      </div>
    </div>

    <div
      v-show="tab === 'special'"
      id="ctx-panel-special"
      role="tabpanel"
      aria-labelledby="ctx-tab-special"
    >
      <SpecialTokenBrowser :selected="selected" />
    </div>

    <div class="row small seq-head">
      <h3>Token sequence</h3>
      <span class="badge" data-testid="token-count"
        >{{ wb.tokens.value.length.toLocaleString() }} tokens</span
      >
      <span
        v-if="ctxLen"
        class="ctx"
        :title="`Model context window: ${ctxLen.toLocaleString()} tokens`"
      >
        <span class="ctx-bar" aria-hidden="true"><span :style="{ width: ctxPct + '%' }" /></span>
        {{ ctxPct.toFixed(ctxPct < 1 ? 2 : 0) }}% of {{ ctxLen.toLocaleString() }}-token context
      </span>
      <span class="spacer" />
      <label class="row"><input v-model="showIds" type="checkbox" /> show IDs</label>
    </div>
    <p v-if="wb.invalidTokenIds.value.length" class="alert alert-error small" role="alert">
      {{ wb.invalidTokenIds.value.length }} token ID(s) are invalid for the loaded model
      (highlighted in red). Remove them to run inference.
    </p>
    <TokenStrip :selected="selected" :show-ids="showIds" @select="onSelect" />
    <TokenInsertBar :selected="selected" :insert-at="insertAt" />
    <TokenInspector
      v-if="selected !== null && selected < wb.tokens.value.length"
      :index="selected"
      @close="selected = null"
      @insert-before="onInsertBefore"
    />
  </section>
</template>

<style scoped>
.panel {
  padding: 0.8rem;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}
[role='tabpanel'] {
  padding-top: 0.3rem;
}
.seq-head h3 {
  font-size: 0.9rem;
}
.ctx {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  color: var(--text-muted);
}
.ctx-bar {
  width: 5rem;
  height: 6px;
  border-radius: 3px;
  background: var(--surface-3);
  overflow: hidden;
  display: inline-block;
}
.ctx-bar span {
  display: block;
  height: 100%;
  background: var(--accent);
}
</style>
