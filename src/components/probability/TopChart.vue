<script setup lang="ts">
/**
 * Horizontal bar chart of the top-N next-token candidates (by original probability).
 * Pure SVG + text; a table with the same numbers is always available next to it.
 */
import { computed } from 'vue'
import { escapeVisible } from '@/tokenization/display'
import { formatLogit, formatProb } from '@/utils/format'

export type ChartMode = 'compare' | 'original' | 'adjusted' | 'logits'

const props = defineProps<{
  ids: number[]
  original: Float64Array
  adjusted: Float64Array | null
  logits: Float32Array
  decoded: readonly string[]
  special: Set<number>
  mode: ChartMode
}>()
const emit = defineEmits<{ pick: [number] }>()

const ROW = 26
const LABEL_W = 150
const VALUE_W = 132
const WIDTH = 640
const barW = WIDTH - LABEL_W - VALUE_W

const rows = computed(() => {
  const ids = props.ids
  const lg = ids.map((id) => props.logits[id] as number).filter(Number.isFinite)
  const lmin = Math.min(...lg, 0)
  const lmax = Math.max(...lg, 1e-9)
  const maxP = Math.max(
    ...ids.map((id) => Math.max(props.original[id] ?? 0, props.adjusted?.[id] ?? 0)),
    1e-12,
  )
  return ids.map((id, i) => {
    const o = props.original[id] ?? 0
    const a = props.adjusted?.[id] ?? 0
    const l = props.logits[id] as number
    return {
      id,
      y: i * ROW,
      label: escapeVisible(props.decoded[id] ?? '') || '∅',
      special: props.special.has(id),
      o,
      a,
      l,
      eligible: a > 0,
      wo: (o / maxP) * barW,
      wa: (a / maxP) * barW,
      wl: Number.isFinite(l) ? ((l - lmin) / (lmax - lmin || 1)) * barW : 0,
    }
  })
})
const height = computed(() => rows.value.length * ROW + 4)
const caption = computed(
  () =>
    ({
      compare: 'Original (blue, upper bar) vs adjusted sampling probability (orange, lower bar)',
      original:
        'Original next-token probability: softmax of the raw logits (temperature 1, no filtering)',
      adjusted: 'Adjusted sampling probability after temperature, top-k, top-p and min-p',
      logits: 'Raw logits (unnormalized scores; bar length relative to the smallest shown logit)',
    })[props.mode],
)
</script>

<template>
  <figure class="chart" data-testid="top-chart">
    <figcaption class="small muted">{{ caption }}</figcaption>
    <svg
      :viewBox="`0 0 ${WIDTH} ${height}`"
      role="img"
      :aria-label="`${caption}. ${rows.length} tokens; see the table for exact values.`"
      preserveAspectRatio="xMinYMin meet"
    >
      <g
        v-for="r in rows"
        :key="r.id"
        :transform="`translate(0 ${r.y})`"
        class="row"
        @click="emit('pick', r.id)"
      >
        <title>
          ID {{ r.id }} · original {{ formatProb(r.o) }} · adjusted {{ formatProb(r.a) }} · logit
          {{ formatLogit(r.l) }} (click to append)
        </title>
        <text
          :x="LABEL_W - 6"
          :y="ROW / 2 + 4"
          text-anchor="end"
          class="label"
          :class="{ special: r.special }"
        >
          {{ r.label.length > 16 ? r.label.slice(0, 15) + '…' : r.label }}
        </text>
        <rect :x="LABEL_W" y="3" :width="barW" :height="ROW - 6" class="track" rx="3" />
        <template v-if="props.mode === 'compare'">
          <rect
            :x="LABEL_W"
            y="3"
            :width="Math.max(1, r.wo)"
            :height="(ROW - 6) / 2"
            class="bar-o"
            rx="2"
          />
          <rect
            :x="LABEL_W"
            :y="3 + (ROW - 6) / 2"
            :width="r.eligible ? Math.max(1, r.wa) : 0"
            :height="(ROW - 6) / 2"
            class="bar-a"
            rx="2"
          />
        </template>
        <rect
          v-else-if="props.mode === 'original'"
          :x="LABEL_W"
          y="3"
          :width="Math.max(1, r.wo)"
          :height="ROW - 6"
          class="bar-o"
          rx="2"
        />
        <rect
          v-else-if="props.mode === 'adjusted'"
          :x="LABEL_W"
          y="3"
          :width="r.eligible ? Math.max(1, r.wa) : 0"
          :height="ROW - 6"
          class="bar-a"
          rx="2"
        />
        <rect
          v-else
          :x="LABEL_W"
          y="3"
          :width="Math.max(1, r.wl)"
          :height="ROW - 6"
          class="bar-l"
          rx="2"
        />
        <text :x="LABEL_W + barW + 6" :y="ROW / 2 + 4" class="value">
          {{
            props.mode === 'logits'
              ? formatLogit(r.l)
              : props.mode === 'adjusted'
                ? formatProb(r.a)
                : formatProb(r.o)
          }}{{ props.mode === 'compare' ? ' → ' + (r.eligible ? formatProb(r.a) : '✗') : '' }}
        </text>
      </g>
    </svg>
    <p class="legend small">
      <span v-if="props.mode === 'compare' || props.mode === 'original'"
        ><span class="sw sw-o" aria-hidden="true" /> original</span
      >
      <span v-if="props.mode === 'compare' || props.mode === 'adjusted'"
        ><span class="sw sw-a" aria-hidden="true" /> adjusted</span
      >
      <span>✗ = filtered out (not eligible for sampling)</span>
    </p>
  </figure>
</template>

<style scoped>
.chart {
  margin: 0;
}
svg {
  width: 100%;
  height: auto;
  display: block;
}
.row {
  cursor: pointer;
}
.row:hover .track {
  fill: var(--surface-3);
}
.label {
  font-family: var(--mono);
  font-size: 12px;
  fill: var(--text);
  white-space: pre;
}
.label.special {
  fill: var(--special);
  font-weight: 600;
}
.value {
  font-family: var(--mono);
  font-size: 11px;
  fill: var(--text-muted);
}
.track {
  fill: var(--surface-2);
}
.bar-o {
  fill: var(--original);
}
.bar-a {
  fill: var(--adjusted);
}
.bar-l {
  fill: var(--text-faint);
}
.legend {
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
  color: var(--text-muted);
}
.sw {
  display: inline-block;
  width: 0.8em;
  height: 0.8em;
  border-radius: 2px;
  vertical-align: middle;
}
.sw-o {
  background: var(--original);
}
.sw-a {
  background: var(--adjusted);
}
</style>
