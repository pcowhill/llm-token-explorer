<script setup lang="ts">
/**
 * Renders a token's text with whitespace / invisible characters made visible.
 * Always rendered as text (never HTML).
 */
import { computed } from 'vue'
import { escapeVisible } from '@/tokenization/display'

const props = withDefaults(defineProps<{ text: string; special?: boolean; empty?: string }>(), {
  special: false,
  empty: '∅',
})
const shown = computed(() => (props.text.length === 0 ? props.empty : escapeVisible(props.text)))
</script>

<template>
  <span class="token-text" :class="{ special: props.special, empty: props.text.length === 0 }">{{
    shown
  }}</span>
</template>

<style scoped>
.token-text {
  font-family: var(--mono);
  white-space: pre;
}
.special {
  color: var(--special);
  font-weight: 600;
}
.empty {
  color: var(--text-faint);
}
</style>
