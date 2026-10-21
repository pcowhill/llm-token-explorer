/**
 * Small persistence helpers (localStorage). Only settings and the token-ID history are
 * stored — never model weights. Everything stays on this device.
 */

import { ref, watch, type Ref } from 'vue'

const PREFIX = 'lte:'

export function readJson<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(PREFIX + key)
    return raw === null ? fallback : (JSON.parse(raw) as T)
  } catch {
    return fallback
  }
}

export function writeJson(key: string, value: unknown): void {
  try {
    localStorage.setItem(PREFIX + key, JSON.stringify(value))
  } catch {
    /* quota or privacy mode: ignore */
  }
}

export function removeKey(key: string): void {
  try {
    localStorage.removeItem(PREFIX + key)
  } catch {
    /* ignore */
  }
}

export type ThemePref = 'system' | 'light' | 'dark'

export function useTheme(): { theme: Ref<ThemePref>; effective: Ref<'light' | 'dark'> } {
  const theme = ref<ThemePref>(readJson<ThemePref>('theme', 'system'))
  const media =
    typeof matchMedia !== 'undefined' ? matchMedia('(prefers-color-scheme: dark)') : null
  const effective = ref<'light' | 'dark'>('light')
  const apply = () => {
    effective.value = theme.value === 'system' ? (media?.matches ? 'dark' : 'light') : theme.value
    document.documentElement.dataset.theme = effective.value
  }
  media?.addEventListener?.('change', apply)
  watch(
    theme,
    (t) => {
      writeJson('theme', t)
      apply()
    },
    { immediate: true },
  )
  return { theme, effective }
}

export function useOnline(): Ref<boolean> {
  const online = ref(typeof navigator === 'undefined' ? true : navigator.onLine)
  if (typeof window !== 'undefined') {
    window.addEventListener('online', () => (online.value = true))
    window.addEventListener('offline', () => (online.value = false))
  }
  return online
}
