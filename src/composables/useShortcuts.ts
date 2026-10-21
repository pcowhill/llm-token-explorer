/**
 * Remappable single-key shortcuts (persisted in localStorage).
 *
 * Bindings are plain printable characters compared case-insensitively against
 * KeyboardEvent.key, without Ctrl/Alt/Meta. They never fire while typing in a text field
 * (App.vue checks the event target).
 */
import { ref } from 'vue'
import { readJson, writeJson } from './usePreferences'

export type ShortcutAction = 'sample' | 'greedy'

export const SHORTCUT_ACTIONS: readonly {
  id: ShortcutAction
  label: string
  defaultKey: string
}[] = [
  { id: 'sample', label: 'Sample one token', defaultKey: 's' },
  { id: 'greedy', label: 'Append the most likely token (greedy)', defaultKey: 'g' },
]

/** Keys that cannot be bound, and why. */
export const RESERVED_KEYS: Readonly<Record<string, string>> = {
  '?': 'it opens the shortcut list',
  ' ': 'Space activates buttons and scrolls the page',
}

const STORAGE_KEY = 'shortcuts'
const defaults = (): Record<ShortcutAction, string> =>
  Object.fromEntries(SHORTCUT_ACTIONS.map((a) => [a.id, a.defaultKey])) as Record<
    ShortcutAction,
    string
  >

function load(): Record<ShortcutAction, string> {
  const saved = readJson<Partial<Record<ShortcutAction, string>>>(STORAGE_KEY, {})
  const out = defaults()
  for (const a of SHORTCUT_ACTIONS) {
    const k = saved[a.id]
    if (typeof k === 'string' && isBindable(k)) out[a.id] = k.toLowerCase()
  }
  // Corrupt storage with duplicate keys: fall back to the defaults.
  return new Set(Object.values(out)).size === SHORTCUT_ACTIONS.length ? out : defaults()
}

function isBindable(key: string): boolean {
  return [...key].length === 1 && !(key in RESERVED_KEYS) && key.trim() !== ''
}

/** Shared state: one set of bindings for the whole app. */
const bindings = ref<Record<ShortcutAction, string>>(load())
/** True while Settings waits for the user to press a new key. */
const capturing = ref(false)

/** How a key is shown in hints: letters upper-case ("S"), other characters as is. */
export function keyLabel(key: string): string {
  return key.length === 1 ? key.toUpperCase() : key
}

export function useShortcuts() {
  function actionForKey(key: string): ShortcutAction | null {
    const k = key.toLowerCase()
    return SHORTCUT_ACTIONS.find((a) => bindings.value[a.id] === k)?.id ?? null
  }

  /** Bind `key` to `action`. Returns an error message (nothing changes) or null. */
  function setBinding(action: ShortcutAction, key: string): string | null {
    if (key in RESERVED_KEYS) return `“${keyLabel(key)}” cannot be used: ${RESERVED_KEYS[key]}.`
    if (!isBindable(key))
      return `“${key}” cannot be used: choose a single letter, digit or symbol key.`
    const k = key.toLowerCase()
    const other = SHORTCUT_ACTIONS.find((a) => a.id !== action && bindings.value[a.id] === k)
    if (other) return `“${keyLabel(k)}” is already used for “${other.label}”.`
    bindings.value = { ...bindings.value, [action]: k }
    writeJson(STORAGE_KEY, bindings.value)
    return null
  }

  function resetBindings() {
    bindings.value = defaults()
    writeJson(STORAGE_KEY, bindings.value)
  }

  return { bindings, capturing, actionForKey, setBinding, resetBindings }
}
