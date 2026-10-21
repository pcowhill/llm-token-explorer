import { createApp, ref } from 'vue'
import App from './App.vue'
import './styles/main.css'
import { createWorkbench } from '@/stores/workbench'
import { WorkerEngineClient } from '@/inference/workerClient'

const workbench = createWorkbench(new WorkerEngineClient(), { appVersion: __APP_VERSION__ })

// Read-only debugging/testing hook, only when explicitly requested via ?debug=1.
if (new URLSearchParams(location.search).has('debug')) {
  ;(window as unknown as { __lte?: unknown }).__lte = workbench
}

const updateAvailable = ref(false)
const offlineReady = ref(false)
let updateSW: (reload?: boolean) => Promise<void> = async () => {}

createApp(App, {
  workbench,
  pwa: { updateAvailable, offlineReady, update: () => void updateSW(true) },
}).mount('#app')

// Service worker (production builds only): caches the app shell for offline use.
if (import.meta.env.PROD && 'serviceWorker' in navigator) {
  void import('virtual:pwa-register').then(({ registerSW }) => {
    updateSW = registerSW({
      immediate: true,
      onNeedRefresh: () => (updateAvailable.value = true),
      onOfflineReady: () => (offlineReady.value = true),
      onRegisteredSW: (_url, reg) => {
        if (reg?.active) offlineReady.value = true
      },
    })
  })
}
