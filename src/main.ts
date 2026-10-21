import { WorkerEngineClient } from '@/inference/workerClient'
;(window as unknown as Record<string, unknown>).__engine = new WorkerEngineClient()
document.getElementById('app')!.textContent = 'proof harness'
