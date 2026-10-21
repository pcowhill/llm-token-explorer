/// <reference lib="webworker" />
/**
 * Inference Web Worker: hosts Transformers.js + ONNX Runtime so that model loading,
 * tokenization and forward passes never block the UI thread.
 *
 * Requests are processed strictly one at a time (ONNX Runtime Web cannot run sessions
 * concurrently), in arrival order. Logits are returned as transferable Float32Arrays.
 */

import { TransformersEngine } from '@/inference/transformersEngine'
import { classifyError } from '@/inference/engine'
import type { WorkerMessage, WorkerRequestEnvelope } from '@/inference/protocol'

const engine = new TransformersEngine()
const scope = self as unknown as DedicatedWorkerGlobalScope
let queue: Promise<void> = Promise.resolve()

function post(msg: WorkerMessage, transfer: Transferable[] = []) {
  scope.postMessage(msg, transfer)
}

async function handle(req: WorkerRequestEnvelope): Promise<void> {
  try {
    let value: unknown
    const transfer: Transferable[] = []
    switch (req.type) {
      case 'capabilities':
        value = await engine.capabilities()
        break
      case 'load':
        value = await engine.load(req.request, (event) => post({ kind: 'progress', id: req.id, event }))
        break
      case 'tokenize':
        value = await engine.tokenize(req.text, req.addSpecialTokens)
        break
      case 'decode':
        value = await engine.decode(req.ids)
        break
      case 'chatTemplate':
        value = await engine.applyChatTemplate(req.messages, req.addGenerationPrompt)
        break
      case 'forward': {
        const out = await engine.forward(req.ids, req.options)
        transfer.push(out.logits.buffer)
        value = out
        break
      }
      case 'unload':
        value = await engine.unload()
        break
    }
    post({ kind: 'result', id: req.id, ok: true, value }, transfer)
  } catch (err) {
    const e = classifyError(err)
    post({ kind: 'result', id: req.id, ok: false, error: { message: e.message, errorKind: e.kind, hint: e.hint } })
  }
}

scope.addEventListener('message', (ev: MessageEvent<WorkerRequestEnvelope>) => {
  const req = ev.data
  queue = queue.then(() => handle(req))
})

post({ kind: 'ready' })
