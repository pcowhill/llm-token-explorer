/**
 * Main-thread client for the inference worker. Implements InferenceEngine by
 * forwarding calls over postMessage, with request IDs to match responses.
 */

import type { Capabilities, ProgressEvent } from '@/types'
import {
  EngineError,
  type ChatMessage,
  type ChatTemplateOptions,
  type ForwardOptions,
  type ForwardOutput,
  type InferenceEngine,
  type LoadRequest,
  type LoadResult,
} from './engine'
import type { WorkerMessage, WorkerRequest } from './protocol'

interface Pending {
  resolve: (v: unknown) => void
  reject: (e: unknown) => void
  onProgress?: (e: ProgressEvent) => void
}

export type WorkerFactory = () => Worker

export const defaultWorkerFactory: WorkerFactory = () =>
  new Worker(new URL('../workers/inference.worker.ts', import.meta.url), {
    type: 'module',
    name: 'inference',
  })

export class WorkerEngineClient implements InferenceEngine {
  private worker: Worker | null = null
  private nextId = 1
  private pending = new Map<number, Pending>()

  constructor(private readonly factory: WorkerFactory = defaultWorkerFactory) {}

  private ensureWorker(): Worker {
    if (this.worker) return this.worker
    const w = this.factory()
    w.addEventListener('message', (ev: MessageEvent<WorkerMessage>) => this.onMessage(ev.data))
    w.addEventListener('error', (ev) => {
      const err = new EngineError(
        `Inference worker crashed: ${ev.message || 'unknown error'}`,
        'backend',
        'Reload the model. If this repeats, try the WASM backend or a smaller model.',
      )
      for (const p of this.pending.values()) p.reject(err)
      this.pending.clear()
      this.worker?.terminate()
      this.worker = null
    })
    this.worker = w
    return w
  }

  private onMessage(msg: WorkerMessage) {
    if (msg.kind === 'ready') return
    const p = this.pending.get(msg.id)
    if (!p) return
    if (msg.kind === 'progress') {
      p.onProgress?.(msg.event)
      return
    }
    this.pending.delete(msg.id)
    if (msg.ok) p.resolve(msg.value)
    else
      p.reject(
        new EngineError(
          msg.error.message,
          msg.error.errorKind as EngineError['kind'],
          msg.error.hint,
        ),
      )
  }

  private call<T>(req: WorkerRequest, onProgress?: (e: ProgressEvent) => void): Promise<T> {
    const w = this.ensureWorker()
    const id = this.nextId++
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject, onProgress })
      w.postMessage({ id, ...req })
    })
  }

  capabilities(): Promise<Capabilities> {
    return this.call({ type: 'capabilities' })
  }

  load(request: LoadRequest, onProgress: (e: ProgressEvent) => void): Promise<LoadResult> {
    return this.call({ type: 'load', request }, onProgress)
  }

  tokenize(text: string, addSpecialTokens: boolean): Promise<{ ids: number[]; ms: number }> {
    return this.call({ type: 'tokenize', text, addSpecialTokens })
  }

  decode(ids: number[]): Promise<string> {
    return this.call({ type: 'decode', ids })
  }

  applyChatTemplate(
    messages: ChatMessage[],
    addGenerationPrompt: boolean,
    templateOptions?: ChatTemplateOptions,
  ): Promise<{ ids: number[]; text: string }> {
    return this.call({ type: 'chatTemplate', messages, addGenerationPrompt, templateOptions })
  }

  forward(ids: number[], options: ForwardOptions): Promise<ForwardOutput> {
    return this.call({ type: 'forward', ids, options })
  }

  unload(): Promise<void> {
    return this.call({ type: 'unload' })
  }

  /** Terminate the worker immediately (e.g. to cancel a download). Pending calls reject. */
  terminate(): void {
    const err = new EngineError('Cancelled', 'aborted')
    for (const p of this.pending.values()) p.reject(err)
    this.pending.clear()
    this.worker?.terminate()
    this.worker = null
  }
}
