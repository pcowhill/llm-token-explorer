/** Message protocol between the main thread and the inference worker. */

import type { ProgressEvent } from '@/types'
import type { ChatMessage, ChatTemplateOptions, ForwardOptions, LoadRequest } from './engine'

export type WorkerRequest =
  | { type: 'capabilities' }
  | { type: 'load'; request: LoadRequest }
  | { type: 'tokenize'; text: string; addSpecialTokens: boolean }
  | { type: 'decode'; ids: number[] }
  | {
      type: 'chatTemplate'
      messages: ChatMessage[]
      addGenerationPrompt: boolean
      templateOptions?: ChatTemplateOptions
    }
  | { type: 'forward'; ids: number[]; options: ForwardOptions }
  | { type: 'unload' }

export type WorkerRequestEnvelope = { id: number } & WorkerRequest

export type WorkerMessage =
  | { kind: 'result'; id: number; ok: true; value: unknown }
  | {
      kind: 'result'
      id: number
      ok: false
      error: { message: string; errorKind: string; hint?: string }
    }
  | { kind: 'progress'; id: number; event: ProgressEvent }
  | { kind: 'ready' }
