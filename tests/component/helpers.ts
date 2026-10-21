import { flushPromises, mount, type MountingOptions } from '@vue/test-utils'
import type { Component } from 'vue'
import { createWorkbench, type Workbench } from '@/stores/workbench'
import { WORKBENCH_KEY } from '@/composables/useWorkbench'
import { FakeEngine } from './fakeEngine'

export async function settle(times = 4) {
  for (let i = 0; i < times; i++) {
    await flushPromises()
    await new Promise((r) => setTimeout(r, 0))
  }
}

/** Create a workbench backed by the fake engine, optionally with a "loaded" model and context. */
export async function makeWorkbench(opts: { loaded?: boolean; text?: string } = {}) {
  const engine = new FakeEngine()
  const wb = createWorkbench(engine, { appVersion: 'test' })
  wb.generation.debounceMs = 0
  await wb.init()
  if (opts.loaded !== false) {
    await wb.loadCatalog('smollm2-135m-instruct', { backend: 'wasm', dtype: 'q8' })
    await settle()
    if (opts.text !== undefined) {
      const { ids } = await engine.tokenize(opts.text)
      wb.setTokens(ids, 'test')
      await settle()
    }
  }
  return { engine, wb }
}

export function mountWith<T extends Component>(
  component: T,
  wb: Workbench,
  options: MountingOptions<unknown> = {},
) {
  return mount(component, {
    attachTo: document.body,
    ...options,
    global: { ...(options.global ?? {}), provide: { [WORKBENCH_KEY as symbol]: wb } },
  } as never)
}
