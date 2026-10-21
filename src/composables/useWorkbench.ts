import { inject, type InjectionKey } from 'vue'
import type { Workbench } from '@/stores/workbench'

export const WORKBENCH_KEY: InjectionKey<Workbench> = Symbol('workbench')

export function useWorkbench(): Workbench {
  const wb = inject(WORKBENCH_KEY)
  if (!wb) throw new Error('Workbench not provided')
  return wb
}
