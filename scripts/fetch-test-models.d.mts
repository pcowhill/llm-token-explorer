/** Types for scripts/fetch-test-models.mjs (imported by tests). */
export interface CatalogFiles {
  key: string
  repo: string
  revision: string
  dtypes: string[]
  /** ONNX files per dtype, graph first, then external-data chunks. */
  files: Record<string, string[]>
}
export const TINY: { repo: string; revision: string; files: string[]; dir: string }
export const CATALOG_FILES: CatalogFiles[]
export function fetchCatalogModel(key: string, dtypes?: string[]): Promise<void>
