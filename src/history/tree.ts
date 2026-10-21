/**
 * Generation history as a tree of exact token sequences.
 *
 * - Every node stores the COMPLETE token-ID sequence (no re-tokenization ever happens
 *   when navigating, exporting or importing).
 * - Undo moves to the parent; redo moves to the most recently visited child.
 * - Appending a different token from an earlier node creates a sibling branch;
 *   appending the same token again re-enters the existing child instead of duplicating it.
 * - Consecutive text/token edits on a leaf are coalesced into one 'edit' node.
 */

import type { PrngState } from '@/sampling/prng'
import type { SamplingSettings, TokenId } from '@/types'

export type NodeKind = 'root' | 'edit' | 'append'
export type AppendSource = 'manual' | 'sample' | 'greedy' | 'auto'

export interface AppendDetails {
  tokenId: TokenId
  source: AppendSource
  /** Original (T=1, unfiltered) probability of the token when it was chosen. */
  originalProbability: number | null
  /** Adjusted (sampling) probability of the token when it was chosen. */
  adjustedProbability: number | null
  /** Raw logit of the token. */
  logit: number | null
  /** Sampling settings in effect (for sampled / greedy / auto tokens). */
  sampling: SamplingSettings | null
  /** PRNG state before the draw (null for manual / greedy picks). */
  prngBefore: PrngState | null
  /** Uniform value drawn, if any. */
  uniform: number | null
}

export interface HistoryNode {
  id: string
  parentId: string | null
  kind: NodeKind
  tokens: TokenId[]
  children: string[]
  /** Child to follow on redo (most recently visited). */
  preferredChild: string | null
  createdAt: string
  label: string
  /** Identity of the model in use when the node was created. */
  modelIdentity: string | null
  /** PRNG state after this node was created; restored when navigating here. */
  prngState: PrngState | null
  append?: AppendDetails
}

export interface HistoryTree {
  rootId: string
  currentId: string
  nodes: Map<string, HistoryNode>
  nextId: number
}

export function createTree(tokens: TokenId[] = [], modelIdentity: string | null = null, prngState: PrngState | null = null): HistoryTree {
  const root: HistoryNode = {
    id: 'n0',
    parentId: null,
    kind: 'root',
    tokens: [...tokens],
    children: [],
    preferredChild: null,
    createdAt: new Date().toISOString(),
    label: 'Start',
    modelIdentity,
    prngState,
  }
  return { rootId: root.id, currentId: root.id, nodes: new Map([[root.id, root]]), nextId: 1 }
}

export function getNode(tree: HistoryTree, id: string): HistoryNode {
  const n = tree.nodes.get(id)
  if (!n) throw new Error(`Unknown history node ${id}`)
  return n
}

export function current(tree: HistoryTree): HistoryNode {
  return getNode(tree, tree.currentId)
}

function sameTokens(a: readonly number[], b: readonly number[]): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false
  return true
}

function addChild(tree: HistoryTree, parent: HistoryNode, node: Omit<HistoryNode, 'id' | 'children' | 'preferredChild' | 'parentId' | 'createdAt'>): HistoryNode {
  const id = `n${tree.nextId++}`
  const full: HistoryNode = {
    ...node,
    id,
    parentId: parent.id,
    children: [],
    preferredChild: null,
    createdAt: new Date().toISOString(),
  }
  tree.nodes.set(id, full)
  parent.children.push(id)
  parent.preferredChild = id
  tree.currentId = id
  return full
}

/**
 * Append one token to the current sequence. Returns the resulting node, which is
 * either a new branch or an existing child with an identical sequence.
 */
export function appendToken(
  tree: HistoryTree,
  details: AppendDetails,
  meta: { modelIdentity: string | null; prngState: PrngState | null; label: string },
): HistoryNode {
  const parent = current(tree)
  const tokens = [...parent.tokens, details.tokenId]
  for (const cid of parent.children) {
    const child = getNode(tree, cid)
    if (sameTokens(child.tokens, tokens)) {
      parent.preferredChild = cid
      tree.currentId = cid
      // Keep the newest PRNG state so that continuing sampling stays reproducible.
      child.prngState = meta.prngState
      return child
    }
  }
  return addChild(tree, parent, {
    kind: 'append',
    tokens,
    label: meta.label,
    modelIdentity: meta.modelIdentity,
    prngState: meta.prngState,
    append: details,
  })
}

/**
 * Record an arbitrary edit (text re-tokenization, token insertion/removal...).
 * Consecutive edits on a childless 'edit' node are coalesced into that node.
 */
export function commitEdit(
  tree: HistoryTree,
  tokens: TokenId[],
  meta: { modelIdentity: string | null; prngState: PrngState | null; label: string; coalesce?: boolean },
): HistoryNode {
  const node = current(tree)
  if (sameTokens(node.tokens, tokens)) return node
  if ((meta.coalesce ?? true) && node.kind === 'edit' && node.children.length === 0) {
    node.tokens = [...tokens]
    node.label = meta.label
    return node
  }
  if (node.kind === 'root' && node.children.length === 0 && node.tokens.length === 0 && (meta.coalesce ?? true)) {
    // An empty, unused root simply becomes the initial context.
    node.tokens = [...tokens]
    node.label = meta.label
    return node
  }
  return addChild(tree, node, {
    kind: 'edit',
    tokens: [...tokens],
    label: meta.label,
    modelIdentity: meta.modelIdentity,
    prngState: meta.prngState,
  })
}

export function canUndo(tree: HistoryTree): boolean {
  return current(tree).parentId !== null
}

export function canRedo(tree: HistoryTree): boolean {
  const n = current(tree)
  return n.children.length > 0
}

export function undo(tree: HistoryTree): HistoryNode {
  const n = current(tree)
  if (n.parentId === null) return n
  const parent = getNode(tree, n.parentId)
  parent.preferredChild = n.id
  tree.currentId = parent.id
  return parent
}

export function redo(tree: HistoryTree): HistoryNode {
  const n = current(tree)
  const next = n.preferredChild ?? n.children[n.children.length - 1]
  if (!next) return n
  tree.currentId = next
  return getNode(tree, next)
}

export function goTo(tree: HistoryTree, id: string): HistoryNode {
  const target = getNode(tree, id)
  // Mark the path as preferred so redo follows it.
  let child = target
  while (child.parentId !== null) {
    const parent = getNode(tree, child.parentId)
    parent.preferredChild = child.id
    child = parent
  }
  tree.currentId = id
  return target
}

/** Path from root to the node (inclusive). */
export function pathTo(tree: HistoryTree, id: string): HistoryNode[] {
  const out: HistoryNode[] = []
  let n: HistoryNode | undefined = getNode(tree, id)
  while (n) {
    out.unshift(n)
    n = n.parentId ? getNode(tree, n.parentId) : undefined
  }
  return out
}

/** Siblings of a node, including itself, in creation order. */
export function siblings(tree: HistoryTree, id: string): HistoryNode[] {
  const n = getNode(tree, id)
  if (n.parentId === null) return [n]
  return getNode(tree, n.parentId).children.map((c) => getNode(tree, c))
}

/** Switch to an adjacent sibling branch (delta = ±1), descending along preferred children. */
export function switchBranch(tree: HistoryTree, delta: number): HistoryNode {
  const n = current(tree)
  const sibs = siblings(tree, n.id)
  const i = sibs.findIndex((s) => s.id === n.id)
  const target = sibs[(i + delta + sibs.length) % sibs.length] as HistoryNode
  return goTo(tree, target.id)
}

export interface FlatNode {
  node: HistoryNode
  depth: number
  isCurrent: boolean
  onCurrentPath: boolean
  branchIndex: number
  branchCount: number
}

/** Depth-first flattening for display. */
export function flatten(tree: HistoryTree): FlatNode[] {
  const onPath = new Set(pathTo(tree, tree.currentId).map((n) => n.id))
  const out: FlatNode[] = []
  const visit = (id: string, depth: number, branchIndex: number, branchCount: number) => {
    const node = getNode(tree, id)
    out.push({
      node,
      depth,
      isCurrent: id === tree.currentId,
      onCurrentPath: onPath.has(id),
      branchIndex,
      branchCount,
    })
    node.children.forEach((c, i) => visit(c, node.children.length > 1 ? depth + 1 : depth, i, node.children.length))
  }
  visit(tree.rootId, 0, 0, 1)
  return out
}

export function leafCount(tree: HistoryTree): number {
  let c = 0
  for (const n of tree.nodes.values()) if (n.children.length === 0) c++
  return c
}
