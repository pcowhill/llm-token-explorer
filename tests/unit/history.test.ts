import { describe, expect, it } from 'vitest'
import {
  appendToken,
  canRedo,
  canUndo,
  commitEdit,
  createTree,
  current,
  flatten,
  goTo,
  leafCount,
  pathTo,
  redo,
  siblings,
  switchBranch,
  undo,
  type AppendDetails,
} from '@/history/tree'

const meta = { modelIdentity: 'm', prngState: null, label: 'x' }
const det = (tokenId: number, source: AppendDetails['source'] = 'manual'): AppendDetails => ({
  tokenId,
  source,
  originalProbability: null,
  adjustedProbability: null,
  logit: null,
  sampling: null,
  prngBefore: null,
  uniform: null,
})

describe('history tree', () => {
  it('starts with a root holding the initial tokens', () => {
    const t = createTree([1, 2])
    expect(current(t).tokens).toEqual([1, 2])
    expect(canUndo(t)).toBe(false)
    expect(canRedo(t)).toBe(false)
  })

  it('append / undo / redo walk the exact token sequences', () => {
    const t = createTree([1])
    appendToken(t, det(5), meta)
    appendToken(t, det(6), meta)
    expect(current(t).tokens).toEqual([1, 5, 6])
    undo(t)
    expect(current(t).tokens).toEqual([1, 5])
    undo(t)
    expect(current(t).tokens).toEqual([1])
    expect(canUndo(t)).toBe(false)
    redo(t)
    redo(t)
    expect(current(t).tokens).toEqual([1, 5, 6])
    expect(canRedo(t)).toBe(false)
  })

  it('choosing a different token after undo creates a branch and keeps the old one', () => {
    const t = createTree([1])
    const a = appendToken(t, det(5), meta)
    undo(t)
    const b = appendToken(t, det(7), meta)
    expect(a.id).not.toBe(b.id)
    expect(siblings(t, b.id).map((n) => n.tokens)).toEqual([
      [1, 5],
      [1, 7],
    ])
    expect(leafCount(t)).toBe(2)
    switchBranch(t, -1)
    expect(current(t).tokens).toEqual([1, 5])
    switchBranch(t, 1)
    expect(current(t).tokens).toEqual([1, 7])
  })

  it('re-appending the same token re-enters the existing child instead of duplicating it', () => {
    const t = createTree([1])
    const a = appendToken(t, det(5), meta)
    undo(t)
    const again = appendToken(t, det(5, 'sample'), meta)
    expect(again.id).toBe(a.id)
    expect(t.nodes.size).toBe(2)
  })

  it('redo follows the most recently visited branch', () => {
    const t = createTree([1])
    appendToken(t, det(5), meta)
    undo(t)
    appendToken(t, det(7), meta)
    undo(t)
    expect(redo(t).tokens).toEqual([1, 7])
    undo(t)
    goTo(t, [...t.nodes.values()].find((n) => n.tokens.join() === '1,5')!.id)
    undo(t)
    expect(redo(t).tokens).toEqual([1, 5])
  })

  it('coalesces consecutive edits on a leaf edit node', () => {
    const t = createTree([])
    commitEdit(t, [1], { ...meta, coalesce: true }) // empty root absorbs first edit
    expect(t.nodes.size).toBe(1)
    appendToken(t, det(2), meta)
    commitEdit(t, [1, 2, 3], { ...meta, coalesce: true })
    commitEdit(t, [1, 2, 3, 4], { ...meta, coalesce: true })
    expect(t.nodes.size).toBe(3)
    expect(current(t).tokens).toEqual([1, 2, 3, 4])
    // Non-coalescing edit creates a new node
    commitEdit(t, [9], { ...meta, coalesce: false })
    expect(t.nodes.size).toBe(4)
    // Identical edit is a no-op
    commitEdit(t, [9], { ...meta, coalesce: false })
    expect(t.nodes.size).toBe(4)
  })

  it('pathTo and flatten describe the tree for display', () => {
    const t = createTree([1])
    appendToken(t, det(2), meta)
    undo(t)
    appendToken(t, det(3), meta)
    expect(pathTo(t, t.currentId).map((n) => n.tokens.length)).toEqual([1, 2])
    const flat = flatten(t)
    expect(flat).toHaveLength(3)
    expect(flat.filter((f) => f.isCurrent)).toHaveLength(1)
    expect(flat.filter((f) => f.branchCount === 2)).toHaveLength(2)
  })
})
