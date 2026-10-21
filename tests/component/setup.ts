import { vi } from 'vitest'

// jsdom lacks a few browser APIs used by the app.
if (!window.matchMedia) {
  Object.defineProperty(window, 'matchMedia', {
    value: (query: string) => ({
      matches: false,
      media: query,
      addEventListener: () => {},
      removeEventListener: () => {},
      addListener: () => {},
      removeListener: () => {},
      onchange: null,
      dispatchEvent: () => false,
    }),
  })
}

if (typeof HTMLDialogElement !== 'undefined' && !HTMLDialogElement.prototype.showModal) {
  HTMLDialogElement.prototype.showModal = function (this: HTMLDialogElement) {
    this.setAttribute('open', '')
  }
  HTMLDialogElement.prototype.close = function (this: HTMLDialogElement) {
    this.removeAttribute('open')
  }
}

// Build-time constants normally injected by Vite's `define`.
vi.stubGlobal('__APP_VERSION__', 'test')
vi.stubGlobal('__TRANSFORMERS_VERSION__', 'test')
vi.stubGlobal('__ORT_VERSION__', 'test')
