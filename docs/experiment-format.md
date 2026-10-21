# Experiment file format (version 1)

Experiments are JSON files containing token IDs, settings and the branching history. They
never contain model weights or executable content. Import validates every field
(`src/history/experiment.ts`); invalid files are rejected with a list of problems.

```jsonc
{
  "format": "llm-token-explorer/experiment",
  "version": 1,
  "createdAt": "2026-10-21T10:00:00.000Z",
  "app": { "name": "LLM Token Explorer", "version": "1.0.0" },
  "model": {
    // null if no model was loaded
    "identity": "hub:onnx-community/SmolLM2-135M-Instruct-ONNX@b8a5c0f…#fp16",
    "displayName": "SmolLM2 135M Instruct",
    "source": "catalog", // catalog | hub | local
    "catalogKey": "smollm2-135m-instruct",
    "repo": "onnx-community/SmolLM2-135M-Instruct-ONNX",
    "revision": "b8a5c0f183b78c55955a5364f610c36668b5e681",
    "dtype": "fp16",
    "logitsSize": 49152,
    "tokenizerSize": 49152,
    "tokenizerFingerprint": "1a2b3c4d", // FNV-1a over the vocabulary
  },
  "sampling": { "temperature": 0.7, "topK": 0, "topP": 1, "minP": 0, "greedy": false, "seed": 42 },
  "generation": {
    "maxNewTokens": 32,
    "stopAtStopTokens": true,
    "debounceMs": 400,
    "autoInfer": true,
    "useKvCache": false,
  },
  "history": {
    "rootId": "n0",
    "currentId": "n3",
    "nodes": [
      {
        "id": "n0",
        "parentId": null,
        "kind": "root",
        "tokens": [],
        "children": ["n1"],
        "preferredChild": "n1",
        "createdAt": "…",
        "label": "Start",
        "modelIdentity": null,
        "prngState": [3654505418, 2465373460, 3864401178, 3307913292],
      },
      {
        "id": "n2",
        "parentId": "n1",
        "kind": "append",
        "tokens": [504, 3575, 282, 4649, 314, 7042],
        "children": [],
        "preferredChild": null,
        "createdAt": "…",
        "label": "Sampled 7042",
        "modelIdentity": "hub:…",
        "prngState": [1, 2, 3, 4],
        "append": {
          "tokenId": 7042,
          "source": "sample",
          "originalProbability": 0.469,
          "adjustedProbability": 0.52,
          "logit": 15.03,
          "sampling": {
            "temperature": 0.7,
            "topK": 0,
            "topP": 1,
            "minP": 0,
            "greedy": false,
            "seed": 42,
          },
          "prngBefore": [3654505418, 2465373460, 3864401178, 3307913292],
          "uniform": 0.1987,
        },
      },
    ],
  },
}
```

## Semantics

- Every node stores the **complete** token sequence. Token IDs are never re-tokenized on
  import.
- `kind`: `root`, `edit` (text/token edit), `append` (one token appended by `manual`,
  `sample`, `greedy` or `auto` selection). For `append` nodes the last token equals
  `append.tokenId`.
- `preferredChild` is the branch followed by _redo_.
- `prngState` is the SFC32 state after the node was created; navigating to the node
  restores it so further sampling is reproducible. `append.prngBefore` + `uniform` record
  the exact draw.

## Validation rules

- `format` and `version` must match; token IDs are integers in [0, 10,000,000];
  at most 20,000 nodes and 200,000 tokens per node; strings are length-limited.
- The tree must have exactly one root, consistent parent/child links, no cycles and no
  unreachable nodes. Sampling settings must be valid. Unknown fields are ignored.

## Importing with a different (or no) model

Import always succeeds if the file is valid. The app then compares the experiment's model
with the loaded one and reports: different model identity, different tokenizer
fingerprint, and token IDs outside the loaded model's range (inference is blocked for
contexts containing them). Load the original model to continue exactly.
