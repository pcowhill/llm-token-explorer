# Loading local model files

**Model → Local files** accepts a folder (directory picker) or individual files (multi-select
or drag-and-drop). Files are passed to the inference worker as `File` objects and read in
the browser; they are **never uploaded** and **never copied into the browser cache**.

## Supported layout

The standard Transformers.js / 🤗 Optimum ONNX export of a decoder-only causal LM:

```
my-model/
├── config.json               required  (model_type, vocab_size, …)
├── tokenizer.json            required  (vocabulary, merges, added tokens)
├── tokenizer_config.json     required  (special tokens, chat template)
├── generation_config.json    optional  (stop tokens)
└── onnx/                     (or the .onnx files directly in my-model/)
    ├── model.onnx            fp32
    ├── model_fp16.onnx       fp16
    ├── model_quantized.onnx  q8
    ├── model_q4.onnx / model_q4f16.onnx / model_bnb4.onnx / model_int8.onnx / model_uint8.onnx
    └── model_q4.onnx_data[, _data_1 …]   external weight chunks, if the export uses them
```

Any repository from `onnx-community/…` downloaded with `git clone` or
`huggingface-cli download` has this layout. Other base names (e.g. the older
`decoder_model_merged.onnx`) are detected too.

## Validation

Before loading, the app (`src/models/localFiles.ts`):

- finds the model root (the shallowest folder containing `config.json`);
- reports missing required files and missing weights;
- detects the dtype from the file-name suffix and counts external-data chunks
  (`<file>.onnx_data`, `_data_1`, …); a tiny `.onnx` without a data file is flagged;
- warns about non-ONNX weights (`.safetensors`, `.bin`, `.gguf` — these cannot run here);
- parses `config.json` and warns for architectures outside the tested decoder-only list or
  for non-causal-LM `architectures`.

## How it works

Transformers.js is pointed at a pseudo path (`/__local_models__/local/m<N>/`) with
`allowRemoteModels = false` and `local_files_only = true`. The worker's `fetch` shim answers
requests for that path from the selected files. (The pseudo path must not be an `http(s)`
URL: Transformers.js only probes `localModelPath` for files when it is a path.) A fresh
model id is used per load because the library memoizes file metadata per id.

## Limitations

- Only decoder-only causal LMs supported by Transformers.js `AutoModelForCausalLM`.
- Encoder-decoder (T5, BART) and multimodal models are not supported.
- GGUF / safetensors / PyTorch checkpoints must be converted to ONNX first.
- Very large files may exceed browser memory (WebAssembly is limited to 4 GB of addressable
  memory per worker).
- Local models are not cached, so they must be re-selected after a reload (but they work
  offline).
