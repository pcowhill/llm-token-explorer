# Special tokens

Special tokens are vocabulary entries reserved for control purposes. The app reads them
from the **loaded tokenizer's metadata** — nothing is hard-coded per model:

- `tokenizer.json` → `added_tokens[]` with `special: true` (and all added tokens);
- the tokenizer's `bos_token`, `eos_token`, `pad_token`, `unk_token`, `sep_token`,
  `mask_token` and `all_special_ids`;
- `generation_config.json` / `config.json` → `eos_token_id` (stop tokens; may be a list);
- the chat template (a special token appearing in it gets the `chat` role).

Each special token is listed with its ID, exact content, roles (`bos`, `eos`, `pad`, `unk`,
`stop`, `chat`, `other`) and a purpose description for well-known tokens (ChatML,
Llama 3, Gemma, code/FIM markers, …).

## Insertion is by ID

Inserting a special token from the browser (or the "Insert token ID" bar) inserts its exact
integer ID into the authoritative sequence. It never inserts the token's spelling as text,
because a tokenizer may or may not map that spelling back to the special ID.

## Model specifics (catalog)

| Model                 | BOS                                                       | Stop tokens (generation)                            | Chat delimiters                            | Notes                                                                                                 |
| --------------------- | --------------------------------------------------------- | --------------------------------------------------- | ------------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| SmolLM2-135M-Instruct | config says `<\|im_start\|>` (1), not added by `encode()` | `<\|im_end\|>` (2)                                  | `<\|im_start\|>` / `<\|im_end\|>` (ChatML) | `<\|endoftext\|>` (0) separated pre-training documents. 17 special tokens incl. code/Jupyter markers. |
| Qwen2.5-0.5B-Instruct | none                                                      | `<\|im_end\|>` (151645), `<\|endoftext\|>` (151643) | ChatML                                     | 14 special tokens incl. vision placeholders.                                                          |
| GPT-2                 | `<\|endoftext\|>` (50256)                                 | `<\|endoftext\|>` (50256)                           | none (base model)                          | The single special token serves as BOS, EOS and document separator.                                   |

The app never invents a universal BOS. Examples use each model's documented start: the
document separator for "minimal" contexts, the real chat template for chat contexts.

## Generation stopping

Automatic generation stops on the model's **stop tokens** only (from the generation
config), not on every special token — e.g. emitting `<|im_start|>` does not stop
generation. Stopping can be disabled.
