"""Pretraining, step 3: talk to your model. Type the start of a text, it continues it.

It's a base model (trained only to continue web text), so it doesn't answer
questions like a chatbot yet. Start sentences instead:
    "The heart pumps blood by"   rather than   "How does the heart work?"

Run from the repo root (works any time a checkpoint exists, even mid-training):
    uv run python pretraining/generate.py
"""

import mlx.core as mx
from tokenizers import Tokenizer

from train import OUT_DIR, TOKENIZER_FILE, GPT, generate  # pyright: ignore[reportImplicitRelativeImport]

TEMPERATURE = 0.8   # lower = safer and more repetitive, higher = wilder
N_TOKENS = 150


def main():
    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
    model = GPT(tokenizer.get_vocab_size())
    if not (OUT_DIR / "model.safetensors").exists():
        raise SystemExit("No checkpoint yet. Run train.py first (a checkpoint is saved every 500 iters).")
    model.load_weights(str(OUT_DIR / "model.safetensors"))
    mx.eval(model.parameters())

    print("Type the start of a text (empty line to quit).")
    while prompt := input("\n> ").strip():
        print(generate(model, tokenizer, prompt, N_TOKENS, TEMPERATURE))


if __name__ == "__main__":
    main()
