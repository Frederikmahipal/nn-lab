"""Pretraining, step 3: talk to your model. Give it the start of a text, it continues it.

It's a base model (trained only to continue web text), so it doesn't answer
questions like a chatbot yet. Start sentences instead:
    "The heart pumps blood by"   rather than   "How does the heart work?"

Run from the repo root (works any time a checkpoint exists, even mid-training):
    uv run python pretraining/generate.py                      # the PROMPTS in train.py
    uv run python pretraining/generate.py "Once upon a time"   # your own prompt(s)
"""

import sys

import mlx.core as mx
from tokenizers import Tokenizer
from train import (  # pyright: ignore[reportImplicitRelativeImport]
    GPT,
    OUT_DIR,
    PROMPTS,
    TOKENIZER_FILE,
    generate,
)

TEMPERATURE = 0.7  # lower = safer and more repetitive, higher = wilder
TOP_P = 0.9  # only pick from the likeliest tokens covering 90% of the probability (1.0 = off)
REPETITION_PENALTY = 1.2  # >1 makes already-used tokens less likely, fights loops (1.0 = off)
N_TOKENS = 150


def main():
    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
    model = GPT(tokenizer.get_vocab_size())
    if not (OUT_DIR / "model.safetensors").exists():
        raise SystemExit(
            "No checkpoint yet. Run train.py first (a checkpoint is saved every 500 steps)."
        )
    model.load_weights(str(OUT_DIR / "model.safetensors"))
    mx.eval(model.parameters())

    for prompt in sys.argv[1:] or PROMPTS:
        print(f"\n> {prompt}")
        print(
            generate(
                model, tokenizer, prompt, N_TOKENS, TEMPERATURE, TOP_P, REPETITION_PENALTY
            )
        )


if __name__ == "__main__":
    main()
