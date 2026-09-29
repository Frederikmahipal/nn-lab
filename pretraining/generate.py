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
