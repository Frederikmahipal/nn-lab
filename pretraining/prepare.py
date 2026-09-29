from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

# Settings 
N_CHARS = 2_500_000_000  
VOCAB_SIZE = 8192  
TOKENIZER_DOCS = 100_000 
VAL_FRACTION = 0.01  

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "fineweb"
TOKENIZER_FILE = Path(__file__).resolve().parent / "tokenizer.json"
EOT = "<|endoftext|>"  # marks where one document ends and the next begins
# The first file of the official 10-billion-token sample (stored as 726 blocks of 1000 documents).
SOURCE = "datasets/HuggingFaceFW/fineweb-edu/sample/10BT/000_00000.parquet"


def read_docs():
    """Stream documents block by block over the network, stop once we have N_CHARS of text."""
    docs, total = [], 0
    with HfFileSystem().open(SOURCE) as f:
        for batch in pq.ParquetFile(f).iter_batches(columns=["text"], batch_size=1000):
            for text in batch.column("text").to_pylist():
                docs.append(text)
                total += len(text)
            print(
                f"  streamed {len(docs):,} documents, {total / 1e6:.0f}M characters",
                end="\r",
            )
            if total >= N_CHARS:
                break
    print()
    return docs


def train_tokenizer(docs):
    """Byte-level BPE: start from single bytes, then repeatedly merge the most common pair."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(docs[:TOKENIZER_DOCS], trainer=trainer)
    return tokenizer


def encode_all(tokenizer, docs):
    """Turn every document into token ids, with an end-of-text marker after each one."""
    eot_id = tokenizer.token_to_id(EOT)
    parts = []
    for start in range(0, len(docs), 10_000):
        ids = []
        for enc in tokenizer.encode_batch(docs[start : start + 10_000]):
            ids.extend(enc.ids)
            ids.append(eot_id)
        parts.append(
            np.array(ids, dtype=np.uint16)
        )  # 8192 < 65536, so 2 bytes per token is enough
        print(
            f"  encoded {min(start + 10_000, len(docs)):,}/{len(docs):,} documents",
            end="\r",
        )
    print()
    return np.concatenate(parts)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Streaming FineWeb-Edu...")
    docs = read_docs()
    print(f"Kept {len(docs):,} documents ({sum(map(len, docs)) / 1e6:.0f}M characters)")

    if TOKENIZER_FILE.exists():
        # Keep the existing tokenizer: models trained with it would break with a new one.
        print(f"\nUsing existing tokenizer {TOKENIZER_FILE.name}")
        tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
    else:
        print(f"\nTraining a BPE tokenizer with {VOCAB_SIZE} word pieces...")
        tokenizer = train_tokenizer(docs)
        tokenizer.save(str(TOKENIZER_FILE))
    sample = "Photosynthesis is how plants turn sunlight into food."
    pieces = tokenizer.encode(sample).tokens
    print(f"Example: {sample!r}\n  -> {len(pieces)} tokens: {pieces}")

    print("\nTokenizing everything...")
    ids = encode_all(tokenizer, docs)
    split = int(len(ids) * (1 - VAL_FRACTION))
    ids[:split].tofile(DATA_DIR / "train.bin")
    ids[split:].tofile(DATA_DIR / "val.bin")
    print(f"{len(ids) / 1e6:.0f}M tokens total -> {DATA_DIR / 'train.bin'} and val.bin")


if __name__ == "__main__":
    main()
