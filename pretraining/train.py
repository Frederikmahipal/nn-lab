"""Pretraining, step 2: train  GPT from scratch on FineWeb-Edu.

Run from the repo root (after prepare.py):
    uv run python pretraining/train.py
"""

import json
import math
import time
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from tokenizers import Tokenizer

# ---- Settings ----
BLOCK_SIZE = 256      # how many tokens the model can look back at (~1000 characters)
DIMS = 384            # size of each token's vector inside the model
N_HEADS = 6           # attention heads per layer
N_LAYERS = 6          # transformer blocks stacked on top of each other
BATCH_SIZE = 32
LEARNING_RATE = 1e-3  # peak; warms up to this, then slowly decays
WARMUP = 200
ITERS = 20_000
EVAL_EVERY = 500      # also saves a checkpoint and prints a sample

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data" / "fineweb"
TOKENIZER_FILE = HERE / "tokenizer.json"
OUT_DIR = HERE / "checkpoints"
SAMPLE_PROMPT = "The most important thing about the ocean is"


# ---- Data ----
def load_tokens(name):
    # memmap: the file stays on disk, we only read the pieces we pick.
    return np.memmap(DATA_DIR / name, dtype=np.uint16, mode="r")


def get_batch(data):
    """Pick random chunks of text. Target = the same chunk shifted one token."""
    starts = np.random.randint(0, len(data) - BLOCK_SIZE - 1, BATCH_SIZE)
    x = np.stack([data[s : s + BLOCK_SIZE] for s in starts]).astype(np.int32)
    y = np.stack([data[s + 1 : s + BLOCK_SIZE + 1] for s in starts]).astype(np.int32)
    return mx.array(x), mx.array(y)


# ---- Model ----
class CausalSelfAttention(nn.Module):
    """Each token looks back at earlier tokens and decides which ones matter."""

    def __init__(self, dims, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(dims, 3 * dims, bias=False)
        self.proj = nn.Linear(dims, dims, bias=False)

    def __call__(self, x):
        B, T, C = x.shape
        head_dim = C // self.n_heads
        q, k, v = mx.split(self.qkv(x), 3, axis=-1)

        # (B, T, C) -> (B, heads, T, head_dim)
        def split_heads(t):
            return t.reshape(B, T, self.n_heads, head_dim).transpose(0, 2, 1, 3)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        # scores -> causal mask -> softmax -> weighted sum, all in one call
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1 / math.sqrt(head_dim), mask="causal"
        )
        return self.proj(out.transpose(0, 2, 1, 3).reshape(B, T, C))


class Block(nn.Module):
    """Attention (communicate between positions) + MLP (think per position)."""

    def __init__(self, dims, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(dims)
        self.attn = CausalSelfAttention(dims, n_heads)
        self.ln2 = nn.LayerNorm(dims)
        self.mlp = nn.Sequential(
            nn.Linear(dims, 4 * dims), nn.GELU(), nn.Linear(4 * dims, dims)
        )

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, DIMS)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, DIMS)
        self.blocks = [Block(DIMS, N_HEADS) for _ in range(N_LAYERS)]
        self.ln_f = nn.LayerNorm(DIMS)

    def __call__(self, idx):
        T = idx.shape[1]
        x = self.tok_emb(idx) + self.pos_emb(mx.arange(T))
        for block in self.blocks:
            x = block(x)
        # Weight tying: reuse the input embedding table as the output layer.
        # Saves vocab_size * DIMS parameters, and works as well or better.
        return self.tok_emb.as_linear(self.ln_f(x))


def loss_fn(model, x, y):
    return nn.losses.cross_entropy(model(x), y, reduction="mean")


def estimate_loss(model, data, n_batches=20):
    losses = [loss_fn(model, *get_batch(data)).item() for _ in range(n_batches)]
    return sum(losses) / len(losses)


def generate(model, tokenizer, prompt, n_tokens=100, temperature=0.8):
    """Predict one token, append it, repeat, until n_tokens or end-of-text."""
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    idx = tokenizer.encode(prompt).ids
    for _ in range(n_tokens):
        context = mx.array(idx[-BLOCK_SIZE:])[None]
        logits = model(context)[0, -1] / temperature
        next_id = mx.random.categorical(logits).item()
        if next_id == eot_id:
            break
        idx.append(next_id)
    return tokenizer.decode(idx)


# ---- Checkpoints ----
def save(model, optimizer, it):
    OUT_DIR.mkdir(exist_ok=True)
    model.save_weights(str(OUT_DIR / "model.safetensors"))
    mx.save_safetensors(str(OUT_DIR / "optimizer.safetensors"), dict(tree_flatten(optimizer.state)))
    (OUT_DIR / "state.json").write_text(json.dumps({"iter": it}))


def load(model, optimizer):
    """Continue from the last checkpoint if there is one. Returns the iteration to start at."""
    if not (OUT_DIR / "state.json").exists():
        return 1
    model.load_weights(str(OUT_DIR / "model.safetensors"))
    saved = mx.load(str(OUT_DIR / "optimizer.safetensors"))
    assert isinstance(saved, dict)
    optimizer.state = tree_unflatten(list(saved.items()))
    it = json.loads((OUT_DIR / "state.json").read_text())["iter"]
    print(f"Resuming from checkpoint at iter {it}")
    return it + 1


# ---- Training ----
def main():
    np.random.seed(0)
    mx.random.seed(0)

    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
    train_data, val_data = load_tokens("train.bin"), load_tokens("val.bin")
    tokens_per_iter = BATCH_SIZE * BLOCK_SIZE
    print(f"{len(train_data) / 1e6:.0f}M training tokens, vocab {tokenizer.get_vocab_size()}")
    print(f"{ITERS} iters x {tokens_per_iter} tokens = {ITERS * tokens_per_iter / 1e6:.0f}M tokens seen")

    model = GPT(tokenizer.get_vocab_size())
    mx.eval(model.parameters())
    n_params = sum(v.size for _, v in tree_flatten(model.parameters()) if isinstance(v, mx.array))
    print(f"Model parameters: {n_params / 1e6:.1f}M")

    schedule = optim.join_schedules(
        [
            optim.linear_schedule(0, LEARNING_RATE, WARMUP),
            optim.cosine_decay(LEARNING_RATE, ITERS - WARMUP, LEARNING_RATE / 10),
        ],
        [WARMUP],
    )
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=0.1)
    start = load(model, optimizer)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    state = [model.state, optimizer.state]

    # mx.compile fuses the whole step into one optimized graph: noticeably faster.
    @partial(mx.compile, inputs=state, outputs=state)
    def step(x, y):
        loss, grads = loss_and_grad(model, x, y)
        grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)  # tame rare huge updates
        optimizer.update(model, grads)
        return loss

    it = start - 1
    tic = time.perf_counter()
    try:
        for it in range(start, ITERS + 1):
            loss = step(*get_batch(train_data))
            mx.eval(state)

            if it % 50 == 0:
                secs = time.perf_counter() - tic
                tok_s = 50 * tokens_per_iter / secs
                print(f"iter {it:6d} | loss {loss.item():.3f} | {tok_s:,.0f} tok/s")
                tic = time.perf_counter()

            if it % EVAL_EVERY == 0 or it == ITERS:
                print(f"\n>>> val loss {estimate_loss(model, val_data):.3f}")
                print(f">>> {generate(model, tokenizer, SAMPLE_PROMPT, n_tokens=60)!r}\n")
                save(model, optimizer, it)
                tic = time.perf_counter()
    except KeyboardInterrupt:
        print("\nStopped. Saving checkpoint...")
        save(model, optimizer, it - 1)  # iter `it` may be half-finished

    print(f"Saved to {OUT_DIR}. Run again to continue, or try generate.py.")


if __name__ == "__main__":
    main()
