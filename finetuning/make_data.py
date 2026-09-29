"""Fine-tuning, step 1: create question-and-answer training data with a bigger model in LM Studio.

The model in LM Studio acts as the "teacher": it answers ~200 questions in a
simple, explain-like-I'm-7 style. A smaller "student" model is then fine-tuned
on those answers, so it explains things simply WITHOUT being told to.
(The adapters-* folders are LoRA runs of this on Qwen, from before the fresh start.)

Before running: open LM Studio, load the teacher model, and start the server
(Developer tab -> Start server). Then, from the repo root:
    uv run python finetuning/make_data.py
"""

import json
import random
import re
import urllib.request
from pathlib import Path
from platform import system

LMSTUDIO_URL = "http://localhost:1234/v1"
MODEL_ID = "qwen/qwen3-4b-2507"  # the teacher model in LM Studio (exact id from the Developer tab)
OUT_DIR = Path(__file__).resolve().parent / "data"

# The style we want the small model to learn. Change this to anything you like.
STYLE = (
    "Explain everything as if talking to a curious 7-year-old. "
    "Use short sentences, simple everyday words, and a fun comparison "
    "to something a child knows (toys, food, animals, playgrounds). "
    "Keep the facts correct. Answer in 2 to 4 sentences. "
    "Do not use lists or headings."
)

TOPICS = [
    "photosynthesis",
    "gravity",
    "the internet",
    "black holes",
    "vaccines",
    "climate change",
    "electricity",
    "the moon",
    "volcanoes",
    "DNA",
    "neural networks",
    "Python programming",
    "passwords",
    "email",
    "Wi-Fi",
    "batteries",
    "smartphones",
    "GPS",
    "cloud storage",
    "video games",
    "coffee",
    "baking bread",
    "cooking pasta",
    "vegetables",
    "sleep",
    "exercise",
    "drinking water",
    "stress",
    "learning a language",
    "reading",
    "saving money",
    "budgeting",
    "job interviews",
    "public speaking",
    "friendship",
    "the Roman Empire",
    "the pyramids",
    "democracy",
    "the printing press",
    "trains",
    "rainbows",
    "earthquakes",
    "the ocean",
    "bees",
    "dinosaurs",
    "music",
    "photography",
    "chess",
    "football",
    "recycling",
]

TEMPLATES = [
    "What is {}?",
    "Can you explain {} simply?",
    "Why does {} matter?",
    "Give me a quick tip about {}.",
]


def request(path, payload=None):
    url = f"{LMSTUDIO_URL}/{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.load(resp)


def pick_model():
    if MODEL_ID:
        return MODEL_ID
    models = [m["id"] for m in request("models")["data"]]
    chat_models = [m for m in models if "embed" not in m.lower()]
    if not chat_models:
        raise SystemExit(
            "LM Studio reports no chat models. Load one and start the server."
        )
    return chat_models[0]


def ask(model_id, question):
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": STYLE},
            {"role": "user", "content": question},
        ],
        "temperature": 0.8,
        "max_tokens": 400,
    }
    text = (
        request("chat/completions", payload)["choices"][0]["message"].get("content")
        or ""
    )
    # Some models "think" out loud first; we only want the final answer.
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def save(examples):
    OUT_DIR.mkdir(exist_ok=True)
    random.shuffle(examples)
    n_valid = max(5, len(examples) // 10)
    splits = {"valid.jsonl": examples[:n_valid], "train.jsonl": examples[n_valid:]}
    for name, rows in splits.items():
        with open(OUT_DIR / name, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rows)} examples to {OUT_DIR / name}")


def main():
    random.seed(0)
    try:
        model_id = pick_model()
    except OSError:
        raise SystemExit("Can't reach LM Studio. Is the server running on port 1234?")
    print(f"Teacher model: {model_id}\n")

    questions = [t.format(topic) for topic in TOPICS for t in TEMPLATES]
    random.shuffle(questions)
    examples = []

    try:
        for i, q in enumerate(questions, 1):
            answer = ask(model_id, q)
            if not answer:
                print(f"[{i}/{len(questions)}] (empty answer, skipped)")
                continue
            # Note: NO system prompt in the saved data. The student must learn
            # the simple style on its own, from the answers alone.
            examples.append(
                {
                    "messages": [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": answer},
                    ]
                }
            )
            print(f"[{i}/{len(questions)}] {q}\n   -> {answer[:90]}...")
    except KeyboardInterrupt:
        print("\nStopped early, saving what we have.")

    save(examples)


if __name__ == "__main__":
    main()
