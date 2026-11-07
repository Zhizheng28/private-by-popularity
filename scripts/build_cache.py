"""
Step A: read WildChat ONCE and save prefix hashes to a small parquet file.
The sweep (scripts/sweep.py) then tests many settings on this file in minutes.

Saves one row per conversation (first user prompt):
  user, ts (seconds), n_tokens, h32, h64, h128, h256, h512, h1024, h2048, h4096
h<k> is a 64-bit hash of the first k tokens, or 0 if the prompt is shorter than k.

Run:
  python scripts/build_cache.py                  # WildChat only
  python scripts/build_cache.py --apps 100       # with app system prompts
  python scripts/build_cache.py --synthetic      # fake data, just to test
"""
import argparse
import hashlib
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd

from pbp.apps import load_system_prompts, make_app_picker, synthetic_system_prompts

ALL_CHECKPOINTS = (32, 64, 128, 256, 512, 1024, 2048, 4096)


def make_tokenizer(synthetic):
    if synthetic:
        def encode_words(text):
            return text.split()
        return encode_words
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")

    def encode(text):
        return enc.encode(text, disallowed_special=())
    return encode


def hash64(tokens, k):
    data = " ".join(str(t) for t in tokens[:k]).encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big", signed=True)
    return value if value != 0 else 1   # 0 is reserved for "too short"


def to_seconds(ts):
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return int(ts.timestamp())


def iter_wildchat(limit):
    from datasets import load_dataset
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        text = ""
        for turn in row["conversation"]:
            if turn.get("role") == "user":
                text = turn.get("content") or ""
                break
        yield row["hashed_ip"], row["timestamp"], text


def iter_synthetic(limit):
    rng = random.Random(0)
    templates = [
        "You are a helpful assistant. " * 30,
        "As a prompt generator for a generative AI called Midjourney, " * 40,
    ]
    words = "the a of my doctor said test result please explain why how what".split()
    for i in range(limit or 20000):
        user = "user%d" % rng.randint(0, 3000)
        ts = datetime(2024, 1 + rng.randint(0, 5), 1 + rng.randint(0, 27), rng.randint(0, 23))
        if rng.random() < 0.3:
            text = rng.choice(templates) + " ".join(rng.choice(words) for _ in range(50))
        else:
            text = " ".join(rng.choice(words) for _ in range(rng.randint(5, 600)))
        yield user, ts, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apps", type=int, default=0, help="number of app system prompts (0 = off)")
    ap.add_argument("--zipf", type=float, default=1.1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out = args.out or "data/processed/prefixes_apps%d%s.parquet" % (args.apps, "_synth" if args.synthetic else "")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    encode = make_tokenizer(args.synthetic)
    system_prompts = []
    if args.apps:
        if args.synthetic:
            system_prompts = synthetic_system_prompts(args.apps)
        else:
            system_prompts = load_system_prompts(args.apps)
        pick_app = make_app_picker(len(system_prompts), args.zipf)
        print("using", len(system_prompts), "app system prompts")

    users, times, lengths = [], [], []
    hashes = {k: [] for k in ALL_CHECKPOINTS}
    source = iter_synthetic(args.limit) if args.synthetic else iter_wildchat(args.limit)
    for user, ts, text in source:
        if not text:
            continue
        if system_prompts:
            text = system_prompts[pick_app(user)] + "\n\n" + text
        tokens = encode(text)
        users.append(user)
        times.append(to_seconds(ts))
        lengths.append(len(tokens))
        for k in ALL_CHECKPOINTS:
            hashes[k].append(hash64(tokens, k) if len(tokens) >= k else 0)
        if len(users) % 50000 == 0:
            print("read", len(users), "prompts")

    df = pd.DataFrame({"user": users, "ts": np.array(times, dtype=np.int64),
                       "n_tokens": np.array(lengths, dtype=np.int32)})
    for k in ALL_CHECKPOINTS:
        df["h%d" % k] = np.array(hashes[k], dtype=np.int64)
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    df.to_parquet(out, index=False)
    print("saved", len(df), "rows to", out)


if __name__ == "__main__":
    main()
