"""
Go/no-go check for "Private by Popularity" (EECS 498/598 project).

Question: in WildChat, are enough prompt prefixes sent by many DIFFERENT users
that a DP threshold (about 15 users for Delta=1, about 60 for Delta=4) would
still let the cache share them?

What it does:
  1. Streams WildChat-1M from Hugging Face (or makes fake data with --synthetic).
  2. Takes the first user prompt of each conversation, tokenizes it, and hashes
     its prefix at each checkpoint length (64, 256, 1024, 4096 tokens).
  3. Counts DISTINCT users per prefix in each time window (one day by default).
  4. Reports how many prefixes clear the DP thresholds, and a first estimate of
     prompt tokens saved under:
       - full sharing (reuse any prefix another user sent earlier),
       - the DP rule (one random prompt per user per window, discrete Laplace
         noise, threshold tau, approved prefixes used in the NEXT window).

Run (on your laptop):
  pip install datasets tiktoken
  python prefix_check.py --limit 50000      # quick test first
  python prefix_check.py                    # full dataset

Outputs go to ./prefix_check_out/ : summary.txt and top_prefixes.csv
"""

import argparse
import csv
import hashlib
import math
import os
import random
from collections import defaultdict
from datetime import datetime

import numpy as np


# ---------------------------------------------------------------- tokenizer

def make_tokenizer():
    """Use tiktoken (GPT-4 tokenizer) if possible, else split on spaces."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")

        def encode(text):
            return enc.encode(text, disallowed_special=())

        return encode, "tiktoken cl100k_base"
    except Exception:
        def encode(text):
            return text.split()

        return encode, "WHITESPACE FALLBACK (install tiktoken for real numbers)"


def prefix_hash(tokens, k):
    """Short stable hash of the first k tokens."""
    data = " ".join(str(t) for t in tokens[:k]).encode("utf-8")
    return hashlib.blake2b(data, digest_size=8).hexdigest()


# ---------------------------------------------------------------- data

def window_of(ts, window_hours):
    """Turn a timestamp into a window number."""
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    seconds = ts.timestamp()
    return int(seconds // (window_hours * 3600))


def first_user_prompt(conversation):
    for turn in conversation:
        if turn.get("role") == "user":
            return turn.get("content") or ""
    return ""


def load_wildchat(limit):
    from datasets import load_dataset
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    count = 0
    for row in ds:
        yield row["hashed_ip"], row["timestamp"], first_user_prompt(row["conversation"])
        count += 1
        if limit and count >= limit:
            break


def load_synthetic(limit):
    """Fake data with the same shape, only for testing the script."""
    rng = random.Random(0)
    templates = [
        "You are a helpful assistant. " * 30,
        "As a prompt generator for a generative AI called Midjourney, " * 40,
        "Translate the following text into English and keep the format: " * 20,
    ]
    words = "the a of my doctor said test result please explain why how what".split()
    n = limit or 20000
    for i in range(n):
        user = "user%d" % rng.randint(0, 3000)
        day = rng.randint(0, 9)
        ts = datetime(2024, 1, 1 + day, rng.randint(0, 23))
        if rng.random() < 0.3:
            text = rng.choice(templates) + " ".join(rng.choice(words) for _ in range(50))
        else:
            text = " ".join(rng.choice(words) for _ in range(rng.randint(5, 600)))
        yield user, ts, text


# ---------------------------------------------------------------- DP pieces

def tau_threshold(eps, delta, contrib):
    """tau = 1 + (Delta/eps) * ln(Delta/delta), from the proposal, Eq. (1)."""
    return 1 + (contrib / eps) * math.log(contrib / delta)


def discrete_laplace(scale, rng):
    """Discrete Laplace noise = difference of two geometric draws."""
    p = 1 - math.exp(-1.0 / scale)
    return int(rng.geometric(p)) - int(rng.geometric(p))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only read this many conversations (0 = all)")
    ap.add_argument("--synthetic", action="store_true", help="use fake data to test the script")
    ap.add_argument("--checkpoints", default="64,256,1024,4096")
    ap.add_argument("--window-hours", type=float, default=24.0)
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="prefix_check_out")
    args = ap.parse_args()

    checkpoints = sorted(int(x) for x in args.checkpoints.split(","))
    contrib = len(checkpoints)  # Delta: each user adds at most one prefix per checkpoint
    encode, tok_name = make_tokenizer()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    # ---- pass 1: read data, keep only hashes (not text) to save memory
    # each request: (window, user, total_tokens, [(k, hash), ...])
    requests = []
    snippets = {}          # hash -> text start, kept only once 2+ users share it
    first_user = {}        # hash -> first user who sent it
    lengths = []

    source = load_synthetic(args.limit) if args.synthetic else load_wildchat(args.limit)
    for user, ts, text in source:
        if not text:
            continue
        tokens = encode(text)
        lengths.append(len(tokens))
        hashes = []
        for k in checkpoints:
            if len(tokens) >= k:
                h = prefix_hash(tokens, k)
                hashes.append((k, h))
                if h not in first_user:
                    first_user[h] = user
                elif first_user[h] != user and h not in snippets:
                    snippets[h] = text[:120].replace("\n", " ")
        requests.append((window_of(ts, args.window_hours), user, len(tokens), hashes))
        if len(requests) % 50000 == 0:
            print("read", len(requests), "prompts")

    requests.sort(key=get_window)
    n = len(requests)
    total_tokens = sum(r[2] for r in requests)
    print("loaded", n, "prompts")

    # ---- distinct users per prefix (whole dataset and per window)
    users_all = defaultdict(set)
    users_win = defaultdict(set)   # (window, hash) -> users
    for win, user, _, hashes in requests:
        for k, h in hashes:
            users_all[(k, h)].add(user)
            users_win[(win, h)].add(user)

    tau1 = tau_threshold(args.eps, args.delta, 1)
    tauD = tau_threshold(args.eps, args.delta, contrib)

    # ---- full sharing: reuse the longest prefix ANOTHER user sent earlier
    seen_by = {}  # hash -> set of users (capped at 2, enough to know "someone else")
    saved_full = 0
    for win, user, _, hashes in requests:
        best = 0
        for k, h in hashes:
            others = seen_by.get(h)
            if others and (len(others) > 1 or user not in others):
                best = k
        saved_full += best
        for k, h in hashes:
            s = seen_by.setdefault(h, set())
            if len(s) < 2:
                s.add(user)

    # ---- DP rule: approve prefixes in window t, use them in window t+1
    by_window = defaultdict(list)
    for r in requests:
        by_window[r[0]].append(r)
    windows = sorted(by_window)

    approved_next = set()
    saved_dp = 0
    approved_counts = []
    for win in windows:
        reqs = by_window[win]
        # use what the previous window approved
        approved_now = approved_next
        for _, _, _, hashes in reqs:
            best = 0
            for k, h in hashes:
                if h in approved_now:
                    best = k
            saved_dp += best
        # one random prompt per user in this window
        per_user = defaultdict(list)
        for r in reqs:
            per_user[r[1]].append(r)
        counts = defaultdict(int)
        for user, items in per_user.items():
            pick = items[int(rng.integers(len(items)))]
            for k, h in pick[3]:
                counts[h] += 1
        approved_next = set()
        for h, c in counts.items():
            if c + discrete_laplace(contrib / args.eps, rng) > tauD:
                approved_next.add(h)
        approved_counts.append(len(approved_next))

    # ---- write summary
    lengths_arr = np.array(lengths)
    lines = []
    lines.append("Tokenizer: %s" % tok_name)
    lines.append("Prompts: %d   windows: %d (%.0f h each)   users: %d"
                 % (n, len(windows), args.window_hours, len({r[1] for r in requests})))
    lines.append("Prompt length (tokens): median %d, 90th pct %d"
                 % (np.median(lengths_arr), np.percentile(lengths_arr, 90)))
    lines.append("")
    lines.append("Thresholds (eps=%g, delta=%g): tau=%.1f for Delta=1, tau=%.1f for Delta=%d"
                 % (args.eps, args.delta, tau1, tauD, contrib))
    lines.append("")
    lines.append("checkpoint | prompts this long | prefixes with >=2 users | >=tau(D=1) | >=tau(D=%d)  [whole dataset]" % contrib)
    for k in checkpoints:
        long_enough = int((lengths_arr >= k).sum())
        sizes = [len(u) for (kk, _), u in users_all.items() if kk == k]
        two = sum(1 for s in sizes if s >= 2)
        a = sum(1 for s in sizes if s >= tau1)
        b = sum(1 for s in sizes if s >= tauD)
        lines.append("%10d | %17d | %23d | %10d | %10d" % (k, long_enough, two, a, b))
    lines.append("")
    per_win_pass = sum(1 for u in users_win.values() if len(u) >= tauD)
    lines.append("Per-window (window, prefix) pairs with >= tau(D=%d) users: %d" % (contrib, per_win_pass))
    lines.append("DP-approved prefixes per window: mean %.1f, max %d"
                 % (np.mean(approved_counts) if approved_counts else 0, max(approved_counts or [0])))
    lines.append("")
    lines.append("Prompt tokens saved (share of all prompt tokens):")
    lines.append("  full sharing  : %.2f%%" % (100.0 * saved_full / max(total_tokens, 1)))
    lines.append("  DP rule       : %.2f%%" % (100.0 * saved_dp / max(total_tokens, 1)))
    lines.append("  (per-user isolation saves 0% here because only first prompts are used)")
    lines.append("")
    lines.append("GO if DP saves a clear share of what full sharing saves.")
    lines.append("NO-GO -> backup plan 1: --window-hours 168, --eps 2, or --checkpoints 64,256")

    summary = "\n".join(lines)
    print("\n" + summary)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(summary + "\n")

    # ---- top shared prefixes, to see WHAT is popular (templates? jailbreaks?)
    rows = []
    for (k, h), u in users_all.items():
        if len(u) >= 2:
            rows.append((len(u), k, h, snippets.get(h, "")))
    rows.sort(key=get_count, reverse=True)
    with open(os.path.join(args.out, "top_prefixes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["distinct_users", "checkpoint", "hash", "text_start"])
        for row in rows[:500]:
            w.writerow(row)
    print("\nWrote", os.path.join(args.out, "summary.txt"), "and top_prefixes.csv")


def get_window(r):
    return r[0]


def get_count(row):
    return row[0]


if __name__ == "__main__":
    main()
