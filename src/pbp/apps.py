"""App system prompts: put a shared system prompt in front of each user's message."""
import hashlib

import numpy as np


def load_system_prompts(n):
    """First n prompts from the public awesome-chatgpt-prompts dataset."""
    from datasets import load_dataset
    ds = load_dataset("fka/awesome-chatgpt-prompts", split="train")
    prompts = []
    for row in ds:
        text = (row.get("prompt") or "").strip()
        if text:
            prompts.append(text)
        if len(prompts) >= n:
            break
    return prompts


def synthetic_system_prompts(n):
    prompts = []
    for i in range(n):
        prompts.append(("You are assistant number %d for a company app. " % i) * 15)
    return prompts


def make_app_picker(n_apps, zipf_s=1.1, seed=0):
    """Return a function user -> app index. App 0 is the most popular (Zipf).
    The same user always gets the same app."""
    weights = np.array([1.0 / ((r + 1) ** zipf_s) for r in range(n_apps)])
    cdf = np.cumsum(weights / weights.sum())

    def pick(user):
        key = ("%s|%d" % (user, seed)).encode("utf-8")
        digest = hashlib.blake2b(key, digest_size=8).digest()
        u = int.from_bytes(digest, "big") / float(2 ** 64)
        return min(int(np.searchsorted(cdf, u)), n_apps - 1)

    return pick
