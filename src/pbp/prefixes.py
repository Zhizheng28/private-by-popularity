"""Tokenize prompts and hash prefixes at checkpoint lengths."""
import hashlib

CHECKPOINTS = (64, 256, 1024, 4096)


def make_tokenizer():
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")

    def encode(text):
        return enc.encode(text, disallowed_special=())

    return encode


def prefix_hash(tokens, k):
    data = " ".join(str(t) for t in tokens[:k]).encode("utf-8")
    return hashlib.blake2b(data, digest_size=8).hexdigest()


def checkpoint_hashes(tokens, checkpoints=CHECKPOINTS):
    """[(k, hash)] for every checkpoint the prompt is long enough to reach."""
    out = []
    for k in checkpoints:
        if len(tokens) >= k:
            out.append((k, prefix_hash(tokens, k)))
    return out
