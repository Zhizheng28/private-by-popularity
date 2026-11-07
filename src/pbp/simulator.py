"""Cache simulator pieces used by the attack experiments.

Latency model (time to first token):
    latency = base + per_token * (prompt tokens NOT served from cache) + noise
A cache hit at checkpoint k means the first k tokens are served from cache.

Policies (what the attacker's probe can hit):
  full       : any prefix any user sent earlier is shared
  isolation  : nothing is shared across users
  dp         : only prefixes the DP rule approved in the previous window are
               shared, and they are pre-computed at the start of the window,
               so a hit reveals only the DP decision
"""
import math

import numpy as np

from pbp.dp import tau_threshold

BASE = 0.050        # seconds
PER_TOKEN = 0.0002  # seconds per uncached prompt token (about 5,000 tokens/s prefill)
NOISE_SD = 0.010    # seconds


def latency(prompt_tokens, cached_tokens, rng, size=None):
    noise = rng.normal(0.0, NOISE_SD, size)
    return BASE + PER_TOKEN * (prompt_tokens - cached_tokens) + np.abs(noise)


def discrete_laplace_array(scale, size, rng):
    p = 1 - math.exp(-1.0 / scale)
    return rng.geometric(p, size) - rng.geometric(p, size)


def dp_admit_arrays(user_ids, hash_lists, eps, delta, contrib, rng):
    """Vectorized DP admission for one window.

    user_ids  : array of user ids, one per request
    hash_lists: list of int64 arrays, one per checkpoint, aligned with user_ids
                (0 = prompt too short for that checkpoint)
    Returns the set of approved hashes.
    """
    n = len(user_ids)
    order = rng.permutation(n)
    seen = set()
    keep = []
    for i in order:                      # one random prompt per user
        u = user_ids[i]
        if u not in seen:
            seen.add(u)
            keep.append(i)
    keep = np.array(keep, dtype=np.int64)

    parts = []
    for col in hash_lists[:contrib]:
        vals = col[keep]
        parts.append(vals[vals != 0])
    all_h = np.concatenate(parts) if parts else np.array([], dtype=np.int64)
    if len(all_h) == 0:
        return set()
    uniq, counts = np.unique(all_h, return_counts=True)
    noisy = counts + discrete_laplace_array(contrib / eps, len(counts), rng)
    tau = tau_threshold(eps, delta, contrib)
    return set(uniq[noisy > tau].tolist())
