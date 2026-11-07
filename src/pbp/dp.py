"""DP admission rule (your main part).

Rule from the proposal: count DISTINCT users per prefix in a window, add
discrete Laplace noise with scale Delta/eps, and share the prefix in the next
window only if the noisy count passes tau = 1 + (Delta/eps) * ln(Delta/delta).
"""
import math
from collections import defaultdict

import numpy as np


def tau_threshold(eps, delta, contrib):
    return 1 + (contrib / eps) * math.log(contrib / delta)


def discrete_laplace(scale, rng):
    """Difference of two geometric draws = discrete Laplace with this scale."""
    p = 1 - math.exp(-1.0 / scale)
    return int(rng.geometric(p)) - int(rng.geometric(p))


def admit(window_requests, eps, delta, contrib, rng):
    """Return the set of prefix hashes approved for the NEXT window.

    window_requests: list of (user, [prefix_hash, ...]) for one window.
    Keeps one random prompt per user, so each user adds at most `contrib` prefixes.
    """
    per_user = defaultdict(list)
    for user, hashes in window_requests:
        per_user[user].append(hashes)

    counts = defaultdict(int)
    for user, prompts in per_user.items():
        pick = prompts[int(rng.integers(len(prompts)))]
        for h in pick[:contrib]:
            counts[h] += 1

    tau = tau_threshold(eps, delta, contrib)
    scale = contrib / eps
    approved = set()
    for h, c in counts.items():
        if c + discrete_laplace(scale, rng) > tau:
            approved.add(h)
    return approved
