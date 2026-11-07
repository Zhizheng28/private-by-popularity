"""Cache simulator (build this in week 2).

Plan:
- Replay requests in time order.
- Shared cache: prefixes the policy allows to be shared. Pre-compute them at
  the START of each window so a hit only reveals the DP decision.
- Private cache: each user's own earlier prefixes (multi-turn reuse).
- Latency model: base + cost_per_token * uncached_tokens + noise.
- Record: hit rate, prompt tokens saved, latency per request.

Policies to plug in (each one decides what goes in the shared cache):
- full_sharing, per_user_isolation, dp_rule (pbp.dp.admit),
  safekv_style (share unless a PII detector flags it),
  prefixwall_style (isolate prefixes that look suspicious).
"""


def latency(uncached_tokens, rng, base=0.05, per_token=0.0002, noise=0.01):
    return base + per_token * uncached_tokens + abs(rng.normal(0, noise))


# TODO: class Simulator with run(requests, policy) -> metrics dict
