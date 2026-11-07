"""
Goal 3: can the attacker still tell whether a victim sent a private prompt?

TIMING AUDIT
  - Take one month of real traffic from the saved prefix file.
  - Add N victims. Each sends one private prompt (400 tokens) that nobody else sends.
    In the app setting, the private text comes after the most popular app's
    system prompt, so its first 64 tokens are shared with many users.
  - Run the DP rule on that month (real traffic + victims).
  - Next month, the attacker probes each victim's prompt, plus N "control"
    prompts that nobody sent, and times the responses.
  - Policies: full sharing, per-user isolation, DP rule.
  Reports, per policy: attack AUC (0.5 = guessing), best attack accuracy,
  the audit's p-value (< 0.05 = sharing detected), and how many victim
  prefixes DP wrongly shared (should be 0).

SYBIL ATTACK
  - Attacker makes n fake accounts that send the guessed prefix, then checks
    whether it got shared. Measures how well that reveals the victim.

Run:
  python scripts/run_audit.py --data data/processed/prefixes_apps0.parquet --name wildchat
  python scripts/run_audit.py --data data/processed/prefixes_apps100.parquet --name apps100 --app
"""
import argparse
import os

import numpy as np
import pandas as pd

from pbp.attacks import attack_auc, audit_p_value, best_accuracy, sybil_success
from pbp.dp import tau_threshold
from pbp.simulator import dp_admit_arrays, latency

PROMPT_TOKENS = 400


def pick_busy_window(df, window_hours):
    win = df["ts"].values // int(window_hours * 3600)
    values, counts = np.unique(win, return_counts=True)
    return win, int(values[np.argmax(counts)])


def most_common_hash(col):
    vals = col[col != 0]
    uniq, counts = np.unique(vals, return_counts=True)
    return int(uniq[np.argmax(counts)])


def cached_tokens_for(probe_hashes, shared, checkpoints):
    """Longest checkpoint whose prefix hash is in the shared set."""
    best = 0
    for k, h in zip(checkpoints, probe_hashes):
        if h != 0 and h in shared:
            best = k
    return best


def run_audit(df, args, rng):
    checkpoints = [int(x) for x in args.checkpoints.split(",")]
    contrib = len(checkpoints)
    win, w = pick_busy_window(df, args.window_hours)
    month = df[win == w]
    print("audit window has", len(month), "real requests from", month["user"].nunique(), "users")

    # victims and controls: unique hashes at every checkpoint
    n = args.victims
    def_rng = np.random.default_rng(12345)
    victim_h = {k: def_rng.integers(1, 2 ** 62, n) for k in checkpoints}
    control_h = {k: def_rng.integers(1, 2 ** 62, n) for k in checkpoints}
    if args.app:
        # the 64-token prefix is the popular app's system prompt, shared by many
        shared64 = most_common_hash(month["h64"].values)
        victim_h[64] = np.full(n, shared64)
        control_h[64] = np.full(n, shared64)

    # DP admission on real traffic + victims
    users = np.concatenate([month["user"].values.astype(str),
                            np.array(["victim%d" % i for i in range(n)])])
    hash_cols = []
    for k in checkpoints:
        hash_cols.append(np.concatenate([month["h%d" % k].values, victim_h[k]]))
    approved = dp_admit_arrays(users, hash_cols, args.eps, args.delta, contrib, rng)

    # cached tokens for each probe under each policy
    shared_prefix = 64 if args.app else 0   # app system prompt part everyone shares
    dp_v = np.array([cached_tokens_for([victim_h[k][i] for k in checkpoints], approved, checkpoints)
                     for i in range(n)])
    dp_c = np.array([cached_tokens_for([control_h[k][i] for k in checkpoints], approved, checkpoints)
                     for i in range(n)])
    cached = {
        # full sharing: the victim's whole prompt is in the shared cache
        "full sharing": (np.full(n, PROMPT_TOKENS), np.full(n, shared_prefix)),
        "per-user isolation": (np.zeros(n, dtype=int), np.zeros(n, dtype=int)),
        "DP rule": (dp_v, dp_c),
    }
    rows = []
    lat_store = {}
    for name, (v_cached, c_cached) in cached.items():
        v_lat = latency(PROMPT_TOKENS, v_cached, rng, n)
        c_lat = latency(PROMPT_TOKENS, c_cached, rng, n)
        lat_store[name] = (v_lat, c_lat)
        leaked = 0
        if name == "DP rule":
            for k in checkpoints:
                if not (args.app and k == 64):
                    leaked += int(np.isin(victim_h[k], list(approved)).sum()) if approved else 0
        rows.append({
            "policy": name,
            "attack_auc": round(attack_auc(v_lat, c_lat), 3),
            "attack_accuracy": round(best_accuracy(v_lat, c_lat), 3),
            "audit_p_value": float("%.3g" % audit_p_value(v_lat, c_lat)),
            "victim_private_prefixes_shared": leaked,
        })
    return pd.DataFrame(rows), lat_store


def run_sybil(args, rng):
    checkpoints = [int(x) for x in args.checkpoints.split(",")]
    contrib = len(checkpoints)
    tau = tau_threshold(args.eps, args.delta, contrib)
    rows = []
    for n_fake in range(0, int(tau) + 15):
        p_sent, p_not = sybil_success(n_fake, args.eps, args.delta, contrib, 200000, rng)
        rows.append({"fake_accounts": n_fake, "p_shared_if_victim_sent": p_sent,
                     "p_shared_if_not": p_not,
                     "attack_accuracy": 0.5 + 0.5 * abs(p_sent - p_not)})
    return pd.DataFrame(rows), tau


def plot(lat_store, sybil, tau, name, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs("results/figures", exist_ok=True)

    fig, axes = plt.subplots(1, len(lat_store), figsize=(11, 3.2), sharey=True)
    for ax, (policy, (v, c)) in zip(axes, lat_store.items()):
        bins = np.linspace(0.0, 0.2, 50)
        ax.hist(c * 1000, bins=bins * 1000, alpha=0.6, label="never sent")
        ax.hist(v * 1000, bins=bins * 1000, alpha=0.6, label="victim sent it")
        ax.set_title(policy)
        ax.set_xlabel("response time (ms)")
    axes[0].set_ylabel("probes")
    axes[0].legend(fontsize=7)
    fig.suptitle("Timing audit (%s, eps=%g, checkpoints %s)" % (name, args.eps, args.checkpoints))
    fig.tight_layout()
    path = "results/figures/audit_%s.png" % name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(sybil["fake_accounts"], sybil["attack_accuracy"], marker="o", markersize=3)
    ax.axvline(tau, linestyle="--", color="gray", label="threshold tau = %.1f" % tau)
    ax.axhline(0.5, linestyle=":", color="black", label="guessing")
    ax.set_xlabel("fake accounts sending the guessed prefix")
    ax.set_ylabel("attack accuracy")
    ax.set_title("Sybil attack (eps=%g, Delta=%d)" % (args.eps, len(args.checkpoints.split(","))))
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = "results/figures/sybil_%s.png" % name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--app", action="store_true", help="victims use the popular app's system prompt")
    ap.add_argument("--checkpoints", default="64")
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=1e-6)
    ap.add_argument("--window-hours", type=float, default=720)
    ap.add_argument("--victims", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    df = pd.read_parquet(args.data)

    audit, lat_store = run_audit(df, args, rng)
    sybil, tau = run_sybil(args, rng)

    os.makedirs("results", exist_ok=True)
    audit.to_csv("results/audit_%s.csv" % args.name, index=False)
    sybil.to_csv("results/sybil_%s.csv" % args.name, index=False)
    print("\nTIMING AUDIT")
    print(audit.to_string(index=False))
    peak = sybil.loc[sybil["attack_accuracy"].idxmax()]
    print("\nSYBIL ATTACK: best accuracy %.3f with %d fake accounts (tau = %.1f)"
          % (peak["attack_accuracy"], peak["fake_accounts"], tau))
    plot(lat_store, sybil, tau, args.name, args)


if __name__ == "__main__":
    main()
