"""
Step B: test many settings on the saved prefix file (no re-download).

For every combination of window length, checkpoint set (Delta = how many
checkpoints), epsilon, and random seed, it measures:
  - full_pct : prompt tokens saved by full sharing (the upper bound)
  - dp_pct   : prompt tokens saved by the DP rule
  - kept     : dp_pct / full_pct (share of the possible benefit we keep)
  - approved_per_window : how many prefixes DP shares per window
  - singletons_shared   : approved prefixes that only ONE user sent
                          (should be ~0; a privacy sanity check)

Run:
  python scripts/sweep.py --data data/processed/prefixes_apps0.parquet --name wildchat
  python scripts/sweep.py --data data/processed/prefixes_apps100.parquet --name apps100

Writes results/sweep_<name>.csv and results/figures/sweep_<name>_<window>h.png
"""
import argparse
import math
import os

import numpy as np
import pandas as pd

from pbp.dp import tau_threshold

CHECKPOINT_SETS = ["256", "64,256", "64,256,1024", "64,256,1024,4096"]
EPS_LIST = [0.25, 0.5, 1.0, 2.0, 4.0]
WINDOWS = [168, 720]


def full_sharing_saved(df, checkpoints):
    """Tokens saved per row if any prefix another user sent earlier is reused."""
    saved = np.zeros(len(df), dtype=np.int64)
    pos = np.arange(len(df))
    for k in checkpoints:
        col = "h%d" % k
        valid = df[col].values != 0
        sub = pd.DataFrame({"h": df[col].values[valid], "user": df["user"].values[valid],
                            "pos": pos[valid]})
        first_user = sub.groupby("h")["user"].transform("first")
        is_other = sub["user"].values != first_user.values
        other_pos = sub["pos"].where(is_other)
        first_other = other_pos.groupby(sub["h"]).transform("min")
        hit = is_other | (sub["pos"].values > first_other.fillna(np.inf).values)
        rows = sub["pos"].values[hit]
        saved[rows] = k          # checkpoints go small -> large, so the longest wins
    return saved


def discrete_laplace_array(scale, size, rng):
    p = 1 - math.exp(-1.0 / scale)
    return rng.geometric(p, size) - rng.geometric(p, size)


def dp_rule_saved(df, win, checkpoints, eps, delta, seed):
    """Tokens saved per row by the DP rule, plus stats."""
    rng = np.random.default_rng(seed)
    contrib = len(checkpoints)
    tau = tau_threshold(eps, delta, contrib)
    scale = contrib / eps

    # one random prompt per user per window
    order = rng.permutation(len(df))
    shuffled = pd.DataFrame({"win": win[order], "user": df["user"].values[order], "row": order})
    chosen_rows = shuffled.drop_duplicates(["win", "user"])["row"].values
    chosen_win = win[chosen_rows]

    saved = np.zeros(len(df), dtype=np.int64)
    approved_counts = []
    singletons = 0
    windows = np.unique(win)
    for w in windows:
        rows_w = chosen_rows[chosen_win == w]
        parts = []
        for k in checkpoints:
            vals = df["h%d" % k].values[rows_w]
            parts.append(vals[vals != 0])
        all_h = np.concatenate(parts) if parts else np.array([], dtype=np.int64)
        if len(all_h) == 0:
            approved_counts.append(0)
            continue
        uniq, counts = np.unique(all_h, return_counts=True)
        noisy = counts + discrete_laplace_array(scale, len(counts), rng)
        passed = noisy > tau
        approved = uniq[passed]
        approved_counts.append(len(approved))
        singletons += int(((counts == 1) & passed).sum())

        # use the approved set in the NEXT window
        next_rows = np.where(win == w + 1)[0]
        if len(next_rows) and len(approved):
            for k in checkpoints:
                hit = np.isin(df["h%d" % k].values[next_rows], approved)
                saved[next_rows[hit]] = k
    return saved, float(np.mean(approved_counts)), singletons, tau


def plot(results, name, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    for wh in sorted(results["window_hours"].unique()):
        sub = results[results["window_hours"] == wh]
        fig, ax = plt.subplots(figsize=(6, 4))
        for cps in CHECKPOINT_SETS:
            s = sub[sub["checkpoints"] == cps]
            if s.empty:
                continue
            means = s.groupby("eps")["dp_pct"].mean()
            delta_contrib = len(cps.split(","))
            ax.plot(means.index, means.values, marker="o", label="DP, Delta=%d (%s)" % (delta_contrib, cps))
        full = sub.groupby("eps")["full_pct"].max()
        ax.axhline(full.max(), linestyle="--", color="gray", label="full sharing (best case)")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("epsilon")
        ax.set_ylabel("prompt tokens saved (%)")
        ax.set_title("%s, %d-hour windows" % (name, wh))
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = os.path.join(out_dir, "sweep_%s_%dh.png" % (name, wh))
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--delta", type=float, default=1e-6)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    total_tokens = float(df["n_tokens"].sum())
    print("loaded", len(df), "rows")

    rows = []
    full_cache = {}
    for wh in WINDOWS:
        win = (df["ts"].values // int(wh * 3600)).astype(np.int64)
        for cps in CHECKPOINT_SETS:
            checkpoints = [int(x) for x in cps.split(",")]
            if cps not in full_cache:
                full_cache[cps] = 100.0 * full_sharing_saved(df, checkpoints).sum() / total_tokens
            full_pct = full_cache[cps]
            for eps in EPS_LIST:
                for seed in range(args.seeds):
                    saved, per_win, singles, tau = dp_rule_saved(df, win, checkpoints, eps, args.delta, seed)
                    dp_pct = 100.0 * saved.sum() / total_tokens
                    rows.append({
                        "window_hours": wh, "checkpoints": cps, "Delta": len(checkpoints),
                        "eps": eps, "tau": round(tau, 1), "seed": seed,
                        "full_pct": round(full_pct, 3), "dp_pct": round(dp_pct, 3),
                        "kept": round(dp_pct / full_pct, 3) if full_pct else 0.0,
                        "approved_per_window": round(per_win, 1), "singletons_shared": singles,
                    })
                print("window %dh  %-18s eps=%-5g  dp=%.2f%%  full=%.2f%%"
                      % (wh, cps, eps, rows[-1]["dp_pct"], full_pct))

    results = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    csv_path = "results/sweep_%s.csv" % args.name
    results.to_csv(csv_path, index=False)
    print("saved", csv_path)

    summary = results.groupby(["window_hours", "checkpoints", "eps"])[["dp_pct", "kept", "approved_per_window"]].mean()
    print(summary.round(3).to_string())
    plot(results, args.name, "results/figures")


if __name__ == "__main__":
    main()
