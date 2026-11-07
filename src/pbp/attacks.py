"""Attack measurements: timing audit and Sybil attack."""
import numpy as np
from scipy.stats import mannwhitneyu

from pbp.dp import tau_threshold
from pbp.simulator import discrete_laplace_array


def attack_auc(victim_lat, control_lat):
    """Chance the attacker ranks a real victim prefix as faster than a fake one.
    0.5 = pure guessing, 1.0 = perfect attack."""
    stat, _ = mannwhitneyu(control_lat, victim_lat, alternative="two-sided")
    return stat / (len(victim_lat) * len(control_lat))


def audit_p_value(victim_lat, control_lat):
    """Gu et al.-style audit: are victim prefixes faster than never-sent ones?
    Small p (< 0.05) = the audit detects cross-user sharing."""
    _, p = mannwhitneyu(victim_lat, control_lat, alternative="less")
    return p


def best_accuracy(victim_lat, control_lat):
    """Best accuracy of a 'faster than t means it was sent' guess, over all t."""
    all_lat = np.concatenate([victim_lat, control_lat])
    labels = np.concatenate([np.ones(len(victim_lat)), np.zeros(len(control_lat))])
    order = np.argsort(all_lat)
    labels = labels[order]
    # guess "sent" for the fastest i requests
    tp = np.concatenate([[0], np.cumsum(labels)])
    fp = np.concatenate([[0], np.cumsum(1 - labels)])
    tn = len(control_lat) - fp
    acc = (tp + tn) / len(labels)
    return float(acc.max())


def sybil_success(n_fake, eps, delta, contrib, trials, rng):
    """Attacker adds n_fake accounts that send the guessed prefix.
    Returns (P[shared | victim sent it], P[shared | victim did not])."""
    tau = tau_threshold(eps, delta, contrib)
    scale = contrib / eps
    noise_a = discrete_laplace_array(scale, trials, rng)
    noise_b = discrete_laplace_array(scale, trials, rng)
    p_sent = float(np.mean(n_fake + 1 + noise_a > tau))
    p_not = float(np.mean(n_fake + noise_b > tau))
    return p_sent, p_not
