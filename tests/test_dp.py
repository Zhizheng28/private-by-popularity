import numpy as np

from pbp.dp import admit, tau_threshold


def test_tau_matches_proposal():
    assert round(tau_threshold(1.0, 1e-6, 1)) == 15
    assert 59 < tau_threshold(1.0, 1e-6, 4) < 62


def test_popular_prefix_admitted_rare_one_not():
    rng = np.random.default_rng(0)
    reqs = []
    for i in range(200):
        reqs.append(("user%d" % i, ["popular"]))
    reqs.append(("victim", ["secret"]))
    approved = admit(reqs, eps=1.0, delta=1e-6, contrib=1, rng=rng)
    assert "popular" in approved
    assert "secret" not in approved
