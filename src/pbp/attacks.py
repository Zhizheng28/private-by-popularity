"""Attacks (week 3).

- audit: plant a fake private prompt for one victim, time the attacker's
  guess vs. random prefixes, run a statistical test (e.g. scipy Mann-Whitney).
  Should detect under full sharing, not under the DP rule.
- sybil: attacker adds tau-1 fake users with the guessed prefix, then checks
  whether the victim pushes it over the threshold.
"""
# TODO
