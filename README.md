# Private by Popularity

DP admission for shared LLM prompt caches (EECS 498/598, Fall 2026).

## Setup
    python -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    pip install -e .
    pytest

## Folders
- `src/pbp/`  reusable code: data, prefixes, dp, simulator, attacks
- `scripts/`  things you run: prefix_check.py (go/no-go), later run_experiments.py
- `tests/`    small checks that the DP rule is right
- `data/`     downloaded / processed data (not in git)
- `results/`  figures and tables (not in git; regenerate from scripts)
- `notebooks/` quick exploration only
- `report/`   LaTeX report

## Order of work
1. scripts/prefix_check.py  -> go / no-go
2. src/pbp/dp.py + proof
3. src/pbp/simulator.py + baselines
4. src/pbp/attacks.py (audit, Sybil)
5. plots -> report
