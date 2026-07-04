"""Learned simulation of whitelisted physics — the ONLY sanctioned home for it.

This subtree exists under an owner-signed carve-out (standalone change,
2026-07-03; see CLAUDE.md "Module map"). It contains no learning code until
a solver change passes the acceptance gates in latentphysics/neural/README.md.

Constitutional constraints (enforced by tests/test_scope_guard.py):

- Learning here is a *simulation method*, never behavior: models may
  approximate whitelisted physics only — deformables, fluids, aerodynamics.
  Nothing in this subtree may output actions, policies, skills, or rewards.
- The rigid-body contact core stays classical in every mode; no network may
  approximate rigid contact, including opt-in "fast modes".
- Every learned solver is a drop-in occupant of a declared solver slot:
  same inputs/outputs as a classical/FEA reference path, switchable off,
  cross-checkable against that reference at verification time.
- Acceptance requires quantitative accuracy-vs-reference gates: committed
  tests with numeric thresholds (per-step error, rollout divergence,
  conservation drift) before any capability claim.
- Training data comes from committed, seeded reference-solver runs — no
  adaptive or online data collection from user workloads.
"""

__all__ = []
