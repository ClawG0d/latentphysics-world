# latentphysics/neural — learned simulation (whitelisted physics only)

Created as an owner-signed standalone change (2026-07-03). Empty of learning
code by design: this file records what any future learned-solver change must
satisfy BEFORE it lands here. These are requirements, not capabilities.

## What may live here

MeshGraphNets-style surrogates and FEA-adjacent learned solvers for the
charter's exhaustive whitelist: **deformables, fluids, aerodynamics** —
nothing else. Rigid contact is never approximated by a network, anywhere,
in any mode.

## Acceptance gates (every learned-solver change must ship all of these)

1. **Solver-slot contract** — the model is a drop-in for a declared solver
   slot: identical inputs/outputs to a classical/FEA reference path,
   consumable only inside the engine step, off by default behind a config
   flag.
2. **Reference dataset** — a committed script + seed that generates the
   training/eval data from the classical/FEA reference solver. No adaptive
   or online collection from user workloads.
3. **Quantitative gates vs reference** — committed tests with numeric
   thresholds: per-step state error, N-step rollout divergence, and
   conservation drift (energy/momentum) on held-out trajectories.
4. **Cross-check at verification** — any benchmark/verifier quantity that
   touches this solver is cross-checked against the classical path at
   verification time; never asserted on network output alone.
5. **Sentinel coverage** — runtime physics sentinels active on the learned
   path (non-finite state, penetration/velocity analogues appropriate to
   the medium).
6. **No behavior code** — no policies, rewards, agents, or action outputs;
   the scope guard enforces name and signature bans inside this subtree.
