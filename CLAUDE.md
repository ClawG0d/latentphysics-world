# LPW Scope Charter — read before any change

Latent Physics World is a **high-speed, high-accuracy physics simulator
built as the substrate for training world models** (physical foundation
models). Engine reference class: MuJoCo / Genesis. We build the world;
users train on it — nothing in this repo trains a user's model.

The core is classical, verifiable rigid-body contact physics. For physics
the rigid core does not cover — an exhaustive whitelist: **deformables,
fluids, aerodynamics** (extending it is an owner decision) — the simulator
may use FEA solvers and **learned simulation** (MeshGraphNets-style
surrogates). Learning is a sanctioned *simulation method*, never anything
that outputs behavior.

## The test every change must pass

Does it make simulation more **accurate**, more **parallel/faster**,
cover **whitelisted new physics** (via FEA or learned solvers), or make
simulatable **worlds richer**? Worlds-richer means procedural/classical
generation and imported assets; generative-model content (LLM scene
composition, diffusion assets, neural sensors) needs explicit owner
sign-off. If none of the four — reject.

## Hard lines around the rigid core

- The rigid contact core stays classical in EVERY mode — no learned
  approximation of rigid contact, including opt-in "fast modes".
- "Faster" is never a justification for a learned solver; only
  whitelisted physics coverage is.
- Calibration residuals adjust physical parameters (friction, restitution,
  inertia, latency) or bounded corrections; a residual that dominates the
  classical forces is a replacement in disguise — banned.
- Benchmark verifier quantities come from classical solver paths, or are
  cross-checked against one at verification time — never asserted on a
  network's output alone.
- A learned solver never touches the rigid contact path: it occupies a
  bounded solver slot with a classical/FEA reference implementation, is
  off by default, and is accepted only through committed
  accuracy-vs-reference tests with numeric thresholds on data from
  seeded, committed reference runs.

## Module map (roles are exhaustive; extending a role = owner decision)

- `backend/` — engine adapter: budgets, snapshots, CUDA-graph stepping
- `assets/` — procedural worlds, articulated furniture, GLB/USD import,
  convex decomposition, SDF voxelization
- `perception/` — batched LiDAR / depth / segmentation
- `envs/` — batched task containers + physically checkable benchmark
  verifiers (poses, distances, joint travel; never learned). Task rewards
  exist as part of the benchmark interface users consume; training loops,
  success-rate leaderboards, and reward engineering for training are not
  this module's role.
- `broadphase/` — large-scene BVH broadphase (engine performance)
- `domain_rand/` — domain randomization + sim-to-real calibration (sysid)
- `latentphysics/neural/` — the ONLY place solver learning lives
  (owner-signed carve-out: standalone change, 2026-07-03). Scope: learned
  simulation of whitelisted physics only (deformables, fluids,
  aerodynamics — the same exhaustive whitelist as above). Every learned
  solver is a drop-in occupant of a declared solver slot — same
  inputs/outputs as a classical/FEA reference path, switchable off,
  cross-checked against that reference — and ships with quantitative
  accuracy-vs-reference acceptance gates (committed tests with numeric
  thresholds) before any capability claim. Training data comes from
  committed, seeded reference-solver runs; no adaptive/online data
  collection. Behavior code stays banned here like everywhere else.
  Widening this carve-out is an owner decision.

## Out of scope — do not add

- RL algorithms, policy/value networks, agents, skill or curriculum
  learning, LLM-in-the-loop task generation — anything that outputs
  behavior rather than physics, even as an "example" or "validation
  tool" (PPO precedent: 9f3f072)
- Training pipelines for user models (world models included): we ship the
  simulator they train on, not the training
- Any README claim not backed by a committed test or a reproducible
  recorded run (committed script + seed + artifact)

## Discipline

- Every gallery clip is a real run from this repo, with a hard assert in
  the script that produced it, and a label linking to the source.
- A model of a real object is "done" only when its shape AND material match
  the real thing — a passing physics or pose assert is necessary, never
  sufficient. Verify procedural and imported assets by rendering them
  (closed and in every open/articulated state) and looking, plus a
  mechanical semantic check (cavities present, handles proud and
  contrasting, seams tight, grounded) — not by joint-travel numbers alone.
- Every roadmap item is a simulator capability with a physics KPI;
  learned-solver items carry accuracy-vs-reference KPIs like everything
  else.
- `tests/test_scope_guard.py` bans learning-code signatures across the
  whole package and examples, with exactly ONE owner-signed carve-out:
  `latentphysics/neural/`. The allowlist is a single hard-coded path, a
  test asserts it stays that way, and behavior-code signatures remain
  banned even inside the carve-out. Keep it green; any widening is a
  standalone owner-approved change.
- This charter and an equivalent scope guard apply to every LPW
  repository, including the engine fork (`latentphysics-engine`), which
  carries its own copy of this file.
- Repo is English-only.
