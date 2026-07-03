# LPW Scope Charter — read before any change

Latent Physics World is a **physics simulator**: contact accuracy and
worlds-per-second, for physical AI. Reference class: MuJoCo / Genesis.
It is NOT a training framework (Isaac Lab), not a robot-learning library,
not a world-model project. Training is the **user's** side of the API.

## Two questions every change must pass

1. Does it make simulation more **accurate**, more **parallel/faster**, or
   make simulatable **worlds richer** (assets, scenes, sensors)? If none of
   the three — it does not belong here.
2. Does it produce or contain a **policy / skill / agent / trainer**?
   If yes — reject, even as an "example" or "validation tool". PPO was
   removed for exactly this reason (commit 9f3f072); do not re-add it
   under a new justification.

## In scope

- Engine work: contact physics, batching, budgets, determinism, throughput
  (engine-level changes go to the private fork repo `latentphysics-engine`)
- Worlds: procedural scenes, articulated furniture, GLB/USD asset import
- Perception: batched LiDAR / depth / segmentation
- Benchmark tasks with **physically checkable verifiers** — poses,
  distances, joint travel; never learned, never subjective
- Sim-to-real calibration, including learned residual dynamics (R5):
  learning is in scope **only when it makes the simulator itself more
  accurate** — never when it produces behavior
- Zero-copy PyTorch interface: users train through it; we never train in it

## Out of scope — do not add

- RL algorithms, trainers, optimizers, policy/value networks, agents
- Curriculum or skill learning; LLM-in-the-loop task generation
- "World model" / "foundation model training" narratives in docs
- Any README claim not backed by a committed test or a real recorded run

## Discipline

- Every gallery clip is a real run from this repo, with a hard assert in
  the script that produced it, and a label linking to the source.
- Every roadmap item is a simulator capability with a physics KPI.
- `tests/test_scope_guard.py` mechanically enforces the no-learning rule —
  keep it green; loosening it requires the owner's explicit decision.
- Repo is English-only. Fork attribution stays intact (NOTICE,
  THIRD_PARTY_NOTICES.md); never hide the mujoco_warp lineage.
