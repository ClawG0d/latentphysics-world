"""Backend layer — the *only* place that talks to the mujoco_warp engine.

Everything above this layer (assets/perception/nav/envs/...) depends on the
`Scene` facade defined here, never on mujoco_warp directly. This keeps the
"upstream engine vs our IP" boundary crisp and lets us swap/patch the engine
(BVH broadphase, contact-overflow detection — see readiness report §4) behind
a stable interface.
"""

from .warp_engine import WarpEngine, Scene, EngineUnavailable

__all__ = ["WarpEngine", "Scene", "EngineUnavailable"]
