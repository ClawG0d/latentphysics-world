"""Environment layer (our IP) — BUILD, readiness report §7.

Manager-based, gym-style env layer (Isaac-Lab / mjlab paradigm) on top of the
`Scene` facade. Composes observations, actions, rewards, resets, and domain
randomization into vectorized environments for RL, with zero-copy torch tensors.
"""

from __future__ import annotations

__all__ = ["Env", "EnvConfig"]


class EnvConfig:
    """Declarative env config: managers for obs / action / reward / termination / DR."""

    def __init__(self, **kw) -> None:
        raise NotImplementedError("TODO(P5): manager-based config schema")


class Env:
    """Vectorized environment.

    Planned API (gym-style, batched over n_worlds):
        env = Env(EnvConfig(...))
        obs = env.reset()
        obs, reward, done, info = env.step(action)   # torch tensors, on device
    """

    def __init__(self, config: "EnvConfig") -> None:
        raise NotImplementedError("TODO(P5): wire Scene + managers")
