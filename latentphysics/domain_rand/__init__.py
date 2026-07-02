"""Domain randomization + sim-to-real calibration (our IP) — BUILD, report §5④.

DR is the primary sim-to-real lever (autodiff is not available on the engine).
Randomizes friction / mass / CoM / armature / joint offsets / lighting / camera
pose, and models action & observation latency (hardware delay). Calibration
(sysid) fits a set of these params to real-robot logs.
"""

from __future__ import annotations

__all__ = ["Randomizer", "LatencyModel", "calibrate"]


class Randomizer:
    """Per-world randomization of model params (batched over the Model dim).

    Planned API:
        rnd = Randomizer(friction=(0.6, 1.2), mass_pct=0.1, com_jitter=0.01)
        model = rnd.apply(model)     # batches params across n_worlds
    """

    def __init__(self, **ranges) -> None:
        raise NotImplementedError("TODO(P4): batch model params across worlds")


class LatencyModel:
    """Randomized action/observation delay to model hardware latency."""

    def __init__(self, action_delay=(0, 2), obs_delay=(0, 1)) -> None:
        raise NotImplementedError("TODO(P4)")


def calibrate(real_logs, model, params):
    """Fit DR/model params to real-robot logs (sysid; finite-difference)."""
    raise NotImplementedError("TODO(P4): port MuJoCo sysid; finite-diff objective")
