"""Adapter over the mujoco_warp GPU physics engine.

Wraps mujoco_warp's ``put_model`` / ``make_data`` / ``step`` behind a small,
stable ``Scene`` facade. The rest of Latent Physics World depends only on
``Scene`` — not on mujoco_warp — so upstream can be patched/swapped freely.

State is batched over ``n_worlds`` (mujoco_warp's ``nworld``): ``qpos`` is
``(n_worlds, nq)`` etc. Engine arrays are exposed as **zero-copy torch tensors**
on the GPU via ``warp.to_torch``.

Platform: needs an NVIDIA CUDA GPU (Linux / WSL2). Importing this module is
cheap and safe everywhere; the engine deps import lazily in ``_require_engine``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import Config, resolve_device


class EngineUnavailable(RuntimeError):
    """Raised when the GPU physics engine (warp / mujoco_warp) can't be used."""


def _require_engine():
    """Lazily import the engine, with an actionable error if unavailable."""
    try:
        import mujoco
        import warp as wp
        import mujoco_warp as mjw
    except ImportError as e:
        raise EngineUnavailable(
            "The GPU physics engine is not available.\n"
            "Latent Physics World runs physics on mujoco_warp, which needs an "
            "NVIDIA CUDA GPU (Linux / WSL2).\n"
            "Install with:  pip install -e '.[gpu]'  (see docs/PLAN.md)\n"
            f"Underlying import error: {e}"
        ) from e
    return mujoco, wp, mjw


@dataclass
class Scene:
    """User-facing handle to a loaded, batched simulation on the GPU."""

    engine: "WarpEngine"
    mjm: Any            # mujoco.MjModel (CPU; kept for compilation/introspection)
    model: Any          # mujoco_warp Model
    data: Any           # mujoco_warp Data
    n_worlds: int = 1

    # --- stepping -------------------------------------------------------------
    def step(self, n: int = 1) -> "Scene":
        """Advance the simulation ``n`` steps across all worlds."""
        self.engine._step(self, n)
        return self

    def reset(self) -> "Scene":
        """Reset all worlds to the model's initial state."""
        self.engine._reset(self)
        return self

    # --- state (zero-copy torch views, shaped (n_worlds, ...)) ----------------
    def state(self, name: str):
        """Zero-copy torch view of any engine Data field (e.g. 'qpos','xpos')."""
        return self.engine._to_torch(getattr(self.data, name))

    def qpos(self):
        """Generalized positions, torch (n_worlds, nq) on GPU."""
        return self.state("qpos")

    def qvel(self):
        """Generalized velocities, torch (n_worlds, nv) on GPU."""
        return self.state("qvel")

    @property
    def time(self) -> float:
        return float(self.engine._to_torch(self.data.time)[0].item())

    # --- contacts (P1; overflow-aware, readiness report §4②) ------------------
    def num_contacts(self) -> int:
        """Current active contact count; warns if the buffer overflowed."""
        d = self.data
        nacon = getattr(d, "nacon", None)
        cap = int(getattr(d, "naconmax", 0) or 0)
        try:
            n = int(self.engine._to_torch(nacon).sum().item()) if nacon is not None else -1
        except Exception:
            n = -1
        if cap and n >= cap:
            import warnings
            warnings.warn(
                f"contact buffer full ({n}>={cap}); contacts were dropped. "
                f"Increase Config.naconmax for cluttered indoor scenes.",
                stacklevel=2,
            )
        return n


class WarpEngine:
    """Owns engine model/data creation and stepping for a given Config."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.device = resolve_device(self.config.device)
        self._mujoco, self._wp, self._mjw = _require_engine()

    def load_mjcf(self, mjcf_path: str) -> Scene:
        """Compile an MJCF and place it on the GPU engine, batched to n_worlds."""
        mujoco, wp, mjw = self._mujoco, self._wp, self._mjw
        mjm = mujoco.MjModel.from_xml_path(mjcf_path)
        if self.config.timestep is not None:
            mjm.opt.timestep = self.config.timestep
        model = mjw.put_model(mjm)
        make_kw = {"nworld": self.config.n_worlds}
        if self.config.naconmax is not None:
            make_kw["naconmax"] = self.config.naconmax
        if self.config.njmax is not None:
            make_kw["njmax"] = self.config.njmax
        data = mjw.make_data(mjm, **make_kw)
        return Scene(engine=self, mjm=mjm, model=model, data=data, n_worlds=self.config.n_worlds)

    # --- internals ------------------------------------------------------------
    def _step(self, scene: Scene, n: int) -> None:
        mjw, wp = self._mjw, self._wp
        for _ in range(n):
            mjw.step(scene.model, scene.data)
        wp.synchronize()

    def _reset(self, scene: Scene) -> None:
        self._mjw.reset_data(scene.model, scene.data)

    def _to_torch(self, warp_array):
        """Zero-copy view of a warp array as a torch tensor (GPU)."""
        return self._wp.to_torch(warp_array)
