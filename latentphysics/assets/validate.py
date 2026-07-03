"""Physics-validity checks and stability bake for imported worlds (R4).

Real 3D assets (scans, exports, procedural mixes) routinely compile into
MJCF that is *loadable* but not *physically sound*: a thin-shell mesh with
near-singular inertia, a dynamic body that lost its collision geometry, two
objects spawned interpenetrating, or a pile that explodes on the first
step. These checks catch that BEFORE a scene is trusted — all classical,
all deterministic, no learning.

    from latentphysics.assets.validate import validate_model, initial_penetration
    report = validate_model(mujoco.MjModel.from_xml_path(path))
    if not report.ok:
        print(report)                 # human-readable issue list

    scene = lpw.load_scene(path, cfg)
    settle(scene)                     # relax to rest, then snapshot as start
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Issue", "ValidationReport", "validate_model",
           "initial_penetration", "settle"]


@dataclass
class Issue:
    body: str
    kind: str          # "zero_mass" | "singular_inertia" | "no_collision" | "penetration"
    detail: str

    def __str__(self):
        return f"[{self.kind}] {self.body}: {self.detail}"


@dataclass
class ValidationReport:
    issues: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.issues) == 0

    def __str__(self):
        if self.ok:
            return "ValidationReport: OK"
        return "ValidationReport: %d issue(s)\n  " % len(self.issues) + \
               "\n  ".join(str(i) for i in self.issues)


def validate_model(mjm, min_mass: float = 1e-6, min_inertia: float = 1e-9) -> ValidationReport:
    """CPU-only structural check of a compiled MjModel (no GPU needed).

    Flags dynamic (free/jointed) bodies that MuJoCo will simulate badly:
      - mass at/under ``min_mass`` (thin shells, missing density)
      - a principal moment of inertia at/under ``min_inertia`` (degenerate
        shape → near-singular inertia → solver blow-up)
      - a movable body with no collision geom (a CoACD "ghost": it renders
        but never touches anything)
    """
    import mujoco

    report = ValidationReport()
    # collision geoms per body (contype|conaffinity != 0)
    col = (mjm.geom_contype | mjm.geom_conaffinity) != 0
    for b in range(1, mjm.nbody):          # skip world body 0
        if mjm.body_dofnum[b] == 0:        # static: mass/inertia irrelevant
            continue
        name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"
        m = float(mjm.body_mass[b])
        if not np.isfinite(m) or m <= min_mass:
            report.issues.append(Issue(name, "zero_mass", f"mass={m:.3e} kg"))
        inertia = np.asarray(mjm.body_inertia[b], dtype=float)
        if not np.isfinite(inertia).all() or inertia.min() <= min_inertia:
            report.issues.append(
                Issue(name, "singular_inertia", f"principal moments={inertia}"))
        has_col = any(col[g] for g in range(mjm.ngeom) if mjm.geom_bodyid[g] == b)
        if not has_col:
            report.issues.append(
                Issue(name, "no_collision", "movable body has no collision geom"))
    return report


def initial_penetration(mjm, tol: float = 0.005) -> ValidationReport:
    """Flag pairs interpenetrating deeper than ``tol`` at the spawn pose.

    Runs one CPU ``mj_forward`` at qpos0 and inspects contacts — resting
    contact is shallow (~0), so a deep negative dist means the importer
    placed objects overlapping. CPU-only.
    """
    import mujoco

    d = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, d)
    report = ValidationReport()
    for i in range(d.ncon):
        c = d.contact[i]
        if c.dist < -tol:
            g1 = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"g{c.geom1}"
            g2 = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"g{c.geom2}"
            report.issues.append(
                Issue(f"{g1}~{g2}", "penetration", f"depth={-c.dist*1000:.1f} mm"))
    return report


def settle(scene, max_steps: int = 600, vel_tol: float = 0.05, chunk: int = 30) -> dict:
    """Step the loaded scene until it comes to rest, then leave it there.

    Relaxes spawn transients (objects dropping the last millimetre onto a
    surface, resolving shallow initial contact) so the scene's usable start
    state is a settled one. Returns ``{"steps", "residual_vel", "converged"}``;
    the caller can ``scene.snapshot()`` afterwards to keep it as the origin.

    Convergence = max |qvel| across all worlds under ``vel_tol``.
    """
    import torch

    qvel = scene.qvel()
    steps = 0
    residual = float("inf")
    while steps < max_steps:
        scene.step(chunk)
        steps += chunk
        residual = float(qvel.abs().max().item())
        if not np.isfinite(residual):
            return {"steps": steps, "residual_vel": residual, "converged": False}
        if residual < vel_tol:
            return {"steps": steps, "residual_vel": residual, "converged": True}
    return {"steps": steps, "residual_vel": residual, "converged": False}
