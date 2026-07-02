"""Manipulation benchmark suite (R2) — auto-verified tasks.

Every task defines a *physically checkable* success predicate (poses,
distances, displacements — never learned or subjective). That is the RSI
requirement: rewards a curriculum engine can trust without a human in the
loop.

All tasks run on the Franka tabletop scene (menagerie ``mjx_single_cube``:
7-DoF arm + gripper + one free cube). Usage:

    from latentphysics.envs.suite import SUITE, make
    env = make("push", scene)                 # or SUITE["push"](scene, cfg)
"""

from __future__ import annotations

import torch

from .base import TaskConfig, VecTask
from .franka_reach import FrankaReach

_TABLE_Z = 0.0            # tabletop height in the mjx_single_cube scene
_CUBE_HALF = 0.03


class FrankaTask(VecTask):
    """Shared plumbing: cube state, gripper site, target buffers."""

    def __init__(self, scene, cfg: TaskConfig | None = None):
        super().__init__(scene, cfg)
        self._gripper = self._site_id("gripper")
        self._cube_adr = self._body_qpos_addr("box")
        self.targets = torch.zeros(self.n, 3, device=self.device)
        self._cube_start = torch.zeros(self.n, 3, device=self.device)
        self.obs_dim = scene.mjm.nq + scene.mjm.nv + 3 + 3 + 3

    # views ------------------------------------------------------------------
    def cube_pos(self) -> torch.Tensor:
        a = self._cube_adr
        return self.qpos[:, a:a + 3]

    def grip_pos(self) -> torch.Tensor:
        return self.site_xpos[:, self._gripper]

    # base hooks ----------------------------------------------------------------
    def _task_reset(self, mask: torch.Tensor) -> None:
        k = int(mask.sum().item())
        if k:
            self.targets[mask] = self._sample_targets(k)
            self._cube_start[mask] = self.cube_pos()[mask]

    def _compute(self):
        grip, cube = self.grip_pos(), self.cube_pos()
        success = self._verify(grip, cube)
        reward = self._reward(grip, cube, success)
        obs = torch.cat([self.qpos, self.qvel, grip, cube, self.targets], dim=-1)
        return obs, reward, success

    # per-task ---------------------------------------------------------------
    def _sample_targets(self, k: int) -> torch.Tensor:  # default: tabletop region
        u = torch.rand(k, 3, device=self.device, generator=self.gen)
        lo = torch.tensor([0.35, -0.30, _TABLE_Z + _CUBE_HALF], device=self.device)
        hi = torch.tensor([0.75, 0.30, _TABLE_Z + _CUBE_HALF], device=self.device)
        return lo + u * (hi - lo)

    def _verify(self, grip, cube) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def _reward(self, grip, cube, success) -> torch.Tensor:
        return success.float()  # sparse by default; tasks add shaping


# --------------------------------------------------------------------------- #
#  the suite                                                                   #
# --------------------------------------------------------------------------- #

class Reach(FrankaReach):
    """Gripper to a random 3D point. Verify: |grip-target| < 5 cm."""


class ReachStable(FrankaTask):
    """Reach and hold still. Verify: within 5 cm AND joint speed < 0.5."""

    def _sample_targets(self, k):
        u = torch.rand(k, 3, device=self.device, generator=self.gen)
        lo = torch.tensor([0.30, -0.35, 0.15], device=self.device)
        hi = torch.tensor([0.75, 0.35, 0.60], device=self.device)
        return lo + u * (hi - lo)

    def _verify(self, grip, cube):
        near = torch.linalg.norm(grip - self.targets, dim=-1) < 0.05
        slow = self.qvel[:, :7].abs().amax(-1) < 0.5
        return near & slow

    def _reward(self, grip, cube, success):
        return -torch.linalg.norm(grip - self.targets, dim=-1) + success.float()


class Hover(FrankaTask):
    """Hover the gripper 10 cm above the cube. Verify: xy < 2 cm, z in [7,13] cm."""

    def _verify(self, grip, cube):
        d_xy = torch.linalg.norm(grip[:, :2] - cube[:, :2], dim=-1)
        dz = grip[:, 2] - cube[:, 2]
        return (d_xy < 0.02) & (dz > 0.07) & (dz < 0.13)

    def _reward(self, grip, cube, success):
        tgt = cube + torch.tensor([0, 0, 0.10], device=self.device)
        return -torch.linalg.norm(grip - tgt, dim=-1) + success.float()


class TouchCube(FrankaTask):
    """Move the cube. Verify: cube displaced > 1 cm from its start."""

    def _verify(self, grip, cube):
        return torch.linalg.norm(cube - self._cube_start, dim=-1) > 0.01

    def _reward(self, grip, cube, success):
        return -torch.linalg.norm(grip - cube, dim=-1) + success.float()


class Push(FrankaTask):
    """Push the cube to a tabletop target. Verify: xy dist < 3 cm, cube on table."""

    def _verify(self, grip, cube):
        d_xy = torch.linalg.norm(cube[:, :2] - self.targets[:, :2], dim=-1)
        on_table = (cube[:, 2] - (_TABLE_Z + _CUBE_HALF)).abs() < 0.02
        return (d_xy < 0.03) & on_table

    def _reward(self, grip, cube, success):
        return (-torch.linalg.norm(grip - cube, dim=-1)
                - 2.0 * torch.linalg.norm(cube[:, :2] - self.targets[:, :2], dim=-1)
                + success.float())


class PushToEdge(Push):
    """Push the cube near the table edge. Verify: |y| > 25 cm, still on table."""

    def _verify(self, grip, cube):
        on_table = (cube[:, 2] - (_TABLE_Z + _CUBE_HALF)).abs() < 0.02
        return (cube[:, 1].abs() > 0.25) & on_table

    def _reward(self, grip, cube, success):
        return (-torch.linalg.norm(grip - cube, dim=-1)
                + cube[:, 1].abs() + success.float())


class CubeToCorner(Push):
    """Push the cube into a fixed corner zone. Verify: inside 6x6 cm zone."""

    _CORNER = (0.68, 0.25)

    def _sample_targets(self, k):
        t = torch.tensor([*self._CORNER, _TABLE_Z + _CUBE_HALF], device=self.device)
        return t.expand(k, 3).clone()


class Lift(FrankaTask):
    """Lift the cube. Verify: cube z > 15 cm above table AND gripper near cube."""

    def _verify(self, grip, cube):
        high = cube[:, 2] > _TABLE_Z + 0.15
        held = torch.linalg.norm(grip - cube, dim=-1) < 0.08
        return high & held

    def _reward(self, grip, cube, success):
        return (-torch.linalg.norm(grip - cube, dim=-1)
                + 5.0 * torch.clamp(cube[:, 2] - (_TABLE_Z + _CUBE_HALF), min=0)
                + success.float())


class PickPlace(FrankaTask):
    """Bring the cube to a 3D target above the table. Verify: dist < 4 cm."""

    def _sample_targets(self, k):
        u = torch.rand(k, 3, device=self.device, generator=self.gen)
        lo = torch.tensor([0.40, -0.25, 0.15], device=self.device)
        hi = torch.tensor([0.70, 0.25, 0.35], device=self.device)
        return lo + u * (hi - lo)

    def _verify(self, grip, cube):
        return torch.linalg.norm(cube - self.targets, dim=-1) < 0.04

    def _reward(self, grip, cube, success):
        return (-torch.linalg.norm(grip - cube, dim=-1)
                - 2.0 * torch.linalg.norm(cube - self.targets, dim=-1)
                + success.float())


class HoldStill(FrankaTask):
    """Anti-task: keep the cube undisturbed for a whole episode.
    Verify (at timeout): cube never displaced > 2 cm."""

    def _verify(self, grip, cube):
        undisturbed = torch.linalg.norm(cube - self._cube_start, dim=-1) < 0.02
        at_end = self.progress >= (self.cfg.episode_len - 1)
        return undisturbed & at_end

    def _reward(self, grip, cube, success):
        disturbed = torch.linalg.norm(cube - self._cube_start, dim=-1)
        return -10.0 * disturbed + success.float()


class GripperUp(FrankaTask):
    """Point the hand straight down (grasp-ready pose). Verify: hand -z axis
    within ~18 deg of world -z."""

    def __init__(self, scene, cfg=None):
        super().__init__(scene, cfg)
        import mujoco
        self._hand = mujoco.mj_name2id(scene.mjm, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self._xmat = scene.state("xmat")   # (n, nbody, 3, 3) rotation matrices

    def _verify(self, grip, cube):
        z_axis = self._xmat.reshape(self.n, -1, 3, 3)[:, self._hand, :, 2]
        return z_axis[:, 2] < -0.95      # hand z pointing down

    def _reward(self, grip, cube, success):
        z_axis = self._xmat.reshape(self.n, -1, 3, 3)[:, self._hand, :, 2]
        return -z_axis[:, 2] + success.float()


class CubeHome(Push):
    """Return the cube to its home position. Verify: within 3 cm of home."""

    def _sample_targets(self, k):
        t = torch.tensor([0.7, 0.0, _TABLE_Z + _CUBE_HALF], device=self.device)
        return t.expand(k, 3).clone()


SUITE: dict[str, type] = {
    "reach": Reach,
    "reach_stable": ReachStable,
    "hover": Hover,
    "touch": TouchCube,
    "push": Push,
    "push_to_edge": PushToEdge,
    "cube_to_corner": CubeToCorner,
    "lift": Lift,
    "pick_place": PickPlace,
    "hold_still": HoldStill,
    "gripper_down": GripperUp,
    "cube_home": CubeHome,
}


def make(name: str, scene, cfg: TaskConfig | None = None) -> VecTask:
    if name not in SUITE:
        raise KeyError(f"unknown task {name!r}; available: {sorted(SUITE)}")
    return SUITE[name](scene, cfg)
