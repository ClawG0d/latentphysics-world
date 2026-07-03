"""Articulated-furniture manipulation benchmarks (R4).

A Franka faces one articulated piece and must open it past a joint-travel
threshold — a physically checkable predicate, the same auto-verified
contract as the R2 suite, with no learning anywhere. Four tasks, one per
R4 archetype:

    open_drawer  — pull the upper drawer out  (slide joint > 15 cm)
    open_door    — swing the cabinet door open (hinge > 0.9 rad)
    open_lid     — lift the chest lid          (hinge > 0.6 rad)
    slide_door   — slide the cabinet door open (slide  > 20 cm)

These need their own scene (arm + one piece), so they are NOT in
``envs.suite.SUITE`` (which shares the tabletop-cube scene). Build with
:func:`build_articulated_scene` and construct via :func:`make`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

from .base import TaskConfig, VecTask
from ..assets.scene_gen import (
    _art_drawer_chest, _art_hinged_cabinet, _art_lid_chest,
    _art_sliding_door_cabinet,
)

__all__ = ["ArtSpec", "ART_SPECS", "ArticulatedOpen", "build_articulated_scene", "make"]

_S = 'contype="1" conaffinity="2"'
_D = 'contype="3" conaffinity="3"'


@dataclass
class ArtSpec:
    archetype: object      # scene_gen archetype fn
    joint: str             # joint name to read (f0_...)
    handle: str            # handle geom name (reward-shaping target)
    thresh: float          # "opened" threshold (rad for hinge, m for slide)
    size: tuple            # (depth, lateral) half-extents for placement
    height: float
    n_joints: int          # articulated dofs the piece adds (keyframe padding)


# tuned per archetype: sizes within each archetype's own clamps, thresholds
# comfortably inside each joint's range (see scene_gen archetypes)
ART_SPECS = {
    "open_drawer": ArtSpec(_art_drawer_chest, "f0_drawer1", "f0w1h", 0.15, (0.20, 0.35), 0.62, 2),
    "open_door":   ArtSpec(_art_hinged_cabinet, "f0_door", "f0dh", 0.9, (0.22, 0.40), 0.95, 1),
    "open_lid":    ArtSpec(_art_lid_chest, "f0_lid", "f0lh", 0.6, (0.28, 0.36), 0.50, 1),
    "slide_door":  ArtSpec(_art_sliding_door_cabinet, "f0_sdoor", "f0sdh", 0.20, (0.26, 0.50), 0.85, 1),
}


def build_articulated_scene(task: str, out_path: str | None = None,
                            menagerie: str | None = None) -> str:
    """Franka at the origin facing the task's articulated piece 0.72 m away.

    The MJCF is written INTO the menagerie panda directory so the panda's
    relative mesh paths resolve (MuJoCo resolves ``meshdir`` against the main
    file, not the include). ``out_path``'s basename names the file.
    """
    spec = ART_SPECS[task]
    men = menagerie or os.path.expanduser("~/lpw/menagerie")
    base = os.path.basename(out_path) if out_path else f"_lpw_{task}.xml"
    out_path = os.path.join(men, "franka_emika_panda", base)
    rng = np.random.default_rng(0)
    # carcass gets the REACHABLE-static mask (S): the robot must collide with
    # the body, not just the moving part
    parts, _ = spec.archetype(rng, 0, (0.72, 0.0), spec.size, spec.height,
                              _S, _D, room_half=(1.0, 1.0))
    # keyframe qpos = arm home (7) + fingers (2) + one zero per articulated dof
    qpos_home = "0 0.3 0 -1.57079 0 2.0 -0.7853 0.04 0.04" + " 0" * spec.n_joints
    xml = f"""<mujoco model="{task}">
  <include file="mjx_panda.xml"/>
  <option timestep="0.005" iterations="8" ls_iterations="10"/>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.45 0.53 0.62"
             rgb2="0.12 0.14 0.18" width="256" height="256"/>
    <texture name="floortex" type="2d" builtin="checker" rgb1="0.78 0.74 0.68"
             rgb2="0.68 0.64 0.58" mark="edge" markrgb="0.55 0.52 0.48"
             width="300" height="300"/>
    <material name="floormat" texture="floortex" texrepeat="10 10" reflectance="0.12"/>
  </asset>
  <worldbody>
    <light name="key" directional="true" pos="0 0 3" dir="-0.3 0.2 -0.9"
           diffuse="0.8 0.78 0.75" castshadow="true"/>
    <geom name="floor" type="plane" size="3 3 0.1" material="floormat" {_S}/>
    {"".join(parts)}
  </worldbody>
  <keyframe>
    <key name="home" qpos="{qpos_home}"
         ctrl="0 0.3 0 -1.57079 0 2.0 -0.7853 0.04"/>
  </keyframe>
</mujoco>
"""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(xml)
    return out_path


class ArticulatedOpen(VecTask):
    """Open an articulated piece past its threshold. Verify: joint > thresh."""

    def __init__(self, scene, spec: ArtSpec, cfg: TaskConfig | None = None):
        super().__init__(scene, cfg)
        import mujoco

        mjm = scene.mjm
        self.spec = spec
        jid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, spec.joint)
        gid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, spec.handle)
        if jid < 0 or gid < 0:
            raise ValueError(f"{spec.joint}/{spec.handle} not found in scene")
        self._qadr = int(mjm.jnt_qposadr[jid])
        self._handle = gid
        self._gripper = self._site_id("gripper")
        self.geom_xpos = scene.state("geom_xpos")
        self.obs_dim = mjm.nq + mjm.nv + 3 + 3 + 1

    def joint_q(self) -> torch.Tensor:
        return self.qpos[:, self._qadr]

    def _task_reset(self, mask: torch.Tensor) -> None:
        pass                       # fixed piece; DR arrives with R3 calibration

    def _compute(self):
        grip = self.site_xpos[:, self._gripper]
        handle = self.geom_xpos[:, self._handle]
        q = self.joint_q()
        success = q > self.spec.thresh
        reward = (-torch.linalg.norm(grip - handle, dim=-1)
                  + 4.0 * q + 2.0 * success.float())
        obs = torch.cat([self.qpos, self.qvel, grip, handle, q.unsqueeze(-1)], dim=-1)
        return obs, reward, success


def make(task: str, scene, cfg: TaskConfig | None = None) -> ArticulatedOpen:
    """Construct the named articulated-open task on a loaded scene."""
    return ArticulatedOpen(scene, ART_SPECS[task], cfg)
