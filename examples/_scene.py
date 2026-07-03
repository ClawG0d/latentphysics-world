"""Shared scene builder for the IK-scripted Franka manipulation demos.

Writes a scene that includes ``mjx_scene.xml`` (the arm + the blue
checkerboard groundplane + skybox + lighting, matching pick & place) plus
a custom worldbody and a ``home`` keyframe. The file goes into the
menagerie panda directory so the panda's relative mesh paths resolve.
"""

from __future__ import annotations

import os

MEN = os.environ.get("LPW_MENAGERIE", os.path.expanduser("~/lpw/menagerie"))
_PANDA_DIR = os.path.join(MEN, "franka_emika_panda")

# arm home pose (7 joints + 2 fingers) and its position-actuator ctrl target
ARM_HOME = "0 0.3 0 -1.57079 0 2.0 -0.7853 0.04 0.04"
ARM_CTRL = "0 0.3 0 -1.57079 0 2.0 -0.7853 0.04"


def franka_scene(name: str, bodies_xml: str, obj_qpos: str, extra_assets: str = "") -> str:
    """Build ``<panda_dir>/_lpw_<name>.xml`` and return its path.

    ``bodies_xml``  : worldbody geoms/bodies to add beside the arm.
    ``obj_qpos``    : qpos values for those bodies, appended after ARM_HOME
                      (order must match declaration; free body = 7 numbers).
    ``extra_assets``: optional extra ``<asset>`` block contents.
    """
    assets = f"<asset>{extra_assets}</asset>" if extra_assets else ""
    xml = f"""<mujoco model="{name}">
  <include file="mjx_scene.xml"/>
  {assets}
  <worldbody>
    {bodies_xml}
  </worldbody>
  <keyframe>
    <key name="home" qpos="{ARM_HOME} {obj_qpos}" ctrl="{ARM_CTRL}"/>
  </keyframe>
</mujoco>
"""
    path = os.path.join(_PANDA_DIR, f"_lpw_{name}.xml")
    with open(path, "w") as f:
        f.write(xml)
    return path
