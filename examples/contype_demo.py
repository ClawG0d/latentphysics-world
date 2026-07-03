"""Rigid: contype — collision bitmask semantics in one glance.

Two identical spheres drop onto a glass shelf. The green one carries a mask
that pairs with the shelf and comes to rest on it; the red one is masked out
of the shelf pair and falls straight through, landing on the floor. MuJoCo
pair rule: collide iff (contype_a & conaffinity_b) | (contype_b & conaffinity_a).

Physics runs on the LPW GPU engine. Run:
  MUJOCO_GL=egl python examples/contype_demo.py --record
"""

import argparse
import os

import numpy as np

import latentphysics as lpw

XML = """<mujoco model="contype_demo">
  <option timestep="0.004" iterations="10" ls_iterations="10"/>
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
    <light name="fill" directional="true" pos="2 -2 2" dir="0.4 0.5 -0.8"
           diffuse="0.3 0.3 0.33" castshadow="false"/>
    <geom name="floor" type="plane" size="3 3 0.1" material="floormat"
          contype="3" conaffinity="3"/>
    <geom name="shelf" type="box" pos="0 0 0.5" size="0.45 0.3 0.015"
          rgba=".7 .85 .9 0.5" contype="1" conaffinity="1"/>
    <body name="green" pos="-0.2 0 1.2"><freejoint/>
      <geom type="sphere" size="0.07" rgba=".35 .65 .35 1" mass="0.2"
            contype="1" conaffinity="1"/></body>
    <body name="red" pos="0.2 0 1.2"><freejoint/>
      <geom type="sphere" size="0.07" rgba=".75 .3 .25 1" mass="0.2"
            contype="2" conaffinity="2"/></body>
  </worldbody>
</mujoco>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--steps", type=int, default=330)
    args = ap.parse_args()

    path = os.path.expanduser("~/lpw/assets/demos/contype.xml")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(XML)

    scene = lpw.load_scene(path, lpw.Config(n_worlds=4))
    traj = []
    for _ in range(args.steps):
        scene.step()
        traj.append(scene.qpos()[0].cpu().numpy().copy())

    green_z, red_z = traj[-1][2], traj[-1][9]
    print(f"green rests at z={green_z:.3f} (on shelf ~0.585), "
          f"red at z={red_z:.3f} (through shelf, on floor ~0.07)")
    assert green_z > 0.4 and red_z < 0.2, "mask semantics demo failed"

    if args.record:
        from _record import record_webp
        record_webp(path, np.asarray(traj), "contype_masks",
                    cam={"lookat": (0, 0, 0.5), "distance": 2.1,
                         "azimuth": 135, "elevation": -12})


if __name__ == "__main__":
    main()
