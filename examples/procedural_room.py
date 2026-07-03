"""Worlds: procedural indoor room — furnished, textured, clutter settling.

Generates a seeded room (wood/fabric grain materials, translucent walls so
the interior reads) and lets its tabletop clutter settle on the LPW GPU
engine, recorded from an orbiting dollhouse view.

Run:  MUJOCO_GL=egl python examples/procedural_room.py --record
"""

import argparse
import os

import numpy as np

import latentphysics as lpw
from latentphysics.assets.scene_gen import RoomSpec, generate_room


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--steps", type=int, default=240)
    args = ap.parse_args()

    path = os.path.expanduser("~/lpw/assets/demos/procedural_room.xml")
    generate_room(RoomSpec(seed=args.seed, n_furniture=14, n_clutter=10), path)
    scene = lpw.load_scene(path, lpw.Config(n_worlds=4))

    traj = []
    for _ in range(args.steps):
        scene.step()
        traj.append(scene.qpos()[0].cpu().numpy().copy())
    assert np.isfinite(traj[-1]).all(), "room state went non-finite"
    print(f"simulated {args.steps} steps on GPU; recorded {len(traj)} frames")

    if args.record:
        from _record import record_webp
        # high oblique orbit; translucent walls keep the interior visible
        record_webp(path, np.asarray(traj), "procedural_room",
                    cam={"lookat": (0.0, 0.0, 0.1), "distance": 6.4,
                         "azimuth": 35, "elevation": -47, "azimuth_rate": 0.25})


if __name__ == "__main__":
    main()
