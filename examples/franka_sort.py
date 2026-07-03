"""Manipulation: franka sort — pick three cubes and drop them into a bin.

Sequential multi-object pick-and-place, IK-scripted (no policy). Shows the
arm handling a task made of several sub-grasps in one run.

Run:  MUJOCO_GL=egl python examples/franka_sort.py --record
"""

import argparse

import numpy as np
import torch

import latentphysics as lpw

H = 0.022                                   # cube half-extent
CUBES = [(0.42, -0.12, "0.85 0.3 0.25"), (0.5, -0.18, "0.3 0.7 0.35"),
         (0.58, -0.10, "0.35 0.55 0.85")]   # (x, y, rgba)
BIN = (0.5, 0.20)                           # bin center xy
BIN_INNER = 0.11                            # bin inner half-size


def bodies():
    parts = []
    for i, (x, y, c) in enumerate(CUBES):
        parts.append(f'<body name="cube{i}" pos="{x} {y} {H}"><freejoint/>'
                     f'<geom type="box" size="{H} {H} {H}" rgba="{c} 1" mass="0.04" '
                     f'condim="4" friction="1 0.05 0.001" contype="1" conaffinity="1"/></body>')
    t, bh = 0.012, 0.06                      # bin wall thickness, height
    for dx, dy, sx, sy in ((BIN_INNER + t, 0, t, BIN_INNER + 2 * t),
                           (-(BIN_INNER + t), 0, t, BIN_INNER + 2 * t),
                           (0, BIN_INNER + t, BIN_INNER + 2 * t, t),
                           (0, -(BIN_INNER + t), BIN_INNER + 2 * t, t)):
        parts.append(f'<geom type="box" pos="{BIN[0]+dx:.3f} {BIN[1]+dy:.3f} {bh/2:.3f}" '
                     f'size="{sx:.3f} {sy:.3f} {bh/2:.3f}" rgba="0.55 0.5 0.42 1" '
                     f'contype="1" conaffinity="1"/>')
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    args = ap.parse_args()

    import mujoco
    from _scene import franka_scene
    from _ik import ArmController

    obj_qpos = " ".join(f"{x} {y} {H} 1 0 0 0" for x, y, _ in CUBES)
    path = franka_scene("sort", bodies(), obj_qpos)
    scene = lpw.load_scene(path, lpw.Config(n_worlds=4, njmax=2048))
    arm = ArmController(scene)
    O, C = arm.OPEN, arm.CLOSED
    gz = H + 0.015

    for i, (x, y, _) in enumerate(CUBES):
        drop = [BIN[0] + (i - 1) * 0.05, BIN[1], gz + 0.16]   # spread drops in the bin
        arm.move([x, y, gz + 0.14], O, 45)     # approach cube
        arm.move([x, y, gz], O, 45)            # descend
        arm.hold(C, 45)                        # grasp
        arm.move([x, y, gz + 0.20], C, 45)     # lift
        arm.move([drop[0], drop[1], gz + 0.22], C, 70)   # carry over bin
        arm.move(drop, C, 40)                  # lower into bin
        arm.hold(O, 30)                        # release
        arm.move([drop[0], drop[1], gz + 0.24], O, 35)   # retract

    mjm = scene.mjm
    inside = 0
    adrs = [int(mjm.jnt_qposadr[mjm.body_jntadr[mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, f"cube{i}")]])
            for i in range(len(CUBES))]
    qpos = scene.qpos()
    per_world_ok = torch.ones(scene.n_worlds, dtype=torch.bool, device="cuda")
    for a in adrs:
        dx = (qpos[:, a] - BIN[0]).abs()
        dy = (qpos[:, a + 1] - BIN[1]).abs()
        per_world_ok &= (dx < BIN_INNER) & (dy < BIN_INNER)
    binned = int(per_world_ok.sum().item())
    print(f"all 3 cubes binned in {binned}/{scene.n_worlds} worlds")
    assert binned == scene.n_worlds, "sort failed"

    if args.record:
        from _record import record_webp
        record_webp(path, np.asarray(arm.traj), "franka_sort",
                    cam={"lookat": (0.5, 0.02, 0.06), "distance": 1.5,
                         "azimuth": 145, "elevation": -20}, every=7, quality=46)


if __name__ == "__main__":
    main()
