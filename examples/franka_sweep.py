"""Manipulation: franka sweep — herd scattered cubes into a corral.

Non-prehensile: the closed gripper makes lateral sweeps that push scattered
cubes into a three-sided low corral (the side walls stop them squirting out).
IK-scripted (no policy).

Run:  MUJOCO_GL=egl python examples/franka_sweep.py --record
"""

import argparse

import numpy as np
import torch

import latentphysics as lpw

H = 0.02
CX = 0.5                                     # corral center x
CORRAL_Y, HALF_W = -0.06, 0.13               # corral center y, inner half-width
BACK_Y = CORRAL_Y - 0.09                     # back wall y
# scattered cubes, all within the corral's x-span so sweeps can corral them
SCATTER = [(0.42, 0.16), (0.5, 0.20), (0.58, 0.14), (0.45, 0.09), (0.56, 0.06)]


def bodies():
    cols = ["0.85 0.35 0.3", "0.35 0.7 0.4", "0.4 0.55 0.85", "0.85 0.7 0.3", "0.6 0.45 0.75"]
    parts = [f'<body name="c{i}" pos="{x} {y} {H}"><freejoint/>'
             f'<geom type="box" size="{H} {H} {H}" rgba="{cols[i]} 1" mass="0.03" '
             f'condim="4" friction="1 0.05 0.001" contype="1" conaffinity="1"/></body>'
             for i, (x, y) in enumerate(SCATTER)]
    t, wh = 0.012, 0.05                       # wall thickness, height
    # three-sided corral open toward +y (where the cubes are)
    walls = [(CX, BACK_Y, HALF_W + t, t),                 # back wall (-y)
             (CX - HALF_W - t, CORRAL_Y, t, 0.09),        # left wall
             (CX + HALF_W + t, CORRAL_Y, t, 0.09)]        # right wall
    for x, y, sx, sy in walls:
        parts.append(f'<geom type="box" pos="{x:.3f} {y:.3f} {wh/2:.3f}" '
                     f'size="{sx:.3f} {sy:.3f} {wh/2:.3f}" rgba="0.55 0.5 0.42 1" '
                     f'contype="1" conaffinity="1"/>')
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    args = ap.parse_args()

    import mujoco
    from _scene import franka_scene
    from _ik import ArmController

    obj_qpos = " ".join(f"{x} {y} {H} 1 0 0 0" for x, y in SCATTER)
    path = franka_scene("sweep", bodies(), obj_qpos)
    scene = lpw.load_scene(path, lpw.Config(n_worlds=4, njmax=2048))
    arm = ArmController(scene)
    C = arm.CLOSED
    z = 0.05

    # sweep each lane from beyond the cubes down into the corral; overlap lanes
    for lane_x in (0.44, 0.5, 0.56, 0.47, 0.53):
        arm.move([lane_x, 0.26, z + 0.15], C, 32)      # above, behind the row
        arm.move([lane_x, 0.26, z], C, 26)             # descend
        arm.move([lane_x, CORRAL_Y + 0.02, z], C, 80)  # sweep into the corral
        arm.move([lane_x, CORRAL_Y + 0.02, z + 0.15], C, 24)   # lift out
    arm.hold(C, 40)

    mjm = scene.mjm
    adrs = [int(mjm.jnt_qposadr[mjm.body_jntadr[mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, f"c{i}")]])
            for i in range(len(SCATTER))]
    qpos = scene.qpos()
    ok = torch.ones(scene.n_worlds, dtype=torch.bool, device="cuda")
    for a in adrs:
        inx = (qpos[:, a] - CX).abs() < HALF_W + 0.03
        iny = (qpos[:, a + 1] > BACK_Y - 0.03) & (qpos[:, a + 1] < CORRAL_Y + 0.10)
        ok &= inx & iny
    corralled = int(ok.sum().item())
    print(f"all cubes corralled in {corralled}/{scene.n_worlds} worlds")
    assert corralled == scene.n_worlds, "sweep failed"

    if args.record:
        from _record import record_webp
        record_webp(path, np.asarray(arm.traj), "franka_sweep",
                    cam={"lookat": (0.5, 0.05, 0.04), "distance": 1.5,
                         "azimuth": 145, "elevation": -24}, every=7, quality=46)


if __name__ == "__main__":
    main()
