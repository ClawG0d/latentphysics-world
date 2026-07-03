"""Manipulation: franka peg-in-hole — grasp a peg and insert it into a socket.

Classic precision / contact-rich benchmark, IK-scripted (no policy). The peg
has ~1 cm clearance in the socket, so it only seats if the grasp + insertion
are accurate.

Run:  MUJOCO_GL=egl python examples/franka_peg_insert.py --record
"""

import argparse

import numpy as np
import torch

import latentphysics as lpw

PEG = (0.5, -0.10)            # peg start xy
HOLE = (0.5, 0.16)           # socket center xy
PEG_HALF = (0.018, 0.018, 0.05)


def bodies():
    ph = PEG_HALF
    walls = []
    inner, t, h = 0.032, 0.012, 0.06     # socket: inner half, wall thickness, height
    for dx, dy, sx, sy in ((inner + t, 0, t, inner + 2 * t),
                           (-(inner + t), 0, t, inner + 2 * t),
                           (0, inner + t, inner + 2 * t, t),
                           (0, -(inner + t), inner + 2 * t, t)):
        walls.append(f'<geom type="box" pos="{HOLE[0]+dx:.3f} {HOLE[1]+dy:.3f} {h/2:.3f}" '
                     f'size="{sx:.3f} {sy:.3f} {h/2:.3f}" rgba="0.5 0.52 0.55 1" '
                     f'contype="1" conaffinity="1"/>')
    peg = (f'<body name="peg" pos="{PEG[0]} {PEG[1]} {ph[2]}"><freejoint/>'
           f'<geom name="peg" type="box" size="{ph[0]} {ph[1]} {ph[2]}" '
           f'rgba="0.90 0.55 0.15 1" mass="0.05" condim="4" friction="1 0.05 0.001" '
           f'contype="1" conaffinity="1"/></body>')
    return peg + "".join(walls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    args = ap.parse_args()

    import mujoco
    from _scene import franka_scene
    from _ik import ArmController

    path = franka_scene("peg_insert", bodies(),
                        f"{PEG[0]} {PEG[1]} {PEG_HALF[2]} 1 0 0 0")
    scene = lpw.load_scene(path, lpw.Config(n_worlds=4))
    arm = ArmController(scene)
    O, C = arm.OPEN, arm.CLOSED

    gz = PEG_HALF[2] + 0.015                  # grasp near the peg's mid/upper body
    arm.move([PEG[0], PEG[1], gz + 0.16], O, 60)     # approach above peg
    arm.move([PEG[0], PEG[1], gz], O, 60)            # descend to peg
    arm.hold(C, 60)                                  # grasp
    arm.move([PEG[0], PEG[1], gz + 0.22], C, 60)     # lift
    arm.move([HOLE[0], HOLE[1], gz + 0.22], C, 90)   # carry over the socket
    arm.move([HOLE[0], HOLE[1], gz + 0.04], C, 90)   # insert down into the well
    arm.hold(O, 50)                                  # release
    arm.move([HOLE[0], HOLE[1], gz + 0.24], O, 60)   # retract

    mjm = scene.mjm
    padr = int(mjm.jnt_qposadr[mjm.body_jntadr[mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "peg")]])
    peg_xy = scene.qpos()[:, padr:padr + 2]
    dxy = torch.linalg.norm(peg_xy - torch.tensor(HOLE, device="cuda"), dim=-1)
    seated = int((dxy < 0.03).sum().item())
    print(f"peg seated in {seated}/{scene.n_worlds} worlds "
          f"(xy err {[round(v, 3) for v in dxy.tolist()]} m)")
    assert seated == scene.n_worlds, "peg insertion failed"

    if args.record:
        from _record import record_webp
        record_webp(path, np.asarray(arm.traj), "franka_peg_insert",
                    cam={"lookat": (0.5, 0.03, 0.1), "distance": 1.35,
                         "azimuth": 140, "elevation": -16}, every=4, quality=52)


if __name__ == "__main__":
    main()
