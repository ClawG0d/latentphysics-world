"""Rigid: franka cube — scripted grasp-and-lift on the GPU engine.

Starts from the menagerie `pickup` keyframe (gripper poised at the cube),
closes the fingers, and lifts back toward the home pose. Contact does the
rest: if friction and contact forces are right, the cube comes along.

Run:  MUJOCO_GL=egl python examples/franka_cube_grasp.py --record
"""

import argparse
import os

import numpy as np
import torch

import latentphysics as lpw

MJCF = os.path.join(
    os.environ.get("LPW_MENAGERIE", os.path.expanduser("~/lpw/menagerie")),
    "franka_emika_panda", "mjx_single_cube.xml",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    args = ap.parse_args()

    import mujoco

    scene = lpw.load_scene(MJCF, lpw.Config(n_worlds=4))
    mjm = scene.mjm
    kid = {n: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_KEY, n) for n in ("home", "pickup")}
    q_pick = torch.as_tensor(mjm.key_qpos[kid["pickup"]], dtype=torch.float32, device="cuda")
    c_home = torch.as_tensor(mjm.key_ctrl[kid["home"]], dtype=torch.float32, device="cuda")

    bid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "box")
    cube_adr = int(mjm.jnt_qposadr[mjm.body_jntadr[bid]])

    qpos, qvel, ctrl = scene.qpos(), scene.qvel(), scene.state("ctrl")
    qpos.copy_(q_pick.expand_as(qpos))
    qvel.zero_()
    # servo target = the keyframe's own arm pose. The keyframe's stored ctrl
    # is a different (hover) target — using it raises the hand off the cube.
    ctrl[:, :7] = q_pick[:7]
    ctrl[:, 7] = 0.04                      # fingers open
    scene.forward()

    traj = []

    def run(n):
        for _ in range(n):
            scene.step()
            traj.append(qpos[0].cpu().numpy().copy())

    run(30)                                # settle at the grasp pose
    ctrl[:, 7] = 0.0                       # close the gripper
    run(80)
    for k in range(150):                   # lift: blend arm ctrl toward home
        a = (k + 1) / 150
        ctrl[:, :7] = (1 - a) * q_pick[:7] + a * c_home[:7]
        scene.step()
        traj.append(qpos[0].cpu().numpy().copy())
    run(60)                                # hold

    z0 = float(q_pick[cube_adr + 2])
    z1 = qpos[:, cube_adr + 2]
    lifted = int((z1 > z0 + 0.10).sum().item())
    print(f"cube z: start {z0:.3f} -> final {[round(v, 3) for v in z1.tolist()]} "
          f"| lifted in {lifted}/{scene.n_worlds} worlds")
    assert lifted == scene.n_worlds, "grasp failed — check friction/keyframe"

    if args.record:
        from _record import record_webp
        record_webp(MJCF, np.asarray(traj), "franka_cube_grasp",
                    cam={"lookat": (0.5, 0.0, 0.25), "distance": 1.5,
                         "azimuth": 140, "elevation": -18})


if __name__ == "__main__":
    main()
