"""Worlds: articulated room — procedural interior with working furniture.

Generates a seeded room whose wall furniture includes all four articulated
archetypes (R4): a hinged-door cabinet, a two-drawer chest, a top-hinged
lid chest, and a sliding-door cabinet. Scripts an open/close cycle on the
GPU engine and records world 0.

Run:  MUJOCO_GL=egl python examples/articulated_room.py --record
"""

import argparse
import math
import os

import numpy as np

import latentphysics as lpw
from latentphysics.assets.scene_gen import RoomSpec, generate_room

# joint-name suffix -> (open velocity, does it fall shut on its own?)
KINDS = {"_drawer": (0.30, False), "_door": (1.0, False),
         "_sdoor": (0.30, False), "_lid": (1.3, True)}


def _kind(name):
    for suf in KINDS:
        if suf in name:
            return suf
    return None


def articulated_joints(mjm):
    import mujoco
    out = []
    for j in range(mjm.njnt):
        name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        k = _kind(name)
        if k:
            out.append((name, int(mjm.jnt_qposadr[j]), int(mjm.jnt_dofadr[j]), k))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    path = os.path.expanduser("~/lpw/assets/demos/artic_room.xml")
    # table shifted off-center frees one end so the two pieces cluster into a
    # single legible shot (all four archetypes are exercised by the tests)
    generate_room(RoomSpec(seed=args.seed, size=(6.2, 4.6), table_pos=(-1.4, 0.0),
                           n_articulated=2, n_furniture=30, n_clutter=5), path)
    scene = lpw.load_scene(path, lpw.Config(n_worlds=4, njmax=2048))
    joints = articulated_joints(scene.mjm)
    print("articulated joints:", [n for n, *_ in joints])

    qpos, qvel = scene.qpos(), scene.qvel()
    traj = []

    def run(n, direction=0.0):
        for _ in range(n):
            for name, _, dof, k in joints:
                qvel[:, dof] = direction * KINDS[k][0]
            scene.step()
            traj.append(qpos[0].cpu().numpy().copy())

    run(40)                       # settle
    run(150, direction=1.0)       # open everything
    peak = {k: 0.0 for k in KINDS}
    for name, qa, _, k in joints:
        peak[k] = max(peak[k], qpos[0, qa].item())
    run(70)                       # hold (lid falls shut; others park)
    run(130, direction=-1.0)      # close
    run(40)

    print("peak openings by kind:", {k: round(v, 3) for k, v in peak.items() if v})
    for k, (_, _falls) in KINDS.items():
        if any(kk == k for *_, kk in joints):
            assert peak[k] > 0.1, f"{k} never opened (peak {peak[k]:.3f})"

    if args.record:
        import mujoco
        from _record import record_webp
        centers = np.array([scene.mjm.body(b).pos[:2] for b in range(scene.mjm.nbody)
                            if (mujoco.mj_id2name(scene.mjm, mujoco.mjtObj.mjOBJ_BODY, b) or "")
                            .endswith("art")])
        mid = centers.mean(axis=0) if len(centers) else np.zeros(2)
        az = math.degrees(math.atan2(mid[1], mid[0]))
        record_webp(path, np.asarray(traj), "articulated_room",
                    cam={"lookat": (float(mid[0]) * 0.8, float(mid[1]) * 0.8, 0.5),
                         "distance": 3.4, "azimuth": az, "elevation": -22})


if __name__ == "__main__":
    main()
