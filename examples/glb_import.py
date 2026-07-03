"""Assets: GLB scene import — a glTF scene becomes a simulatable world.

Builds a small GLB indoor corner (table + rings + cups + ball), runs it
through the importer (scene graph -> baked transforms -> CoACD hulls ->
MJCF), then drops the dynamic objects onto the imported table on the GPU
engine and records world 0.

Run:  MUJOCO_GL=egl python examples/glb_import.py --record
"""

import argparse
import os

import numpy as np
import trimesh

import latentphysics as lpw
from latentphysics.assets.import_3d import ImportSpec, import_glb

T = trimesh.transformations.translation_matrix


def _tint(mesh, rgb):
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh, face_colors=np.array([*rgb, 255], dtype=np.uint8))
    return mesh


def build_glb(path):
    s = trimesh.Scene()
    wood, dark = (150, 108, 62), (96, 68, 40)
    s.add_geometry(_tint(trimesh.creation.box(extents=(1.3, 0.9, 0.06)), wood),
                   node_name="table_top", transform=T((0, 0, 0.5)))
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        s.add_geometry(_tint(trimesh.creation.box(extents=(0.07, 0.07, 0.47)), dark),
                       node_name=f"leg{i}", transform=T((sx * 0.55, sy * 0.36, 0.235)))
    accents = ((194, 84, 64), (219, 173, 77), (87, 140, 158))
    for i, c in enumerate(accents):
        s.add_geometry(_tint(trimesh.creation.annulus(r_min=0.05, r_max=0.11, height=0.045), c),
                       node_name=f"ring{i}", transform=T((-0.3 + 0.3 * i, 0.1 - 0.1 * i, 0.9 + 0.18 * i)))
    for i in range(2):
        s.add_geometry(_tint(trimesh.creation.cylinder(radius=0.045, height=0.11), (210, 202, 184)),
                       node_name=f"cup{i}", transform=T((0.15 - 0.4 * i, -0.15 + 0.35 * i, 1.25)))
    s.add_geometry(_tint(trimesh.creation.icosphere(subdivisions=3, radius=0.06), (120, 105, 150)),
                   node_name="ball", transform=T((0.05, 0.22, 1.45)))
    s.export(path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--steps", type=int, default=460)
    args = ap.parse_args()

    root = os.path.expanduser("~/lpw/assets/demos/glb")
    os.makedirs(root, exist_ok=True)
    glb = build_glb(os.path.join(root, "corner.glb"))
    mjcf = import_glb(glb, root, name="corner",
                      spec=ImportSpec(up="z", dynamic=("ring", "cup", "ball")))
    print("imported:", mjcf)

    scene = lpw.load_scene(mjcf, lpw.Config(n_worlds=4))
    traj = []
    for _ in range(args.steps):
        scene.step()
        traj.append(scene.qpos()[0].cpu().numpy().copy())

    import mujoco
    m = scene.mjm
    settled = []
    for b in range(m.nbody):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if m.body_dofnum[b] == 6:
            adr = int(m.jnt_qposadr[m.body_jntadr[b]])
            settled.append((name, round(traj[-1][adr + 2], 3)))
    print("settled z:", settled)
    assert all(-0.01 < z < 0.85 for _, z in settled), "an object escaped the world"

    if args.record:
        from _record import record_webp
        record_webp(mjcf, np.asarray(traj), "glb_import",
                    cam={"lookat": (0, 0, 0.45), "distance": 2.4,
                         "azimuth": 130, "elevation": -18, "azimuth_rate": 0.04})


if __name__ == "__main__":
    main()
