"""Assets: import validity + stability bake — trust an imported world before use.

Imports a small GLB scene, runs the classical validity checks (mass /
inertia / no-collision ghosts, initial interpenetration), then bakes it to
rest with settle(). Prints each report; asserts the scene ends healthy.

Run:  MUJOCO_GL=egl python examples/validate_import.py
"""

import os

import numpy as np
import trimesh

import latentphysics as lpw
from latentphysics.assets.import_3d import ImportSpec, import_glb
from latentphysics.assets.validate import (
    initial_penetration, settle, validate_model,
)

T = trimesh.transformations.translation_matrix


def build_glb(path):
    s = trimesh.Scene()
    s.add_geometry(trimesh.creation.box(extents=(1.0, 0.7, 0.06)),
                   node_name="table_top", transform=T((0, 0, 0.5)))
    for i in range(3):
        s.add_geometry(trimesh.creation.annulus(r_min=0.05, r_max=0.1, height=0.04),
                       node_name=f"ring{i}", transform=T((-0.2 + 0.2 * i, 0, 0.75 + 0.12 * i)))
    s.export(path)
    return path


def main():
    import mujoco

    root = os.path.expanduser("~/lpw/assets/demos/validate")
    os.makedirs(root, exist_ok=True)
    mjcf = import_glb(build_glb(os.path.join(root, "scene.glb")), root,
                      name="vscene", spec=ImportSpec(up="z", dynamic=("ring",)))

    mjm = mujoco.MjModel.from_xml_path(mjcf)
    print("structural:", validate_model(mjm))
    print("spawn contacts:", initial_penetration(mjm))
    assert validate_model(mjm).ok, "imported model has structural defects"

    scene = lpw.load_scene(mjcf, lpw.Config(n_worlds=4))
    info = settle(scene)
    print(f"settle: {info['steps']} steps -> residual |v| = "
          f"{info['residual_vel']:.4f} m/s (converged={info['converged']})")
    assert info["converged"], "scene did not reach rest"
    print("scene is validated and baked to rest — safe to use as an origin state")


if __name__ == "__main__":
    main()
