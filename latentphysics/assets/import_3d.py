"""GLB/glTF indoor scene import (R4) — real 3D assets to engine-ready MJCF.

Traverses the glTF scene graph, bakes every node transform into world-space
vertices (sidesteps MuJoCo's inability to express non-uniform node scale),
convex-decomposes each object via CoACD, and emits one MJCF: original meshes
as collision-free visual geoms, convex hulls as collision geoms, with the
same collision-mask scheme as the procedural generator (static x static
pruned at model build).

glTF is +y-up; MuJoCo is +z-up — ``ImportSpec.up`` controls the conversion.
USD import lands later on the same composer.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np

from . import convex_decompose

__all__ = ["ImportSpec", "import_glb"]

# masks match scene_gen: static never pairs with itself, dynamic pairs with all
_S = 'contype="1" conaffinity="2"'
_D = 'contype="3" conaffinity="3"'


@dataclass
class ImportSpec:
    threshold: float = 0.06        # CoACD concavity (lower = tighter hulls)
    max_hulls: int = 32            # hull cap per object
    dynamic: tuple = ()            # node-name substrings imported as free bodies
    up: str = "y"                  # source up-axis: "y" (glTF standard) or "z"
    scale: float = 1.0             # uniform rescale (e.g. cm assets -> 0.01)
    add_floor: bool = True         # add a ground plane (off if the scan has one)
    density: float = 400.0         # dynamic-body geom density (kg/m^3)
    solver_iterations: int = 8
    ls_iterations: int = 10


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name) or "node"


def _up_matrix(spec) -> np.ndarray:
    T = np.eye(4)
    if spec.up == "y":             # rotate +90 deg about x: y-up -> z-up
        T[:3, :3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    T[:3, :3] *= spec.scale
    return T


def _rgba(mesh) -> str:
    """Mean vertex color (how GLB color data survives a trimesh round-trip),
    falling back to the PBR base color, then neutral gray."""
    for get in (lambda: np.asarray(mesh.visual.vertex_colors, dtype=float).mean(axis=0),
                lambda: np.asarray(mesh.visual.material.main_color, dtype=float)):
        try:
            c = get()[:4] / 255.0
            return "%.3f %.3f %.3f %.3f" % (c[0], c[1], c[2], max(c[3], 0.05))
        except Exception:
            continue
    return "0.7 0.7 0.7 1"


def import_glb(glb_path: str, out_dir: str, name: str = "scene",
               spec: ImportSpec | None = None) -> str:
    """Import a GLB/glTF scene into ``<out_dir>/<name>.xml``; returns the path."""
    import trimesh

    spec = spec or ImportSpec()
    os.makedirs(out_dir, exist_ok=True)
    loaded = trimesh.load(glb_path)
    if isinstance(loaded, trimesh.Trimesh):
        scene = trimesh.Scene()
        scene.add_geometry(loaded, node_name="object")
    else:
        scene = loaded

    up = _up_matrix(spec)
    assets, bodies = [], []
    seen = {}
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph.get(node)
        mesh = scene.geometry[gname]
        if not hasattr(mesh, "vertices") or len(mesh.faces) == 0:
            continue
        raw = _sanitize(node)
        n_prev = seen.get(raw, 0)
        seen[raw] = n_prev + 1
        base = raw if n_prev == 0 else f"{raw}_{n_prev}"

        world = mesh.copy()
        world.apply_transform(up @ T)          # bake node transform + up-axis
        rgba = _rgba(mesh)
        is_dyn = any(s in node for s in spec.dynamic)

        # dynamic bodies get a local frame at their centroid so the free
        # joint / inertia are well-conditioned; statics stay in world frame
        origin = world.vertices.mean(axis=0) if is_dyn else np.zeros(3)
        world.apply_translation(-origin)

        vis = f"{base}_visual.obj"
        world.export(os.path.join(out_dir, vis))
        assets.append(f'<mesh name="{base}_v" file="{vis}"/>')

        hulls = convex_decompose(world, threshold=spec.threshold,
                                 max_hulls=spec.max_hulls)
        geoms = [f'<geom type="mesh" mesh="{base}_v" rgba="{rgba}" '
                 f'contype="0" conaffinity="0" group="2"/>']
        for i, part in enumerate(hulls):
            fn = f"{base}_c{i}.obj"
            trimesh.Trimesh(vertices=part.vertices, faces=part.faces).export(
                os.path.join(out_dir, fn))
            assets.append(f'<mesh name="{base}_c{i}" file="{fn}"/>')
            mask = _D if is_dyn else _S
            dens = f' density="{spec.density}"' if is_dyn else ""
            geoms.append(f'<geom type="mesh" mesh="{base}_c{i}" group="3" '
                         f'rgba="{rgba}"{dens} {mask}/>')

        joint = "<freejoint/>" if is_dyn else ""
        pos = "%.5f %.5f %.5f" % tuple(origin)
        bodies.append(f'<body name="{base}" pos="{pos}">{joint}'
                      + "".join(geoms) + "</body>")

    floor = (f'<geom name="floor" type="plane" size="10 10 0.1" '
             f'material="floormat" {_S}/>' if spec.add_floor else "")
    assets.append('<texture type="skybox" builtin="gradient" rgb1="0.45 0.53 0.62" '
                  'rgb2="0.12 0.14 0.18" width="256" height="256"/>')
    if spec.add_floor:
        assets.append('<texture name="floortex" type="2d" builtin="checker" '
                      'rgb1="0.78 0.74 0.68" rgb2="0.68 0.64 0.58" mark="edge" '
                      'markrgb="0.55 0.52 0.48" width="300" height="300"/>')
        assets.append('<material name="floormat" texture="floortex" '
                      'texrepeat="16 16" reflectance="0.12"/>')
    xml = f"""<mujoco model="{name}">
  <compiler meshdir="." angle="radian"/>
  <option timestep="0.005" iterations="{spec.solver_iterations}" ls_iterations="{spec.ls_iterations}"/>
  <asset>
    {chr(10).join('    ' + a for a in assets).lstrip()}
  </asset>
  <worldbody>
    <light name="key" directional="true" pos="0 0 3" dir="-0.3 0.2 -0.9"
           diffuse="0.8 0.78 0.75" castshadow="true"/>
    <light name="fill" directional="true" pos="2 -2 2" dir="0.4 0.5 -0.8"
           diffuse="0.3 0.3 0.33" castshadow="false"/>
    {floor}
    {chr(10).join('    ' + b for b in bodies).lstrip()}
  </worldbody>
</mujoco>
"""
    path = os.path.join(out_dir, f"{name}.xml")
    with open(path, "w") as f:
        f.write(xml)
    return path
