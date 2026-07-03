"""Indoor scene asset pipeline (our IP) — readiness report §5①.

Turns diverse indoor assets into engine-ready MJCF:

    OBJ / GLB / trimesh  --coacd-->  convex collision parts  -->  MJCF

Offline CPU preprocessing (convex decomposition is CPU in every major stack);
the GPU-accelerated step is SDF voxelization for concave large geometry, in
``latentphysics.assets.sdf_voxelize``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

__all__ = ["convex_decompose", "mesh_to_mjcf", "ConvexPart"]


@dataclass
class ConvexPart:
    vertices: "any"   # (n,3) float array
    faces: "any"      # (m,3) int array


def _load_trimesh(mesh):
    import trimesh
    if isinstance(mesh, str):
        m = trimesh.load(mesh, force="mesh")
    else:
        m = mesh
    return m


def _sanitize_mesh(m):
    """Drop the degeneracies that make CoACD/MuJoCo misbehave: non-finite
    vertices (and any face touching them), zero-area faces, and duplicate
    vertices. Returns a cleaned trimesh, or None if nothing usable remains.

    Real scanned/exported assets routinely carry these — feeding them to
    CoACD raw yields silent empty output (a collision-less "ghost" body) or
    a hard crash, so this runs before every decomposition.
    """
    import numpy as np
    import trimesh

    v = np.asarray(m.vertices, dtype=np.float64)
    f = np.asarray(m.faces, dtype=np.int64)
    if v.size == 0 or f.size == 0:
        return None

    finite = np.isfinite(v).all(axis=1)
    if not finite.all():
        remap = -np.ones(len(v), dtype=np.int64)
        remap[finite] = np.arange(int(finite.sum()))
        v = v[finite]
        f = remap[f]
        f = f[(f >= 0).all(axis=1)]
        if len(v) == 0 or len(f) == 0:
            return None

    clean = trimesh.Trimesh(vertices=v, faces=f, process=True)  # merges dupes
    clean.update_faces(clean.nondegenerate_faces())             # drops zero-area
    clean.remove_unreferenced_vertices()
    if len(clean.vertices) < 4 or len(clean.faces) == 0:
        return None
    return clean


def convex_decompose(mesh, threshold: float = 0.05, max_hulls: int = -1):
    """Decompose a (possibly concave) mesh into convex parts via CoACD (CPU).

    Parameters
    ----------
    mesh:       path to a mesh file, or a trimesh.Trimesh.
    threshold:  CoACD concavity threshold (lower = more parts, tighter fit).
    max_hulls:  cap on number of convex hulls (-1 = unlimited).

    Returns a non-empty list[ConvexPart]. The mesh is sanitized first; if
    CoACD fails or returns nothing, falls back to the mesh's single convex
    hull so the caller never silently gets a collision-less body. Raises
    ValueError if the mesh has no usable geometry at all.

    Contact-sensitive furniture (edges/handles) wants a lower threshold /
    higher hull budget.
    """
    import coacd
    import numpy as np

    m = _sanitize_mesh(_load_trimesh(mesh))
    if m is None:
        raise ValueError("convex_decompose: mesh has no usable geometry "
                         "after sanitization (empty / non-finite / degenerate)")

    parts = []
    try:
        cmesh = coacd.Mesh(np.asarray(m.vertices, dtype=np.float64),
                           np.asarray(m.faces, dtype=np.int32))
        parts = coacd.run_coacd(cmesh, threshold=threshold, max_convex_hull=max_hulls)
    except Exception:
        parts = []

    if not parts:   # CoACD gave up — a single convex hull still collides correctly
        hull = m.convex_hull
        parts = [(np.asarray(hull.vertices), np.asarray(hull.faces))]

    return [ConvexPart(vertices=np.asarray(v, dtype=np.float32), faces=np.asarray(f, dtype=np.int32))
            for (v, f) in parts]


def mesh_to_mjcf(mesh, out_dir: str, name: str = "object", mass: float = 1.0,
                 pos=(0.0, 0.0, 1.0), free: bool = True, threshold: float = 0.05):
    """Full path: mesh -> convex collision parts -> a loadable MJCF.

    Writes the visual mesh + per-part collision OBJs into ``out_dir`` and emits
    ``<out_dir>/<name>.xml`` with the convex parts as collision geoms and the
    original mesh as a (collision-free) visual geom. Returns the MJCF path.
    """
    import trimesh

    os.makedirs(out_dir, exist_ok=True)
    m = _load_trimesh(mesh)

    # visual mesh
    vis_obj = f"{name}_visual.obj"
    m.export(os.path.join(out_dir, vis_obj))

    # collision convex parts
    parts = convex_decompose(m, threshold=threshold)
    part_objs = []
    for i, p in enumerate(parts):
        fn = f"{name}_coll_{i}.obj"
        trimesh.Trimesh(vertices=p.vertices, faces=p.faces).export(os.path.join(out_dir, fn))
        part_objs.append(fn)

    joint = "<freejoint/>" if free else ""
    assets = [f'    <mesh name="{name}_visual" file="{vis_obj}"/>']
    assets += [f'    <mesh name="{name}_c{i}" file="{fn}"/>' for i, fn in enumerate(part_objs)]
    coll_geoms = "\n".join(
        f'      <geom type="mesh" mesh="{name}_c{i}" group="3"/>' for i in range(len(part_objs))
    )
    xml = f"""<mujoco model="{name}">
  <compiler meshdir="." angle="radian"/>
  <option timestep="0.005"/>
  <asset>
{os.linesep.join(assets)}
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}">
      {joint}
      <geom type="mesh" mesh="{name}_visual" contype="0" conaffinity="0" group="2"/>
{coll_geoms}
    </body>
  </worldbody>
</mujoco>
"""
    mjcf_path = os.path.join(out_dir, f"{name}.xml")
    with open(mjcf_path, "w") as f:
        f.write(xml)
    return mjcf_path
