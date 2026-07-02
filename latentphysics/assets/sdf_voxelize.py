"""GPU SDF voxelization (our IP) — readiness report §5④, the GPU-accelerated
step of the asset pipeline.

Builds a signed-distance-field volume from a triangle mesh using NVIDIA Warp:
a BVH-accelerated closest-point query is launched per voxel (embarrassingly
parallel), 1-2 orders of magnitude faster than CPU. Used to give concave
furniture (which convex decomposition would over-simplify) an accurate SDF
collision representation for MuJoCo's SDF plugin.
"""

from __future__ import annotations


def voxelize_sdf(mesh, resolution: int = 64, padding: float = 0.1, device: str = "cuda"):
    """Compute a signed distance field of ``mesh`` on the GPU.

    Parameters
    ----------
    mesh:        path to a mesh file, or a trimesh.Trimesh.
    resolution:  grid cells per axis (res^3 voxels).
    padding:     fraction of the AABB diagonal added around the mesh.

    Returns (sdf, lower, upper): sdf is a (res,res,res) numpy array of signed
    distances (negative inside); lower/upper are the grid world bounds.
    """
    import numpy as np
    import warp as wp

    from . import _load_trimesh

    m = _load_trimesh(mesh)
    verts = np.asarray(m.vertices, dtype=np.float32)
    faces = np.asarray(m.faces, dtype=np.int32).reshape(-1)

    lo = verts.min(0)
    hi = verts.max(0)
    diag = float(np.linalg.norm(hi - lo))
    pad = diag * padding
    lower = lo - pad
    upper = hi + pad

    wp_mesh = wp.Mesh(
        points=wp.array(verts, dtype=wp.vec3, device=device),
        indices=wp.array(faces, dtype=wp.int32, device=device),
    )
    sdf = wp.zeros(resolution ** 3, dtype=wp.float32, device=device)
    wp.launch(
        _sdf_kernel,
        dim=resolution ** 3,
        inputs=[wp_mesh.id, wp.vec3(*lower), wp.vec3(*upper), resolution, diag + 2.0 * pad, sdf],
        device=device,
    )
    wp.synchronize_device(device)
    return sdf.numpy().reshape(resolution, resolution, resolution), lower, upper


try:
    import warp as wp

    @wp.kernel
    def _sdf_kernel(
        mesh: wp.uint64,
        lower: wp.vec3,
        upper: wp.vec3,
        res: int,
        max_dist: float,
        out: wp.array(dtype=wp.float32),
    ):
        tid = wp.tid()
        # unravel flat index -> (i,j,k)
        i = tid // (res * res)
        j = (tid // res) % res
        k = tid % res
        span = upper - lower
        p = wp.vec3(
            lower[0] + span[0] * (float(i) + 0.5) / float(res),
            lower[1] + span[1] * (float(j) + 0.5) / float(res),
            lower[2] + span[2] * (float(k) + 0.5) / float(res),
        )
        q = wp.mesh_query_point_sign_normal(mesh, p, max_dist)
        if q.result:
            cp = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
            out[tid] = q.sign * wp.length(p - cp)
        else:
            out[tid] = max_dist
except Exception:  # warp unavailable (non-GPU host) — kernel defined lazily
    pass
