"""Procedural lofted meshes + textures for the office-chair asset (visuals).

Pure numpy: parts are superellipse cross-sections swept along an axis (a
loft) or tubes swept along a 3D path (piping, frame, lumbar bar), written
as OBJ with UVs. Deterministic — same code, same bytes. Assets are
generated on demand into ~/lpw/assets/chair_meshes_v2 (never committed,
same policy as fetched assets). Collision stays primitive — these meshes
carry no mass and no contype.
"""

from __future__ import annotations

import os

import numpy as np

MESH_DIR = os.path.expanduser("~/lpw/assets/chair_meshes_v2")


# ------------------------------------------------------------ mesh plumbing
def _superellipse(K, a, b, p):
    """(K,2) rounded-rectangle ring, CCW, p>=2 (2=ellipse, high=boxy)."""
    th = np.linspace(0, 2 * np.pi, K, endpoint=False)
    c, s = np.cos(th), np.sin(th)
    return np.stack([a * np.sign(c) * np.abs(c) ** (2 / p),
                     b * np.sign(s) * np.abs(s) ** (2 / p)], axis=1)


def _loft(rings, uv=(1.0, 1.0), closed=False):
    """Loft through same-K rings -> (V, F, VT). Open lofts get centroid-fan
    caps; ``closed=True`` wraps the last ring to the first (a torus-like
    sweep, no caps). ``uv`` is the texture repeat count around/along the
    sweep (MuJoCo ignores material texrepeat for meshes, so tiling lives
    in the texcoords)."""
    rings = [np.asarray(r, dtype=float) for r in rings]
    K = rings[0].shape[0]
    n = len(rings)
    V = np.concatenate(rings, axis=0)
    VT = np.stack([np.tile(np.linspace(0, uv[0], K, endpoint=False), n),
                   np.repeat(np.linspace(0, uv[1], n), K)], axis=1)
    F = []
    strips = n if closed else n - 1
    for i in range(strips):
        i1 = (i + 1) % n
        for k in range(K):
            k1 = (k + 1) % K
            a, b = i * K + k, i * K + k1
            c, d = i1 * K + k1, i1 * K + k
            F.append((a, b, c))
            F.append((a, c, d))
    if not closed:
        c0 = len(V)
        V = np.concatenate([V, rings[0].mean(0, keepdims=True),
                            rings[-1].mean(0, keepdims=True)], axis=0)
        VT = np.concatenate([VT, [[0.5, 0.0], [0.5, 1.0]]], axis=0)
        for k in range(K):
            k1 = (k + 1) % K
            F.append((c0, k1, k))                                  # bottom cap
            F.append((c0 + 1, (n - 1) * K + k, (n - 1) * K + k1))  # top cap
    return V, np.asarray(F, dtype=int), VT


def _sweep_tube(path, r, K=12, closed=True, uv=(1.0, 1.0)):
    """Circular tube swept along a 3D path. Ring normals point from the
    path centroid outward (stable frames on convex-ish loops — no twist)."""
    P = np.asarray(path, dtype=float)
    n = len(P)
    nxt, prv = np.roll(P, -1, 0), np.roll(P, 1, 0)
    tan = nxt - prv
    if not closed:
        tan[0], tan[-1] = P[1] - P[0], P[-1] - P[-2]
    tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-12
    centroid = P.mean(0)
    th = np.linspace(0, 2 * np.pi, K, endpoint=False)
    rings = []
    for i in range(n):
        radial = P[i] - centroid
        radial -= tan[i] * radial.dot(tan[i])
        nrm = np.linalg.norm(radial)
        if nrm < 1e-9:
            radial = np.array([1.0, 0, 0]) - tan[i] * tan[i][0]
            nrm = np.linalg.norm(radial)
        nvec = radial / nrm
        bvec = np.cross(tan[i], nvec)
        rings.append(P[i] + r * (np.outer(np.cos(th), nvec)
                                 + np.outer(np.sin(th), bvec)))
    return _loft(rings, uv=uv, closed=closed)


def _smooth_loop(P, passes=2):
    """Circular moving average — rounds polyline corners."""
    P = np.asarray(P, dtype=float)
    for _ in range(passes):
        P = (np.roll(P, 1, 0) + P + np.roll(P, -1, 0)) / 3.0
    return P


def _write_obj(path, V, F, VT):
    with open(path, "w") as f:
        for v in V:
            f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
        for t in VT:
            f.write(f"vt {t[0]:.4f} {t[1]:.4f}\n")
        for a, b, c in F:
            f.write(f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}\n")


def _shrink_caps(build_ring, ts, shrink=(1.0, 0.82, 0.5, 0.18), pad=0.014):
    """Wrap a ring builder with rounded end closures: extra rings at each
    end, scaled toward the centroid and pushed outward along the sweep."""
    t0, t1 = ts[0], ts[-1]
    rings = []
    for s, dz in zip(shrink[:0:-1], np.linspace(pad, 0, len(shrink) - 1, endpoint=False)):
        rings.append(build_ring(t0, scale=s, axis_pad=-dz))
    rings += [build_ring(t) for t in ts]
    for s, dz in zip(shrink[1:], np.linspace(0, pad, len(shrink) - 1, endpoint=False) + pad / (len(shrink) - 1)):
        rings.append(build_ring(t1, scale=s, axis_pad=dz))
    return rings


# ---------------------------------------------------- backrest silhouette
# shared by the membrane, the frame tube and the collision-slab centers
def _back_w2(t):
    return 0.186 + 0.040 * np.sin(np.pi * (t * 0.92 + 0.04))


def _back_x_off(t):
    return (0.050 * np.sin(np.pi * min(t * 1.8, 1.0))
            - 0.090 * max(t - 0.45, 0.0) ** 1.5)


# ------------------------------------------------------------ chair pieces
def back_membrane(back_x, z0, z1, K=40):
    """Suspension-mesh backrest membrane: the sculpted silhouette of the
    upholstered version, thinned to a stretched-fabric shell."""
    ts = np.linspace(0, 1, 44)

    def ring(t, scale=1.0, axis_pad=0.0):
        z = z0 + 0.010 + t * (z1 - z0 - 0.020) + axis_pad
        w2 = _back_w2(t) * scale
        r = _superellipse(K, 0.011 * scale, w2, 4.5)
        curl = 0.014 * (np.abs(r[:, 1]) / max(w2, 1e-6)) ** 3.5
        x = back_x + _back_x_off(t) + r[:, 0] + curl
        return np.stack([x, r[:, 1], np.full(K, z)], axis=1)

    return _loft(_shrink_caps(ring, ts, pad=0.006), uv=(16.0, 9.0))


def _back_outline(back_x, z0, z1, inset=0.0, n_side=46, n_arc=16):
    """Closed loop around the backrest silhouette (frame / piping path)."""
    pts = []
    ts = np.linspace(0, 1, n_side)
    zline = z0 + 0.012 + ts * (z1 - z0 - 0.024)
    for t, z in zip(ts, zline):                       # right edge, up
        pts.append((back_x + _back_x_off(float(t)) + 0.004,
                    _back_w2(float(t)) - inset, z))
    for phi in np.linspace(0, np.pi, n_arc)[1:-1]:    # across the top
        w = (_back_w2(1.0) - inset) * np.cos(phi)
        pts.append((back_x + _back_x_off(1.0) + 0.004, w,
                    zline[-1] + 0.012 * np.sin(phi)))
    for t, z in zip(ts[::-1], zline[::-1]):           # left edge, down
        pts.append((back_x + _back_x_off(float(t)) + 0.004,
                    -(_back_w2(float(t)) - inset), z))
    for phi in np.linspace(np.pi, 2 * np.pi, n_arc)[1:-1]:  # across the bottom
        w = (_back_w2(0.0) - inset) * np.cos(phi)
        pts.append((back_x + _back_x_off(0.0) + 0.004, w,
                    zline[0] - 0.010 * np.abs(np.sin(phi))))
    return _smooth_loop(pts, passes=3)


def back_frame(back_x, z0, z1):
    """Plastic perimeter frame the mesh membrane is stretched onto."""
    return _sweep_tube(_back_outline(back_x, z0, z1, inset=-0.002),
                       r=0.013, K=14, closed=True)


def lumbar_bar(back_x):
    """Adjustable lumbar support bar bowed behind the membrane."""
    ys = np.linspace(-0.155, 0.155, 24)
    z = 0.625
    t = (z - 0.497) / 0.538
    x0 = back_x + _back_x_off(t) - 0.020
    path = [(x0 - 0.012 * (1 - (y / 0.155) ** 2), y, z) for y in ys]
    return _sweep_tube(path, r=0.0095, K=12, closed=False)


def seat_cushion(K=40):
    """Plump seat: crowned top, side bolsters, waterfall front edge."""
    ts = np.linspace(0, 1, 40)
    x0, x1 = -0.223, 0.232

    def ring(t, scale=1.0, axis_pad=0.0):
        x = x0 + t * (x1 - x0) + axis_pad + 0.012
        w2 = 0.240 * (1 - 0.10 * (1 - t) ** 2 - 0.06 * t ** 4) * scale
        # cross-section rounds toward the front (waterfall)
        p = 4.6 - 2.2 * max(t - 0.72, 0.0) / 0.28
        h2 = (0.042 - 0.004 * max(t - 0.8, 0.0) / 0.2) * scale
        r = _superellipse(K, w2, h2, max(p, 2.2))
        y, z = r[:, 0], r[:, 1]
        # crown + side bolsters on the top half only
        top = z > 0
        z = z + top * (0.004 * np.cos(y / 0.24 * np.pi / 2)
                       + 0.008 * np.exp(-((np.abs(y) - 0.205) / 0.035) ** 2)
                       * np.sin(np.pi * min(max(t, 0.08), 0.92)))
        return np.stack([np.full(K, x), y, 0.442 + z], axis=1)

    return _loft(_shrink_caps(ring, ts, pad=0.010), uv=(9.0, 7.0))


def seat_piping(K=10):
    """Piped seam cord around the seat cushion's top shoulder."""
    th = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    c, s = np.cos(th), np.sin(th)
    p = 3.6
    x = 0.012 + 0.212 * np.sign(c) * np.abs(c) ** (2 / p)
    y = 0.226 * np.sign(s) * np.abs(s) ** (2 / p)
    z = (0.442 + 0.0315
         + 0.006 * np.exp(-((np.abs(y) - 0.205) / 0.035) ** 2)   # bolsters
         - 0.010 * np.clip((x - 0.14) / 0.09, 0, 1))             # waterfall
    path = _smooth_loop(np.stack([x, y, z], axis=1), passes=2)
    return _sweep_tube(path, r=0.0045, K=K, closed=True)


def headrest_pillow(K=36):
    """Wide flat pillow, rounded everywhere."""
    ts = np.linspace(0, 1, 30)
    W = 0.168

    def ring(t, scale=1.0, axis_pad=0.0):
        y = (t * 2 - 1) * W + axis_pad
        env = (1 - min(abs(t * 2 - 1), 1.0) ** 3.2) ** (1 / 3.2)
        r = _superellipse(K, 0.042 * env * scale + 1e-4, 0.090 * env * scale + 1e-4, 2.6)
        return np.stack([r[:, 0], np.full(K, y), r[:, 1]], axis=1)

    return _loft(_shrink_caps(ring, ts, shrink=(1.0, 0.7, 0.3), pad=0.004), uv=(4.0, 5.0))


def headrest_piping(K=10):
    """Piped seam around the pillow's silhouette (pillow local frame)."""
    W = 0.168
    pts = []
    for phi in np.linspace(0, np.pi, 40)[1:-1]:           # top edge
        y = W * 0.985 * np.cos(phi)
        env = (1 - min(abs(y / W), 1.0) ** 3.2) ** (1 / 3.2)
        pts.append((0.0, y, 0.090 * env))
    for phi in np.linspace(np.pi, 2 * np.pi, 40)[1:-1]:   # bottom edge
        y = W * 0.985 * np.cos(phi)
        env = (1 - min(abs(y / W), 1.0) ** 3.2) ** (1 / 3.2)
        pts.append((0.0, y, -0.090 * env))
    return _sweep_tube(_smooth_loop(pts, passes=3), r=0.0035, K=K, closed=True)


def arm_pad(K=28):
    """Soft rounded armrest pad, lofted along its length."""
    ts = np.linspace(0, 1, 26)
    L = 0.150

    def ring(t, scale=1.0, axis_pad=0.0):
        x = (t * 2 - 1) * L + axis_pad
        env = (1 - min(abs(t * 2 - 1), 1.0) ** 4.0) ** (1 / 4.0)
        r = _superellipse(K, 0.046 * env * scale + 1e-4, 0.017 * env * scale + 1e-4, 3.0)
        return np.stack([np.full(K, x), r[:, 0], r[:, 1]], axis=1)

    return _loft(_shrink_caps(ring, ts, shrink=(1.0, 0.7, 0.3), pad=0.004))


def star_arm(reach, K=24):
    """One tapered base arm along +x (positioned/rotated by the geom)."""
    ts = np.linspace(0, 1, 16)
    L2 = reach / 2

    def ring(t, scale=1.0, axis_pad=0.0):
        x = (t * 2 - 1) * L2 + axis_pad
        w = (0.034 - 0.010 * t) * scale
        h = (0.021 - 0.007 * t) * scale
        arch = 0.004 * np.sin(np.pi * t)
        r = _superellipse(K, w, h, 3.2)
        return np.stack([np.full(K, x), r[:, 0], r[:, 1] + arch], axis=1)

    return _loft(_shrink_caps(ring, ts, shrink=(1.0, 0.75, 0.35), pad=0.008))


def lift_boot(K=28):
    """Accordion bellows around the gas lift."""
    zs = np.linspace(0.125, 0.255, 40)
    rings = []
    for i, z in enumerate(zs):
        t = i / (len(zs) - 1)
        r = 0.0295 + 0.0045 * (0.5 + 0.5 * np.cos(2 * np.pi * 6.0 * t))
        ring = _superellipse(K, r, r, 2.0)
        rings.append(np.stack([ring[:, 0], ring[:, 1], np.full(K, z)], axis=1))
    return _loft(rings)


def wheel_disc(R=0.031, W=0.0080, K=28):
    """Caster wheel: flat disc with rounded rim (lofted along the axle)."""
    ts = np.linspace(0, 1, 14)

    def ring(t, scale=1.0, axis_pad=0.0):
        y = (t * 2 - 1) * W + axis_pad
        env = (1 - min(abs(t * 2 - 1), 1.0) ** 6.0) ** (1 / 6.0)
        r = R * env * scale + 2e-4
        ring = _superellipse(K, r, r, 2.0)
        return np.stack([ring[:, 0], np.full(K, y), ring[:, 1]], axis=1)

    return _loft(_shrink_caps(ring, ts, shrink=(1.0, 0.6), pad=0.001))


_BUILDERS = {
    "back_membrane": lambda: back_membrane(-0.215, 0.487, 1.045),
    "back_frame": lambda: back_frame(-0.215, 0.487, 1.045),
    "lumbar_bar": lambda: lumbar_bar(-0.215),
    "seat_cushion": seat_cushion,
    "seat_piping": seat_piping,
    "headrest_pillow": headrest_pillow,
    "headrest_piping": headrest_piping,
    "arm_pad": arm_pad,
    "star_arm": lambda: star_arm(0.215),
    "lift_boot": lift_boot,
    "wheel_disc": wheel_disc,
}


# ------------------------------------------------------- chair textures
def _to_png(arr):
    a = np.clip(arr, 0.0, 1.0)
    return (np.repeat(a[:, :, None], 3, axis=2) * 255).astype(np.uint8)


def _stitch_fabric(res=256):
    """Woven grain + one dashed stitch channel per tile: quilted seams."""
    rng = np.random.default_rng(11)
    u = np.linspace(0, 2 * np.pi, res, endpoint=False)
    warp = 0.5 + 0.5 * np.sin(24 * u)[:, None]
    weft = 0.5 + 0.5 * np.sin(24 * u)[None, :]
    base = 0.82 + 0.14 * np.maximum(warp, weft)
    base = base + 0.025 * rng.standard_normal((res, res))
    col = res // 2
    base[:, col - 2:col + 3] *= 0.90                        # seam groove
    dash = ((np.arange(res) // 7) % 2 == 0).astype(float)   # stitch dashes
    base[:, col - 1:col + 2] *= (1.0 - 0.22 * dash)[:, None]
    return _to_png(base)


def _mesh_weave(res=256):
    """Open suspension-mesh weave: bright threads, dark pores (tinted by
    the geom rgba; the pores plus alpha give the see-through read)."""
    u = np.linspace(0, 2 * np.pi, res, endpoint=False)
    wa = 0.5 + 0.5 * np.sin(14 * u)[:, None]
    we = 0.5 + 0.5 * np.sin(14 * u)[None, :]
    thread = np.maximum(wa, we) ** 2.5
    return _to_png(0.42 + 0.58 * thread)


_TEXTURES = {"chair_stitch": _stitch_fabric, "chair_weave": _mesh_weave}


def ensure_meshes(cache_dir: str = MESH_DIR) -> dict:
    """Generate any missing OBJ/PNG into ``cache_dir``; return name -> path."""
    import imageio.v2 as imageio

    os.makedirs(cache_dir, exist_ok=True)
    paths = {}
    for name, build in _BUILDERS.items():
        p = os.path.join(cache_dir, f"{name}.obj")
        if not os.path.exists(p):
            V, F, VT = build()
            _write_obj(p, V, F, VT)
        paths[name] = p
    for name, gen in _TEXTURES.items():
        p = os.path.join(cache_dir, f"{name}.png")
        if not os.path.exists(p):
            imageio.imwrite(p, gen())
        paths[name] = p
    return paths


def mesh_assets(cache_dir: str = MESH_DIR) -> str:
    """MJCF <asset> inner block: chair meshes + chair-specific materials."""
    paths = ensure_meshes(cache_dir)
    parts = [f'<mesh name="{n}" file="{paths[n]}" smoothnormal="true"/>'
             for n in _BUILDERS]
    parts.append(f'<texture name="chair_stitch" type="2d" file="{paths["chair_stitch"]}"/>')
    parts.append(f'<texture name="chair_weave" type="2d" file="{paths["chair_weave"]}"/>')
    parts.append('<material name="mat_fabric_stitch" texture="chair_stitch" '
                 'reflectance="0.05"/>')
    parts.append('<material name="mat_mesh_weave" texture="chair_weave" '
                 'specular="0.15" shininess="0.25"/>')
    return "".join(parts)
