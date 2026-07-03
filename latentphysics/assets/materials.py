"""Procedural diffuse materials for the procedural worlds (R4 visual fidelity).

MuJoCo's renderer is diffuse-only (not PBR), and flat rgba furniture reads as
untextured blocks. This generates small, tileable GRAYSCALE grain textures
(wood, fabric, plaster) with numpy — self-authored, so there is zero external
asset or licensing dependency — and exposes them as MuJoCo materials.

The textures are near-white (mean ~0.9) so a geom's own ``rgba`` still tints
them: a geom keeps its palette color and gains grain detail on top (verified:
geom rgba modulates the material texture). Materials are visual only — physics,
collision, and mass are untouched.
"""

from __future__ import annotations

import os

import numpy as np

__all__ = ["ensure_textures", "material_assets", "TEX_DIR"]

TEX_DIR = os.path.expanduser("~/lpw/assets/textures")
_RES = 256


def _to_png(arr):
    """float [0,1] HxW -> uint8 HxWx3."""
    a = np.clip(arr, 0.0, 1.0)
    return (np.repeat(a[:, :, None], 3, axis=2) * 255).astype(np.uint8)


def _wood(res=_RES):
    """Vertical grain: seamless low-freq streaks along y + fine noise."""
    rng = np.random.default_rng(7)
    y = np.linspace(0, 2 * np.pi, res, endpoint=False)
    grain = 0.5 + 0.5 * np.sin(6 * y + 1.2 * np.sin(3 * y))     # tiles along y
    streaks = 0.06 * np.sin(np.linspace(0, 2 * np.pi, res, endpoint=False) * 11)
    base = 0.86 + 0.12 * grain[:, None] + streaks[None, :]
    base = base + 0.03 * rng.standard_normal((res, res))
    return _to_png(base)


def _fabric(res=_RES):
    """Woven weave: two perpendicular gratings (seamless) + speckle."""
    rng = np.random.default_rng(11)
    u = np.linspace(0, 2 * np.pi, res, endpoint=False)
    warp = 0.5 + 0.5 * np.sin(24 * u)[:, None]
    weft = 0.5 + 0.5 * np.sin(24 * u)[None, :]
    weave = np.maximum(warp, weft)
    base = 0.82 + 0.14 * weave + 0.025 * rng.standard_normal((res, res))
    return _to_png(base)


def _plaster(res=_RES):
    """Subtle wall/floor speckle — near-flat, faint mottling."""
    rng = np.random.default_rng(3)
    base = 0.9 + 0.04 * rng.standard_normal((res, res))
    return _to_png(base)


_GENERATORS = {"grain_wood": _wood, "grain_fabric": _fabric, "grain_plaster": _plaster}


def ensure_textures(cache_dir: str = TEX_DIR) -> dict:
    """Generate the grain PNGs into ``cache_dir`` if missing; return name->path."""
    import imageio.v2 as imageio

    os.makedirs(cache_dir, exist_ok=True)
    paths = {}
    for name, gen in _GENERATORS.items():
        p = os.path.join(cache_dir, f"{name}.png")
        if not os.path.exists(p):
            imageio.imwrite(p, gen())
        paths[name] = p
    return paths


def material_assets(cache_dir: str = TEX_DIR, skybox: bool = True) -> str:
    """Return a MuJoCo ``<asset>`` inner block: a gradient skybox plus wood /
    fabric / plaster diffuse materials backed by the generated grain textures.
    Attach with ``material="mat_wood"`` etc.; the geom's rgba still tints it."""
    tex = ensure_textures(cache_dir)
    parts = []
    if skybox:
        parts.append('<texture type="skybox" builtin="gradient" '
                     'rgb1="0.46 0.54 0.63" rgb2="0.11 0.13 0.17" width="256" height="256"/>')
    for name, mat, rep in (("grain_wood", "mat_wood", "0.6 0.6"),
                           ("grain_fabric", "mat_fabric", "0.35 0.35"),
                           ("grain_plaster", "mat_plaster", "1 1")):
        parts.append(f'<texture name="{name}" type="2d" file="{tex[name]}"/>')
        parts.append(f'<material name="{mat}" texture="{name}" texuniform="true" '
                     f'texrepeat="{rep}" reflectance="0.05"/>')
    return "".join(parts)
