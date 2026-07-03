"""CPU tests: procedural materials are present and VISUAL-ONLY (no physics change)."""

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("imageio")
import mujoco  # noqa: E402

from latentphysics.assets.materials import ensure_textures, material_assets  # noqa: E402
from latentphysics.assets.scene_gen import RoomSpec, generate_room  # noqa: E402


def test_textures_generate_and_are_neutral(tmp_path):
    import imageio.v2 as imageio
    paths = ensure_textures(str(tmp_path / "tex"))
    assert set(paths) >= {"grain_wood", "grain_fabric", "grain_plaster"}
    for p in paths.values():
        img = np.asarray(imageio.imread(p), dtype=float) / 255.0
        # near-white so a geom's rgba still tints the grain, with real variation
        assert img.mean() > 0.7, f"{p} too dark to tint"
        assert img.std() > 0.01, f"{p} has no grain"


def test_material_assets_block_wellformed(tmp_path):
    xml = f"""<mujoco><asset>{material_assets(str(tmp_path / 't'))}</asset>
      <worldbody><geom type="box" size="1 1 1" material="mat_wood"/></worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    assert m.nmat >= 3 and m.ntex >= 4          # 3 grain materials + skybox + 3 tex


def test_room_has_materials_but_same_collision(tmp_path):
    p = generate_room(RoomSpec(seed=5, n_furniture=30, n_clutter=6), str(tmp_path / "r.xml"))
    m = mujoco.MjModel.from_xml_path(p)
    # materials present on furniture...
    assert m.nmat >= 3
    assert (m.geom_matid >= 0).sum() > 10, "furniture geoms should carry materials"
    # ...but collision is untouched: every furniture/clutter geom still has a
    # nonzero collision mask (materials never change contype/conaffinity)
    named = [(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "", g) for g in range(m.ngeom)]
    fgeoms = [g for n, g in named if n.startswith("f") or n.startswith("clutter") or n == "table"]
    assert fgeoms, "expected furniture geoms"
    for g in fgeoms:
        assert (m.geom_contype[g] | m.geom_conaffinity[g]) != 0, "material broke a collision mask"


def test_room_physics_still_finite_with_materials(tmp_path):
    # a material room must simulate identically-well on the CPU reference
    p = generate_room(RoomSpec(seed=5, n_furniture=20, n_clutter=8), str(tmp_path / "r.xml"))
    m = mujoco.MjModel.from_xml_path(p)
    d = mujoco.MjData(m)
    for _ in range(300):
        mujoco.mj_step(m, d)
    assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
