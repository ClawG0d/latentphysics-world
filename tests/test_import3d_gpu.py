"""GPU tests for the GLB scene importer (R4): scene graph -> MJCF -> stable sim."""

import os

import numpy as np
import pytest

lpw = pytest.importorskip("latentphysics")
pytest.importorskip("mujoco_warp")
trimesh = pytest.importorskip("trimesh")
pytest.importorskip("coacd")
torch = pytest.importorskip("torch")
mujoco = pytest.importorskip("mujoco")

if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

from latentphysics.assets.import_3d import ImportSpec, import_glb  # noqa: E402

N = 8


def _make_glb(path):
    """Synthetic indoor corner: table (static), concave torus + cup (dynamic)."""
    s = trimesh.Scene()
    s.add_geometry(trimesh.creation.box(extents=(1.2, 0.8, 0.06)), node_name="table_top",
                   transform=trimesh.transformations.translation_matrix((0, 0, 0.5)))
    s.add_geometry(trimesh.creation.box(extents=(0.08, 0.08, 0.47)), node_name="table_leg",
                   transform=trimesh.transformations.translation_matrix((0.5, 0.3, 0.235)))
    s.add_geometry(trimesh.creation.annulus(r_min=0.06, r_max=0.12, height=0.05),
                   node_name="ring",
                   transform=trimesh.transformations.translation_matrix((0, 0, 0.9)))
    s.add_geometry(trimesh.creation.cylinder(radius=0.04, height=0.1), node_name="cup",
                   transform=trimesh.transformations.translation_matrix((0.3, 0.1, 0.8)))
    s.export(path)
    return path


@pytest.fixture(scope="module")
def scene_path(tmp_path_factory):
    d = tmp_path_factory.mktemp("glb")
    glb = _make_glb(str(d / "corner.glb"))
    # assets authored z-up in trimesh; real glTF assets use the default up="y"
    return import_glb(glb, str(d / "out"), name="corner",
                      spec=ImportSpec(up="z", dynamic=("ring", "cup")))


def test_import_emits_hulls_and_masks(scene_path):
    m = mujoco.MjModel.from_xml_path(scene_path)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]
    # concave ring must decompose into multiple hulls
    ring_hulls = sum(1 for g in range(m.ngeom)
                     if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g]) or "")
                     .startswith("ring_c"))
    assert ring_hulls >= 3, f"torus produced only {ring_hulls} hulls"
    # dynamic bodies are free, statics are welded
    for bname, free in (("ring", True), ("cup", True), ("table_top", False)):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bname)
        assert bid >= 0
        assert (m.body_dofnum[bid] == 6) == free


def test_up_axis_conversion(tmp_path):
    s = trimesh.Scene()
    s.add_geometry(trimesh.creation.box(extents=(0.2, 0.2, 0.2)), node_name="probe",
                   transform=trimesh.transformations.translation_matrix((0, 2.0, 0)))
    glb = str(tmp_path / "yup.glb")
    s.export(glb)
    path = import_glb(glb, str(tmp_path / "out"), name="yup",
                      spec=ImportSpec(up="y", add_floor=False))
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    gid = next(g for g in range(m.ngeom) if m.geom_contype[g] != 0)
    # +y in glTF becomes +z in MuJoCo
    assert abs(d.geom_xpos[gid][2] - 2.0) < 1e-3


def test_imported_scene_steps_stably(scene_path):
    scene = lpw.load_scene(scene_path, lpw.Config(n_worlds=N))
    scene.step(300)
    qpos = scene.qpos()
    assert torch.isfinite(qpos).all() and torch.isfinite(scene.qvel()).all()
    # both free bodies fell from their spawn height and came to rest above floor
    m = scene.mjm
    for bname in ("ring", "cup"):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bname)
        adr = int(m.jnt_qposadr[m.body_jntadr[bid]])
        z = qpos[:, adr + 2]
        assert (z > -0.01).all(), f"{bname} fell through the floor"
        assert (z < 0.85).all(), f"{bname} did not fall/settle"
