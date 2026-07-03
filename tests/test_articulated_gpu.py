"""GPU tests for articulated furniture (R4): hinged doors + sliding drawers.

Verifies the joints are drivable on the batched engine, joint limits hold,
friction/damping park the parts when released, and the default room stays
articulation-free (reproducibility guard for existing seeds).
"""

import numpy as np
import pytest

lpw = pytest.importorskip("latentphysics")
pytest.importorskip("mujoco_warp")
torch = pytest.importorskip("torch")
mujoco = pytest.importorskip("mujoco")

if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

from latentphysics.assets.scene_gen import RoomSpec, generate_room  # noqa: E402

N = 8


def _articulated_joints(mjm):
    out = []
    for j in range(mjm.njnt):
        name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if "_door" in name or "_drawer" in name:
            out.append((name, int(mjm.jnt_qposadr[j]), int(mjm.jnt_dofadr[j]),
                        tuple(mjm.jnt_range[j])))
    return out


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("artic") / "room.xml")
    generate_room(RoomSpec(seed=3, n_articulated=2, n_furniture=40, n_clutter=4), path)
    return lpw.load_scene(path, lpw.Config(n_worlds=N))


@pytest.fixture(scope="module")
def scene4(tmp_path_factory):
    # all four archetypes: hinged door, drawers, lid chest, sliding door
    path = str(tmp_path_factory.mktemp("artic4") / "room.xml")
    generate_room(RoomSpec(seed=3, size=(7.5, 6.0), table_pos=(-1.6, 0.0),
                           n_articulated=4, n_furniture=40, n_clutter=3), path)
    return lpw.load_scene(path, lpw.Config(n_worlds=N, njmax=2048))


def _joint(mjm, suffix):
    for j in range(mjm.njnt):
        name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if suffix in name:
            return name, int(mjm.jnt_qposadr[j]), int(mjm.jnt_dofadr[j]), tuple(mjm.jnt_range[j])
    return None


def test_default_room_has_no_articulation(tmp_path):
    path = str(tmp_path / "room0.xml")
    generate_room(RoomSpec(seed=0), path)
    with open(path) as f:
        xml = f.read()
    assert "_door" not in xml and "_drawer" not in xml


def test_articulated_joints_exist(scene):
    joints = _articulated_joints(scene.mjm)
    assert any("_door" in n for n, *_ in joints)
    assert any("_drawer" in n for n, *_ in joints)


def test_joints_drivable_and_limited(scene):
    joints = _articulated_joints(scene.mjm)
    qpos, qvel = scene.qpos(), scene.qvel()
    scene.reset()

    # drive every articulated joint open (kinematic push, physics resolves)
    for _ in range(80):
        for name, _, dof, _ in joints:
            qvel[:, dof] = 1.5 if "_door" in name else 0.4
        scene.step()
    for name, qadr, _, (lo, hi) in joints:
        q = qpos[:, qadr]
        assert torch.isfinite(q).all(), f"{name} went non-finite"
        assert (q > lo + 0.05).all(), f"{name} did not open (q={q[0].item():.3f})"
        assert (q < hi + 1e-2).all(), f"{name} blew past its limit"

    # park test: once stopped, friction/damping must hold the parts in place
    # (no creep under gravity or solver noise) — coasting after a hard shove
    # is legitimate physics and NOT what this asserts
    for _, _, dof, _ in joints:
        qvel[:, dof] = 0.0
    held = {n: qpos[:, qadr].clone() for n, qadr, *_ in joints}
    scene.step(120)
    for name, qadr, _, (lo, hi) in joints:
        q = qpos[:, qadr]
        assert torch.isfinite(q).all()
        assert (q >= lo - 1e-3).all() and (q <= hi + 1e-2).all()
        drift = (q - held[name]).abs().max().item()
        assert drift < 0.05, f"{name} crept {drift:.3f} after being parked"


def test_room_state_stays_finite(scene):
    scene.reset()
    scene.step(200)
    assert torch.isfinite(scene.qpos()).all() and torch.isfinite(scene.qvel()).all()


def test_all_four_archetypes_present(scene4):
    mjm = scene4.mjm
    for suffix in ("_door", "_drawer", "_lid", "_sdoor"):
        assert _joint(mjm, suffix) is not None, f"missing archetype joint {suffix}"


def test_sliding_door_drivable_and_parks(scene4):
    """Sliding door: gravity-neutral, so it opens under a push, respects its
    travel limit, and stays put when released."""
    name, qadr, dof, (lo, hi) = _joint(scene4.mjm, "_sdoor")
    scene4.reset()
    qpos, qvel = scene4.qpos(), scene4.qvel()
    for _ in range(120):
        qvel[:, dof] = 0.35
        scene4.step()
    q = qpos[:, qadr]
    assert torch.isfinite(q).all()
    assert (q > lo + 0.05).all(), f"{name} did not slide open (q={q[0].item():.3f})"
    assert (q < hi + 1e-2).all(), f"{name} exceeded travel limit"
    qvel[:, dof] = 0.0
    held = q.clone()
    scene4.step(120)
    assert (qpos[:, qadr] - held).abs().max().item() < 0.05, "sliding door crept"


def test_lid_opens_and_falls_shut(scene4):
    """Lid chest: gravity-loaded, so it opens under a push but swings back
    toward closed when released (no stay) — the physically correct behavior."""
    name, qadr, dof, (lo, hi) = _joint(scene4.mjm, "_lid")
    scene4.reset()
    qpos, qvel = scene4.qpos(), scene4.qvel()
    for _ in range(70):
        qvel[:, dof] = 1.2
        scene4.step()
    opened = qpos[:, qadr].clone()
    assert torch.isfinite(opened).all()
    assert (opened > lo + 0.2).all(), f"{name} did not lift (q={opened[0].item():.3f})"
    assert (opened < hi + 1e-2).all(), f"{name} exceeded its limit"
    # release: gravity pulls it back down toward closed
    qvel[:, dof] = 0.0
    scene4.step(160)
    closed = qpos[:, qadr]
    assert torch.isfinite(closed).all()
    assert (closed < opened - 0.1).all(), "released lid did not fall back toward closed"
