"""GPU backend tests: budget autoscaling + world-state snapshot/branching.

Skipped automatically on hosts without the GPU engine (warp/mujoco_warp/CUDA).
Run on Linux/WSL2 + NVIDIA GPU:  pytest tests/test_backend_gpu.py -v
"""

import io
import contextlib
import math

import pytest

lpw = pytest.importorskip("latentphysics")
pytest.importorskip("mujoco_warp")
torch = pytest.importorskip("torch")

if not torch.cuda.is_available():  # engine importable but no device
    pytest.skip("CUDA device required", allow_module_level=True)


@pytest.fixture(scope="module")
def torus_scene_path(tmp_path_factory):
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("coacd")
    from latentphysics.assets import mesh_to_mjcf

    out = tmp_path_factory.mktemp("assets")
    mesh = trimesh.creation.torus(major_radius=0.3, minor_radius=0.1)
    return mesh_to_mjcf(mesh, str(out), name="torus", pos=(0, 0, 0.6), threshold=0.05)


@pytest.fixture(scope="module")
def scene(torus_scene_path):
    return lpw.load_scene(torus_scene_path, lpw.Config(n_worlds=8))


def test_auto_budgets_no_overflow(scene):
    """11-hull concave body must not overflow auto-scaled contact/constraint buffers."""
    assert scene.data.njmax >= 256 and scene.data.naconmax >= 8 * 96
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        scene.step(300)
    assert "overflow" not in buf.getvalue()
    z = scene.qpos()[:, 2]
    assert 0.0 < z[0].item() < 0.3  # settled on plane, no tunneling


def test_snapshot_restore_roundtrip(scene):
    scene.reset()
    scene.step(20)  # mid-fall: state evolves after this point
    snap = scene.snapshot()
    q0 = scene.qpos().clone()
    scene.step(100)
    assert not torch.allclose(q0, scene.qpos())
    scene.restore(snap)
    assert torch.allclose(q0, scene.qpos(), atol=1e-6)


def test_partial_restore_branches_worlds(scene):
    import numpy as np

    scene.reset()
    scene.step(20)
    snap = scene.snapshot()
    q0 = scene.qpos().clone()
    scene.step(100)
    mask = np.zeros(scene.n_worlds, dtype=bool)
    mask[:4] = True
    scene.restore(snap, worlds=mask)
    q = scene.qpos()
    assert (q[:4] - q0[:4]).abs().max().item() < 1e-6
    assert (q[4:] - q0[4:]).abs().max().item() > 1e-4


@pytest.fixture(scope="module")
def pile_scene_path(tmp_path_factory):
    """Dense free-body pile: 6x8 ring tower of boxes + a heavy ball to topple
    it (the examples/collision_tower.py geometry, stripped of visuals). Once
    collapsed it needs ~1152 constraint rows per world — past the old flat
    1024 njmax cap whose overflow silently corrupted the solve to NaN."""
    layers, per_layer, radius = 6, 8, 0.24
    bx, by, bz = 0.085, 0.032, 0.045
    body = ['<geom name="floor" type="plane" size="4 4 0.1"/>']
    for lay in range(layers):
        z = bz * (2 * lay + 1)
        for i in range(per_layer):
            a = 2 * math.pi * (i + 0.5 * (lay % 2)) / per_layer
            body.append(
                f'<body pos="{radius * math.cos(a):.4f} {radius * math.sin(a):.4f} {z:.4f}" '
                f'euler="0 0 {math.degrees(a) + 90:.2f}"><freejoint/>'
                f'<geom type="box" size="{bx} {by} {bz}" mass="0.08"/></body>')
    body.append('<body pos="-1.6 0 0.35"><freejoint/>'
                '<geom type="sphere" size="0.09" mass="2.5"/></body>')
    xml = ('<mujoco model="dense_pile">'
           '<option timestep="0.004" iterations="10" ls_iterations="10"/>'
           '<worldbody>' + "".join(body) + '</worldbody></mujoco>')
    out = tmp_path_factory.mktemp("pile") / "pile.xml"
    out.write_text(xml)
    return str(out)


def test_auto_njmax_covers_dense_pile(pile_scene_path):
    """48 free boxes + ball must survive collapse on the AUTO budget alone."""
    scene = lpw.load_scene(pile_scene_path, lpw.Config(n_worlds=2))
    assert scene.data.njmax >= 1536  # scaled past the old flat 1024 cap
    qv = scene.qvel()
    qv[:, -6] = 5.5   # fling the ball at the tower, as in the demo
    qv[:, -4] = 1.2
    scene.forward()
    for _ in range(8):
        scene.step(70)  # 560 steps, chunked like a rollout loop
    assert torch.isfinite(scene.qpos()).all(), "solve corrupted to NaN"
    peak = int(scene._nefc_peak.numpy()[0])
    assert peak > 1024, "pile never exceeded the old cap — regression test is inert"
    assert peak <= scene.data.njmax


def test_njmax_overflow_raises_instead_of_nan(pile_scene_path):
    """An undersized explicit njmax must fail loudly at step time: the engine's
    own warning is a GPU-kernel printf, and the solve corrupts to NaN."""
    from latentphysics.backend import BudgetOverflow

    scene = lpw.load_scene(pile_scene_path, lpw.Config(n_worlds=2, njmax=256))
    with pytest.raises(BudgetOverflow, match="njmax"):
        for _ in range(20):
            scene.step(10)


def test_replay_determinism_within_atomic_noise(scene):
    """Restore + identical steps must reproduce trajectories to float-atomic noise.

    Known engine limitation: contacts accumulate via GPU float atomics in
    nondeterministic order (~1e-9 noise / 50 steps). Bit-exact replay needs a
    fork-level deterministic contact ordering patch (roadmap R4).
    """
    scene.reset()
    scene.step(20)
    snap = scene.snapshot()
    scene.step(50)
    qa = scene.qpos().clone()
    scene.restore(snap)
    scene.step(50)
    qb = scene.qpos().clone()
    assert (qa - qb).abs().max().item() < 1e-7
