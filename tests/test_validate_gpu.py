"""GPU test for settle() — the stability bake relaxes spawn transients."""

import pytest

lpw = pytest.importorskip("latentphysics")
pytest.importorskip("mujoco_warp")
torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

from latentphysics.assets.validate import settle  # noqa: E402

SCENE = """<mujoco>
  <option timestep="0.004"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="drop" pos="0 0 0.6"><freejoint/>
      <geom type="box" size="0.08 0.08 0.08" density="500"/></body>
  </worldbody>
</mujoco>"""


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    p = str(tmp_path_factory.mktemp("settle") / "drop.xml")
    with open(p, "w") as f:
        f.write(SCENE)
    return lpw.load_scene(p, lpw.Config(n_worlds=8))


def test_settle_converges(scene):
    scene.reset()
    info = settle(scene, max_steps=800, vel_tol=0.05)
    assert info["converged"], f"did not settle: {info}"
    assert info["residual_vel"] < 0.05
    # box fell from 0.6 m and rests on the floor at half-extent 0.08
    z = scene.qpos()[:, 2]
    assert (z > 0.0).all() and (z < 0.12).all(), f"resting z={z[0].item():.3f}"


def test_settle_reports_nonfinite(scene, tmp_path):
    # a wildly overlapping spawn can explode; settle must report, not hang
    bad = """<mujoco><option timestep="0.05"/><worldbody>
      <geom type="plane" size="5 5 0.1"/>
      <body pos="0 0 0.02"><freejoint/><geom type="box" size="0.3 0.3 0.3" density="9000"/></body>
      <body pos="0.05 0 0.02"><freejoint/><geom type="box" size="0.3 0.3 0.3" density="9000"/></body>
    </worldbody></mujoco>"""
    p = str(tmp_path / "bad.xml")
    with open(p, "w") as f:
        f.write(bad)
    s = lpw.load_scene(p, lpw.Config(n_worlds=4))
    info = settle(s, max_steps=120, vel_tol=0.05)
    assert "residual_vel" in info and isinstance(info["converged"], bool)
