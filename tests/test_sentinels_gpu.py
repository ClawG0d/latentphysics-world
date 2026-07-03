"""GPU tests for the physics sentinels (anti-exploit guardrails).

The sentinels are advertised as shipped (README R1); these tests turn that
claim into a committed gate: injected violations must be flagged in exactly
the offending world and nowhere else.
"""

import os

import pytest

lpw = pytest.importorskip("latentphysics")
pytest.importorskip("mujoco_warp")
torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

from latentphysics.envs.sentinels import PhysicsSentinel  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(ROOT, "examples", "scenes", "falling_box.xml")
N = 8


@pytest.fixture()
def scene():
    return lpw.load_scene(SCENE, lpw.Config(n_worlds=N))


def test_velocity_sentinel_flags_only_offending_world(scene):
    scene.step(5)
    s = PhysicsSentinel(scene)
    report = s.check()
    assert not report["any"].any(), "a plainly falling box must not trip sentinels"
    scene.qvel()[3, :] = 200.0  # inject a velocity explosion into world 3 only
    report = s.check()
    assert report["velocity"][3]
    assert report["velocity"].sum().item() == 1, "only world 3 may be flagged"
    assert report["any"][3] and report["any"].sum().item() == 1
    assert s.violation_counts[3].item() == 1


def test_nonfinite_sentinel_flags_nan_state(scene):
    scene.step(5)
    s = PhysicsSentinel(scene)
    scene.qpos()[1, 0] = float("nan")
    report = s.check()
    assert report["nonfinite"][1]
    assert report["nonfinite"].sum().item() == 1, "only world 1 may be flagged"
    assert report["any"][1]
