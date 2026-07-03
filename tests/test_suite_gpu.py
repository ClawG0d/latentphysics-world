"""GPU smoke tests for the manipulation benchmark suite (12 auto-verified tasks)."""

import os

import pytest

lpw = pytest.importorskip("latentphysics")
pytest.importorskip("mujoco_warp")
torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

MJCF = os.path.join(
    os.environ.get("LPW_MENAGERIE", os.path.expanduser("~/lpw/menagerie")),
    "franka_emika_panda", "mjx_single_cube.xml",
)
if not os.path.exists(MJCF):
    pytest.skip("mujoco_menagerie Franka scene not found", allow_module_level=True)

from latentphysics.envs import TaskConfig  # noqa: E402
from latentphysics.envs.suite import SUITE, make  # noqa: E402

N = 128


@pytest.fixture(scope="module")
def scene():
    return lpw.load_scene(MJCF, lpw.Config(n_worlds=N))


def test_suite_has_at_least_12_tasks():
    # the README advertises a 12-task suite; gate the count it claims
    assert len(SUITE) >= 12


@pytest.mark.parametrize("name", sorted(SUITE))
def test_task_mechanics(scene, name):
    env = make(name, scene, TaskConfig(episode_len=40, substeps=4, seed=1))
    obs = env.reset()
    assert obs.shape == (N, env.obs_dim) and obs.is_cuda
    for _ in range(45):  # crosses episode boundary -> auto-reset exercised
        a = torch.rand(N, env.act_dim, device="cuda") * 2 - 1
        obs, rew, done, info = env.step(a)
    assert obs.shape == (N, env.obs_dim)
    assert rew.shape == (N,) and torch.isfinite(rew).all()
    assert done.dtype == torch.bool and info["success"].dtype == torch.bool


def test_verifiers_discriminate(scene):
    """Verifier sanity: 'touch' succeeds under random flailing (arm hits cube),
    'hold_still' does not; both verify physical facts, not reward hacks."""
    import mujoco
    touch = make("touch", scene, TaskConfig(episode_len=200, substeps=4, seed=2))
    touch.reset()
    # physically kick the cube (write its free-joint velocity) — the
    # displacement verifier must fire off actual state change
    mjm = scene.mjm
    bid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "box")
    dadr = int(mjm.jnt_dofadr[mjm.body_jntadr[bid]])
    touch.qvel[:, dadr:dadr + 2] = 0.5  # 0.5 m/s sideways
    scene.forward()
    hits = 0
    a = torch.zeros(N, touch.act_dim, device="cuda")
    for _ in range(20):
        _, _, _, info = touch.step(a)
        hits += int(info["success"].sum().item())
    assert hits > 0, "a kicked cube must trip the displacement verifier"

    # same perturbation, opposite verdicts: the kicked cube that satisfies
    # "touch" must FAIL "hold_still" — verifiers judge physical facts
    hold = make("hold_still", scene, TaskConfig(episode_len=30, substeps=4, seed=2))
    hold.reset()
    hold.qvel[:, dadr:dadr + 2] = 0.5
    scene.forward()
    succ = 0
    a = torch.zeros(N, hold.act_dim, device="cuda")
    for _ in range(35):
        _, _, _, info = hold.step(a)
        succ += int(info["success"].sum().item())
    assert succ == 0, "a disturbed cube must fail the hold_still verifier"
