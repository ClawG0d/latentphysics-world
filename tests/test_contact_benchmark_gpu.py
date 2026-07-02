"""Contact benchmark: LPW (mujoco_warp GPU) vs the C MuJoCo engine.

The C engine is the semantics oracle. This is the R1 fidelity gate and the
first anti-exploit guardrail: if GPU contact forces drift from the reference,
RL will learn simulator artifacts instead of physics.

Checks (falling-box scene, 1kg box settling on a plane):
  1. ballistic phase: GPU trajectory matches C step-for-step (contact-free)
  2. settled pose: both engines rest the box at z ~= half-extent
  3. settled contact normal force: |GPU - C| / C < 5%  (and both ~= m*g)
"""

import os

import numpy as np
import pytest

lpw = pytest.importorskip("latentphysics")
mujoco = pytest.importorskip("mujoco")
mjw = pytest.importorskip("mujoco_warp")
torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

SCENE = os.path.join(os.path.dirname(__file__), "..", "examples", "scenes", "falling_box.xml")
MASS = 1.0
G = 9.81


def _c_engine_rollout(n_steps):
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    traj = []
    for _ in range(n_steps):
        mujoco.mj_step(m, d)
        traj.append(d.qpos.copy())
    # settled total contact normal force on the box
    f_total = 0.0
    for i in range(d.ncon):
        f6 = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, f6)
        f_total += f6[0]  # normal component in contact frame
    return np.asarray(traj), f_total


def _gpu_rollout(n_steps):
    import warp as wp

    scene = lpw.load_scene(SCENE, lpw.Config(n_worlds=1))
    traj = []
    for _ in range(n_steps):
        scene.step()
        traj.append(scene.qpos()[0].cpu().numpy().copy())
    # settled contact normal forces via engine helper
    n = int(scene.engine._to_torch(scene.data.nacon).sum().item())
    f_total = 0.0
    if n > 0:
        ids = wp.array(np.arange(n, dtype=np.int32), dtype=wp.int32, device="cuda")
        out = wp.zeros(n, dtype=wp.spatial_vectorf, device="cuda")
        mjw.contact_force(scene.model, scene.data, ids, False, out)
        wp.synchronize()
        fn = out.numpy()  # (n, 6): torque(3) + force(3) or force-first? use max norm axis
        # contact frame normal force is component [0] of the force triplet;
        # spatial_vector layout in mujoco_warp mirrors mjContactForce: [f, t]
        f_total = float(np.abs(fn[:, 0]).sum())
    return np.asarray(traj), f_total


def test_contact_benchmark_vs_c_engine():
    n_steps = 400  # 2.0 s at dt=0.005 -> fully settled
    c_traj, c_force = _c_engine_rollout(n_steps)
    g_traj, g_force = _gpu_rollout(n_steps)

    # 1) ballistic phase (first 30 steps, contact-free): step-for-step match
    ballistic_err = np.abs(c_traj[:30] - g_traj[:30]).max()
    assert ballistic_err < 1e-3, f"ballistic divergence {ballistic_err}"

    # 2) settled pose agreement
    cz, gz = c_traj[-1][2], g_traj[-1][2]
    assert abs(cz - gz) < 5e-3, f"settled z: C={cz} GPU={gz}"
    assert 0.08 < gz < 0.12

    # 3) settled contact force: both ~= m*g, relative gap < 5%
    assert c_force > 0 and g_force > 0, "both engines must report settled contacts"
    rel = abs(g_force - c_force) / c_force
    assert rel < 0.05, f"contact force gap {rel:.1%}: C={c_force:.3f}N GPU={g_force:.3f}N"
    assert abs(c_force - MASS * G) / (MASS * G) < 0.1  # oracle sanity
