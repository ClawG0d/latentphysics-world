"""Physics: 8192 parallel worlds — one batched GPU sim, 16 worlds shown.

All 8192 worlds run the same six-box drop in ONE batched simulation; a
seeded per-world velocity nudge makes every world diverge, so the mosaic
tiles (worlds 0-15) visibly disagree while sharing one engine step. The
hard asserts cover ALL 8192 worlds, not just the rendered 16.
Run:  MUJOCO_GL=egl python examples/parallel_worlds.py --record
"""

import argparse
import os

import numpy as np
import torch

import latentphysics as lpw

N_WORLDS = 8192
N_BOXES = 6
SHOW = 16          # worlds rendered into the 4x4 mosaic
TILE = (160, 100)  # per-world tile (w, h) -> 640x400 canvas

ASSETS = ('<asset>'
          '<texture type="skybox" builtin="gradient" rgb1="0.45 0.53 0.62" '
          'rgb2="0.12 0.14 0.18" width="256" height="256"/>'
          '<texture name="floortex" type="2d" builtin="checker" rgb1="0.78 0.74 0.68" '
          'rgb2="0.68 0.64 0.58" mark="edge" markrgb="0.55 0.52 0.48" width="300" height="300"/>'
          '<material name="floormat" texture="floortex" texrepeat="12 12" reflectance="0.12"/>'
          '</asset>')

COLORS = ((0.85, 0.30, 0.25), (0.25, 0.55, 0.85), (0.95, 0.75, 0.20),
          (0.35, 0.75, 0.40), (0.70, 0.40, 0.80), (0.90, 0.55, 0.30))


def build_scene(path: str) -> str:
    body = ['<geom name="floor" type="plane" size="3 3 0.1" material="floormat"/>']
    # no cast shadows: at 160px tiles the directional shadowmap renders as
    # black speckle (acne), and shadows are unreadable at that size anyway
    body.append('<light name="key" directional="true" pos="0 0 3" dir="-0.3 0.2 -0.9" '
                'diffuse="0.8 0.78 0.75" castshadow="false"/>')
    body.append('<light name="fill" directional="true" pos="2 -2 2" dir="0.4 0.5 -0.8" '
                'diffuse="0.25 0.25 0.28" castshadow="false"/>')
    # six boxes staggered in a loose 3x2 grid at two heights so they collide
    # mid-air and on the floor — contact chaos amplifies the per-world nudges
    for k in range(N_BOXES):
        x = ((k % 3) - 1) * 0.13
        y = ((k // 3) - 0.5) * 0.15
        z = 0.30 + 0.22 * (k % 2)
        c = COLORS[k]
        body.append(
            f'<body name="b{k}" pos="{x:.3f} {y:.3f} {z:.3f}" '
            f'euler="{7 * k} {11 * k} {17 * k}"><freejoint/>'
            f'<geom type="box" size="0.05 0.05 0.05" '
            f'rgba="{c[0]} {c[1]} {c[2]} 1" mass="0.1"/></body>')
    xml = ('<mujoco model="parallel_worlds">'
           '<option timestep="0.004" iterations="10" ls_iterations="10"/>'
           + ASSETS +
           '<worldbody>' + "".join(body) + '</worldbody></mujoco>')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(xml)
    return path


def record_mosaic(scene_path, traj, every=4, fps=12):
    import mujoco
    from _record import save_frames

    m = mujoco.MjModel.from_xml_path(scene_path)
    d = mujoco.MjData(m)
    tw, th = TILE
    ssaa = 2  # render 2x and box-downscale: MuJoCo's Renderer has no MSAA
    r = mujoco.Renderer(m, height=th * ssaa, width=tw * ssaa)
    c = mujoco.MjvCamera()
    c.lookat[:] = (0, 0, 0.15)
    c.distance = 1.7
    c.azimuth = 130
    c.elevation = -22
    frames = []
    for k in range(0, len(traj), every):
        canvas = np.zeros((th * 4, tw * 4, 3), dtype=np.uint8)
        for w in range(SHOW):
            d.qpos[:] = traj[k][w]
            mujoco.mj_forward(m, d)
            r.update_scene(d, camera=c)
            big = r.render()
            tile = (big.reshape(th, ssaa, tw, ssaa, 3).mean(axis=(1, 3))).astype(np.uint8)
            tile[:1, :] = 24  # hairline separators between worlds
            tile[:, :1] = 24
            row, col = divmod(w, 4)
            canvas[row * th:(row + 1) * th, col * tw:(col + 1) * tw] = tile
        frames.append(canvas)
    r.close()
    save_frames(frames, "parallel_worlds", fps=fps, quality=55)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--steps", type=int, default=320)
    args = ap.parse_args()

    scene_path = os.path.expanduser("~/lpw/assets/demos/parallel_worlds.xml")
    build_scene(scene_path)
    scene = lpw.load_scene(scene_path, lpw.Config(n_worlds=N_WORLDS))

    # seeded per-world nudges: committed script + seed -> reproducible run
    gen = torch.Generator(device="cuda").manual_seed(0)
    qv = scene.qvel()
    qv[:] = 0.6 * torch.randn(qv.shape, generator=gen, device="cuda")
    scene.forward()

    traj = []
    for _ in range(args.steps):
        scene.step()
        traj.append(scene.qpos()[:SHOW].cpu().numpy().copy())

    q = scene.qpos()
    assert torch.isfinite(q).all(), "some of the 8192 worlds went non-finite"
    z = q.view(N_WORLDS, N_BOXES, 7)[:, :, 2]
    assert (z > -0.02).all(), "a box tunneled through the floor"
    spread = q.std(dim=0).max().item()
    assert spread > 0.05, f"8192 worlds failed to diverge (max std {spread:.4f})"
    print(f"simulated {args.steps} steps x {N_WORLDS} worlds on GPU; "
          f"max cross-world std = {spread:.3f} (diverged)")

    if args.record:
        record_mosaic(scene_path, traj)


if __name__ == "__main__":
    main()
