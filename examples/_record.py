"""Shared demo recorder: replay a GPU-simulated qpos trajectory offscreen
and write an animated webp for the gallery.

Physics always runs on the LPW GPU engine; this module only re-renders the
recorded trajectory with the CPU reference renderer (visuals, not physics).
"""

from __future__ import annotations

import os
import subprocess
import tempfile


def record_webp(mjcf_path, qpos_traj, out_name, cam=None, every=3, fps=15,
                size=(480, 300), media_dir=None, quality=60):
    """Render qpos_traj (list/array of qpos) into docs/media/<out_name>.webp."""
    import imageio.v2 as imageio
    import imageio_ffmpeg
    import mujoco

    media_dir = media_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "media")
    os.makedirs(media_dir, exist_ok=True)

    m = mujoco.MjModel.from_xml_path(mjcf_path)
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, height=size[1], width=size[0])
    c = mujoco.MjvCamera()
    cam = cam or {}
    c.lookat[:] = cam.get("lookat", (0, 0, 0.3))
    c.distance = cam.get("distance", 2.0)
    c.azimuth = cam.get("azimuth", 120)
    c.elevation = cam.get("elevation", -20)

    frames = []
    for k, q in enumerate(qpos_traj):
        if k % every:
            continue
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        if "azimuth_rate" in cam:
            c.azimuth = cam.get("azimuth", 120) + k * cam["azimuth_rate"]
        r.update_scene(d, camera=c)
        frames.append(r.render().copy())
    r.close()

    tmp = os.path.join(tempfile.gettempdir(), out_name + ".mp4")
    imageio.mimsave(tmp, frames, fps=fps, quality=8)
    out = os.path.join(media_dir, out_name + ".webp")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    res = subprocess.run([ffmpeg, "-y", "-i", tmp, "-vcodec", "libwebp", "-lossless", "0",
                          "-q:v", str(quality), "-loop", "0", "-an", out], capture_output=True)
    if res.returncode != 0:
        out = os.path.join(media_dir, out_name + ".gif")
        imageio.mimsave(out, frames, fps=fps, loop=0)
    print(f"wrote {out} ({len(frames)} frames)")
    return out
