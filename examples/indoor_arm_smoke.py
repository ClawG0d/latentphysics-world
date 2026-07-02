"""Smoke example — intended API for an indoor arm scene.

Runs on Linux + CUDA once P0/P1 land. On a non-GPU host it raises
EngineUnavailable with an actionable message (by design).
"""

import latentphysics as lpw


def main() -> None:
    cfg = lpw.Config(n_worlds=256, device="auto", naconmax=256)
    print(f"Latent Physics World {lpw.__version__} | device={lpw.resolve_device(cfg.device)}")

    # P3: build an indoor scene from assets (kitchen with articulated furniture)
    #   from latentphysics.assets import SceneBuilder
    #   mjcf = SceneBuilder().add_room(...).add_articulated(...).add_robot(...).build("kitchen.xml")
    mjcf = "scenes/indoor_arm.xml"  # placeholder path

    scene = lpw.load_scene(mjcf, cfg)   # -> raises EngineUnavailable off-GPU
    for _ in range(1000):
        scene.step()
    print("done:", scene.qpos().shape)


if __name__ == "__main__":
    try:
        main()
    except lpw.backend.EngineUnavailable as e:  # type: ignore[attr-defined]
        print(e)
