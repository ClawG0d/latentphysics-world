# Third-Party Notices

Latent Physics World incorporates and/or depends on the following third-party
open-source components. Each is used under its own license. Latent Physics
World's own original code is proprietary (see `LICENSE`).

> Action (P0): keep this file authoritative. Run a license scan
> (`pip-licenses`, ScanCode, or FOSSA) in CI and reconcile any additions here.
> Reject any transitive dependency under a non-commercial or copyleft (GPL/AGPL)
> license, or replace it.

| Component | Version (pin) | License | Copyright | Notes |
|---|---|---|---|---|
| mujoco_warp | _pin in P0_ | Apache-2.0 | The Newton Developers | GPU physics engine core; forked into `third_party/`. Modified files carry a "Modified by" notice. |
| MuJoCo | >=3.2.0 | Apache-2.0 | Google DeepMind | Model compilation, MJCF parsing, CPU reference oracle. |
| NVIDIA Warp (`warp-lang`) | >=1.5.0 | Apache-2.0 | NVIDIA Corporation | Kernel runtime for the engine. **Verify the pinned version's license in P0.** |
| NumPy | >=1.24 | BSD-3-Clause | NumPy Developers | Numerics. |
| trimesh | >=4.0 | MIT | Michael Dawson-Haggerty et al. | Asset/mesh IO (assets extra). |
| CoACD | _pin_ | MIT | Xinyue Wei et al. | Convex decomposition of furniture meshes (assets extra). |
| PyTorch | >=2.2 | BSD-3-Clause | Meta / PyTorch | Training interop (train extra). |

## Apache-2.0 compliance checklist (this repo)

- [x] Retain upstream `LICENSE`, `NOTICE`, and per-file copyright/SPDX headers in `third_party/`.
- [ ] Mark every modified upstream file with `Modified by Latent Physics World, <year>`.
- [ ] Do not use the "MuJoCo", "Newton", or "NVIDIA" trademarks to name or endorse this product.
- [ ] Ship this file + upstream `NOTICE` contents with any binary/source distribution.
- [ ] CI license-scan gate green.
