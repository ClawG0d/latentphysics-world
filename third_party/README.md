# third_party/

Upstream, Apache-2.0 components. **Do not** remove or alter their copyright,
LICENSE, NOTICE, or SPDX headers. Any file we modify must carry a
`Modified by Latent Physics World, <year>` notice (Apache-2.0 §4(b)).

## mujoco_warp (the physics engine core)

Brought in on the Linux/CUDA machine as a **git submodule** pointing at our fork
of upstream (so we can patch it — BVH broadphase, contact-overflow detection;
see `docs/MUJOCO_WARP_FORK_READINESS.md` §4):

```bash
# on Linux + CUDA
git submodule add https://github.com/<org>/mujoco_warp third_party/mujoco_warp
git -C third_party/mujoco_warp checkout <pinned-sha>   # pin; record in THIRD_PARTY_NOTICES.md
```

Upstream: https://github.com/google-deepmind/mujoco_warp — © The Newton Developers, Apache-2.0.

Not vendored into this repo yet: it requires Linux + CUDA to build/run, and our
patches live on a branch of the fork. `latentphysics/backend/` is the only code
that imports it.
