"""Runtime configuration for Latent Physics World.

Deliberately does NOT touch global torch/warp state (no set_default_device) —
config is threaded explicitly so the library stays a good citizen in a host
training process.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Simulation configuration.

    Attributes
    ----------
    n_worlds:   number of parallel environments (the engine's `nworld`).
    device:     "cuda" | "cpu" | "auto".
    dtype:      "float32" (engine works in fp32).
    timestep:   physics dt (s). None -> take from the MJCF <option timestep>.
    naconmax:   per-batch contact-buffer cap; indoor scenes need it large
                (see readiness report §4/§5). None -> engine default (small!).
    njmax:      per-world constraint cap. None -> engine default.
    """

    n_worlds: int = 1
    device: str = "auto"
    dtype: str = "float32"
    timestep: float | None = None
    naconmax: int | None = None
    njmax: int | None = None


def resolve_device(device: str = "auto") -> str:
    """Resolve "auto" to "cuda" if a CUDA device is available, else "cpu".

    Import of torch/warp is optional; we probe defensively so this works even
    where the GPU engine isn't installed (e.g. a Windows dev box).
    """
    if device != "auto":
        return device
    try:
        import torch  # optional

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"
