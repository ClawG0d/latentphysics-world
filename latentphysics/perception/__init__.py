"""Perception (our IP) — BUILD, readiness report §5③.

Sensors for indoor navigation + manipulation, on top of the engine's ray/render
primitives. The engine has single-ray + a native BVH batch renderer (RGB/depth/
segmentation); this layer adds scanning-LiDAR semantics and point clouds.
"""

from __future__ import annotations

__all__ = ["Lidar", "DepthCamera", "PointCloud"]


class Lidar:
    """Scanning LiDAR built on batched multi-ray casting.

    Planned API:
        lidar = Lidar(mount_link="base", channels=16, h_fov=360, rate_hz=10)
        cloud = lidar.scan(scene)   # -> PointCloud, batched over n_worlds
    """

    def __init__(self, **kw) -> None:
        raise NotImplementedError("TODO(P3): scan-line angles + batched multi-ray")


class DepthCamera:
    """Depth + segmentation via the engine's batch renderer (RGB-D+seg)."""

    def __init__(self, **kw) -> None:
        raise NotImplementedError("TODO(P3): bind engine batch renderer")


class PointCloud:
    """Batched point cloud with export (PCD/PLY) and noise/dropout models."""

    def __init__(self, points=None) -> None:
        raise NotImplementedError("TODO(P3)")
