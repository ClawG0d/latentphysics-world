"""Navigation (our IP) — BUILD.

Mobile-robot navigation semantics the engine doesn't provide: occupancy grid
from the scene, path planning, and obstacle-avoidance reward terms for RL.
"""

from __future__ import annotations

__all__ = ["OccupancyGrid", "Planner"]


class OccupancyGrid:
    """2D/2.5D occupancy map extracted from static scene geometry.

    Planned API:
        grid = OccupancyGrid.from_scene(scene, resolution=0.05)
        grid.is_free(x, y)
    """

    def __init__(self, **kw) -> None:
        raise NotImplementedError("TODO(P5): rasterize static collision geoms")


class Planner:
    """Global path planner (A*/RRT) over the occupancy grid."""

    def __init__(self, grid: "OccupancyGrid") -> None:
        raise NotImplementedError("TODO(P5)")
