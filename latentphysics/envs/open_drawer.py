"""Open-drawer benchmark (R4) — backward-compatible shim.

The articulated-furniture task family now lives in :mod:`articulated_tasks`
(open_drawer / open_door / open_lid / slide_door). This module keeps the
original ``build_scene`` / ``OpenDrawer`` names working.
"""

from __future__ import annotations

from .articulated_tasks import ART_SPECS, ArticulatedOpen, build_articulated_scene
from .base import TaskConfig

__all__ = ["build_scene", "OpenDrawer"]


def build_scene(out_path: str | None = None, menagerie: str | None = None) -> str:
    return build_articulated_scene("open_drawer", out_path, menagerie)


class OpenDrawer(ArticulatedOpen):
    """Pull the upper drawer open. Verify: slide joint > 15 cm."""

    OPEN_THRESH = ART_SPECS["open_drawer"].thresh

    def __init__(self, scene, cfg: TaskConfig | None = None):
        super().__init__(scene, ART_SPECS["open_drawer"], cfg)

    def drawer_q(self):
        return self.joint_q()
