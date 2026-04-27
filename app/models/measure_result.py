"""
VideoFIT — Measure Result Model
Pure data structure for a single fitted shape (circle, arc, or line).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ShapeKind(str, Enum):
    CIRCLE = "circle"
    ARC = "arc"
    LINE = "line"


@dataclass
class MeasureResult:
    kind: ShapeKind
    residual_rms: float          # RMS geometric residual in pixels
    n_points: int                # Number of edge points used

    # ── Circle / Arc ─────────────────────────────────────────────────
    cx: float = 0.0              # Centre x  (pixels)
    cy: float = 0.0              # Centre y  (pixels)
    radius: float = 0.0         # Radius    (pixels)

    # ── Arc only: angular range in screen-coord convention ───────────
    # Angles from np.arctan2(dy_screen, dx); right=0°, down=90°.
    arc_start_deg: float = 0.0  # Start of arc
    arc_span_deg: float = 360.0 # Span of arc (always positive, CW on screen)

    # ── Line ─────────────────────────────────────────────────────────
    line_p1: tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    line_p2: tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
