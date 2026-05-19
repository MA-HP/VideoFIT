"""
VideoFIT — Measure Result Model
Pure data structure for a single fitted shape (circle, arc, or line).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class ShapeKind(str, Enum):
    CIRCLE = "circle"
    ARC = "arc"
    LINE = "line"


@dataclass
class MeasureResult:
    kind: ShapeKind
    residual_rms: float          # RMS geometric residual in pixels
    n_points: int                # Number of edge points used
    confidence: float = 1.0      # Quality score (0–1), higher is better
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Edge points used for this fit (N, 2) float32
    edge_points: np.ndarray | None = field(default=None, repr=False)

    # ── Circle / Arc ─────────────────────────────────────────────────
    cx: float = 0.0
    cy: float = 0.0
    radius: float = 0.0

    # ── Arc only ─────────────────────────────────────────────────────
    arc_start_deg: float = 0.0
    arc_span_deg: float = 360.0

    # ── Line ─────────────────────────────────────────────────────────
    line_p1: tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    line_p2: tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))

