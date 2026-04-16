"""
VideoFIT — DXF Model
Pure data structure holding the result of a DXF file parse.
All parsing and geometry logic lives in app.services.dxf_service.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Dxf:
    """Result of parsing a DXF file."""
    polylines: list[np.ndarray] = field(default_factory=list)  # Nx2 float32, pixel coords
    dxf_center_mm: tuple[float, float] = (0.0, 0.0)           # centre of DXF bounding box (mm)
