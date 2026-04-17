"""
VideoFIT — DXF Model
Pure data structure holding the result of a DXF file parse.
All parsing and geometry logic lives in app.services.dxf_service.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import ezdxf


@dataclass
class Dxf:
    """Result of parsing a DXF file."""
    polylines: list[np.ndarray] = field(default_factory=list)  # Nx2 float32, pixel coords (all layers)
    dxf_center_mm: tuple[float, float] = (0.0, 0.0)           # centre of DXF bounding box (mm)
    doc: ezdxf.document.Drawing | None = None                 # Original ezdxf document for native rendering
    px_per_mm: float = 1.0                                    # camera calibration used
    canvas_shape: tuple[int, int] = (0, 0)                    # (H, W) canvas shape used
    # Layer-separated polylines for Complete mode
    polylines_global: list[np.ndarray] = field(default_factory=list)   # GLOBAL layer
    polylines_refine: list[np.ndarray] = field(default_factory=list)   # REFINE layer
