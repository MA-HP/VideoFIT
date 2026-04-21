"""
VideoFIT — Fit Result Model
Pure data structure holding the result of a rigid-body alignment.
All fitting logic (Chamfer cost, Nelder-Mead) lives in app.services.fit_service.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FitResult:
    """Rigid-body transform that maps DXF pixel coords → aligned pixel coords."""
    tx: float                             # translation X (pixels)
    ty: float                             # translation Y (pixels)
    angle_deg: float                      # rotation (degrees)
    cost: float                           # final mean distance-transform cost
    dxf_cx: float                         # DXF centroid X used for rotation pivot (pixels)
    dxf_cy: float                         # DXF centroid Y used for rotation pivot (pixels)
    inlier_frac: float                    # fraction of DXF points within 3px of edges
    dist_t: Optional[np.ndarray] = None              # smoothed distance field (for optimizer)
    distance_field_raw: Optional[np.ndarray] = None  # raw distance field (for heatmap display)
