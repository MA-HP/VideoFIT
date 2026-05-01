"""
VideoFIT — Edge Result Model
Pure data structure holding the result of the edge-detection pipeline.
All processing logic lives in app.services.edge_service.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EdgeResult:
    """Intermediate products of the edge-detection pipeline."""
    edges: np.ndarray                        # uint8  rendered edge map (for display)
    distance_field: np.ndarray               # float32 distance transform
    silhouette_centroid: np.ndarray          # (x, y) in pixel coords — median of edge pts
    mask: np.ndarray | None = None           # uint8 silhouette mask, or None if maskless
    edge_points: np.ndarray | None = None    # (N, 2) float32 sub-pixel (x, y) positions
