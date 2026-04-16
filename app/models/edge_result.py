"""
VideoFIT — Edge Result Model
Pure data structure holding the result of the edge-detection pipeline.
All processing logic lives in app.services.edge_service.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EdgeResult:
    """Intermediate products of the edge-detection pipeline."""
    mask: np.ndarray                 # uint8  binary silhouette mask
    edges: np.ndarray                # uint8  rendered edge map (for display)
    distance_field: np.ndarray       # float32 distance transform (fallback / display)
    silhouette_centroid: np.ndarray  # (x, y) in pixel coords
    edge_points: np.ndarray = None   # (N, 2) float32 sub-pixel (x, y) positions
