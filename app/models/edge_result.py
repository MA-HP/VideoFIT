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
    edges: np.ndarray                # uint8  Canny edge map (within mask)
    distance_field: np.ndarray       # float32 distance transform
    silhouette_centroid: np.ndarray  # (x, y) in pixel coords