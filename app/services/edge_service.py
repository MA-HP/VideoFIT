"""
VideoFIT — Edge Service
Business logic for edge detection and distance-transform computation.
Uses a fully vectorised Canny-Devernay sub-pixel edge detector.
The EdgeResult dataclass (pure structure) lives in app.models.edge_result.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.models.edge_result import EdgeResult
from app.services.devernay_service import devernay_edges

# ── Pre-built gamma LUT (γ = 0.8, computed once at import time) ──────────────
_GAMMA = 0.8
_GAMMA_LUT: np.ndarray = np.array(
    [((i / 255.0) ** (1.0 / _GAMMA)) * 255 for i in range(256)],
    dtype=np.uint8,
)


def compute_edges(frame_bgr: np.ndarray) -> EdgeResult:
    """
    Run the full sub-pixel edge pipeline on a **BGR** frame.

    Steps
    -----
    1. Silhouette mask  (largest contour via adaptive threshold)
    2. Gamma correction → CLAHE → bilateral filter
    3. Canny-Devernay sub-pixel edge detection (vectorised)
    4. Silhouette contour drawn onto edge map for boundary completeness
    5. Distance transform on the inverted edge map

    Parameters
    ----------
    frame_bgr : np.ndarray
        Input image in BGR colour order (as delivered by OpenCV / IC4).

    Returns
    -------
    EdgeResult
        Mask, sub-pixel edge map, distance field, and silhouette centroid.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # ── Step 1: Silhouette mask ───────────────────────────────────────
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 51, 5,
    )
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(gray)
    largest_contour = None
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Silhouette centroid
    M = cv2.moments(mask)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        h, w = gray.shape[:2]
        cx, cy = w / 2.0, h / 2.0

    # ── Step 2: Pre-processing (full resolution) ─────────────────────
    gamma_corrected = cv2.LUT(gray, _GAMMA_LUT)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq = clahe.apply(gamma_corrected)

    # ── Step 3: Sub-pixel Canny-Devernay edge detection ───────────────
    # sigma=1.2: fast OpenCV GaussianBlur (SIMD) replaces the slow bilateral.
    # Full resolution preserved → sub-pixel precision ~0.5/66.83 ≈ 0.007 mm.
    ex, ey, final_edges = devernay_edges(
        eq,
        sigma=1.2,
        high_thresh=20.0,
        low_thresh=10.0,
        mask=mask,
        downsample=1.0,
    )

    # Collect sub-pixel edge points as a (N, 2) float32 array
    valid = ex >= 0.0
    edge_points = np.column_stack([ex[valid], ey[valid]]).astype(np.float32)


    # ── Step 4: Add silhouette boundary for completeness ─────────────
    if largest_contour is not None:
        cv2.drawContours(final_edges, [largest_contour], -1, 255, thickness=1)

    # ── Step 5: Distance transform ────────────────────────────────────
    dist = cv2.distanceTransform(~final_edges, cv2.DIST_L2, 5)

    return EdgeResult(
        mask=mask,
        edges=final_edges,
        distance_field=dist.astype(np.float32),
        silhouette_centroid=np.array([cx, cy]),
        edge_points=edge_points,
    )

