"""
VideoFIT — Edge Service
GPU edge pipeline, no spatial mask.

The pipeline runs Devernay on the full grayscale frame with strict
gradient / length / curvature thresholds so only real structural edges
survive.  The scene centroid is derived from the coordinate-wise median
of all detected sub-pixel edge points — robust to outliers because the
part boundary always contributes the majority of edge pixels.

Benefits over the mask-based approach
──────────────────────────────────────
• Zero sensitivity to illumination mode (DIA / EPI / DIA+EPI) — no
  border-brightness heuristics, no threshold to tune.
• No morphological overhead (Gaussian blur → Otsu → flood-fill → erode
  / dilate) — saves ~10–20 ms per frame.
• No mask failure modes (part touching border, unusual reflectivity …).
• Centroid from edge-point median is more stable than mask moments for
  non-convex or partially-occluded parts.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.models.edge_result import EdgeResult
from app.services.devernay_service import devernay_edges


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_edges(
    frame_bgr: np.ndarray, capture_stages: bool = False
) -> "EdgeResult | tuple[EdgeResult, dict]":
    """
    Run the full edge pipeline on a BGR frame.

    Returns
    -------
    EdgeResult, or (EdgeResult, stages_dict) when *capture_stages* is True.
    """
    stages: dict = {} if capture_stages else None

    # ── Step 1: Grayscale ─────────────────────────────────────────────
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]
    if capture_stages:
        stages["gray"] = gray.copy()

    # ── Step 2: GPU Devernay — full image, no spatial mask ────────────
    # Strict thresholds act as the quality gate:
    #   high_thresh=80  → only strong, well-defined gradients anchor chains
    #   low_thresh =30  → faint scratches / texture never grow into chains
    #   min_curvature=3 → rejects erratic / jagged noise responses
    #   min_edge_length=25 → eliminates isolated short bursts
    # Together these leave only clean geometric boundary edges.
    ex, ey, edge_map = devernay_edges(
        gray=gray,
        sigma=0.0,
        high_thresh=80.0,
        low_thresh=30.0,
        mask=None,          # no spatial restriction
        downsample=1.0,
        min_curvature=3.0,
        min_edge_length=25,
    )
    if capture_stages:
        stages["edges_dev"] = edge_map.copy()

    # ── Step 3: Sub-pixel edge points ─────────────────────────────────
    valid = ex >= 0.0
    edge_points = np.column_stack([ex[valid], ey[valid]]).astype(np.float32)

    if capture_stages:
        viz = np.zeros((H, W), dtype=np.uint8)
        if len(edge_points) > 0:
            xs_i = edge_points[:, 0].astype(np.int32).clip(0, W - 1)
            ys_i = edge_points[:, 1].astype(np.int32).clip(0, H - 1)
            viz[ys_i, xs_i] = 255
        stages["edge_points_viz"] = viz

    # ── Step 4: Centroid from coordinate-wise median ──────────────────
    # The part boundary always contributes the majority of edge pixels, so
    # the median is pulled to the part centre even when some background or
    # internal texture edges survive the threshold.  The mean would be
    # distorted by a handful of far-away outlier edges; the median is not.
    if len(edge_points) > 0:
        cx = float(np.median(edge_points[:, 0]))
        cy = float(np.median(edge_points[:, 1]))
    else:
        cx, cy = W / 2.0, H / 2.0

    # ── Step 5: Distance Transform ────────────────────────────────────
    if len(edge_points) > 0:
        dist = cv2.distanceTransform(
            cv2.bitwise_not(edge_map), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
    else:
        dist = np.full((H, W), float(max(H, W)), dtype=np.float32)

    if capture_stages:
        stages["distance_field"] = dist

    result = EdgeResult(
        mask=None,          # no mask — centroid lives in silhouette_centroid
        edges=edge_map,
        distance_field=dist,
        silhouette_centroid=np.array([cx, cy]),
        edge_points=edge_points,
    )

    if capture_stages:
        return result, stages
    return result
