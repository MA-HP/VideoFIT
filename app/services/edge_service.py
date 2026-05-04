"""
VideoFIT — Edge Service
GPU edge pipeline, no spatial mask.

The pipeline runs Devernay on the full grayscale frame with strict
gradient / length / curvature thresholds so only real structural edges
survive. The scene centroid is derived from the coordinate-wise median
of all detected sub-pixel edge points — robust to outliers because the
part boundary always contributes the majority of edge pixels.

Benefits over the mask-based approach
──────────────────────────────────────
• Zero sensitivity to illumination mode (DIA / EPI / DIA+EPI) — no
  border-brightness heuristics, no threshold to tune.
• No morphological overhead (Gaussian blur → Otsu → flood-fill → erode
  / dilate) — saves ~10–20 ms per frame[cite: 1].
• No mask failure modes (part touching border, unusual reflectivity …)[cite: 1].
• Centroid from edge-point median is more stable than mask moments for
  non-convex or partially-occluded parts[cite: 1].
"""

from __future__ import annotations

import cv2
import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import distance_transform_edt

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
    # OpenCV's CPU cvtColor is incredibly fast. Leaving this on the CPU
    # avoids unnecessary complexity before the GPU handoff.
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]
    if capture_stages:
        stages["gray"] = gray.copy()

    # ── Step 2: GPU Devernay — full image, no spatial mask ────────────
    ex, ey, edge_map = devernay_edges(
        gray=gray,
        sigma=0.0,
        high_thresh=80.0,
        low_thresh=30.0,
        mask=None,
        downsample=1.0,
        min_curvature=3.0,
        min_edge_length=25,
    )
    if capture_stages:
        stages["edges_dev"] = edge_map.copy()

    # ── GPU Handover for Post-Processing ──────────────────────────────
    # Upload the arrays returned by devernay_edges back to the GPU.
    # PRO TIP: If you ever refactor devernay_edges to natively return
    # CuPy arrays, you can delete these three lines and eliminate the
    # PCIe roundtrip completely!
    ex_cp = cp.asarray(ex)
    ey_cp = cp.asarray(ey)
    edge_map_cp = cp.asarray(edge_map)

    # ── Step 3: Sub-pixel edge points (GPU) ───────────────────────────
    valid_cp = ex_cp >= 0.0
    xs_cp = ex_cp[valid_cp]
    ys_cp = ey_cp[valid_cp]

    edge_points_cp = cp.column_stack([xs_cp, ys_cp]).astype(cp.float32)

    if capture_stages:
        viz_cp = cp.zeros((H, W), dtype=cp.uint8)
        if len(edge_points_cp) > 0:
            xs_i = cp.clip(xs_cp.astype(cp.int32), 0, W - 1)
            ys_i = cp.clip(ys_cp.astype(cp.int32), 0, H - 1)
            viz_cp[ys_i, xs_i] = 255
        stages["edge_points_viz"] = cp.asnumpy(viz_cp)

    # ── Step 4: Centroid from coordinate-wise median (GPU) ────────────
    if len(edge_points_cp) > 0:
        cx = float(cp.median(xs_cp))
        cy = float(cp.median(ys_cp))
    else:
        cx, cy = W / 2.0, H / 2.0

    # ── Step 5: Distance Transform (GPU) ──────────────────────────────
    if len(edge_points_cp) > 0:
        # EDT calculates exact Euclidean distance to the nearest 0.
        # By checking `== 0`, edge pixels (255) become False (0),
        # and open space (0) becomes True (1), achieving the exact
        # same result as cv2.bitwise_not.
        dist_input = edge_map_cp == 0
        dist_cp = distance_transform_edt(dist_input).astype(cp.float32)
    else:
        dist_cp = cp.full((H, W), float(max(H, W)), dtype=cp.float32)

    if capture_stages:
        stages["distance_field"] = cp.asnumpy(dist_cp)

    # ── Final PCIe Download ───────────────────────────────────────────
    result = EdgeResult(
        mask=None,
        edges=edge_map,  # Already a NumPy array from Step 2
        distance_field=cp.asnumpy(dist_cp),
        silhouette_centroid=np.array([cx, cy]),
        edge_points=cp.asnumpy(edge_points_cp),
    )

    if capture_stages:
        return result, stages
    return result