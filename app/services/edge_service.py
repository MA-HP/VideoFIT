"""
VideoFIT — Edge Service
GPU edge pipeline (exact test) + mode-adaptive silhouette mask.

Illumination modes
──────────────────
DIA       bright background (green/white), part is a dark silhouette.
EPI       dark background, part is bright.
DIA+EPI   bright background with visible surface detail — treated as DIA
          for masking purposes (outer silhouette is still the dominant dark blob).

The GPU Devernay pipeline is identical for all modes.
The silhouette mask is derived by a separate, mode-aware Otsu pass on the
raw grayscale so the fitters always receive a clean filled mask and centroid.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.models.edge_result import EdgeResult
from app.services.devernay_service import devernay_edges

# Structuring element reused every frame
_DIL_K = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))


# ─────────────────────────────────────────────────────────────────────────────
# Mask helpers
# ─────────────────────────────────────────────────────────────────────────────

def _border_brightness(gray: np.ndarray, margin_frac: float = 0.05) -> float:
    """
    Mean intensity of a thin border strip around the image.
    High → DIA (bright background).  Low → EPI (dark background).
    """
    h, w = gray.shape[:2]
    m = max(4, int(min(h, w) * margin_frac))
    strips = [
        gray[:m, :],        # top
        gray[h - m:, :],    # bottom
        gray[:, :m],        # left
        gray[:, w - m:],    # right
    ]
    return float(np.mean([s.mean() for s in strips]))


def _largest_interior_blob(binary: np.ndarray) -> np.ndarray:
    """
    Largest connected component that does NOT touch the image border.
    Falls back to the globally largest component on degenerate images.
    """
    h, w = binary.shape
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    best_label,     best_area     = 0, 0
    fallback_label, fallback_area = 0, 0

    for lbl in range(1, n_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        comp = (labels == lbl).view(np.uint8)
        touches = (
            comp[0, :].any() or comp[-1, :].any()
            or comp[:, 0].any() or comp[:, -1].any()
        )
        if not touches and area > best_area:
            best_area,     best_label     = area, lbl
        if area > fallback_area:
            fallback_area, fallback_label = area, lbl

    chosen = best_label if best_label > 0 else fallback_label
    mask = np.zeros((h, w), dtype=np.uint8)
    if chosen > 0:
        mask[labels == chosen] = 255
    return mask


def _silhouette_mask(gray: np.ndarray) -> np.ndarray:
    """
    Mode-adaptive silhouette mask (DIA / EPI / DIA+EPI).

    1. Detect background brightness from border pixels.
    2. Light Gaussian blur for Otsu stability.
    3. Otsu threshold → invert only for bright backgrounds (DIA / DIA+EPI).
    4. Keep the largest non-border-touching blob.
    """
    bg = _border_brightness(gray)
    smooth = cv2.GaussianBlur(gray, (7, 7), 2.0)
    _, binary = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if bg > 100:                        # DIA / DIA+EPI: bright background
        binary = cv2.bitwise_not(binary)

    return _largest_interior_blob(binary)


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

    # ── Step 2: Mode-adaptive silhouette mask ─────────────────────────
    mask = _silhouette_mask(gray)
    if capture_stages:
        stages["mask"] = mask.copy()

    M = cv2.moments(mask)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx, cy = W / 2.0, H / 2.0

    # ── Step 3: GPU Devernay (masked to silhouette + 7 px margin) ────
    # Dilating the mask keeps boundary edges that land just outside the
    # filled silhouette while rejecting all background noise in EPI mode.
    mask_for_gpu = cv2.dilate(mask, _DIL_K, iterations=2)

    ex, ey, edge_map = devernay_edges(
        gray=gray,
        sigma=0.0,
        high_thresh=50.0,
        low_thresh=15.0,
        mask=mask_for_gpu,
        downsample=1.0,
        min_curvature=2.0,
        min_edge_length=15,
    )
    if capture_stages:
        stages["edges_dev"] = edge_map.copy()

    # ── Step 4: Sub-pixel edge points ─────────────────────────────────
    valid = ex >= 0.0
    edge_points = np.column_stack([ex[valid], ey[valid]]).astype(np.float32)

    if capture_stages:
        viz = np.zeros((H, W), dtype=np.uint8)
        if len(edge_points) > 0:
            xs_i = edge_points[:, 0].astype(np.int32).clip(0, W - 1)
            ys_i = edge_points[:, 1].astype(np.int32).clip(0, H - 1)
            viz[ys_i, xs_i] = 255
        stages["edge_points_viz"] = viz

    # ── Step 5: Distance Transform ────────────────────────────────────
    # cv2.distanceTransform with DIST_MASK_PRECISE gives pixel-accurate L2
    # distances. The fitter pre-smooths the field with gaussian_filter(σ=1.5)
    # making any cKDTree sub-pixel correction (< 0.3 px) irrelevant.
    # Dropping the cKDTree removes 20–50 ms of build + query overhead.
    if len(edge_points) > 0:
        dist = cv2.distanceTransform(
            cv2.bitwise_not(edge_map), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
    else:
        dist = np.full((H, W), float(max(H, W)), dtype=np.float32)

    if capture_stages:
        stages["distance_field"] = dist

    result = EdgeResult(
        mask=mask,
        edges=edge_map,
        distance_field=dist,
        silhouette_centroid=np.array([cx, cy]),
        edge_points=edge_points,
    )

    if capture_stages:
        return result, stages
    return result
