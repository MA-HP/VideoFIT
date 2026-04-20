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


def compute_edges(frame_bgr: np.ndarray, capture_stages: bool = False) -> "EdgeResult | tuple[EdgeResult, dict]":
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
    capture_stages : bool
        When ``True``, also return a ``dict`` of intermediate images keyed by
        stage name (useful for the debug preprocessing window).

    Returns
    -------
    EdgeResult
        Mask, sub-pixel edge map, distance field, and silhouette centroid.
    tuple[EdgeResult, dict]
        Only when *capture_stages* is ``True``.
    """
    stages: dict = {} if capture_stages else None

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if capture_stages:
        stages["gray"] = gray.copy()

    # ── Step 1: Silhouette mask ───────────────────────────────────────
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    if capture_stages:
        stages["blur"] = blur.copy()

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 51, 5,
    )
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k, iterations=2)
    if capture_stages:
        stages["thresh"] = thresh.copy()

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(gray)
    largest_contour = None
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    if capture_stages:
        stages["mask"] = mask.copy()

    # Silhouette centroid
    M = cv2.moments(mask)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        h, w = gray.shape[:2]
        cx, cy = w / 2.0, h / 2.0

    # ── Step 2: Pre-processing (full resolution) ──────────────────────
    gamma_corrected = cv2.LUT(gray, _GAMMA_LUT)
    if capture_stages:
        stages["gamma"] = gamma_corrected.copy()

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq = clahe.apply(gamma_corrected)
    if capture_stages:
        stages["clahe"] = eq.copy()

    # ── Step 3: Sub-pixel Canny-Devernay edge detection ───────────────
    ex, ey, edges_raw = devernay_edges(
        eq,
        sigma=1.2,
        high_thresh=20.0,
        low_thresh=10.0,
        mask=mask,
        downsample=1.0,
    )
    if capture_stages:
        stages["edges_raw"] = edges_raw.copy()

    # Collect sub-pixel edge points as a (N, 2) float32 array
    valid = ex >= 0.0
    edge_points = np.column_stack([ex[valid], ey[valid]]).astype(np.float32)

    # ── Step 4: Add silhouette boundary for completeness ─────────────
    final_edges = edges_raw.copy()
    if largest_contour is not None:
        cv2.drawContours(final_edges, [largest_contour], -1, 255, thickness=1)
    if capture_stages:
        stages["edges_final"] = final_edges.copy()

    # ── Step 5: Distance transform ────────────────────────────────────
    dist = cv2.distanceTransform(~final_edges, cv2.DIST_L2, 5)
    if capture_stages:
        stages["distance_field"] = dist.astype(np.float32)

        h_img, w_img = gray.shape[:2]

        # ── ⑩ Sub-pixel edge points (rounded) ────────────────────────
        viz = np.zeros((h_img, w_img), dtype=np.uint8)
        if len(edge_points) > 0:
            xs = np.clip(edge_points[:, 0].astype(np.int32), 0, w_img - 1)
            ys = np.clip(edge_points[:, 1].astype(np.int32), 0, h_img - 1)
            viz[ys, xs] = 255
        stages["edge_points_viz"] = viz

        # ── ⑪ Sub-pixel offset colour map ────────────────────────────
        # R = |frac(ex)|*2, G = |frac(ey)|*2, mapped to 0-255.
        # Coloured pixels → real sub-pixel offsets (Devernay working).
        # All black / uniform grey → offsets are zero (something is wrong).
        offset_map = np.zeros((h_img, w_img, 3), dtype=np.uint8)
        if len(edge_points) > 0:
            frac_x = np.abs(edge_points[:, 0] - np.round(edge_points[:, 0]))
            frac_y = np.abs(edge_points[:, 1] - np.round(edge_points[:, 1]))
            xs_i = np.clip(edge_points[:, 0].astype(np.int32), 0, w_img - 1)
            ys_i = np.clip(edge_points[:, 1].astype(np.int32), 0, h_img - 1)
            offset_map[ys_i, xs_i, 0] = (frac_x * 510).clip(0, 255).astype(np.uint8)  # R = offset X
            offset_map[ys_i, xs_i, 1] = (frac_y * 510).clip(0, 255).astype(np.uint8)  # G = offset Y
            offset_map[ys_i, xs_i, 2] = 50  # dim blue baseline so zero-offset pixels show as dark blue
        stages["subpixel_offset_map"] = offset_map

        # ── ⑫ Anti-aliased sub-pixel overlay on CLAHE ─────────────────
        # Uses OpenCV fixed-point SHIFT rendering so dots are drawn at
        # the *true* fractional position — compare with edges_raw to see
        # the sub-pixel smoothness / continuity improvement.
        overlay = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
        if len(edge_points) > 0:
            SHIFT = 4
            SCALE = 1 << SHIFT  # 16
            pts_shifted = (edge_points * SCALE).astype(np.int32)
            radius_shifted = max(1, SCALE // 4)
            step = max(1, len(pts_shifted) // 8000)  # thin for speed
            for pt in pts_shifted[::step]:
                cv2.circle(overlay, (pt[0], pt[1]), radius_shifted,
                           (0, 90, 255), -1, cv2.LINE_AA, shift=SHIFT)
        stages["subpixel_overlay"] = overlay

    result = EdgeResult(
        mask=mask,
        edges=final_edges,
        distance_field=dist.astype(np.float32),
        silhouette_centroid=np.array([cx, cy]),
        edge_points=edge_points,
    )

    if capture_stages:
        return result, stages
    return result
