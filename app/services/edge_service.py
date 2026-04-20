"""
VideoFIT — Edge Service
Business logic for edge detection and distance-transform computation.

Pipeline
--------
1.  Grayscale
2.  Bilateral filter  (d=10, σ_color=120, σ_space=90)
3.  Otsu threshold   → find the largest blob NOT touching any border
                       → silhouette mask (frame / parasites filtered out)
4.  Canny (50 – 150) masked to silhouette
5.  Silhouette contour drawn onto edge map for boundary completeness
6.  Distance transform on the inverted edge map
"""

from __future__ import annotations

import cv2
import numpy as np

from app.models.edge_result import EdgeResult


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _largest_interior_blob(binary: np.ndarray) -> np.ndarray:
    """
    Return a mask containing only the largest connected component in *binary*
    that does **not** touch the image border (border-connected blobs are
    assumed to be the frame or parasitic reflections).

    Falls back to the globally largest component if every component touches
    the border (degenerate image).
    """
    h, w = binary.shape
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    best_label = 0
    best_area = 0
    fallback_label = 0
    fallback_area = 0

    for lbl in range(1, n_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        comp = (labels == lbl).view(np.uint8)
        touches_border = (
            comp[0, :].any()
            or comp[-1, :].any()
            or comp[:, 0].any()
            or comp[:, -1].any()
        )
        if not touches_border and area > best_area:
            best_area = area
            best_label = lbl
        if area > fallback_area:
            fallback_area = area
            fallback_label = lbl

    chosen = best_label if best_label > 0 else fallback_label
    mask = np.zeros((h, w), dtype=np.uint8)
    if chosen > 0:
        mask[labels == chosen] = 255
    return mask


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
    if capture_stages:
        stages["gray"] = gray.copy()

    # ── Step 2: Bilateral filter ──────────────────────────────────────
    bilateral = cv2.bilateralFilter(gray, d=10, sigmaColor=120, sigmaSpace=90)
    if capture_stages:
        stages["bilateral"] = bilateral.copy()

    # ── Step 3: Otsu threshold → largest interior blob = silhouette ───
    _, otsu = cv2.threshold(
        bilateral, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    if capture_stages:
        stages["thresh"] = otsu.copy()

    mask = _largest_interior_blob(otsu)
    if capture_stages:
        stages["mask"] = mask.copy()

    # Silhouette centroid
    M = cv2.moments(mask)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        h_fr, w_fr = gray.shape[:2]
        cx, cy = w_fr / 2.0, h_fr / 2.0

    # Largest contour from mask (for boundary drawing)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea) if contours else None

    # ── Step 4: Canny edge detection on binary image (masked) ────────
    # Running Canny on the Otsu binary guarantees perfectly continuous
    # edges: step gradients (0↔255) are always above any threshold, so
    # NMS and hysteresis never break the contour.
    #
    # IMPORTANT: Canny edge pixels land exactly ON the mask boundary
    # (sometimes 1 px outside due to connectedComponents rounding).
    # Masking with the raw mask would zero those boundary pixels and
    # create artificial discontinuities.  We dilate the mask by 2 px
    # first so no valid edge pixel is ever accidentally clipped.
    # The silhouette contour is redrawn cleanly in Step 5 anyway.
    _dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_dilated = cv2.dilate(mask, _dilate_k, iterations=1)
    edges_raw = cv2.Canny(otsu, 50, 150)
    edges_raw[mask_dilated == 0] = 0    # restrict to silhouette (with 2px margin)
    if capture_stages:
        stages["edges_raw"] = edges_raw.copy()

    # ── Step 5: Add silhouette boundary ──────────────────────────────
    final_edges = edges_raw.copy()
    if largest_contour is not None:
        cv2.drawContours(final_edges, [largest_contour], -1, 255, thickness=1)
    if capture_stages:
        stages["edges_final"] = final_edges.copy()

    # ── Step 6: Distance transform ────────────────────────────────────
    dist = cv2.distanceTransform(~final_edges, cv2.DIST_L2, 5)
    if capture_stages:
        stages["distance_field"] = dist.astype(np.float32)

    # Edge points as (N, 2) float32 [x, y]
    ys_ep, xs_ep = np.where(final_edges > 0)
    edge_points = np.column_stack([xs_ep, ys_ep]).astype(np.float32)

    # ── Debug extra stages ─────────────────────────────────────────────
    if capture_stages:
        h_img, w_img = gray.shape[:2]

        # ⑩ Edge point map
        viz = np.zeros((h_img, w_img), dtype=np.uint8)
        if len(edge_points) > 0:
            xs_i = edge_points[:, 0].astype(np.int32).clip(0, w_img - 1)
            ys_i = edge_points[:, 1].astype(np.int32).clip(0, h_img - 1)
            viz[ys_i, xs_i] = 255
        stages["edge_points_viz"] = viz

        # ⑪ Edge overlay on bilateral image
        overlay = cv2.cvtColor(bilateral, cv2.COLOR_GRAY2BGR)
        overlay[final_edges > 0] = [0, 90, 255]
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
