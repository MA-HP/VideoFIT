"""
VideoFIT — Edge Service
Business logic for edge detection and distance-transform computation.
The EdgeResult dataclass (pure structure) lives in app.models.edge_processor.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.models.edge_result import EdgeResult


def compute_edges(frame_bgr: np.ndarray) -> EdgeResult:
    """
    Run the full two-step edge pipeline on a **BGR** frame.

    Steps
    -----
    1. Silhouette mask  (largest contour via adaptive threshold)
    2. Gamma correction → CLAHE → bilateral filter → Canny
    3. Silhouette contour drawn onto edge map for boundary completeness
    4. Edges masked to the silhouette region
    5. Distance transform on the inverted edge map

    Parameters
    ----------
    frame_bgr : np.ndarray
        Input image in BGR colour order (as delivered by OpenCV / IC4).

    Returns
    -------
    EdgeResult
        Mask, edge map, distance field, and silhouette centroid.
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

    # ── Step 2: Enhanced edge detection ──────────────────────────────
    gamma = 0.8
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    ).astype("uint8")
    gamma_corrected = cv2.LUT(gray, table)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq = clahe.apply(gamma_corrected)

    filtered = cv2.bilateralFilter(eq, d=5, sigmaColor=50, sigmaSpace=50)
    edges = cv2.Canny(filtered, 95, 145)

    if largest_contour is not None:
        cv2.drawContours(edges, [largest_contour], -1, 255, thickness=1)

    final_edges = cv2.bitwise_and(edges, edges, mask=mask)

    # ── Step 3: Distance transform ────────────────────────────────────
    dist = cv2.distanceTransform(~final_edges, cv2.DIST_L2, 5)

    return EdgeResult(
        mask=mask,
        edges=final_edges,
        distance_field=dist.astype(np.float32),
        silhouette_centroid=np.array([cx, cy]),
    )

