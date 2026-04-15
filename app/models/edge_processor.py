"""
Metrology Vision Pro — Edge Processor
Two-step edge detection and distance-transform computation.

Faithfully reproduces the proven POC pipeline:
  1. Silhouette mask (largest contour)
  2. Gamma correction → CLAHE → bilateral filter → Canny
  3. Contour outline drawn onto edge map
  4. Masked to silhouette
  5. Distance transform
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class EdgeResult:
    """Intermediate products of the edge-detection pipeline."""
    mask: np.ndarray              # uint8  binary silhouette mask
    edges: np.ndarray             # uint8  Canny edge map (within mask)
    distance_field: np.ndarray    # float32 distance transform
    silhouette_centroid: np.ndarray  # (x, y) in pixel coords


def compute_edges(frame_bgr: np.ndarray) -> EdgeResult:
    """
    Run the full two-step edge pipeline on a **BGR** frame.

    This matches the POC's ``preprocess_image`` + distance transform exactly.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # ── Step 1: Silhouette mask ──────────────────────────────────────
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
    # Gamma correction
    gamma = 0.8
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255 for i in range(256)
    ]).astype("uint8")
    gamma_corrected = cv2.LUT(gray, table)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq = clahe.apply(gamma_corrected)

    # Bilateral filter (edge-preserving smoothing)
    filtered = cv2.bilateralFilter(eq, d=5, sigmaColor=50, sigmaSpace=50)

    # Canny
    edges = cv2.Canny(filtered, 95, 145)

    # Draw the silhouette contour onto edges for better boundary matching
    if largest_contour is not None:
        cv2.drawContours(edges, [largest_contour], -1, 255, thickness=1)

    # Mask edges to the silhouette region only
    final_edges = cv2.bitwise_and(edges, edges, mask=mask)

    # ── Step 3: Distance transform ───────────────────────────────────
    dist = cv2.distanceTransform(~final_edges, cv2.DIST_L2, 5)

    return EdgeResult(
        mask=mask,
        edges=final_edges,
        distance_field=dist.astype(np.float32),
        silhouette_centroid=np.array([cx, cy]),
    )

