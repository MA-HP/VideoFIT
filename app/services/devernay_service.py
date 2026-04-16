"""
VideoFIT — Canny-Devernay Sub-pixel Edge Detector
Vectorised NumPy + OpenCV implementation optimised for large sensors.

Reference
---------
F. Devernay, "A Non-Maxima Suppression Method for Edge Detection with
Sub-Pixel Accuracy", INRIA Research Report RR-2724, 1995.
"""

from __future__ import annotations

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def devernay_edges(
    gray: np.ndarray,
    sigma: float = 0.0,
    high_thresh: float = 20.0,
    low_thresh: float = 10.0,
    mask: np.ndarray | None = None,
    downsample: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sub-pixel Canny-Devernay edge detection.

    Parameters
    ----------
    gray        : uint8 2-D image (H, W).
    sigma       : Gaussian pre-smoothing (0 → skip).
    high_thresh : Hysteresis high threshold on gradient magnitude.
    low_thresh  : Hysteresis low threshold on gradient magnitude.
    mask        : Optional uint8 binary mask (full-res).
    downsample  : Scale factor applied before detection (0.5 = half-res).
                  Sub-pixel coordinates are scaled back to full resolution.

    Returns
    -------
    ex, ey   : float32 arrays (H, W) sub-pixel coords (−1 where no edge).
    edge_map : uint8 binary image (H, W).
    """
    H, W = gray.shape[:2]

    # ── 1. Downsample ────────────────────────────────────────────────
    if downsample != 1.0:
        small_w = max(1, int(W * downsample))
        small_h = max(1, int(H * downsample))
        small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
        small_mask = (
            cv2.resize(mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
            if mask is not None else None
        )
    else:
        small, small_mask = gray, mask
        small_h, small_w = H, W

    # ── 2. Pre-smoothing ─────────────────────────────────────────────
    img = small.astype(np.float32)
    if sigma > 0.0:
        ks = int(2 * np.ceil(3 * sigma) + 1) | 1
        img = cv2.GaussianBlur(img, (ks, ks), sigma)

    # ── 3. Gradient via Sobel (OpenCV SIMD) ──────────────────────────
    Gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    Gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    modG = cv2.magnitude(Gx, Gy)   # in-place via OpenCV, avoids np.hypot copy

    # ── 4. Sub-pixel NMS ─────────────────────────────────────────────
    ex_s, ey_s = _subpixel_nms(Gx, Gy, modG, small_h, small_w)

    # ── 5. Hysteresis ────────────────────────────────────────────────
    ex_s, ey_s = _hysteresis(ex_s, ey_s, modG, high_thresh, low_thresh)

    # ── 6. Apply mask (small-res) ────────────────────────────────────
    if small_mask is not None:
        ex_s[small_mask == 0] = -1.0
        ey_s[small_mask == 0] = -1.0

    # ── 7. Scale coordinates back to full resolution ─────────────────
    if downsample != 1.0:
        inv = 1.0 / downsample
        valid = ex_s >= 0.0
        ex = np.full((H, W), -1.0, dtype=np.float32)
        ey = np.full((H, W), -1.0, dtype=np.float32)
        # Map small-pixel coords → full-res sub-pixel coords
        ys_s = np.round(ey_s[valid]).astype(np.int32).clip(0, small_h - 1)
        xs_s = np.round(ex_s[valid]).astype(np.int32).clip(0, small_w - 1)
        ys_f = np.round(ey_s[valid] * inv).astype(np.int32).clip(0, H - 1)
        xs_f = np.round(ex_s[valid] * inv).astype(np.int32).clip(0, W - 1)
        ex[ys_f, xs_f] = ex_s[ys_s, xs_s] * inv
        ey[ys_f, xs_f] = ey_s[ys_s, xs_s] * inv
    else:
        ex, ey = ex_s, ey_s

    # ── 8. Render edge map ───────────────────────────────────────────
    edge_map = _render_edge_map(ex, ey, H, W)

    return ex, ey, edge_map


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _subpixel_nms(
    Gx: np.ndarray,
    Gy: np.ndarray,
    modG: np.ndarray,
    h: int,
    w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised sub-pixel non-maximum suppression (Devernay parabola fit)."""
    ex = np.full((h, w), -1.0, dtype=np.float32)
    ey = np.full((h, w), -1.0, dtype=np.float32)

    # Work on interior [1:-1, 1:-1] — one-pixel border is enough
    mod  = modG[1:-1, 1:-1]
    gx_a = np.abs(Gx[1:-1, 1:-1])
    gy_a = np.abs(Gy[1:-1, 1:-1])

    # Neighbours (no extra padding needed — slice directly from modG)
    L = modG[1:-1, :-2]    # x-1
    R = modG[1:-1, 2:]     # x+1
    D = modG[:-2, 1:-1]    # y-1
    U = modG[2:,  1:-1]    # y+1

    # ── Case A: horizontal suppression (|Gx| >= |Gy|) ────────────────
    caseA = (gx_a >= gy_a) & (mod > L) & (mod >= R)
    dA = L - 2.0 * mod + R
    offA = np.where(np.abs(dA) > 1e-6, np.clip(0.5 * (L - R) / np.where(np.abs(dA) > 1e-6, dA, 1.0), -0.5, 0.5), 0.0)
    ys_A, xs_A = np.where(caseA)
    ex[ys_A + 1, xs_A + 1] = xs_A + 1 + offA[ys_A, xs_A]
    ey[ys_A + 1, xs_A + 1] = ys_A + 1

    # ── Case B: vertical suppression (|Gy| > |Gx|) ───────────────────
    caseB = (gy_a > gx_a) & (mod > D) & (mod >= U)
    dB = D - 2.0 * mod + U
    offB = np.where(np.abs(dB) > 1e-6, np.clip(0.5 * (D - U) / np.where(np.abs(dB) > 1e-6, dB, 1.0), -0.5, 0.5), 0.0)
    ys_B, xs_B = np.where(caseB)
    ex[ys_B + 1, xs_B + 1] = xs_B + 1
    ey[ys_B + 1, xs_B + 1] = ys_B + 1 + offB[ys_B, xs_B]

    return ex, ey


def _hysteresis(
    ex: np.ndarray,
    ey: np.ndarray,
    modG: np.ndarray,
    high: float,
    low: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Hysteresis via OpenCV connectedComponents — O(N), no Python BFS.
    Only connected components that contain at least one strong pixel are kept.
    """
    has_edge = ex >= 0.0
    strong = has_edge & (modG >= high)
    weak   = has_edge & (modG >= low)        # superset of strong

    # Label connected weak regions; keep those touching a strong pixel
    _, labels = cv2.connectedComponents(weak.view(np.uint8), connectivity=8)

    # Unique labels that contain a strong pixel (fast: boolean index → unique)
    strong_labels = np.unique(labels[strong])
    # Build a keep-lookup via a boolean array indexed by label id
    keep_lut = np.zeros(labels.max() + 1, dtype=bool)
    keep_lut[strong_labels] = True
    keep_lut[0] = False   # background

    keep = keep_lut[labels]
    ex = np.where(keep, ex, np.float32(-1.0))
    ey = np.where(keep, ey, np.float32(-1.0))
    return ex, ey


def _render_edge_map(ex: np.ndarray, ey: np.ndarray, h: int, w: int) -> np.ndarray:
    """Round sub-pixel coordinates to the nearest pixel and paint a binary map."""
    edge_map = np.zeros((h, w), dtype=np.uint8)
    valid = ex >= 0.0
    ys = np.round(ey[valid]).astype(np.int32).clip(0, h - 1)
    xs = np.round(ex[valid]).astype(np.int32).clip(0, w - 1)
    edge_map[ys, xs] = 255
    return edge_map
