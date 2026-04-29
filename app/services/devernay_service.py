"""
VideoFIT — Canny-Devernay Sub-pixel Edge Detector
Vectorised NumPy + OpenCV implementation optimised for large sensors.
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
    modG = cv2.magnitude(Gx, Gy)

    # ── 4. Sub-pixel NMS (Optimized Mask-Early approach) ─────────────
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
    """Optimized pure-NumPy sub-pixel NMS (Mask Early, Compute Late)."""
    ex = np.full((h, w), -1.0, dtype=np.float32)
    ey = np.full((h, w), -1.0, dtype=np.float32)

    mod  = modG[1:-1, 1:-1]
    gx_a = np.abs(Gx[1:-1, 1:-1])
    gy_a = np.abs(Gy[1:-1, 1:-1])

    L = modG[1:-1, :-2]
    R = modG[1:-1, 2:]
    D = modG[:-2, 1:-1]
    U = modG[2:,  1:-1]

    # ── Case A: horizontal suppression (|Gx| >= |Gy|) ────────────────
    caseA = (gx_a >= gy_a) & (mod > L) & (mod >= R)
    ys_A, xs_A = np.nonzero(caseA)

    if len(ys_A) > 0:
        mod_A = mod[ys_A, xs_A]
        L_A = L[ys_A, xs_A]
        R_A = R[ys_A, xs_A]

        dA = L_A - 2.0 * mod_A + R_A
        valid_A = np.abs(dA) > 1e-6

        offA = np.zeros_like(dA)
        offA[valid_A] = np.clip(0.5 * (L_A[valid_A] - R_A[valid_A]) / dA[valid_A], -0.5, 0.5)

        ex[ys_A + 1, xs_A + 1] = xs_A + 1 + offA
        ey[ys_A + 1, xs_A + 1] = ys_A + 1

    # ── Case B: vertical suppression (|Gy| > |Gx|) ───────────────────
    caseB = (gy_a > gx_a) & (mod > D) & (mod >= U)
    ys_B, xs_B = np.nonzero(caseB)

    if len(ys_B) > 0:
        mod_B = mod[ys_B, xs_B]
        D_B = D[ys_B, xs_B]
        U_B = U[ys_B, xs_B]

        dB = D_B - 2.0 * mod_B + U_B
        valid_B = np.abs(dB) > 1e-6

        offB = np.zeros_like(dB)
        offB[valid_B] = np.clip(0.5 * (D_B[valid_B] - U_B[valid_B]) / dB[valid_B], -0.5, 0.5)

        ex[ys_B + 1, xs_B + 1] = xs_B + 1
        ey[ys_B + 1, xs_B + 1] = ys_B + 1 + offB

    return ex, ey


def _hysteresis(
    ex: np.ndarray,
    ey: np.ndarray,
    modG: np.ndarray,
    high: float,
    low: float,
) -> tuple[np.ndarray, np.ndarray]:
    has_edge = ex >= 0.0
    strong = has_edge & (modG >= high)
    weak   = has_edge & (modG >= low)

    _, labels = cv2.connectedComponents(weak.view(np.uint8), connectivity=8)

    strong_labels = np.unique(labels[strong])
    keep_lut = np.zeros(labels.max() + 1, dtype=bool)
    keep_lut[strong_labels] = True
    keep_lut[0] = False

    keep = keep_lut[labels]
    ex = np.where(keep, ex, np.float32(-1.0))
    ey = np.where(keep, ey, np.float32(-1.0))
    return ex, ey


def _render_edge_map(ex: np.ndarray, ey: np.ndarray, h: int, w: int) -> np.ndarray:
    edge_map = np.zeros((h, w), dtype=np.uint8)
    valid = ex >= 0.0
    ys = np.round(ey[valid]).astype(np.int32).clip(0, h - 1)
    xs = np.round(ex[valid]).astype(np.int32).clip(0, w - 1)
    edge_map[ys, xs] = 255
    return edge_map