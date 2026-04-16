"""
VideoFIT — Fit Service
Aligns DXF polylines onto a distance field built from sub-pixel Devernay edges.

Algorithm
---------
1. Sample DXF polylines densely at sub-pixel spacing (stratified, deterministic).
2. Pre-smooth the distance field once (gaussian) to widen the basin of attraction.
3. Coarse angle sweep (144 angles, vectorised) → keep top 5 candidates.
4. Powell refinement only — fully deterministic, no Nelder-Mead.
5. Cost = trimmed mean of bilinear-interpolated distance field values
   (worst 15 % dropped, soft OOB penalty).

Why map_coordinates instead of KDTree
--------------------------------------
The distance_field is built from the Devernay sub-pixel edge map, so it already
encodes sub-pixel geometry. Bilinear interpolation at float (nx, ny) gives
continuous sub-pixel cost without the O(N·log M) overhead of a KDTree query
inside the optimizer hot loop.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
from scipy.optimize import minimize

from app.models.fit_result import FitResult

# Fraction of worst-distance points ignored in cost (robustness to spurious edges)
_TRIM_FRAC = 0.15


def _sample_polylines(polylines: list[np.ndarray], spacing: float = 0.5) -> np.ndarray:
    """
    Densely and deterministically sample points along DXF polylines.
    Points are placed every *spacing* pixels along each segment — no randomness.
    Returns (N, 2) float32 array of (x, y) positions.
    """
    pts = []
    for poly in polylines:
        if len(poly) < 2:
            continue
        for i in range(len(poly) - 1):
            p0 = poly[i].astype(np.float64)
            p1 = poly[i + 1].astype(np.float64)
            seg_len = np.hypot(p1[0] - p0[0], p1[1] - p0[1])
            n = max(2, int(np.ceil(seg_len / spacing)))
            t = np.linspace(0.0, 1.0, n, endpoint=False)
            pts.append(p0 + np.outer(t, p1 - p0))
    if not pts:
        return np.empty((0, 2), dtype=np.float32)
    return np.vstack(pts).astype(np.float32)


def _stride_subsample(pts: np.ndarray, n: int) -> np.ndarray:
    """
    Return exactly *n* evenly-strided points from *pts*.
    Fully deterministic — no RNG, same result every call.
    """
    if len(pts) <= n:
        return pts
    idx = np.linspace(0, len(pts) - 1, n, dtype=np.int32)
    return pts[idx]


def fit(
    polylines: list[np.ndarray],
    edge_points: np.ndarray,
    silhouette_mask: np.ndarray,
    distance_field: np.ndarray | None = None,
) -> FitResult:
    """
    Align pixel-space DXF polylines onto the sub-pixel distance field.

    Parameters
    ----------
    polylines       : list of Nx2 float32 arrays (DXF geometry in pixel coords).
    edge_points     : (N, 2) float32 — kept for signature compat, not used here.
    silhouette_mask : uint8 binary mask for centroid estimation.
    distance_field  : float32 (H, W) built from Devernay edges (required).
    """
    if distance_field is None:
        raise ValueError("distance_field is required for fitting.")

    H, W = distance_field.shape

    # ── Step 1: Pre-smooth distance field once ────────────────────────
    # Widens the basin of attraction; done here, not inside the cost loop.
    dist_t = gaussian_filter(distance_field.astype(np.float32), sigma=1.5)

    # ── Step 2: Dense deterministic DXF sample ────────────────────────
    dxf_all = _sample_polylines(polylines, spacing=0.5)
    if len(dxf_all) == 0:
        raise ValueError("DXF polylines produced no sample points.")

    # Stride subsample — identical indices every call (no RNG)
    dxf_sample = _stride_subsample(dxf_all, n=4000)

    # ── Step 3: Centroids ─────────────────────────────────────────────
    # Use the mean of all sampled DXF points as centroid (robust for open
    # entities like individual arcs/lines which don't form closed polygons).
    dxf_cx = float(dxf_all[:, 0].mean())
    dxf_cy = float(dxf_all[:, 1].mean())

    M_scene = cv2.moments(silhouette_mask)
    scene_cx = float(M_scene["m10"] / max(M_scene["m00"], 1))
    scene_cy = float(M_scene["m01"] / max(M_scene["m00"], 1))

    tx_init = scene_cx - dxf_cx
    ty_init = scene_cy - dxf_cy

    # ── Step 4: Cost function ─────────────────────────────────────────
    # Pre-centre once — avoids repeated subtraction inside the optimizer loop
    dxf_c = (dxf_sample - np.array([dxf_cx, dxf_cy], dtype=np.float32)).astype(np.float64)

    _OOB_DIST = float(max(H, W))   # soft penalty for out-of-bounds points
    _n_keep   = max(1, int(len(dxf_sample) * (1.0 - _TRIM_FRAC)))

    # Pre-allocate output arrays reused across every cost call
    _all_vals = np.empty(len(dxf_c), dtype=np.float64)

    def cost(params: np.ndarray) -> float:
        tx, ty, theta = float(params[0]), float(params[1]), float(params[2])
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        nx = cos_t * dxf_c[:, 0] - sin_t * dxf_c[:, 1] + dxf_cx + tx
        ny = sin_t * dxf_c[:, 0] + cos_t * dxf_c[:, 1] + dxf_cy + ty

        # Soft boundary: OOB points keep _OOB_DIST (continuous cost surface)
        in_b = (nx >= 0) & (nx < W - 1) & (ny >= 0) & (ny < H - 1)
        _all_vals[:] = _OOB_DIST
        if in_b.sum() < 10:
            return _OOB_DIST

        # Bilinear interpolation on the distance field — sub-pixel, O(N)
        _all_vals[in_b] = map_coordinates(
            dist_t, [ny[in_b], nx[in_b]], order=1, mode="nearest", prefilter=False
        )

        # Trimmed mean: ignore the worst _TRIM_FRAC
        _all_vals.sort()
        return float(_all_vals[:_n_keep].mean())

    # ── Step 5: Vectorised coarse angle sweep (144 angles) ────────────
    angles = np.linspace(-np.pi, np.pi, 144, endpoint=False)
    sweep_costs = np.fromiter((cost([tx_init, ty_init, a]) for a in angles),
                              dtype=np.float64, count=len(angles))
    top_idx = np.argsort(sweep_costs)[:5]
    top_candidates = [[tx_init, ty_init, float(angles[i])] for i in top_idx]

    # ── Step 6: Powell refinement only — deterministic ────────────────
    best_res = None
    for candidate in top_candidates:
        res = minimize(
            cost,
            np.array(candidate, dtype=np.float64),
            method="Powell",
            options={"xtol": 1e-4, "ftol": 1e-5, "maxiter": 20_000},
        )
        if best_res is None or res.fun < best_res.fun:
            best_res = res

    tx_opt, ty_opt, angle_opt = best_res.x

    # ── Step 7: Inlier fraction (threshold = 2 px) ────────────────────
    all_pts = _sample_polylines(polylines, spacing=1.0)
    all_c   = (all_pts - np.array([dxf_cx, dxf_cy], dtype=np.float32)).astype(np.float64)
    cos_t, sin_t = np.cos(angle_opt), np.sin(angle_opt)
    nx_all = cos_t * all_c[:, 0] - sin_t * all_c[:, 1] + dxf_cx + tx_opt
    ny_all = sin_t * all_c[:, 0] + cos_t * all_c[:, 1] + dxf_cy + ty_opt
    valid_all = (nx_all >= 0) & (nx_all < W - 1) & (ny_all >= 0) & (ny_all < H - 1)
    if valid_all.sum() > 0:
        vals = map_coordinates(dist_t, [ny_all[valid_all], nx_all[valid_all]],
                               order=1, mode="nearest", prefilter=False)
        inlier_frac = float(np.mean(vals < 2.0))
    else:
        inlier_frac = 0.0

    return FitResult(
        tx=float(tx_opt),
        ty=float(ty_opt),
        angle_deg=float(np.degrees(angle_opt)),
        cost=float(best_res.fun),
        dxf_cx=dxf_cx,
        dxf_cy=dxf_cy,
        inlier_frac=inlier_frac,
        dist_t=dist_t,
    )
