"""
VideoFIT — Fit Service
Business logic for aligning DXF polylines onto a distance-transform guide map.
The FitResult dataclass (pure structure) lives in app.models.dxf_fitter.

Algorithm (POC-proven)
----------------------
1. Rasterise DXF polylines → edge image → sample points
2. Compute DXF filled centroid and scene silhouette centroid
3. Coarse angle sweep (36 angles)
4. Nelder-Mead refinement (tx, ty, θ)
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize

from app.models.fit_result import FitResult


def fit(
    polylines: list[np.ndarray],
    distance_field: np.ndarray,
    silhouette_mask: np.ndarray,
) -> FitResult:
    """
    Align pixel-space DXF polylines onto the distance field.

    Parameters
    ----------
    polylines : list of Nx2 float32 arrays
        DXF geometry already converted to pixel coordinates.
    distance_field : 2-D float32 array
        Guide map produced by :func:`app.services.edge_service.compute_edges`.
    silhouette_mask : 2-D uint8 array
        Binary mask of the detected piece silhouette.

    Returns
    -------
    FitResult
        Rigid-body transform (tx, ty, angle) that aligns the DXF to the piece.
    """
    H, W = distance_field.shape
    # Smooth the distance field to eliminate noisy local minima
    # that cause the optimizer to settle at slightly different spots each run
    from scipy.ndimage import gaussian_filter
    dist_t = gaussian_filter(distance_field.astype(np.float32), sigma=1.5)

    # ── Step 1: Rasterise DXF polylines → sample edge points ─────────
    dxf_edge_img = np.zeros((H, W), dtype=np.uint8)
    for poly in polylines:
        if len(poly) >= 2:
            cv2.polylines(
                dxf_edge_img,
                [poly.astype(np.int32).reshape(-1, 1, 2)],
                isClosed=False,
                color=255,
                thickness=1,
            )

    dxf_edge_pts = np.column_stack(np.where(dxf_edge_img > 0))[:, [1, 0]].astype(np.float32)

    # Use a larger, deterministic sample for stability
    rng = np.random.default_rng(0)
    rng.shuffle(dxf_edge_pts)
    n_sample = min(2000, len(dxf_edge_pts))
    dxf_sample = dxf_edge_pts[:n_sample]

    # ── Step 2: Compute centroids ─────────────────────────────────────
    dxf_filled = np.zeros((H, W), np.uint8)
    for poly in polylines:
        if len(poly) >= 3:
            cv2.fillPoly(dxf_filled, [poly.astype(np.int32).reshape(-1, 1, 2)], 255)

    M_dxf = cv2.moments(dxf_filled)
    dxf_cx = M_dxf["m10"] / max(M_dxf["m00"], 1)
    dxf_cy = M_dxf["m01"] / max(M_dxf["m00"], 1)

    M_scene = cv2.moments(silhouette_mask)
    scene_cx = M_scene["m10"] / max(M_scene["m00"], 1)
    scene_cy = M_scene["m01"] / max(M_scene["m00"], 1)

    tx_init = scene_cx - dxf_cx
    ty_init = scene_cy - dxf_cy

    # ── Step 3: Chamfer cost function ─────────────────────────────────
    def chamfer_cost(params):
        tx, ty, theta = params
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        xs = dxf_sample[:, 0] - dxf_cx
        ys = dxf_sample[:, 1] - dxf_cy
        nx = cos_t * xs - sin_t * ys + dxf_cx + tx   # float coords
        ny = sin_t * xs + cos_t * ys + dxf_cy + ty   # float coords
        valid = (nx >= 0) & (nx < W) & (ny >= 0) & (ny < H)
        if valid.sum() < 10:
            return 1e6
        # Bilinear interpolation → smooth, continuous surface (no quantisation steps)
        costs = map_coordinates(dist_t, [ny[valid], nx[valid]], order=1, mode="nearest")
        return float(costs.mean())

    # ── Step 4: Coarse angle sweep (72 candidates) — keep top 3 ──────
    sweep_results = []
    for angle_try in np.linspace(-np.pi, np.pi, 72, endpoint=False):
        c = chamfer_cost([tx_init, ty_init, angle_try])
        sweep_results.append((c, [tx_init, ty_init, angle_try]))
    sweep_results.sort(key=lambda x: x[0])
    top_candidates = sweep_results[:3]

    # ── Step 5: Refine each top candidate with Nelder-Mead then Powell
    best_res = None
    for _, candidate_params in top_candidates:
        # Pass 1: Nelder-Mead (robust basin finder)
        r1 = minimize(
            chamfer_cost,
            np.array(candidate_params, dtype=np.float64),
            method="Nelder-Mead",
            options={"xatol": 0.5, "fatol": 0.5, "maxiter": 5000},
        )
        # Pass 2: Powell (deterministic, sub-pixel precision)
        r2 = minimize(
            chamfer_cost,
            r1.x,
            method="Powell",
            options={"xtol": 0.01, "ftol": 0.001, "maxiter": 10000},
        )
        if best_res is None or r2.fun < best_res.fun:
            best_res = r2

    tx_opt, ty_opt, angle_opt = best_res.x

    # ── Step 6: Inlier fraction ───────────────────────────────────────
    cos_t = np.cos(angle_opt)
    sin_t = np.sin(angle_opt)
    xs = dxf_edge_pts[:, 0] - dxf_cx
    ys = dxf_edge_pts[:, 1] - dxf_cy
    nx = np.clip((cos_t * xs - sin_t * ys + dxf_cx + tx_opt).astype(int), 0, W - 1)
    ny = np.clip((sin_t * xs + cos_t * ys + dxf_cy + ty_opt).astype(int), 0, H - 1)
    inlier_frac = float((dist_t[ny, nx] < 3.0).mean())

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

