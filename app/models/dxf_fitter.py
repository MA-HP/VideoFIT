"""
Metrology Vision Pro — DXF Fitter
Aligns DXF polylines (already in pixel space) onto a distance-transform
guide map using the proven POC approach:

  1. Rasterise DXF polylines → edge image → sample points
  2. Compute DXF filled centroid and scene silhouette centroid
  3. Coarse angle sweep (36 angles)
  4. Nelder-Mead refinement (tx, ty, θ)
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import minimize


@dataclass
class FitResult:
    """Rigid-body transform that maps DXF pixel coords → aligned pixel coords."""
    tx: float          # translation X (pixels)
    ty: float          # translation Y (pixels)
    angle_deg: float   # rotation (degrees)
    cost: float        # final mean distance-transform cost
    dxf_cx: float      # DXF centroid X used for rotation pivot (pixels)
    dxf_cy: float      # DXF centroid Y used for rotation pivot (pixels)
    inlier_frac: float # fraction of DXF points within 3px of edges


def fit(
    polylines: list[np.ndarray],
    distance_field: np.ndarray,
    silhouette_mask: np.ndarray,
) -> FitResult:
    """
    Align pixel-space DXF polylines onto the distance field.

    This reproduces the POC's ``align()`` function exactly.

    Parameters
    ----------
    polylines : list of Nx2 float32 arrays
        DXF geometry already converted to pixel coordinates.
    distance_field : 2-D float32 array
        Guide map from :func:`edge_processor.compute_edges`.
    silhouette_mask : 2-D uint8 array
        Binary mask of the detected piece silhouette.

    Returns
    -------
    FitResult
    """
    H, W = distance_field.shape
    dist_t = distance_field

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

    # Subsample for speed
    rng = np.random.default_rng(0)
    rng.shuffle(dxf_edge_pts)
    n_sample = min(800, len(dxf_edge_pts))
    dxf_sample = dxf_edge_pts[:n_sample]

    # ── Step 2: Compute centroids ────────────────────────────────────
    # DXF filled centroid
    dxf_filled = np.zeros((H, W), np.uint8)
    for poly in polylines:
        if len(poly) >= 3:
            cv2.fillPoly(
                dxf_filled,
                [poly.astype(np.int32).reshape(-1, 1, 2)],
                255,
            )
    M_dxf = cv2.moments(dxf_filled)
    dxf_cx = M_dxf["m10"] / max(M_dxf["m00"], 1)
    dxf_cy = M_dxf["m01"] / max(M_dxf["m00"], 1)

    # Scene silhouette centroid
    M_scene = cv2.moments(silhouette_mask)
    scene_cx = M_scene["m10"] / max(M_scene["m00"], 1)
    scene_cy = M_scene["m01"] / max(M_scene["m00"], 1)

    tx_init = scene_cx - dxf_cx
    ty_init = scene_cy - dxf_cy

    # ── Step 3: Chamfer cost function ────────────────────────────────
    def chamfer_cost(params):
        tx, ty, theta = params
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        xs = dxf_sample[:, 0] - dxf_cx
        ys = dxf_sample[:, 1] - dxf_cy
        nx = (cos_t * xs - sin_t * ys + dxf_cx + tx).astype(int)
        ny = (sin_t * xs + cos_t * ys + dxf_cy + ty).astype(int)
        valid = (nx >= 0) & (nx < W) & (ny >= 0) & (ny < H)
        if valid.sum() < 10:
            return 1e6
        return float(dist_t[ny[valid], nx[valid]].mean())

    # ── Step 4: Coarse angle sweep ───────────────────────────────────
    best_cost = 1e9
    best_params = [tx_init, ty_init, 0.0]
    for angle_try in np.linspace(-np.pi, np.pi, 36, endpoint=False):
        c = chamfer_cost([tx_init, ty_init, angle_try])
        if c < best_cost:
            best_cost = c
            best_params = [tx_init, ty_init, angle_try]

    # ── Step 5: Nelder-Mead refinement ───────────────────────────────
    res = minimize(
        chamfer_cost,
        best_params,
        method="Nelder-Mead",
        options={"xatol": 0.05, "fatol": 0.05, "maxiter": 10000},
    )
    tx_opt, ty_opt, angle_opt = res.x

    # ── Step 6: Compute inlier fraction ──────────────────────────────
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
        cost=float(res.fun),
        dxf_cx=dxf_cx,
        dxf_cy=dxf_cy,
        inlier_frac=inlier_frac,
    )

