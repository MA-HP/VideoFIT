"""
VideoFIT — Fit Service
Aligns DXF polylines onto a distance field built from sub-pixel Devernay edges.

Algorithm
---------
1. Sample DXF polylines densely at sub-pixel spacing (stratified, deterministic).
2. Pre-smooth the distance field once (gaussian) to widen the basin of attraction.
3. Coarse angle sweep (180 angles, vectorised) → keep top 10 candidates.
4. Powell refinement only — fully deterministic, no Nelder-Mead.
5. Cost = trimmed mean of bilinear-interpolated distance field values
   (worst 10 % dropped, soft OOB penalty).

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
_TRIM_FRAC = 0.10


def _make_cost_fn(
    dxf_c: np.ndarray,
    dxf_cx: float,
    dxf_cy: float,
    dist_t: np.ndarray,
    objective: str = "Strict",
    max_error_px: float = 2.0,
    locked_theta: float | None = None,
) -> callable:
    """
    Build and return a cost function for the optimizer.

    Parameters
    ----------
    dxf_c         : (N, 2) centred DXF sample points.
    dxf_cx/cy     : DXF centroid used to un-centre.
    dist_t        : Smoothed distance field (H, W).
    objective     : "Strict"    → trimmed mean of distances (minimise).
                    "Tolerance" → composite cost: primary hinge loss (tolerance violations
                                  weighted 10×) + secondary trimmed mean (precision within
                                  tolerance). Still aims for the best fit, but strongly
                                  prefers placements where all points are within max_error_px.
    max_error_px  : Tolerance threshold in pixels (Tolerance mode only).
    locked_theta  : If not None, θ is fixed and params are (tx, ty) only.
    """
    H, W = dist_t.shape
    _OOB_DIST = float(max(H, W))
    _n_keep = max(1, int(len(dxf_c) * (1.0 - _TRIM_FRAC)))
    _all_vals = np.empty(len(dxf_c), dtype=np.float64)

    def _sample_dist(nx, ny):
        """Bilinear-interpolate dist_t at (nx, ny); OOB points get _OOB_DIST."""
        in_b = (nx >= 0) & (nx < W - 1) & (ny >= 0) & (ny < H - 1)
        out = np.full(len(nx), _OOB_DIST, dtype=np.float64)
        if in_b.sum() >= 10:
            out[in_b] = map_coordinates(
                dist_t, [ny[in_b], nx[in_b]], order=1, mode="nearest", prefilter=False
            )
        return out, in_b

    def _transform(tx, ty, theta):
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        nx = cos_t * dxf_c[:, 0] - sin_t * dxf_c[:, 1] + dxf_cx + tx
        ny = sin_t * dxf_c[:, 0] + cos_t * dxf_c[:, 1] + dxf_cy + ty
        return nx, ny

    if locked_theta is not None:
        theta_fixed = float(locked_theta)
        cos_l = np.cos(theta_fixed)
        sin_l = np.sin(theta_fixed)
        nx_base = cos_l * dxf_c[:, 0] - sin_l * dxf_c[:, 1] + dxf_cx
        ny_base = sin_l * dxf_c[:, 0] + cos_l * dxf_c[:, 1] + dxf_cy

        if objective == "Tolerance":
            def cost_fn(params):
                tx, ty = float(params[0]), float(params[1])
                nx, ny = nx_base + tx, ny_base + ty
                vals, in_b = _sample_dist(nx, ny)
                if in_b.sum() < 10:
                    return _OOB_DIST
                # Primary: hinge loss (penalise tolerance violations heavily)
                hinge = float(np.mean(np.maximum(0.0, vals - max_error_px)))
                # Secondary: trimmed mean (precision — still aims for best fit)
                sorted_vals = np.sort(vals)
                precision = float(sorted_vals[:_n_keep].mean())
                return 10.0 * hinge + precision
        else:
            def cost_fn(params):
                tx, ty = float(params[0]), float(params[1])
                nx, ny = nx_base + tx, ny_base + ty
                vals, in_b = _sample_dist(nx, ny)
                if in_b.sum() < 10:
                    return _OOB_DIST
                vals.sort()
                # Trimmed mean + soft penalty for the worst 10% to preserve gradient
                return float(vals[:_n_keep].mean() + 0.1 * vals[_n_keep:].mean())
    else:
        if objective == "Tolerance":
            def cost_fn(params):
                tx, ty, theta = float(params[0]), float(params[1]), float(params[2])
                nx, ny = _transform(tx, ty, theta)
                vals, in_b = _sample_dist(nx, ny)
                if in_b.sum() < 10:
                    return _OOB_DIST
                hinge = float(np.mean(np.maximum(0.0, vals - max_error_px)))
                sorted_vals = np.sort(vals)
                precision = float(sorted_vals[:_n_keep].mean())
                return 10.0 * hinge + precision
        else:
            def cost_fn(params):
                tx, ty, theta = float(params[0]), float(params[1]), float(params[2])
                nx, ny = _transform(tx, ty, theta)
                vals, in_b = _sample_dist(nx, ny)
                if in_b.sum() < 10:
                    return _OOB_DIST
                vals.sort()
                # Trimmed mean + soft penalty for the worst 10% to preserve gradient
                return float(vals[:_n_keep].mean() + 0.1 * vals[_n_keep:].mean())

    return cost_fn


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
    objective: str = "Strict",
    max_error_px: float = 2.0,
) -> FitResult:
    """
    Align pixel-space DXF polylines onto the sub-pixel distance field.

    Parameters
    ----------
    polylines       : list of Nx2 float32 arrays (DXF geometry in pixel coords).
    edge_points     : (N, 2) float32 — kept for signature compat, not used here.
    silhouette_mask : uint8 binary mask for centroid estimation.
    distance_field  : float32 (H, W) built from Devernay edges (required).
    objective       : "Strict" (minimise trimmed mean distance) or
                      "Tolerance" (maximise inlier fraction within max_error_px).
    max_error_px    : Inlier threshold in pixels (Tolerance mode only).
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

    cost = _make_cost_fn(dxf_c, dxf_cx, dxf_cy, dist_t,
                         objective=objective, max_error_px=max_error_px)

    # ── Step 5: Vectorised coarse angle sweep (180 angles) ────────────
    angles = np.linspace(-np.pi, np.pi, 180, endpoint=False)
    sweep_costs = np.fromiter((cost([tx_init, ty_init, a]) for a in angles),
                              dtype=np.float64, count=len(angles))
    top_idx = np.argsort(sweep_costs)[:10]
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

    # ── Step 7: Inlier fraction (threshold = max_error_px) ───────────
    all_pts = _sample_polylines(polylines, spacing=1.0)
    all_c   = (all_pts - np.array([dxf_cx, dxf_cy], dtype=np.float32)).astype(np.float64)
    cos_t, sin_t = np.cos(angle_opt), np.sin(angle_opt)
    nx_all = cos_t * all_c[:, 0] - sin_t * all_c[:, 1] + dxf_cx + tx_opt
    ny_all = sin_t * all_c[:, 0] + cos_t * all_c[:, 1] + dxf_cy + ty_opt
    valid_all = (nx_all >= 0) & (nx_all < W - 1) & (ny_all >= 0) & (ny_all < H - 1)
    if valid_all.sum() > 0:
        vals = map_coordinates(dist_t, [ny_all[valid_all], nx_all[valid_all]],
                               order=1, mode="nearest", prefilter=False)
        inlier_frac = float(np.mean(vals < max_error_px))
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


# ======================================================================
# Refine mode — 3-step pipeline (GLOBAL+REFINE → GLOBAL+REFINE → REFINE)
# ======================================================================

def _refine_from(
        polylines: list[np.ndarray],
        init_params: np.ndarray,
        dist_t: np.ndarray,
        n_sample: int = 4000,
        maxiter: int = 20_000,
        objective: str = "Strict",
        max_error_px: float = 2.0,
        dxf_cx: float | None = None,
        dxf_cy: float | None = None,
) -> tuple:
    """
    Run a single Powell refinement starting from *init_params*.
    Returns (optimized_params, cost, dxf_cx, dxf_cy).
    """
    H, W = dist_t.shape
    dxf_all = _sample_polylines(polylines, spacing=0.5)
    if len(dxf_all) == 0:
        raise ValueError("Polylines produced no sample points.")

    dxf_sample = _stride_subsample(dxf_all, n=n_sample)

    # Use provided global centroids, otherwise compute local ones
    if dxf_cx is None or dxf_cy is None:
        dxf_cx = float(dxf_all[:, 0].mean())
        dxf_cy = float(dxf_all[:, 1].mean())

    dxf_c = (dxf_sample - np.array([dxf_cx, dxf_cy], dtype=np.float32)).astype(np.float64)

    cost_fn = _make_cost_fn(dxf_c, dxf_cx, dxf_cy, dist_t,
                            objective=objective, max_error_px=max_error_px)

    res = minimize(
        cost_fn,
        init_params.astype(np.float64),
        method="Powell",
        options={"xtol": 1e-4, "ftol": 1e-5, "maxiter": maxiter},
    )
    return res.x, res.fun, dxf_cx, dxf_cy


def fit_complete(
        polylines_all: list[np.ndarray],
        polylines_refine: list[np.ndarray],
        edge_points: np.ndarray,
        silhouette_mask: np.ndarray,
        distance_field: np.ndarray | None = None,
        polylines_rot: list[np.ndarray] | None = None,
        polylines_pan: list[np.ndarray] | None = None,
        objective: str = "Strict",
        max_error_px: float = 2.0,
) -> FitResult:
    if distance_field is None:
        raise ValueError("distance_field is required for fitting.")

    effective_refine = polylines_refine
    if not effective_refine:
        effective_refine = (polylines_rot or []) + (polylines_pan or [])
        if effective_refine:
            print("Refine mode: no REFINE layer found — using ROT + PAN as fallback.")

    # ── Step 1: Coarse fit on all layers ──────────────────────────────
    result_coarse = fit(
        polylines=polylines_all,
        edge_points=edge_points,
        silhouette_mask=silhouette_mask,
        distance_field=distance_field,
        objective=objective,
        max_error_px=max_error_px,
    )
    dist_t = result_coarse.dist_t
    angle_rad = np.radians(result_coarse.angle_deg)
    params_1 = np.array([result_coarse.tx, result_coarse.ty, angle_rad])

    # CAPTURE GLOBAL CENTROID
    global_cx = result_coarse.dxf_cx
    global_cy = result_coarse.dxf_cy

    # ── Step 2: Refine on all layers ──────────────────────────────────
    params_2, cost_2, _, _ = _refine_from(
        polylines_all, params_1, dist_t, n_sample=6000,
        objective=objective, max_error_px=max_error_px,
        dxf_cx=global_cx, dxf_cy=global_cy
    )

    # ── Step 3: Finetune on REFINE layer ─────────────────────────────
    if effective_refine:
        params_3, cost_3, _, _ = _refine_from(
            effective_refine, params_2, dist_t, n_sample=6000,
            objective=objective, max_error_px=max_error_px,
            dxf_cx=global_cx, dxf_cy=global_cy
        )
    else:
        params_3, cost_3 = params_2, cost_2

    inlier_frac = _compute_inlier_frac(
        polylines_all, params_3, dist_t,
        dxf_cx=global_cx, dxf_cy=global_cy,
        max_error_px=max_error_px
    )

    return FitResult(
        tx=float(params_3[0]),
        ty=float(params_3[1]),
        angle_deg=float(np.degrees(params_3[2])),
        cost=float(cost_3),
        dxf_cx=global_cx,
        dxf_cy=global_cy,
        inlier_frac=inlier_frac,
        dist_t=dist_t,
    )


# ======================================================================
# POC mode — 4-step pipeline (Position / Orientation / Centre)
# Layers: GLOBAL, ROT, PAN
# Step 1: Coarse   — all layers (GLOBAL + ROT + PAN), full (tx, ty, θ)
# Step 2: Refine   — all layers (GLOBAL + ROT + PAN), full (tx, ty, θ)
# Step 3: Finetune — ROT + PAN layers only,           full (tx, ty, θ)
# Step 4: Pan      — PAN layer only,                  translation only (tx, ty), θ locked
# ======================================================================

def _refine_translation_only(
        polylines: list[np.ndarray],
        init_params: np.ndarray,
        dist_t: np.ndarray,
        n_sample: int = 6000,
        maxiter: int = 20_000,
        objective: str = "Strict",
        max_error_px: float = 2.0,
        dxf_cx: float | None = None,
        dxf_cy: float | None = None,
) -> tuple:
    """
    Powell refinement optimising **only tx, ty** while keeping θ locked.
    Returns (optimized_params_3d, cost, dxf_cx, dxf_cy).
    """
    H, W = dist_t.shape
    dxf_all = _sample_polylines(polylines, spacing=0.5)
    if len(dxf_all) == 0:
        raise ValueError("Polylines produced no sample points.")

    dxf_sample = _stride_subsample(dxf_all, n=n_sample)

    # Use provided global centroids, otherwise compute local ones
    if dxf_cx is None or dxf_cy is None:
        dxf_cx = float(dxf_all[:, 0].mean())
        dxf_cy = float(dxf_all[:, 1].mean())

    dxf_c = (dxf_sample - np.array([dxf_cx, dxf_cy], dtype=np.float32)).astype(np.float64)

    theta_locked = float(init_params[2])
    cost_fn = _make_cost_fn(dxf_c, dxf_cx, dxf_cy, dist_t,
                            objective=objective, max_error_px=max_error_px,
                            locked_theta=theta_locked)

    res = minimize(
        cost_fn,
        np.array([init_params[0], init_params[1]], dtype=np.float64),
        method="Powell",
        options={"xtol": 1e-4, "ftol": 1e-5, "maxiter": maxiter},
    )
    out_params = np.array([res.x[0], res.x[1], theta_locked])
    return out_params, res.fun, dxf_cx, dxf_cy


def _compute_inlier_frac(
        polylines: list[np.ndarray],
        params: np.ndarray,
        dist_t: np.ndarray,
        dxf_cx: float,
        dxf_cy: float,
        max_error_px: float = 2.0,
) -> float:
    """Compute inlier fraction (threshold = max_error_px) for given polylines + transform."""
    H, W = dist_t.shape
    all_pts = _sample_polylines(polylines, spacing=1.0)
    if len(all_pts) == 0:
        return 0.0

    # Removed local centroid calculation to enforce global coordinate frame
    all_c = (all_pts - np.array([dxf_cx, dxf_cy], dtype=np.float32)).astype(np.float64)
    tx, ty, angle = float(params[0]), float(params[1]), float(params[2])
    cos_t, sin_t = np.cos(angle), np.sin(angle)
    nx_all = cos_t * all_c[:, 0] - sin_t * all_c[:, 1] + dxf_cx + tx
    ny_all = sin_t * all_c[:, 0] + cos_t * all_c[:, 1] + dxf_cy + ty
    valid = (nx_all >= 0) & (nx_all < W - 1) & (ny_all >= 0) & (ny_all < H - 1)
    if valid.sum() > 0:
        vals = map_coordinates(dist_t, [ny_all[valid], nx_all[valid]],
                               order=1, mode="nearest", prefilter=False)
        return float(np.mean(vals < max_error_px))
    return 0.0


def fit_poc(
        polylines_all: list[np.ndarray],
        polylines_rot: list[np.ndarray],
        polylines_pan: list[np.ndarray],
        edge_points: np.ndarray,
        silhouette_mask: np.ndarray,
        distance_field: np.ndarray | None = None,
        objective: str = "Strict",
        max_error_px: float = 2.0,
) -> FitResult:
    if distance_field is None:
        raise ValueError("distance_field is required for fitting.")

    # ── Step 1: Coarse fit on all layers ──────────────────────────────
    result_coarse = fit(
        polylines=polylines_all,
        edge_points=edge_points,
        silhouette_mask=silhouette_mask,
        distance_field=distance_field,
        objective=objective,
        max_error_px=max_error_px,
    )
    dist_t = result_coarse.dist_t
    angle_rad = np.radians(result_coarse.angle_deg)
    params_1 = np.array([result_coarse.tx, result_coarse.ty, angle_rad])

    # CAPTURE GLOBAL CENTROID
    global_cx = result_coarse.dxf_cx
    global_cy = result_coarse.dxf_cy

    # ── Step 2: Refine on all layers ──────────────────────────────────
    params_2, cost_2, _, _ = _refine_from(
        polylines_all, params_1, dist_t, n_sample=6000,
        objective=objective, max_error_px=max_error_px,
        dxf_cx=global_cx, dxf_cy=global_cy
    )

    # ── Step 3: Finetune on ROT layer only ───────────────────────────
    if polylines_rot:
        params_3, cost_3, _, _ = _refine_from(
            polylines_rot, params_2, dist_t, n_sample=6000,
            objective=objective, max_error_px=max_error_px,
            dxf_cx=global_cx, dxf_cy=global_cy
        )
    else:
        params_3, cost_3 = params_2, cost_2

    # ── Step 4: Pan on PAN layer only ────────────────────────────────
    if polylines_pan:
        params_4, cost_4, _, _ = _refine_translation_only(
            polylines_pan, params_3, dist_t, n_sample=6000,
            objective=objective, max_error_px=max_error_px,
            dxf_cx=global_cx, dxf_cy=global_cy
        )
    else:
        params_4, cost_4 = params_3, cost_3

    inlier_frac = _compute_inlier_frac(
        polylines_all, params_4, dist_t,
        dxf_cx=global_cx, dxf_cy=global_cy,
        max_error_px=max_error_px
    )

    return FitResult(
        tx=float(params_4[0]),
        ty=float(params_4[1]),
        angle_deg=float(np.degrees(params_4[2])),
        cost=float(cost_4),
        dxf_cx=global_cx,
        dxf_cy=global_cy,
        inlier_frac=inlier_frac,
        dist_t=dist_t,
    )
