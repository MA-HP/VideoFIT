"""
VideoFIT — Fit Service
Aligns DXF polylines onto a distance field built from sub-pixel Devernay edges.

Texture-Robust Metrology Refactor:
- Removed heavy EDT blurring (which creates sinkholes in texture regions).
- Dual-Loss Pipeline: Uses Huber loss for global pulling, and Welsch (redescending)
  loss for final refinement to mathematically ignore texture and outliers without
  creating cliffs for the Powell optimizer.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
from scipy.optimize import minimize

from app.models.fit_result import FitResult


def _make_cost_fn(
        dxf_c: np.ndarray,
        dxf_cx: float,
        dxf_cy: float,
        dist_t: np.ndarray,
        loss_type: str = "Huber",  # "Huber" (pulls globally) or "Welsch" (ignores texture)
        objective: str = "Tolerance",
        max_error_px: float = 1.0,
        locked_theta: float | None = None,
        interp_order: int = 1,
) -> callable:
    H, W = dist_t.shape
    _OOB_DIST = float(max(H, W))

    def _sample_dist(nx, ny):
        in_b = (nx >= 0) & (nx < W - 1) & (ny >= 0) & (ny < H - 1)
        out = np.full(len(nx), _OOB_DIST, dtype=np.float64)
        if in_b.sum() >= 10:
            out[in_b] = map_coordinates(
                dist_t, [ny[in_b], nx[in_b]],
                order=interp_order, mode="nearest", prefilter=False
            )
        return out, in_b

    def _transform(tx, ty, theta):
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        nx = cos_t * dxf_c[:, 0] - sin_t * dxf_c[:, 1] + dxf_cx + tx
        ny = sin_t * dxf_c[:, 0] + cos_t * dxf_c[:, 1] + dxf_cy + ty
        return nx, ny

    def _calc_huber_cost(vals):
        """Pulls from anywhere in the image. Great for initial sweeps."""
        delta = max(0.5, max_error_px)
        weight = 0.5 if objective == "Tolerance" else 1.0
        return float(np.mean(weight * delta**2 * (np.sqrt(1 + (vals / delta)**2) - 1)))

    def _calc_welsch_cost(vals):
        """
        Redescending Robust Loss.
        As 'vals' exceeds max_error_px, the penalty caps out at 1.0 and the gradient
        drops to 0. This makes distant texture completely invisible to the optimizer.
        """
        delta = max(0.5, max_error_px)
        # 1.0 - exp(...) maps distance 0 -> 0, and distance infinity -> 1.0
        return float(np.mean(1.0 - np.exp(-0.5 * (vals / delta)**2)))

    _loss_func = _calc_welsch_cost if loss_type == "Welsch" else _calc_huber_cost

    if locked_theta is not None:
        theta_fixed = float(locked_theta)
        cos_l = np.cos(theta_fixed)
        sin_l = np.sin(theta_fixed)
        nx_base = cos_l * dxf_c[:, 0] - sin_l * dxf_c[:, 1] + dxf_cx
        ny_base = sin_l * dxf_c[:, 0] + cos_l * dxf_c[:, 1] + dxf_cy

        def cost_fn(params):
            tx, ty = float(params[0]), float(params[1])
            nx, ny = nx_base + tx, ny_base + ty
            vals, in_b = _sample_dist(nx, ny)
            if in_b.sum() < 10:
                return _OOB_DIST
            return _loss_func(vals)

    else:
        def cost_fn(params):
            tx, ty, theta = float(params[0]), float(params[1]), float(params[2])
            nx, ny = _transform(tx, ty, theta)
            vals, in_b = _sample_dist(nx, ny)
            if in_b.sum() < 10:
                return _OOB_DIST
            return _loss_func(vals)

    return cost_fn


def _sample_polylines(polylines: list[np.ndarray], spacing: float = 0.5) -> np.ndarray:
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
    if len(pts) <= n:
        return pts
    idx = np.linspace(0, len(pts) - 1, n, dtype=np.int32)
    return pts[idx]


def fit(
        polylines: list[np.ndarray],
        edge_points: np.ndarray,
        silhouette_mask: np.ndarray | None = None,
        distance_field: np.ndarray | None = None,
        objective: str = "Tolerance",
        max_error_px: float = 1.0,
) -> FitResult:
    if distance_field is None:
        raise ValueError("distance_field is required for fitting.")

    H, W = distance_field.shape

    # ── Step 1: Micro-Smooth Only ─────────────────────────────────────
    # Do NOT blur heavily. We only use a tiny blur to fix pixel-grid aliasing.
    dist_smooth = gaussian_filter(distance_field.astype(np.float32), sigma=0.5)

    # ── Step 2: DXF samples ───────────────────────────────────────────
    dxf_all = _sample_polylines(polylines, spacing=0.5)
    if len(dxf_all) == 0:
        raise ValueError("DXF polylines produced no sample points.")

    dxf_sweep  = _stride_subsample(dxf_all, n=600)
    dxf_sample = _stride_subsample(dxf_all, n=4000)

    # ── Step 3: Centroids ─────────────────────────────────────────────
    dxf_cx = float(dxf_all[:, 0].mean())
    dxf_cy = float(dxf_all[:, 1].mean())

    if silhouette_mask is not None and silhouette_mask.any():
        M_scene = cv2.moments(silhouette_mask)
        scene_cx = float(M_scene["m10"] / max(M_scene["m00"], 1))
        scene_cy = float(M_scene["m01"] / max(M_scene["m00"], 1))
    elif edge_points is not None and len(edge_points) > 0:
        scene_cx = float(np.median(edge_points[:, 0]))
        scene_cy = float(np.median(edge_points[:, 1]))
    else:
        scene_cx, scene_cy = W / 2.0, H / 2.0

    tx_init = scene_cx - dxf_cx
    ty_init = scene_cy - dxf_cy

    # ── Step 4: Centred coordinate arrays & Cost Functions ────────────
    centroid = np.array([dxf_cx, dxf_cy], dtype=np.float32)
    dxf_c_sweep = (dxf_sweep  - centroid).astype(np.float64)
    dxf_c       = (dxf_sample - centroid).astype(np.float64)

    # SWEEP/PULL: Huber loss (to pull from far away)
    cost_pull = _make_cost_fn(dxf_c, dxf_cx, dxf_cy, dist_smooth,
                              loss_type="Huber", objective=objective,
                              max_error_px=max_error_px, interp_order=1)

    # POLISH: Welsch loss + Bicubic (ignores texture entirely, snaps to edge)
    cost_polish = _make_cost_fn(dxf_c, dxf_cx, dxf_cy, dist_smooth,
                                loss_type="Welsch", objective=objective,
                                max_error_px=max_error_px, interp_order=3)

    # ── Step 5: Fully-vectorised coarse X/Y grid + angle sweep ────────
    angles = np.linspace(-np.pi, np.pi, 180, endpoint=False)
    grid_offsets = np.linspace(-24.0, 24.0, 9)

    _OOB_D   = float(max(H, W))
    delta_c  = max(0.5, max_error_px * 5.0)
    weight_c = 0.5 if objective == "Tolerance" else 1.0
    n_angles = len(angles)
    n_sweep  = len(dxf_c_sweep)

    cos_a  = np.cos(angles)
    sin_a  = np.sin(angles)
    nx_rot = cos_a[:, None] * dxf_c_sweep[:, 0] - sin_a[:, None] * dxf_c_sweep[:, 1]
    ny_rot = sin_a[:, None] * dxf_c_sweep[:, 0] + cos_a[:, None] * dxf_c_sweep[:, 1]

    all_costs = np.full((len(grid_offsets), len(grid_offsets), n_angles), _OOB_D)

    for i, dx in enumerate(grid_offsets):
        for j, dy in enumerate(grid_offsets):
            tx = tx_init + dx
            ty = ty_init + dy
            nx = nx_rot + (dxf_cx + tx)
            ny = ny_rot + (dxf_cy + ty)
            nx_f = nx.ravel()
            ny_f = ny.ravel()
            in_b = (nx_f >= 0) & (nx_f < W - 1) & (ny_f >= 0) & (ny_f < H - 1)
            vals_f = np.full(len(nx_f), _OOB_D, dtype=np.float64)
            if in_b.any():
                vals_f[in_b] = map_coordinates(
                    dist_smooth, [ny_f[in_b], nx_f[in_b]],
                    order=1, mode="nearest", prefilter=False,
                )
            vals = vals_f.reshape(n_angles, n_sweep)

            # Vectorized Huber calculation for the sweep
            all_costs[i, j] = np.mean(
                weight_c * delta_c ** 2 * (np.sqrt(1.0 + (vals / delta_c) ** 2) - 1.0),
                axis=1,
            )

    flat     = all_costs.ravel()
    top_flat = np.argpartition(flat, min(5, len(flat) - 1))[:5]
    top_flat = top_flat[np.argsort(flat[top_flat])]
    top_candidates = []
    for flat_i in top_flat:
        gi, gj, ak = np.unravel_index(int(flat_i), all_costs.shape)
        top_candidates.append(
            [tx_init + grid_offsets[gi], ty_init + grid_offsets[gj], angles[ak]]
        )

    # ── Step 6: Powell Refinement ─────────────────────────────────────
    best_res = None

    # Stage A: Loose pass on top-5 using Huber (pulls out of local ruts)
    for candidate in top_candidates:
        res = minimize(
            cost_pull,
            np.array(candidate, dtype=np.float64),
            method="Powell",
            options={"xtol": 1e-2, "ftol": 1e-2, "maxiter": 25},
        )
        if best_res is None or res.fun < best_res.fun:
            best_res = res

    # Stage B: Strict tightening on winner using Welsch.
    # This zeroes out the texture gradient and snaps to the true edge.
    final_res = minimize(
        cost_polish,
        best_res.x,
        method="Powell",
        options={"xtol": 1e-5, "ftol": 1e-6, "maxiter": 150},
    )

    tx_opt, ty_opt, angle_opt = final_res.x

    # ── Step 7: Inlier fraction ───────────────────────────────────────
    inlier_frac = _compute_inlier_frac(
        polylines, final_res.x, dist_smooth, dxf_cx, dxf_cy, max_error_px
    )

    return FitResult(
        tx=float(tx_opt),
        ty=float(ty_opt),
        angle_deg=float(np.degrees(angle_opt)),
        cost=float(final_res.fun),
        dxf_cx=dxf_cx,
        dxf_cy=dxf_cy,
        inlier_frac=inlier_frac,
        dist_t=dist_smooth,
        dist_raw=distance_field.astype(np.float32),
    )


# ======================================================================
# Refine / Sub-functions
# ======================================================================

def _refine_from(
        polylines: list[np.ndarray],
        init_params: np.ndarray,
        dist_t: np.ndarray,
        n_sample: int = 4000,
        maxiter: int = 300,
        objective: str = "Tolerance",
        max_error_px: float = 1.0,
        dxf_cx: float | None = None,
        dxf_cy: float | None = None,
) -> tuple:
    H, W = dist_t.shape
    dxf_all = _sample_polylines(polylines, spacing=0.5)
    if len(dxf_all) == 0:
        raise ValueError("Polylines produced no sample points.")

    dxf_sample = _stride_subsample(dxf_all, n=n_sample)

    if dxf_cx is None or dxf_cy is None:
        dxf_cx = float(dxf_all[:, 0].mean())
        dxf_cy = float(dxf_all[:, 1].mean())

    dxf_c = (dxf_sample - np.array([dxf_cx, dxf_cy], dtype=np.float32)).astype(np.float64)

    # Downstream refinements are strictly metrology polish. Use Welsch.
    cost_fn = _make_cost_fn(dxf_c, dxf_cx, dxf_cy, dist_t, loss_type="Welsch",
                            objective=objective, max_error_px=max_error_px, interp_order=3)

    res = minimize(
        cost_fn,
        init_params.astype(np.float64),
        method="Powell",
        options={"xtol": 1e-5, "ftol": 1e-6, "maxiter": maxiter},
    )
    return res.x, res.fun, dxf_cx, dxf_cy


def _refine_translation_only(
        polylines: list[np.ndarray],
        init_params: np.ndarray,
        dist_t: np.ndarray,
        n_sample: int = 4000,
        maxiter: int = 300,
        objective: str = "Tolerance",
        max_error_px: float = 1.0,
        dxf_cx: float | None = None,
        dxf_cy: float | None = None,
) -> tuple:
    H, W = dist_t.shape
    dxf_all = _sample_polylines(polylines, spacing=0.5)
    if len(dxf_all) == 0:
        raise ValueError("Polylines produced no sample points.")

    dxf_sample = _stride_subsample(dxf_all, n=n_sample)

    if dxf_cx is None or dxf_cy is None:
        dxf_cx = float(dxf_all[:, 0].mean())
        dxf_cy = float(dxf_all[:, 1].mean())

    dxf_c = (dxf_sample - np.array([dxf_cx, dxf_cy], dtype=np.float32)).astype(np.float64)
    theta_locked = float(init_params[2])

    cost_fn = _make_cost_fn(dxf_c, dxf_cx, dxf_cy, dist_t, loss_type="Welsch",
                            objective=objective, max_error_px=max_error_px,
                            locked_theta=theta_locked, interp_order=3)

    res = minimize(
        cost_fn,
        np.array([init_params[0], init_params[1]], dtype=np.float64),
        method="Powell",
        options={"xtol": 1e-5, "ftol": 1e-6, "maxiter": maxiter},
    )
    out_params = np.array([res.x[0], res.x[1], theta_locked])
    return out_params, res.fun, dxf_cx, dxf_cy


def _compute_inlier_frac(
        polylines: list[np.ndarray],
        params: np.ndarray,
        dist_t: np.ndarray,
        dxf_cx: float,
        dxf_cy: float,
        max_error_px: float = 1.0,
) -> float:
    H, W = dist_t.shape
    all_pts = _sample_polylines(polylines, spacing=1.0)
    if len(all_pts) == 0:
        return 0.0

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


def fit_complete(
        polylines_all: list[np.ndarray],
        polylines_refine: list[np.ndarray],
        edge_points: np.ndarray,
        silhouette_mask: np.ndarray | None = None,
        distance_field: np.ndarray | None = None,
        polylines_rot: list[np.ndarray] | None = None,
        polylines_pan: list[np.ndarray] | None = None,
        objective: str = "Tolerance",
        max_error_px: float = 1.0,
) -> FitResult:
    if distance_field is None:
        raise ValueError("distance_field is required for fitting.")

    effective_refine = polylines_refine
    if not effective_refine:
        effective_refine = (polylines_rot or []) + (polylines_pan or [])
        if effective_refine:
            print("Refine mode: no REFINE layer found — using ROT + PAN as fallback.")

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

    global_cx, global_cy = result_coarse.dxf_cx, result_coarse.dxf_cy

    params_2, cost_2, _, _ = _refine_from(
        polylines_all, params_1, dist_t, n_sample=4000,
        objective=objective, max_error_px=max_error_px,
        dxf_cx=global_cx, dxf_cy=global_cy
    )

    if effective_refine:
        params_3, cost_3, _, _ = _refine_from(
            effective_refine, params_2, dist_t, n_sample=4000,
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
        dist_raw=result_coarse.dist_raw,
    )


def fit_poc(
        polylines_all: list[np.ndarray],
        polylines_rot: list[np.ndarray],
        polylines_pan: list[np.ndarray],
        edge_points: np.ndarray,
        silhouette_mask: np.ndarray | None = None,
        distance_field: np.ndarray | None = None,
        objective: str = "Tolerance",
        max_error_px: float = 1.0,
) -> FitResult:
    if distance_field is None:
        raise ValueError("distance_field is required for fitting.")

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

    global_cx, global_cy = result_coarse.dxf_cx, result_coarse.dxf_cy

    params_2, cost_2, _, _ = _refine_from(
        polylines_all, params_1, dist_t, n_sample=4000,
        objective=objective, max_error_px=max_error_px,
        dxf_cx=global_cx, dxf_cy=global_cy
    )

    if polylines_rot:
        params_3, cost_3, _, _ = _refine_from(
            polylines_rot, params_2, dist_t, n_sample=4000,
            objective=objective, max_error_px=max_error_px,
            dxf_cx=global_cx, dxf_cy=global_cy
        )
    else:
        params_3, cost_3 = params_2, cost_2

    if polylines_pan:
        params_4, cost_4, _, _ = _refine_translation_only(
            polylines_pan, params_3, dist_t, n_sample=4000,
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
        dist_raw=result_coarse.dist_raw,
    )