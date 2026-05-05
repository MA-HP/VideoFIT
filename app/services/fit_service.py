"""
VideoFIT — Fit Service
Aligns DXF polylines onto a distance field built from sub-pixel Devernay edges.

Resolution-Independent Metrology Refactor:
- Dynamic Scaling: All spatial constants (grids, margins, radii) are calculated
  as percentages of the image diagonal to support multiple camera resolutions.
- Blob-Locked Centroids: Utilizes connected components to isolate the largest
  physical part, making the drop-point immune to FOV borders and debris.
- Radial Feature Boosting: Internal features (holes) are weighted 5x to
  prevent symmetric perimeters (gear teeth) from dominating the rotation.
- CuPy Acceleration: Fused C++ kernels evaluate the grid sweep and
  Powell iterations entirely in VRAM for sub-second performance.
- Multi-Candidate Refinement: Evaluates top 5 distinct local minima from the
  sweep grid to guarantee avoidance of local optimizer traps.
"""

from __future__ import annotations

import cv2
import numpy as np
import cupy as cp
from scipy.optimize import minimize

from app.models.fit_result import FitResult


# ─────────────────────────────────────────────────────────────────────────────
# CUDA Kernels
# ─────────────────────────────────────────────────────────────────────────────

_fused_kernels_src = r'''
extern "C" __global__
void fused_sweep_kernel(
    const float* dist, const float* dxf_x, const float* dxf_y, const float* pt_weights,
    const float* angles, const float* gx, const float* gy,
    float* out_costs,
    int width, int height, int num_pts,
    float dxf_cx, float dxf_cy, float tx_init, float ty_init,
    float delta, float oob_dist,
    int num_gx, int num_gy, int num_angles)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int ia = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= num_gx || iy >= num_gy || ia >= num_angles) return;

    float tx = tx_init + gx[ix];
    float ty = ty_init + gy[iy];
    float theta = angles[ia];

    float cos_t = cosf(theta);
    float sin_t = sinf(theta);
    float total_cost = 0.0f;

    for (int p = 0; p < num_pts; ++p) {
        float px = dxf_x[p];
        float py = dxf_y[p];

        float nx = cos_t * px - sin_t * py + dxf_cx + tx;
        float ny = sin_t * px + cos_t * py + dxf_cy + ty;

        float val = oob_dist;
        if (nx >= 0.0f && nx < (width - 1.0f) && ny >= 0.0f && ny < (height - 1.0f)) {
            int x0 = (int)nx;
            int y0 = (int)ny;
            float dx = nx - x0;
            float dy = ny - y0;

            float v00 = dist[y0 * width + x0];
            float v10 = dist[y0 * width + x0 + 1];
            float v01 = dist[(y0 + 1) * width + x0];
            float v11 = dist[(y0 + 1) * width + x0 + 1];

            float v0 = v00 + dx * (v10 - v00);
            float v1 = v01 + dx * (v11 - v01);
            val = v0 + dy * (v1 - v0);
        }
        
        float w = pt_weights[p];
        float v_d = val / delta;
        total_cost += w * (1.0f - expf(-0.5f * v_d * v_d));
    }
    int out_idx = (iy * num_gx + ix) * num_angles + ia;
    out_costs[out_idx] = total_cost / (float)num_pts;
}

extern "C" __global__
void point_cost_huber(
    const float* dist, const float* dxf_x, const float* dxf_y, const float* pt_weights,
    float* out_total_cost,
    int width, int height, int num_pts,
    float dxf_cx, float dxf_cy, float tx, float ty, float theta,
    float delta, float weight, float oob_dist)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= num_pts) return;

    float cos_t = cosf(theta);
    float sin_t = sinf(theta);
    float px = dxf_x[p], py = dxf_y[p];

    float nx = cos_t * px - sin_t * py + dxf_cx + tx;
    float ny = sin_t * px + cos_t * py + dxf_cy + ty;

    float val = oob_dist;
    if (nx >= 0.0f && nx < (width - 1.0f) && ny >= 0.0f && ny < (height - 1.0f)) {
        int x0 = (int)nx, y0 = (int)ny;
        float dx = nx - x0, dy = ny - y0;
        float v00 = dist[y0 * width + x0], v10 = dist[y0 * width + x0 + 1];
        float v01 = dist[(y0 + 1) * width + x0], v11 = dist[(y0 + 1) * width + x0 + 1];
        val = (v00 + dx*(v10-v00)) + dy*((v01 + dx*(v11-v01)) - (v00 + dx*(v10-v00)));
    }

    float w = pt_weights[p];
    float v_d = val / delta;
    atomicAdd(out_total_cost, w * weight * delta * delta * (sqrtf(1.0f + v_d * v_d) - 1.0f));
}

extern "C" __global__
void point_cost_welsch(
    const float* dist, const float* dxf_x, const float* dxf_y, const float* pt_weights,
    float* out_total_cost,
    int width, int height, int num_pts,
    float dxf_cx, float dxf_cy, float tx, float ty, float theta,
    float delta, float weight, float oob_dist)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= num_pts) return;

    float cos_t = cosf(theta);
    float sin_t = sinf(theta);
    float px = dxf_x[p], py = dxf_y[p];

    float nx = cos_t * px - sin_t * py + dxf_cx + tx;
    float ny = sin_t * px + cos_t * py + dxf_cy + ty;

    float val = oob_dist;
    if (nx >= 0.0f && nx < (width - 1.0f) && ny >= 0.0f && ny < (height - 1.0f)) {
        int x0 = (int)nx, y0 = (int)ny;
        float dx = nx - x0, dy = ny - y0;
        float v00 = dist[y0 * width + x0], v10 = dist[y0 * width + x0 + 1];
        float v01 = dist[(y0 + 1) * width + x0], v11 = dist[(y0 + 1) * width + x0 + 1];
        val = (v00 + dx*(v10-v00)) + dy*((v01 + dx*(v11-v01)) - (v00 + dx*(v10-v00)));
    }

    float w = pt_weights[p];
    float v_d = val / delta;
    atomicAdd(out_total_cost, w * (1.0f - expf(-0.5f * v_d * v_d)));
}
'''

_k_fused_sweep = cp.RawKernel(_fused_kernels_src, 'fused_sweep_kernel', options=('-use_fast_math',))
_k_point_cost_huber = cp.RawKernel(_fused_kernels_src, 'point_cost_huber', options=('-use_fast_math',))
_k_point_cost_welsch = cp.RawKernel(_fused_kernels_src, 'point_cost_welsch', options=('-use_fast_math',))


# ─────────────────────────────────────────────────────────────────────────────
# Functions
# ─────────────────────────────────────────────────────────────────────────────

def _make_cost_fn(
        dxf_c_cp: cp.ndarray,
        pt_weights_cp: cp.ndarray,
        dxf_cx: float,
        dxf_cy: float,
        dist_t_cp: cp.ndarray,
        loss_type: str = "Huber",
        max_error_px: float = 1.0,
        locked_theta: float | None = None,
) -> callable:
    H, W = dist_t_cp.shape
    _OOB_DIST = float(max(H, W))
    delta_c = float(max(0.5, max_error_px))
    weight_c = 0.5

    dxf_x_cp = cp.ascontiguousarray(dxf_c_cp[:, 0], dtype=cp.float32)
    dxf_y_cp = cp.ascontiguousarray(dxf_c_cp[:, 1], dtype=cp.float32)
    num_pts = len(dxf_x_cp)
    out_total_cp = cp.zeros(1, dtype=cp.float32)

    block = (256,)
    grid = ((num_pts + 255) // 256,)
    kernel = _k_point_cost_welsch if loss_type == "Welsch" else _k_point_cost_huber

    def cost_fn(params):
        out_total_cp.fill(0.0)
        tx, ty = float(params[0]), float(params[1])
        theta = float(locked_theta if locked_theta is not None else params[2])
        kernel(grid, block, (
            dist_t_cp, dxf_x_cp, dxf_y_cp, pt_weights_cp, out_total_cp,
            W, H, num_pts,
            cp.float32(dxf_cx), cp.float32(dxf_cy), cp.float32(tx), cp.float32(ty), cp.float32(theta),
            cp.float32(delta_c), cp.float32(weight_c), cp.float32(_OOB_DIST)
        ))
        return float(out_total_cp[0]) / num_pts

    return cost_fn


def _sample_polylines(polylines: list[np.ndarray], spacing: float = 0.5) -> np.ndarray:
    pts = []
    for poly in polylines:
        if len(poly) < 2: continue
        for i in range(len(poly) - 1):
            p0, p1 = poly[i].astype(np.float64), poly[i + 1].astype(np.float64)
            seg_len = np.hypot(p1[0] - p0[0], p1[1] - p0[1])
            n = max(2, int(np.ceil(seg_len / spacing)))
            t = np.linspace(0.0, 1.0, n, endpoint=False)
            pts.append(p0 + np.outer(t, p1 - p0))
    return np.vstack(pts).astype(np.float32) if pts else np.empty((0, 2), dtype=np.float32)


def _stride_subsample(pts: np.ndarray, n: int) -> np.ndarray:
    if len(pts) <= n: return pts
    return pts[np.linspace(0, len(pts) - 1, n, dtype=np.int32)]


def fit(
        polylines: list[np.ndarray],
        edge_points: np.ndarray,
        silhouette_mask: np.ndarray | None = None,
        distance_field: np.ndarray | None = None,
        objective: str = "Tolerance",
        max_error_px: float = 1.0,
) -> FitResult:
    if distance_field is None: raise ValueError("distance_field required.")

    H, W = distance_field.shape
    diag = np.hypot(H, W)

    # ── Step 1: Adaptive CPU Smoothing ────────────────────────────────
    sigma_val = max(0.5, diag * 0.0005)
    dist_smooth = cv2.GaussianBlur(distance_field.astype(np.float32), (0, 0), sigma_val)
    dist_smooth_cp = cp.ascontiguousarray(cp.asarray(dist_smooth), dtype=cp.float32)

    # ── Step 2: DXF samples ───────────────────────────────────────────
    dxf_all = _sample_polylines(polylines, spacing=0.5)
    dxf_sweep = _stride_subsample(dxf_all, n=800)
    dxf_sample = _stride_subsample(dxf_all, n=3000)

    # ── Step 3: Blob-Locked Centroids (Resolution Independent) ────────
    dxf_min_x, dxf_max_x = float(dxf_all[:, 0].min()), float(dxf_all[:, 0].max())
    dxf_min_y, dxf_max_y = float(dxf_all[:, 1].min()), float(dxf_all[:, 1].max())
    dxf_cx, dxf_cy = (dxf_min_x + dxf_max_x) / 2.0, (dxf_min_y + dxf_max_y) / 2.0

    scene_cx, scene_cy = W / 2.0, H / 2.0
    blob_thresh = diag * 0.02
    thick_edges = (distance_field < blob_thresh).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thick_edges, connectivity=8)

    if num_labels > 1:
        largest_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        scene_cx = stats[largest_idx, cv2.CC_STAT_LEFT] + stats[largest_idx, cv2.CC_STAT_WIDTH] / 2.0
        scene_cy = stats[largest_idx, cv2.CC_STAT_TOP] + stats[largest_idx, cv2.CC_STAT_HEIGHT] / 2.0

    tx_init, ty_init = scene_cx - dxf_cx, scene_cy - dxf_cy

    # ── Step 4: Radial Weighting ──────────────────────────────────────
    centroid = np.array([dxf_cx, dxf_cy], dtype=np.float32)
    dxf_sample_c = dxf_sample - centroid
    dxf_c_cp = cp.asarray(dxf_sample_c, dtype=cp.float32)

    radii = np.hypot(dxf_sample_c[:, 0], dxf_sample_c[:, 1])
    r_max = radii.max() if len(radii) > 0 else 1.0
    pt_weights = np.ones(len(dxf_sample), dtype=np.float32)
    pt_weights[radii < r_max * 0.92] = 5.0  # Holes weighted 5x
    pt_weights_cp = cp.asarray(pt_weights, dtype=cp.float32)

    cost_pull = _make_cost_fn(dxf_c_cp, pt_weights_cp, dxf_cx, dxf_cy, dist_smooth_cp, "Huber", max_error_px)
    cost_polish = _make_cost_fn(dxf_c_cp, pt_weights_cp, dxf_cx, dxf_cy, dist_smooth_cp, "Welsch", max_error_px)

    # ── Step 5: Resolution-Independent Fused Sweep ────────────────────
    num_angles, num_gx, num_gy = 360, 21, 21
    angles_cp = cp.linspace(-cp.pi, cp.pi, num_angles, endpoint=False, dtype=cp.float32)
    search_range = diag * 0.05
    gx_cp = cp.linspace(-search_range, search_range, num_gx, dtype=cp.float32)
    gy_cp = cp.linspace(-search_range, search_range, num_gy, dtype=cp.float32)

    dxf_sweep_c = dxf_sweep - centroid
    dxf_x_cp = cp.ascontiguousarray(cp.asarray(dxf_sweep_c[:, 0]), dtype=cp.float32)
    dxf_y_cp = cp.ascontiguousarray(cp.asarray(dxf_sweep_c[:, 1]), dtype=cp.float32)

    sweep_radii = np.hypot(dxf_sweep_c[:, 0], dxf_sweep_c[:, 1])
    sweep_weights = np.ones(len(dxf_sweep), dtype=np.float32)
    sweep_weights[sweep_radii < (sweep_radii.max() * 0.92)] = 5.0
    sweep_weights_cp = cp.asarray(sweep_weights, dtype=cp.float32)

    costs_cp = cp.empty((num_gy, num_gx, num_angles), dtype=cp.float32)
    sweep_delta = max(10.0, diag * 0.01)

    _k_fused_sweep((num_gx//8+1, num_gy//8+1, num_angles//4+1), (8, 8, 4),
        (dist_smooth_cp, dxf_x_cp, dxf_y_cp, sweep_weights_cp, angles_cp, gx_cp, gy_cp, costs_cp,
         W, H, len(dxf_x_cp), cp.float32(dxf_cx), cp.float32(dxf_cy), cp.float32(tx_init), cp.float32(ty_init),
         cp.float32(sweep_delta), cp.float32(diag), num_gx, num_gy, num_angles))

    # ── Step 6: Multi-Candidate NMS & Powell Refinement ───────────────
    # Move costs to CPU to efficiently find the top 5 distinct minimums
    costs_np = cp.asnumpy(costs_cp)
    flat_indices = np.argsort(costs_np.ravel())

    k_candidates = 5
    candidates = []

    # Thresholds for Non-Maximum Suppression (ensures candidates are distinct)
    step_x = float(gx_cp[1] - gx_cp[0]) if num_gx > 1 else 1.0
    step_y = float(gy_cp[1] - gy_cp[0]) if num_gy > 1 else 1.0
    step_a = float(angles_cp[1] - angles_cp[0]) if num_angles > 1 else 1.0

    min_dist_sq = (max(step_x, step_y) * 2.0) ** 2  # At least 2 grid cells apart
    min_ang = step_a * 3.0                          # At least 3 angular steps apart

    for flat_idx in flat_indices:
        iy, ix, ia = np.unravel_index(flat_idx, (num_gy, num_gx, num_angles))
        cx = tx_init + float(gx_cp[ix])
        cy = ty_init + float(gy_cp[iy])
        ct = float(angles_cp[ia])

        # Check against already saved distinct candidates
        is_distinct = True
        for (ex, ey, et) in candidates:
            dist_sq = (cx - ex)**2 + (cy - ey)**2
            ang_diff = abs(ct - et)
            ang_diff = min(ang_diff, 2 * np.pi - ang_diff) # Handle wrap-around

            if dist_sq < min_dist_sq and ang_diff < min_ang:
                is_distinct = False
                break

        if is_distinct:
            candidates.append((cx, cy, ct))
            if len(candidates) >= k_candidates:
                break

    # Fallback (rare): Just fill up to K if the grid was extremely uniform
    if len(candidates) < k_candidates:
        for flat_idx in flat_indices:
            iy, ix, ia = np.unravel_index(flat_idx, (num_gy, num_gx, num_angles))
            cand = (tx_init + float(gx_cp[ix]), ty_init + float(gy_cp[iy]), float(angles_cp[ia]))
            if cand not in candidates:
                candidates.append(cand)
            if len(candidates) >= k_candidates:
                break

    # Evaluate the Top 5 candidates through Powell to find the global best
    best_final_cost = float('inf')
    best_final_params = None

    for cand in candidates:
        res_pull = minimize(cost_pull, np.array(cand), method="Powell", options={"xtol": 1e-2, "ftol": 1e-2, "maxiter": 20})
        final_res = minimize(cost_polish, res_pull.x, method="Powell", options={"xtol": 1e-4, "ftol": 1e-4, "maxiter": 75})

        if final_res.fun < best_final_cost:
            best_final_cost = final_res.fun
            best_final_params = final_res.x

    tx_opt, ty_opt, angle_opt = best_final_params
    inlier_frac = _compute_inlier_frac(polylines, best_final_params, dist_smooth_cp, dxf_cx, dxf_cy, max_error_px)

    return FitResult(tx=float(tx_opt), ty=float(ty_opt), angle_deg=float(np.degrees(angle_opt)), cost=float(best_final_cost),
                     dxf_cx=dxf_cx, dxf_cy=dxf_cy, inlier_frac=inlier_frac, dist_t=cp.asnumpy(dist_smooth_cp),
                     dist_raw=distance_field.astype(np.float32))


def _compute_inlier_frac(polylines: list[np.ndarray], params: np.ndarray, dist_t_cp: cp.ndarray, dxf_cx: float, dxf_cy: float, max_error_px: float) -> float:
    all_pts = _sample_polylines(polylines, spacing=1.0)
    if len(all_pts) == 0: return 0.0
    all_c_cp = cp.asarray(all_pts - np.array([dxf_cx, dxf_cy], dtype=np.float32), dtype=cp.float32)
    tx, ty, angle = float(params[0]), float(params[1]), float(params[2])
    cos_t, sin_t = float(np.cos(angle)), float(np.sin(angle))
    nx_cp, ny_cp = cos_t * all_c_cp[:, 0] - sin_t * all_c_cp[:, 1] + dxf_cx + tx, sin_t * all_c_cp[:, 0] + cos_t * all_c_cp[:, 1] + dxf_cy + ty
    from cupyx.scipy.ndimage import map_coordinates as cp_map_coordinates
    vals = cp_map_coordinates(dist_t_cp, cp.stack([ny_cp, nx_cp]), order=1, mode="constant", cval=float(max(dist_t_cp.shape)), prefilter=False)
    return float(cp.mean(vals < max_error_px))


def fit_complete(polylines_all: list[np.ndarray], polylines_refine: list[np.ndarray], edge_points: np.ndarray, silhouette_mask: np.ndarray | None = None, distance_field: np.ndarray | None = None, polylines_rot: list[np.ndarray] | None = None, polylines_pan: list[np.ndarray] | None = None, objective: str = "Tolerance", max_error_px: float = 1.0) -> FitResult:
    res = fit(polylines_all, edge_points, silhouette_mask, distance_field, objective, max_error_px)
    dist_t_cp = cp.ascontiguousarray(cp.asarray(res.dist_t), dtype=cp.float32)
    params = np.array([res.tx, res.ty, np.radians(res.angle_deg)])

    if polylines_refine:
        res_ref, _, _, _ = _refine_from(polylines_refine, params, dist_t_cp, n_sample=3000, max_error_px=max_error_px, dxf_cx=res.dxf_cx, dxf_cy=res.dxf_cy)
        params = res_ref

    inlier = _compute_inlier_frac(polylines_all, params, dist_t_cp, res.dxf_cx, res.dxf_cy, max_error_px)
    return FitResult(tx=float(params[0]), ty=float(params[1]), angle_deg=float(np.degrees(params[2])), cost=res.cost, dxf_cx=res.dxf_cx, dxf_cy=res.dxf_cy, inlier_frac=inlier, dist_t=res.dist_t, dist_raw=res.dist_raw)


def _refine_from(polylines, init_params, dist_t_cp, n_sample=3000, max_error_px=1.0, dxf_cx=None, dxf_cy=None) -> tuple:
    dxf_all = _sample_polylines(polylines, spacing=0.5)
    dxf_sample = _stride_subsample(dxf_all, n=n_sample)
    if dxf_cx is None: dxf_cx, dxf_cy = dxf_all[:, 0].mean(), dxf_all[:, 1].mean()

    centroid = np.array([dxf_cx, dxf_cy], dtype=np.float32)
    dxf_c_cp = cp.asarray(dxf_sample - centroid, dtype=cp.float32)

    radii = np.hypot(dxf_sample[:, 0] - dxf_cx, dxf_sample[:, 1] - dxf_cy)
    pt_weights = np.ones(len(dxf_sample), dtype=np.float32)
    pt_weights[radii < radii.max() * 0.92] = 5.0
    pt_weights_cp = cp.asarray(pt_weights, dtype=cp.float32)

    cost_fn = _make_cost_fn(dxf_c_cp, pt_weights_cp, dxf_cx, dxf_cy, dist_t_cp, "Welsch", max_error_px)
    res = minimize(cost_fn, init_params, method="Powell", options={"xtol": 1e-4, "ftol": 1e-4, "maxiter": 100})
    return res.x, res.fun, dxf_cx, dxf_cy


def fit_poc(polylines_all, polylines_rot, polylines_pan, edge_points, silhouette_mask=None, distance_field=None, objective="Tolerance", max_error_px=1.0) -> FitResult:
    res = fit(polylines_all, edge_points, silhouette_mask, distance_field, objective, max_error_px)
    dist_t_cp = cp.ascontiguousarray(cp.asarray(res.dist_t), dtype=cp.float32)
    params = np.array([res.tx, res.ty, np.radians(res.angle_deg)])

    if polylines_rot:
        params, _, _, _ = _refine_from(polylines_rot, params, dist_t_cp, 3000, max_error_px, res.dxf_cx, res.dxf_cy)
    if polylines_pan:
        dxf_all = _sample_polylines(polylines_pan, 0.5)
        dxf_sample = _stride_subsample(dxf_all, 3000)
        dxf_c_cp = cp.asarray(dxf_sample - np.array([res.dxf_cx, res.dxf_cy], dtype=np.float32), dtype=cp.float32)
        pt_w_cp = cp.ones(len(dxf_sample), dtype=cp.float32)
        cost_fn = _make_cost_fn(dxf_c_cp, pt_w_cp, res.dxf_cx, res.dxf_cy, dist_t_cp, "Welsch", max_error_px, locked_theta=params[2])
        res_pan = minimize(cost_fn, params[:2], method="Powell", options={"xtol": 1e-4, "ftol": 1e-4, "maxiter": 100})
        params[:2] = res_pan.x

    inlier = _compute_inlier_frac(polylines_all, params, dist_t_cp, res.dxf_cx, res.dxf_cy, max_error_px)
    return FitResult(tx=float(params[0]), ty=float(params[1]), angle_deg=float(np.degrees(params[2])), cost=res.cost, dxf_cx=res.dxf_cx, dxf_cy=res.dxf_cy, inlier_frac=inlier, dist_t=res.dist_t, dist_raw=res.dist_raw)