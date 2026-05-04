"""
VideoFIT — GPU-accelerated Canny-Devernay Sub-pixel Edge Detector
CUDA kernels via CuPy: bilateral pre-smooth → fused Sobel → sub-pixel NMS
with curvature rejection.

Hysteresis runs on the CPU with OpenCV connected-components (faster than
CuPy ndi.label for typical frame sizes due to lower kernel-launch overhead).
Valid sub-pixel coordinates are reconstructed and scaled entirely on the GPU,
with only the final results downloaded over PCIe.
"""

from __future__ import annotations

import cv2
import numpy as np
import cupy as cp


# ─────────────────────────────────────────────────────────────────────────────
# CUDA kernels  (compiled once at import time)
# ─────────────────────────────────────────────────────────────────────────────

_bilateral_src = r'''
extern "C" __global__
void bilateral_kernel(
    const float* input, float* output,
    int width, int height,
    float sigma_s, float sigma_r, int radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float center_val = input[y * width + x];
    float sum = 0.0f, norm = 0.0f;
    float var_s = 2.0f * sigma_s * sigma_s;
    float var_r = 2.0f * sigma_r * sigma_r;

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            int nx = max(0, min(width - 1, x + dx));
            int ny = max(0, min(height - 1, y + dy));
            float val = input[ny * width + nx];
            float space_dist2 = (float)(dx * dx + dy * dy);
            float color_dist  = val - center_val;
            float weight = expf(-(space_dist2 / var_s)
                              - ((color_dist * color_dist) / var_r));
            sum  += val * weight;
            norm += weight;
        }
    }
    output[y * width + x] = sum / norm;
}
'''

_sobel_src = r'''
extern "C" __global__
void fused_sobel_kernel(
    const float* input, float* Gx, float* Gy, float* G_mag,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) return;

    float p00 = input[(y-1)*width + (x-1)];
    float p01 = input[(y-1)*width + x    ];
    float p02 = input[(y-1)*width + (x+1)];
    float p10 = input[y    *width + (x-1)];
    float p12 = input[y    *width + (x+1)];
    float p20 = input[(y+1)*width + (x-1)];
    float p21 = input[(y+1)*width + x    ];
    float p22 = input[(y+1)*width + (x+1)];

    float gx = (p02 - p00) + 2.0f*(p12 - p10) + (p22 - p20);
    float gy = (p20 - p00) + 2.0f*(p21 - p01) + (p22 - p02);

    int idx = y * width + x;
    Gx[idx]    = gx;
    Gy[idx]    = gy;
    G_mag[idx] = sqrtf(gx * gx + gy * gy);
}
'''

_devernay_src = r'''
extern "C" __global__
void fast_devernay_kernel(
    const float* Gx, const float* Gy, const float* G_mag,
    float* out_x, float* out_y, bool* out_mask,
    int width, int height, float low_thresh, float min_curvature)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) return;

    int idx = y * width + x;
    float mag0 = G_mag[idx];

    if (mag0 < low_thresh) { out_mask[idx] = false; return; }

    float gx = Gx[idx], gy = Gy[idx];
    float abs_gx = abs(gx), abs_gy = abs(gy);
    float mag_plus, mag_minus, dx_norm, dy_norm;

    if (abs_gx > abs_gy) {
        float weight = gy / (gx + 1e-6f);
        mag_plus  = G_mag[y * width + (x + 1)];
        mag_minus = G_mag[y * width + (x - 1)];
        dx_norm = 1.0f; dy_norm = weight;
    } else {
        float weight = gx / (gy + 1e-6f);
        mag_plus  = G_mag[(y + 1) * width + x];
        mag_minus = G_mag[(y - 1) * width + x];
        dx_norm = weight; dy_norm = 1.0f;
    }

    if (mag0 >= mag_plus && mag0 >= mag_minus) {
        float curvature = abs(mag_minus - 2.0f * mag0 + mag_plus);
        if (curvature < min_curvature) { out_mask[idx] = false; return; }

        float denom = 2.0f * (mag_minus - 2.0f * mag0 + mag_plus) + 1e-6f;
        float delta = (mag_minus - mag_plus) / denom;
        delta = fmaxf(-0.5f, fminf(delta, 0.5f));

        out_x[idx]    = (float)x + (delta * dx_norm);
        out_y[idx]    = (float)y + (delta * dy_norm);
        out_mask[idx] = true;
    } else {
        out_mask[idx] = false;
    }
}
'''

_k_bilateral = cp.RawKernel(_bilateral_src, 'bilateral_kernel',   options=('-use_fast_math',))
_k_sobel     = cp.RawKernel(_sobel_src,     'fused_sobel_kernel',  options=('-use_fast_math',))
_k_devernay  = cp.RawKernel(_devernay_src,  'fast_devernay_kernel', options=('-use_fast_math',))

# Structuring element reused for CPU morphological close
_CPU_CLOSE_K = np.ones((3, 3), dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def devernay_edges(
    gray: np.ndarray,
    sigma: float = 0.0,
    high_thresh: float = 50.0,
    low_thresh: float = 15.0,
    mask: np.ndarray | None = None,
    downsample: float = 1.0,
    min_curvature: float = 2.0,
    min_edge_length: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    GPU sub-pixel Canny-Devernay edge detection.

    Returns
    -------
    ex, ey   : float32 (H, W) — sub-pixel coordinates; -1 where no edge.
    edge_map : uint8  (H, W) — rasterised edge pixels.
    """
    H, W = gray.shape[:2]

    # ── 1. Optional downsample ───────────────────────────────────────
    if downsample != 1.0:
        small_w = max(1, int(W * downsample))
        small_h = max(1, int(H * downsample))
        work = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
        work_mask = (
            cv2.resize(mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
            if mask is not None else None
        )
    else:
        work, work_mask = gray, mask
        small_h, small_w = H, W

    # ── 2. Upload → normalise ────────────────────────────────────────
    cp_frame = cp.asarray(work, dtype=cp.float32)
    height, width = cp_frame.shape

    block = (16, 16)
    grid  = ((width + 15) // 16, (height + 15) // 16)

    f_min, f_max = cp_frame.min(), cp_frame.max()
    cp_frame = (cp_frame - f_min) * (255.0 / (f_max - f_min + 1e-6))

    # ── 3. GPU bilateral pre-smooth ──────────────────────────────────
    bil_sigma_s = max(1.0, sigma * 10.0) if sigma > 0.0 else 2.0
    bil_radius  = max(2, int(3 * sigma)) if sigma > 0.0 else 2
    filtered = cp.empty_like(cp_frame)
    _k_bilateral(
        grid, block,
        (cp_frame, filtered, width, height,
         cp.float32(bil_sigma_s), cp.float32(25.0), bil_radius)
    )

    # ── 4. GPU fused Sobel ───────────────────────────────────────────
    Gx    = cp.empty_like(cp_frame)
    Gy    = cp.empty_like(cp_frame)
    G_mag = cp.empty_like(cp_frame)
    _k_sobel(grid, block, (filtered, Gx, Gy, G_mag, width, height))

    # ── 5. GPU sub-pixel NMS + curvature rejection ───────────────────
    out_x    = cp.empty_like(G_mag)
    out_y    = cp.empty_like(G_mag)
    out_mask = cp.zeros(G_mag.shape, dtype=cp.bool_)
    _k_devernay(
        grid, block,
        (Gx, Gy, G_mag, out_x, out_y, out_mask,
         width, height, cp.float32(low_thresh), cp.float32(min_curvature))
    )

    # ── 6. Minimal PCIe download for CPU hysteresis ──────────────────
    out_mask_np = cp.asnumpy(out_mask)   # bool   (H, W)
    G_mag_np    = cp.asnumpy(G_mag)      # float32 (H, W)

    # ── 7. CPU hysteresis ────────────────────────────────────────────
    closed = cv2.morphologyEx(
        out_mask_np.view(np.uint8), cv2.MORPH_CLOSE, _CPU_CLOSE_K
    )

    _, labels, _, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    num_features = int(labels.max())

    if num_features > 0:
        comp_sizes    = np.bincount(labels.ravel())
        strong_pixels = out_mask_np & (G_mag_np > high_thresh)
        strong_labels = np.unique(labels[strong_pixels])
        has_strong    = np.zeros(num_features + 1, dtype=bool)
        has_strong[strong_labels] = True
        valid_labels  = (comp_sizes >= min_edge_length) & has_strong
        valid_labels[0] = False
        clean_np      = out_mask_np & valid_labels[labels]
    else:
        clean_np = out_mask_np

    if work_mask is not None:
        clean_np = clean_np & (work_mask > 0)

    # ── 8. Upload valid mask back to GPU ─────────────────────────────
    # Send the cleaned boolean mask back to filter the original arrays
    clean_gpu = cp.asarray(clean_np)

    # ── 9. Reconstruct (H, W) coordinate arrays on GPU ───────────────
    xs = out_x[clean_gpu]
    ys = out_y[clean_gpu]

    ex_s = cp.full((small_h, small_w), -1.0, dtype=cp.float32)
    ey_s = cp.full((small_h, small_w), -1.0, dtype=cp.float32)

    if xs.size > 0:
        xs_i = cp.round(xs).astype(cp.int32).clip(0, small_w - 1)
        ys_i = cp.round(ys).astype(cp.int32).clip(0, small_h - 1)
        ex_s[ys_i, xs_i] = xs
        ey_s[ys_i, xs_i] = ys

    # ── 10. Optional upscale back to full resolution (GPU) ────────────
    if downsample != 1.0:
        inv = 1.0 / downsample
        valid = ex_s >= 0.0

        ex = cp.full((H, W), -1.0, dtype=cp.float32)
        ey = cp.full((H, W), -1.0, dtype=cp.float32)

        ex_valid = ex_s[valid]
        ey_valid = ey_s[valid]

        ys_s = cp.round(ey_valid).astype(cp.int32).clip(0, small_h - 1)
        xs_s = cp.round(ex_valid).astype(cp.int32).clip(0, small_w - 1)
        ys_f = cp.round(ey_valid * inv).astype(cp.int32).clip(0, H - 1)
        xs_f = cp.round(ex_valid * inv).astype(cp.int32).clip(0, W - 1)

        ex[ys_f, xs_f] = ex_s[ys_s, xs_s] * inv
        ey[ys_f, xs_f] = ey_s[ys_s, xs_s] * inv
    else:
        ex, ey = ex_s, ey_s

    # ── 11. Render edge map on GPU & Final Download ───────────────────
    edge_map = cp.zeros((H, W), dtype=cp.uint8)
    valid_final = ex >= 0.0

    if valid_final.any():
        ys_render = cp.round(ey[valid_final]).astype(cp.int32).clip(0, H - 1)
        xs_render = cp.round(ex[valid_final]).astype(cp.int32).clip(0, W - 1)
        edge_map[ys_render, xs_render] = 255

    return cp.asnumpy(ex), cp.asnumpy(ey), cp.asnumpy(edge_map)