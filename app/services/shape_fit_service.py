"""
VideoFIT — Shape Fit Service
Robust sub-pixel shape fitting: circle, arc, line, automatic detection.

Pipeline per connected component
──────────────────────────────────
1. Categorise    — PCA eigenvalue ratio (λ₂/λ₁) guides the order of fitting
                   attempts.  Both shapes are always attempted so that PCA
                   misclassification never silently drops a valid shape.

2. MAGSAC++-like — Minimal-sample RANSAC with soft multi-sigma scoring.
                   Each hypothesis is scored by marginalising Gaussian weights
                   over σ ∈ {0.5, 1.0, 2.0, 3.5} px.  Line RANSAC uses
                   locality-aware sampling (sort by x+y diagonal, second point
                   drawn from ±20 neighbours) to bias toward same-feature pairs.
                   Adaptive iteration count via RANSAC formula.

3. Refine        — Deterministic weighted Gauss-Newton (circle/arc) or
                   weighted SVD total-least-squares (line) on RANSAC inliers,
                   with per-point weights w_i = exp(−½(r_i/σ_ref)²).

4. Quality gate  — Density check (inliers per pixel of shape length) rejects
                   sparse phantom shapes.  Score = n_inliers² / rms with a
                   ×2.5 circle bonus and ×0.1 large-radius arc penalty drives
                   the line-vs-arc decision.

5. Inlier strip  — After accepting a shape its inliers are removed from the
                   component and the loop repeats.  One L-shaped component
                   (line meeting arc) yields two distinct shapes.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial import cKDTree

from app.models.measure_result import MeasureResult, ShapeKind

# ── Tuneable constants ────────────────────────────────────────────────────────

_MIN_PTS_LINE   = 5
_MIN_PTS_CIRCLE = 8

_MAX_RESIDUAL_PX = 4.0
_MAX_RADIUS_PX   = 8000.0

# Angular gap below which a fitted shape is classified as a full circle vs arc.
# gap < 60° → coverage > 300° (conservative; avoids misclassifying open arcs).
_FULL_CIRCLE_GAP_DEG = 60.0

_MIN_COMPONENT_AREA = 30

STROKE_BRUSH_RADIUS_PX: float = 15.0

# MAGSAC sigma levels (px)
_MAGSAC_SIGMAS = (0.5, 1.0, 2.0, 3.5)

# Inlier gate (used for RANSAC model selection and multi-shape stripping)
_SIGMA_REF           = 2.0
_INLIER_THRESHOLD_PX = 3.0 * _SIGMA_REF   # 6 px

# Per-shape minimum inlier fraction
_MIN_INLIER_FRAC_LINE   = 0.20
_MIN_INLIER_FRAC_CIRCLE = 0.40

# RANSAC budgets
_RANSAC_CIRCLE_ITERS    = 300
_RANSAC_LINE_ITERS      = 200

# Locality window for line RANSAC: second sample drawn from within ±N sorted
# neighbours (sorted by x+y diagonal so spatial locality ≈ index locality).
_RANSAC_LINE_LOCAL_HALF = 20

# Minimum inliers per pixel of shape length — rejects sparse phantom shapes.
# The check is bypassed when n_inliers >= _MIN_DENSITY_N_EXEMPT: a large inlier
# count is itself strong evidence the shape is real regardless of arc length.
_MIN_DENSITY          = 0.05   # inliers-per-px threshold
_MIN_DENSITY_N_EXEMPT = 20     # skip density check when inliers >= this

# Arc–vs–line hard guards (lens distortion → large-R / small-span phantom arcs).
# _MIN_ARC_SPAN_DEG is deliberately conservative (12°) so that 15–20° arcs with
# slightly noisy RANSAC coverage are not wrongly forced to LINE.  Genuine
# micro-arcs below 12° are essentially indistinguishable from lines anyway.
_MAX_ARC_RADIUS_PREFER_LINE = 1500.0
_MIN_ARC_SPAN_DEG           = 12.0

# Scoring: score = n_inliers² / (rms + ε) × shape_bonus.
# Arc vs line is decided by score (not a raw RMS ratio) so that wider-coverage
# arc fits beat narrower-inlier line fits even when absolute RMS values are close.
_SCORE_CIRCLE_BONUS      = 2.5   # closed contour bonus
_SCORE_LARGE_ARC_PENALTY = 0.10  # soft penalty for likely-distortion arcs (R > 1500)

# Maximum distinct shapes extracted per connected component (inlier stripping)
_MAX_SHAPES_PER_COMPONENT = 5


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _circle_from_3pts(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
) -> tuple[float, float, float] | None:
    """Unique circle through three non-collinear points (perpendicular bisectors).
    Returns (cx, cy, r) or None on degenerate input.
    """
    ax, ay = float(p1[0]), float(p1[1])
    bx, by = float(p2[0]), float(p2[1])
    cx, cy = float(p3[0]), float(p3[1])

    D = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(D) < 1e-8:
        return None

    a2, b2, c2 = ax*ax + ay*ay, bx*bx + by*by, cx*cx + cy*cy
    ux = (a2*(by - cy) + b2*(cy - ay) + c2*(ay - by)) / D
    uy = (a2*(cx - bx) + b2*(ax - cx) + c2*(bx - ax)) / D
    r  = float(np.hypot(ax - ux, ay - uy))

    if r <= 0.0 or r > _MAX_RADIUS_PX:
        return None
    return ux, uy, r


def _magsac_score(residuals: np.ndarray) -> float:
    """Gaussian soft-weights marginalised over _MAGSAC_SIGMAS. Higher = better."""
    s = 0.0
    for sigma in _MAGSAC_SIGMAS:
        s += float(np.exp(-0.5 * (residuals / sigma) ** 2).sum())
    return s / len(_MAGSAC_SIGMAS)


def _shape_score(r: MeasureResult) -> float:
    """Ranking score: n_inliers² / rms with shape-type bonuses/penalties.

    Replaces pure-RMS comparison so shapes with more coverage are preferred.
    Large-radius arcs (likely lens distortion) are soft-penalised rather than
    hard-rejected, keeping them detectable when they genuinely dominate.
    """
    base = (r.n_points ** 2) / (r.residual_rms + 1e-5)
    if r.kind == ShapeKind.CIRCLE:
        return base * _SCORE_CIRCLE_BONUS
    if r.kind == ShapeKind.ARC and r.radius > _MAX_ARC_RADIUS_PREFER_LINE:
        return base * _SCORE_LARGE_ARC_PENALTY
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Categorise (advisory only — both shapes are always attempted)
# ─────────────────────────────────────────────────────────────────────────────

def _classify_component(pts: np.ndarray) -> str:
    """PCA eigenvalue ratio classifier → ``'line'``, ``'arc_or_circle'``, ``'both'``."""
    if len(pts) < 3:
        return "line"
    centered = pts.astype(np.float64) - pts.mean(axis=0)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    if s[0] < 1e-10:
        return "both"
    ratio = s[1] / s[0]
    if ratio < 0.04:
        return "line"
    if ratio > 0.14:
        return "arc_or_circle"
    return "both"


# ─────────────────────────────────────────────────────────────────────────────
# Step 2a — MAGSAC++-like circle RANSAC (global sampling)
# ─────────────────────────────────────────────────────────────────────────────

def _ransac_circle(
    pts: np.ndarray,
    n_iter: int = _RANSAC_CIRCLE_ITERS,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float, np.ndarray] | None:
    """Robust circle via 3-point minimal samples + MAGSAC soft scoring.
    Returns (cx, cy, r, inlier_mask) or None.
    """
    if len(pts) < _MIN_PTS_CIRCLE:
        return None

    rng   = rng or np.random.default_rng()
    pts64 = pts.astype(np.float64)
    N     = len(pts64)

    best_score                = -1.0
    best_cx = best_cy = best_r = 0.0
    best_inliers              = np.zeros(N, dtype=bool)

    max_iter  = n_iter
    iteration = 0

    while iteration < max_iter:
        iteration += 1
        idx    = rng.choice(N, 3, replace=False)
        circle = _circle_from_3pts(pts64[idx[0]], pts64[idx[1]], pts64[idx[2]])
        if circle is None:
            continue

        cx, cy, r = circle
        residuals = np.abs(np.hypot(pts64[:, 0] - cx, pts64[:, 1] - cy) - r)
        score     = _magsac_score(residuals)

        if score > best_score:
            best_score                = score
            best_cx, best_cy, best_r = cx, cy, r
            best_inliers              = residuals < _INLIER_THRESHOLD_PX

            inlier_frac = best_inliers.sum() / N
            p3 = inlier_frac ** 3
            if 0.0 < p3 < 1.0:
                needed   = int(np.ceil(np.log(0.01) / np.log1p(-p3)))
                max_iter = min(n_iter, iteration + max(needed, 8))

    n_inliers = int(best_inliers.sum())
    if n_inliers < _MIN_PTS_CIRCLE:
        return None
    if n_inliers / N < _MIN_INLIER_FRAC_CIRCLE:
        return None

    return best_cx, best_cy, best_r, best_inliers


# ─────────────────────────────────────────────────────────────────────────────
# Step 2b — MAGSAC++-like line RANSAC (locality-aware sampling)
# ─────────────────────────────────────────────────────────────────────────────

def _ransac_line(
    pts: np.ndarray,
    n_iter: int = _RANSAC_LINE_ITERS,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Robust line via 2-point minimal samples + MAGSAC soft scoring.

    Points are sorted by their x+y diagonal so that local index proximity
    approximates spatial proximity.  The second sample is drawn from within
    ±_RANSAC_LINE_LOCAL_HALF of the first, biasing hypotheses toward pairs
    from the same feature rather than random cross-component pairs.

    Returns (direction_unit, point_on_line, inlier_mask) or None.
    """
    if len(pts) < _MIN_PTS_LINE:
        return None

    rng   = rng or np.random.default_rng()
    pts64 = pts.astype(np.float64)
    N     = len(pts64)

    # Sort by diagonal; RANSAC operates on sorted view, mask mapped back
    diag_order = np.argsort(pts64[:, 0] + pts64[:, 1])
    spts       = pts64[diag_order]

    best_score   = -1.0
    best_dir     = np.array([1.0, 0.0])
    best_pt      = pts64[0]
    best_sinliers = np.zeros(N, dtype=bool)  # inliers in sorted space

    max_iter  = n_iter
    iteration = 0

    while iteration < max_iter:
        iteration += 1

        # First point: any sorted index
        i1 = int(rng.integers(0, N))
        # Second point: local neighbourhood in sorted array
        lo = max(0, i1 - _RANSAC_LINE_LOCAL_HALF)
        hi = min(N, i1 + _RANSAC_LINE_LOCAL_HALF + 1)
        candidates = [j for j in range(lo, hi) if j != i1]
        if not candidates:
            continue
        i2 = int(rng.choice(candidates))

        p1, p2 = spts[i1], spts[i2]
        d = p2 - p1
        d_len = np.linalg.norm(d)
        if d_len < 1e-8:
            continue
        d    /= d_len
        n_vec = np.array([-d[1], d[0]])

        residuals = np.abs((spts - p1) @ n_vec)
        score     = _magsac_score(residuals)

        if score > best_score:
            best_score    = score
            best_dir      = d.copy()
            best_pt       = p1.copy()
            best_sinliers = residuals < _INLIER_THRESHOLD_PX

            inlier_frac = best_sinliers.sum() / N
            p2_frac = inlier_frac ** 2
            if 0.0 < p2_frac < 1.0:
                needed   = int(np.ceil(np.log(0.01) / np.log1p(-p2_frac)))
                max_iter = min(n_iter, iteration + max(needed, 5))

    n_inliers = int(best_sinliers.sum())
    if n_inliers < _MIN_PTS_LINE:
        return None
    if n_inliers / N < _MIN_INLIER_FRAC_LINE:
        return None

    # Map inlier mask back to original point ordering
    orig_inliers = np.zeros(N, dtype=bool)
    orig_inliers[diag_order[best_sinliers]] = True

    return best_dir, best_pt, orig_inliers


# ─────────────────────────────────────────────────────────────────────────────
# Step 3a — Weighted Gauss-Newton circle refinement
# ─────────────────────────────────────────────────────────────────────────────

def _weighted_circle_refine(
    pts: np.ndarray,
    cx0: float, cy0: float, r0: float,
    max_iter: int = 60, tol: float = 1e-9,
) -> tuple[float, float, float, float]:
    """Minimise Σ wᵢ(dᵢ − r)² via Gauss-Newton with w = exp(−½(res/σ)²).
    Returns (cx, cy, r, inlier_rms).  Raises ValueError on degenerate result.
    """
    if len(pts) < _MIN_PTS_CIRCLE:
        raise ValueError("too few inlier points for circle refinement")

    pts64      = pts.astype(np.float64)
    cx, cy, r  = float(cx0), float(cy0), float(r0)

    for _ in range(max_iter):
        dx     = pts64[:, 0] - cx
        dy     = pts64[:, 1] - cy
        d      = np.hypot(dx, dy)
        d_safe = np.where(d < 1e-10, 1e-10, d)

        res = d - r
        w   = np.exp(-0.5 * (res / _SIGMA_REF) ** 2)

        J    = np.column_stack([-dx / d_safe, -dy / d_safe, -np.ones(len(pts64))])
        WJ   = w[:, None] * J
        JtWJ = WJ.T @ J
        JtWr = WJ.T @ res
        try:
            delta = np.linalg.solve(JtWJ, -JtWr)
        except np.linalg.LinAlgError:
            break

        cx += delta[0]; cy += delta[1]; r += delta[2]
        if np.linalg.norm(delta) < tol:
            break

    if r <= 0.0 or r > _MAX_RADIUS_PX:
        raise ValueError(f"refinement produced invalid radius r={r:.2f}")

    dx  = pts64[:, 0] - cx
    dy  = pts64[:, 1] - cy
    res = np.abs(np.hypot(dx, dy) - r)
    mask = res < _INLIER_THRESHOLD_PX
    rms  = float(np.sqrt(np.mean(res[mask] ** 2))) if mask.any() else float(np.sqrt(np.mean(res ** 2)))
    return float(cx), float(cy), float(r), rms


# ─────────────────────────────────────────────────────────────────────────────
# Step 3b — Weighted TLS line refinement
# ─────────────────────────────────────────────────────────────────────────────

def _weighted_line_refine(
    pts: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Weighted TLS via sqrt(w)-scaled SVD.  Returns (p1, p2, inlier_rms)."""
    if len(pts) < _MIN_PTS_LINE:
        raise ValueError("too few inlier points for line refinement")

    pts64 = pts.astype(np.float64)

    # Initial unweighted pass → distances → weights
    mean0     = pts64.mean(axis=0)
    c0        = pts64 - mean0
    _, _, vt0 = np.linalg.svd(c0, full_matrices=False)
    n_vec0    = np.array([-vt0[0, 1], vt0[0, 0]])
    perp0     = np.abs(c0 @ n_vec0)
    w         = np.exp(-0.5 * (perp0 / _SIGMA_REF) ** 2)

    w_sum  = w.sum()
    w_mean = (w[:, None] * pts64).sum(axis=0) / w_sum

    wc     = pts64 - w_mean
    sqrt_w = np.sqrt(w / w_sum)
    _, _, vt = np.linalg.svd(sqrt_w[:, None] * wc, full_matrices=False)
    direction = vt[0]

    proj = wc @ direction
    p1   = w_mean + proj.min() * direction
    p2   = w_mean + proj.max() * direction

    n_vec     = np.array([-direction[1], direction[0]])
    residuals = np.abs(wc @ n_vec)
    mask      = residuals < _INLIER_THRESHOLD_PX
    rms       = float(np.sqrt(np.mean(residuals[mask] ** 2))) if mask.any() else float(np.sqrt(np.mean(residuals ** 2)))

    return (float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1])), rms


# ─────────────────────────────────────────────────────────────────────────────
# Arc angular analysis
# ─────────────────────────────────────────────────────────────────────────────

def _angular_gap(
    cx: float, cy: float, pts: np.ndarray
) -> tuple[float, float, float]:
    """Return (arc_start_deg, arc_span_deg, largest_gap_deg).
    Screen convention: right=0°, down=+90°.
    """
    angles   = np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
    sorted_a = np.sort(angles)
    n        = len(sorted_a)

    diffs       = np.diff(np.append(sorted_a, sorted_a[0] + 360.0))
    gap_idx     = int(np.argmax(diffs))
    largest_gap = float(diffs[gap_idx])

    arc_start = float(sorted_a[(gap_idx + 1) % n])
    arc_end   = float(sorted_a[gap_idx])
    arc_span  = arc_end - arc_start
    if arc_span <= 0:
        arc_span += 360.0

    return arc_start, arc_span, largest_gap


# ─────────────────────────────────────────────────────────────────────────────
# Internal: full pipeline, returns (MeasureResult, inlier_mask) or None
# The inlier mask covers the *input* pts and is used for multi-shape stripping.
# ─────────────────────────────────────────────────────────────────────────────

def _fit_circle_arc_wm(
    pts: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[MeasureResult, np.ndarray] | None:
    """RANSAC → weighted refine → density + quality checks.
    Returns (result, inlier_mask_on_pts) or None.
    """
    ransac = _ransac_circle(pts, rng=rng)
    if ransac is None:
        return None
    cx0, cy0, r0, ransac_inliers = ransac
    try:
        cx, cy, r, rms = _weighted_circle_refine(pts[ransac_inliers], cx0, cy0, r0)
    except ValueError:
        return None

    # Final inlier mask on *all* input points (used for stripping and arc geometry).
    # Computed on the REFINED centre/radius so it is more accurate than the
    # RANSAC mask and ensures n_points is consistent with what gets stripped.
    dx           = pts[:, 0].astype(np.float64) - cx
    dy           = pts[:, 1].astype(np.float64) - cy
    final_inlier = np.abs(np.hypot(dx, dy) - r) < _INLIER_THRESHOLD_PX
    n_final      = int(final_inlier.sum())

    if n_final < _MIN_PTS_CIRCLE:
        return None

    # Arc / circle classification from FINAL inliers (more accurate coverage)
    arc_start, arc_span, largest_gap = _angular_gap(cx, cy, pts[final_inlier])
    kind = ShapeKind.CIRCLE if largest_gap < _FULL_CIRCLE_GAP_DEG else ShapeKind.ARC

    # Density check on final inliers
    arc_span_used = 360.0 if kind == ShapeKind.CIRCLE else arc_span
    shape_length  = r * np.radians(arc_span_used)
    if (shape_length > 0
            and n_final < _MIN_DENSITY_N_EXEMPT
            and n_final / shape_length < _MIN_DENSITY):
        return None

    result = MeasureResult(
        kind=kind,
        residual_rms=rms,
        n_points=n_final,          # consistent with final_inlier mask
        cx=cx, cy=cy, radius=r,
        arc_start_deg=0.0       if kind == ShapeKind.CIRCLE else arc_start,
        arc_span_deg=360.0      if kind == ShapeKind.CIRCLE else arc_span,
    )
    return result, final_inlier


def _fit_line_wm(
    pts: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[MeasureResult, np.ndarray] | None:
    """RANSAC → weighted TLS refine → density check.
    Returns (result, inlier_mask_on_pts) or None.
    """
    ransac = _ransac_line(pts, rng=rng)
    if ransac is None:
        return None
    _, _, ransac_inliers = ransac
    try:
        p1, p2, rms = _weighted_line_refine(pts[ransac_inliers])
    except ValueError:
        return None

    p1_arr    = np.array(p1, dtype=np.float64)
    p2_arr    = np.array(p2, dtype=np.float64)
    length    = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    if length < 1e-8:
        return None
    direction = (p2_arr - p1_arr) / length
    normal    = np.array([-direction[1], direction[0]])
    pts64     = pts.astype(np.float64)

    # Final inlier mask: perpendicular distance only (no projection clamp).
    # Endpoints are then derived from the inlier projections so they always
    # span the true inlier extent even when RANSAC found only a subset.
    perp         = np.abs((pts64 - p1_arr) @ normal)
    final_inlier = perp < _INLIER_THRESHOLD_PX
    n_final      = int(final_inlier.sum())

    if n_final < _MIN_PTS_LINE:
        return None

    # Recompute p1/p2 from final inlier projections for accurate endpoints
    proj_in = (pts64[final_inlier] - p1_arr) @ direction
    p1_new  = p1_arr + proj_in.min() * direction
    p2_new  = p1_arr + proj_in.max() * direction
    p1      = (float(p1_new[0]), float(p1_new[1]))
    p2      = (float(p2_new[0]), float(p2_new[1]))
    length  = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))

    # Density check on final inliers
    if (length > 0
            and n_final < _MIN_DENSITY_N_EXEMPT
            and n_final / length < _MIN_DENSITY):
        return None

    result = MeasureResult(
        kind=ShapeKind.LINE,
        residual_rms=rms,
        n_points=n_final,          # consistent with final_inlier mask
        line_p1=p1, line_p2=p2,
    )
    return result, final_inlier


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Select best candidate
# ─────────────────────────────────────────────────────────────────────────────

def _select_best(
    line_pair: tuple[MeasureResult, np.ndarray] | None,
    circ_pair: tuple[MeasureResult, np.ndarray] | None,
) -> tuple[MeasureResult, np.ndarray] | None:
    """Choose between line and arc/circle using score + hard guards.

    Score = n_inliers² / rms (with shape-type bonuses/penalties).
    Hard arc guards (radius, span) override scoring for lens-distortion arcs.
    """
    if line_pair is None:
        return circ_pair
    if circ_pair is None:
        return line_pair

    line_r, line_m = line_pair
    circ_r, circ_m = circ_pair

    # Full circles: pure score comparison (no lens-distortion guard needed)
    if circ_r.kind == ShapeKind.CIRCLE:
        return circ_pair if _shape_score(circ_r) >= _shape_score(line_r) else line_pair

    # Arc hard guards (large radius OR shallow span → lens distortion)
    if circ_r.radius > _MAX_ARC_RADIUS_PREFER_LINE:
        return line_pair
    if circ_r.arc_span_deg < _MIN_ARC_SPAN_DEG:
        return line_pair

    # Score-based selection: n²/rms rewards both coverage and precision.
    # This correctly handles the case where the arc has more inliers but
    # slightly higher RMS than a partial line fit through the arc's chord.
    return circ_pair if _shape_score(circ_r) >= _shape_score(line_r) else line_pair


def _fit_auto_wm(
    pts: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[MeasureResult, np.ndarray] | None:
    """Full pipeline: both fits → select → quality gate.
    Returns (result, inlier_mask) or None.
    """
    if len(pts) < _MIN_PTS_LINE:
        return None

    rng = rng or np.random.default_rng()

    line_pair = _fit_line_wm(pts, rng=rng)

    circ_pair = None
    if len(pts) >= _MIN_PTS_CIRCLE:
        circ_pair = _fit_circle_arc_wm(pts, rng=rng)

    best = _select_best(line_pair, circ_pair)
    if best is None:
        return None

    result, mask = best
    return (result, mask) if result.residual_rms <= _MAX_RESIDUAL_PX else None


# ─────────────────────────────────────────────────────────────────────────────
# Public: individual shape fits (backward-compatible API)
# ─────────────────────────────────────────────────────────────────────────────

def fit_circle(pts: np.ndarray) -> MeasureResult | None:
    """Fit a full circle using robust RANSAC + weighted Gauss-Newton."""
    pair = _fit_circle_arc_wm(pts)
    if pair is None:
        return None
    r, _ = pair
    return MeasureResult(
        kind=ShapeKind.CIRCLE, residual_rms=r.residual_rms,
        n_points=r.n_points, cx=r.cx, cy=r.cy, radius=r.radius,
        arc_start_deg=0.0, arc_span_deg=360.0,
    )


def fit_arc(pts: np.ndarray) -> MeasureResult | None:
    """Fit a circular arc using robust RANSAC + weighted Gauss-Newton."""
    pair = _fit_circle_arc_wm(pts)
    if pair is None:
        return None
    r, _ = pair
    return MeasureResult(
        kind=ShapeKind.ARC, residual_rms=r.residual_rms,
        n_points=r.n_points, cx=r.cx, cy=r.cy, radius=r.radius,
        arc_start_deg=r.arc_start_deg, arc_span_deg=r.arc_span_deg,
    )


def fit_line(pts: np.ndarray) -> MeasureResult | None:
    """Fit a line using robust RANSAC + weighted TLS."""
    pair = _fit_line_wm(pts)
    return pair[0] if pair is not None else None


def fit_auto(pts: np.ndarray) -> MeasureResult | None:
    """Full auto-fit (public API, returns result only)."""
    pair = _fit_auto_wm(pts)
    return pair[0] if pair is not None else None


def fit_shape(pts: np.ndarray, kind: str = "auto") -> MeasureResult | None:
    """Dispatch to the appropriate fitter. kind ∈ {auto, circle, arc, line}."""
    k = kind.lower().strip()
    if k == "circle": return fit_circle(pts)
    if k == "arc":    return fit_arc(pts)
    if k == "line":   return fit_line(pts)
    return fit_auto(pts)


# ─────────────────────────────────────────────────────────────────────────────
# Public: stroke-brush utilities (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def interpolate_stroke(pts: np.ndarray, spacing: float = 2.0) -> np.ndarray:
    """Densify pts so consecutive points are at most spacing px apart."""
    if len(pts) < 2:
        return pts.astype(np.float32)
    dense: list[np.ndarray] = [pts[0]]
    for i in range(1, len(pts)):
        p0, p1 = pts[i - 1].astype(np.float64), pts[i].astype(np.float64)
        d = np.linalg.norm(p1 - p0)
        if d > spacing:
            n = max(1, int(np.ceil(d / spacing)))
            for t in np.linspace(0, 1, n + 1)[1:]:
                dense.append(p0 + t * (p1 - p0))
        else:
            dense.append(p1)
    return np.array(dense, dtype=np.float32)


def collect_near_stroke(
    edge_pts: np.ndarray,
    stroke_pts: np.ndarray,
    radius: float = STROKE_BRUSH_RADIUS_PX,
) -> np.ndarray:
    """Return all edge_pts within radius pixels of any point on stroke_pts."""
    if len(edge_pts) == 0 or len(stroke_pts) == 0:
        return np.empty((0, 2), dtype=np.float32)
    tree  = cKDTree(stroke_pts.astype(np.float64))
    dists, _ = tree.query(edge_pts.astype(np.float64), k=1, workers=-1)
    return edge_pts[dists <= radius]


# ─────────────────────────────────────────────────────────────────────────────
# Public: auto-detection with inlier stripping (Run button)
# ─────────────────────────────────────────────────────────────────────────────

def auto_detect_shapes(
    edge_image: np.ndarray,
    edge_points: np.ndarray,
    max_results: int = 40,
) -> list[MeasureResult]:
    """Segment edge_image into connected components; robustly fit each one.

    After accepting a shape its inliers are stripped from the component and
    the loop repeats up to _MAX_SHAPES_PER_COMPONENT times.  This lets a
    single L-shaped component (line meeting an arc at a corner) yield two
    distinct shapes rather than forcing a single poor fit.

    Sub-pixel points are mapped to components by rounded pixel coordinate so
    concentric inner/outer circle edges remain in separate clusters.
    """
    if edge_points is None or len(edge_points) == 0:
        return []

    H, W = edge_image.shape[:2]
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        edge_image, connectivity=8
    )

    ix = np.clip(np.round(edge_points[:, 0]).astype(np.int32), 0, W - 1)
    iy = np.clip(np.round(edge_points[:, 1]).astype(np.int32), 0, H - 1)
    pt_labels = labels[iy, ix]

    results: list[MeasureResult] = []
    rng = np.random.default_rng()

    for lbl in range(1, n_labels):
        if int(stats[lbl, cv2.CC_STAT_AREA]) < _MIN_COMPONENT_AREA:
            continue

        remaining = edge_points[pt_labels == lbl]

        for _ in range(_MAX_SHAPES_PER_COMPONENT):
            if len(remaining) < _MIN_PTS_LINE:
                break

            pair = _fit_auto_wm(remaining, rng=rng)
            if pair is None:
                break

            result, inlier_mask = pair
            results.append(result)

            # Strip inliers; stop if no progress
            n_stripped = int(inlier_mask.sum())
            if n_stripped < _MIN_PTS_LINE:
                break
            remaining = remaining[~inlier_mask]

    results.sort(key=lambda r: r.residual_rms)
    return results[:max_results]
