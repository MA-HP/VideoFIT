"""
VideoFIT — Shape Fit Service
True sub-pixel shape fitting: circle, arc, line, and automatic selection.

Fitting hierarchy
─────────────────
1.  Algebraic initialisation (Kasa linear system, closed-form, unbiased for
    uniform arc coverage) — fast and gives a good starting point.
2.  Geometric refinement (Gauss-Newton) — minimises the true orthogonal
    distance to the shape, eliminating the systematic drift of pure
    algebraic fits on non-uniform point distributions.

All functions operate on (N, 2) float32/float64 arrays of sub-pixel edge
points and return MeasureResult dataclasses.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial import cKDTree

from app.models.measure_result import MeasureResult, ShapeKind

# ── Tuneable thresholds ───────────────────────────────────────────────────────

_MIN_PTS_LINE   = 5       # minimum points to attempt any fit
_MIN_PTS_CIRCLE = 8       # minimum points to attempt a circle / arc fit

# Maximum RMS geometric residual accepted as a "good" fit during auto-detect
_MAX_RESIDUAL_PX = 3.0

# Reject circles with radii above this (image-diagonal guard)
_MAX_RADIUS_PX = 8000.0

# If the largest angular gap is below this threshold the cluster covers enough
# of the circumference to be called a full circle rather than an arc.
_FULL_CIRCLE_GAP_DEG = 60.0

# Minimum connected-component pixel area to process during auto-detect
_MIN_COMPONENT_AREA = 30

# Stroke brush radius (image pixels): edge points within this distance of
# any point on the drawn stroke are included in the manual fit.
STROKE_BRUSH_RADIUS_PX: float = 15.0


# ─────────────────────────────────────────────────────────────────────────────
# Internal: algebraic initialisation (Kasa 1976)
# ─────────────────────────────────────────────────────────────────────────────

def _kasa_init(pts: np.ndarray) -> tuple[float, float, float]:
    """
    Solve  2·cx·x + 2·cy·y + c = x² + y²  in the least-squares sense.
    Returns (cx, cy, radius).  Raises ValueError on rank-deficient input.
    """
    x = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)
    A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
    b = x ** 2 + y ** 2
    result, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    if rank < 3:
        raise ValueError("rank-deficient matrix — points are collinear or identical")
    cx, cy, c = result
    r_sq = c + cx ** 2 + cy ** 2
    if r_sq <= 0:
        raise ValueError("degenerate circle (r² ≤ 0)")
    return float(cx), float(cy), float(np.sqrt(r_sq))


# ─────────────────────────────────────────────────────────────────────────────
# Internal: geometric refinement (Gauss-Newton)
# ─────────────────────────────────────────────────────────────────────────────

def _geometric_circle_fit(
    pts: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-9,
) -> tuple[float, float, float, float]:
    """
    Minimise sum of (dist_i − r)² via Gauss-Newton.

    Parameters
    ----------
    pts : (N, 2) float array of sub-pixel edge coordinates.

    Returns
    -------
    (cx, cy, radius, rms_residual_px)
    Raises ValueError when the initialisation or the iteration fails.
    """
    cx, cy, r = _kasa_init(pts)
    pts64 = pts.astype(np.float64)

    for _ in range(max_iter):
        dx = pts64[:, 0] - cx
        dy = pts64[:, 1] - cy
        d = np.hypot(dx, dy)
        d_safe = np.where(d < 1e-10, 1e-10, d)

        res = d - r                                    # (N,) residuals
        J = np.column_stack([-dx / d_safe,             # ∂res/∂cx
                              -dy / d_safe,             # ∂res/∂cy
                              -np.ones(len(pts64))])    # ∂res/∂r

        JtJ = J.T @ J
        Jtr = J.T @ res
        try:
            delta = np.linalg.solve(JtJ, -Jtr)
        except np.linalg.LinAlgError:
            break

        cx += delta[0]
        cy += delta[1]
        r  += delta[2]

        if np.linalg.norm(delta) < tol:
            break

    if r <= 0 or r > _MAX_RADIUS_PX:
        raise ValueError(f"geometric fit produced invalid radius r={r:.2f}")

    dx = pts64[:, 0] - cx
    dy = pts64[:, 1] - cy
    d  = np.hypot(dx, dy)
    rms = float(np.sqrt(np.mean((d - r) ** 2)))
    return float(cx), float(cy), float(r), rms


# ─────────────────────────────────────────────────────────────────────────────
# Internal: arc angular analysis
# ─────────────────────────────────────────────────────────────────────────────

def _angular_gap(
    cx: float, cy: float, pts: np.ndarray
) -> tuple[float, float, float]:
    """
    Return (arc_start_deg, arc_span_deg, largest_gap_deg).

    Angles use the screen-coordinate atan2 convention:
      right = 0°, down = +90°  (Y increases downward).
    arc_span_deg is always positive and represents the CW sweep on screen.
    """
    angles = np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
    sorted_a = np.sort(angles)
    n = len(sorted_a)

    diffs = np.diff(np.append(sorted_a, sorted_a[0] + 360.0))
    gap_idx = int(np.argmax(diffs))
    largest_gap = float(diffs[gap_idx])

    arc_start = float(sorted_a[(gap_idx + 1) % n])
    arc_end   = float(sorted_a[gap_idx])

    arc_span = arc_end - arc_start
    if arc_span <= 0:
        arc_span += 360.0

    return arc_start, arc_span, largest_gap


# ─────────────────────────────────────────────────────────────────────────────
# Public: individual shape fits
# ─────────────────────────────────────────────────────────────────────────────

def fit_circle(pts: np.ndarray) -> MeasureResult | None:
    """Fit a full circle to *pts* using sub-pixel geometric optimisation."""
    if len(pts) < _MIN_PTS_CIRCLE:
        return None
    try:
        cx, cy, r, rms = _geometric_circle_fit(pts)
    except ValueError:
        return None
    return MeasureResult(
        kind=ShapeKind.CIRCLE,
        residual_rms=rms, n_points=len(pts),
        cx=cx, cy=cy, radius=r,
        arc_start_deg=0.0, arc_span_deg=360.0,
    )


def fit_arc(pts: np.ndarray) -> MeasureResult | None:
    """Fit a circular arc to *pts* using sub-pixel geometric optimisation."""
    if len(pts) < _MIN_PTS_CIRCLE:
        return None
    try:
        cx, cy, r, rms = _geometric_circle_fit(pts)
    except ValueError:
        return None
    arc_start, arc_span, _ = _angular_gap(cx, cy, pts)
    return MeasureResult(
        kind=ShapeKind.ARC,
        residual_rms=rms, n_points=len(pts),
        cx=cx, cy=cy, radius=r,
        arc_start_deg=arc_start, arc_span_deg=arc_span,
    )


def fit_line(pts: np.ndarray) -> MeasureResult | None:
    """
    Fit a line segment via orthogonal distance regression (PCA / total
    least-squares).  Already optimal — no iterative refinement needed.
    """
    if len(pts) < _MIN_PTS_LINE:
        return None
    pts64 = pts.astype(np.float64)
    mean = pts64.mean(axis=0)
    centered = pts64 - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]

    proj = centered @ direction
    p1 = mean + proj.min() * direction
    p2 = mean + proj.max() * direction

    residuals = centered - proj[:, None] * direction[None, :]
    rms = float(np.sqrt((residuals ** 2).sum(axis=1).mean()))

    return MeasureResult(
        kind=ShapeKind.LINE,
        residual_rms=rms, n_points=len(pts),
        line_p1=(float(p1[0]), float(p1[1])),
        line_p2=(float(p2[0]), float(p2[1])),
    )


def fit_auto(pts: np.ndarray) -> MeasureResult | None:
    """
    Try circle/arc and line; return whichever geometric RMS is lower.
    Circles and arcs are distinguished by the largest angular gap in the
    fitted circle (gap < 60° → full circle, otherwise arc).
    When both residuals exceed *_MAX_RESIDUAL_PX*, the best candidate is
    still returned (so the caller decides how to use it).
    """
    candidates: list[MeasureResult] = []

    line_r = fit_line(pts)
    if line_r is not None:
        candidates.append(line_r)

    if len(pts) >= _MIN_PTS_CIRCLE:
        try:
            cx, cy, r, rms = _geometric_circle_fit(pts)
            arc_start, arc_span, largest_gap = _angular_gap(cx, cy, pts)
            kind = ShapeKind.CIRCLE if largest_gap < _FULL_CIRCLE_GAP_DEG else ShapeKind.ARC
            candidates.append(MeasureResult(
                kind=kind,
                residual_rms=rms, n_points=len(pts),
                cx=cx, cy=cy, radius=r,
                arc_start_deg=0.0 if kind == ShapeKind.CIRCLE else arc_start,
                arc_span_deg=360.0 if kind == ShapeKind.CIRCLE else arc_span,
            ))
        except ValueError:
            pass

    if not candidates:
        return None

    valid = [r for r in candidates if r.residual_rms <= _MAX_RESIDUAL_PX]
    return min(valid or candidates, key=lambda r: r.residual_rms)


def fit_shape(pts: np.ndarray, kind: str = "auto") -> MeasureResult | None:
    """
    Dispatch to the appropriate fitter.
    *kind* ∈ {``"auto"``, ``"circle"``, ``"arc"``, ``"line"``} (case-insensitive).
    """
    k = kind.lower().strip()
    if k == "circle":
        return fit_circle(pts)
    if k == "arc":
        return fit_arc(pts)
    if k == "line":
        return fit_line(pts)
    return fit_auto(pts)


# ─────────────────────────────────────────────────────────────────────────────
# Public: stroke-brush utilities
# ─────────────────────────────────────────────────────────────────────────────

def interpolate_stroke(pts: np.ndarray, spacing: float = 2.0) -> np.ndarray:
    """
    Densify *pts* so that consecutive points are at most *spacing* px apart.
    This ensures the brush covers all edge pixels even when the mouse moved
    quickly (few events per image pixel at low zoom).
    """
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
    """
    Return all *edge_pts* within *radius* pixels of any point on *stroke_pts*.

    Uses a cKDTree on the (already-dense) stroke path for O(N log M) queries.
    """
    if len(edge_pts) == 0 or len(stroke_pts) == 0:
        return np.empty((0, 2), dtype=np.float32)
    tree = cKDTree(stroke_pts.astype(np.float64))
    dists, _ = tree.query(edge_pts.astype(np.float64), k=1, workers=-1)
    return edge_pts[dists <= radius]


# ─────────────────────────────────────────────────────────────────────────────
# Public: auto-detection (Run button)
# ─────────────────────────────────────────────────────────────────────────────

def auto_detect_shapes(
    edge_image: np.ndarray,
    edge_points: np.ndarray,
    max_results: int = 40,
) -> list[MeasureResult]:
    """
    Segment the binary *edge_image* into connected components and geometrically
    fit a shape to each significant cluster.

    Sub-pixel points are assigned to components by mapping each point's
    rounded pixel coordinate to the CC label image — not by bounding-box
    intersection, which was incorrect and caused inner/outer circles to
    contaminate each other.

    Parameters
    ----------
    edge_image  : uint8 binary edge map (non-zero = edge pixel).
    edge_points : (N, 2) float32 sub-pixel positions from the same frame.
    max_results : cap on returned shapes, ordered by ascending RMS residual.
    """
    if edge_points is None or len(edge_points) == 0:
        return []

    H, W = edge_image.shape[:2]
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        edge_image, connectivity=8
    )

    # ── Map each sub-pixel point to its parent pixel's CC label ──────────
    # Rounding to nearest integer gives the correct pixel, which is always
    # within the (dilated) edge region that generated the sub-pixel refine.
    ix = np.clip(np.round(edge_points[:, 0]).astype(np.int32), 0, W - 1)
    iy = np.clip(np.round(edge_points[:, 1]).astype(np.int32), 0, H - 1)
    pt_labels = labels[iy, ix]

    results: list[MeasureResult] = []

    for lbl in range(1, n_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < _MIN_COMPONENT_AREA:
            continue

        cluster_pts = edge_points[pt_labels == lbl]
        if len(cluster_pts) < _MIN_PTS_LINE:
            continue

        result = fit_auto(cluster_pts)
        if result is not None and result.residual_rms <= _MAX_RESIDUAL_PX:
            results.append(result)

    results.sort(key=lambda r: r.residual_rms)
    return results[:max_results]
