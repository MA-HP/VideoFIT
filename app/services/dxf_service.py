"""
VideoFIT — DXF Service
Business logic for loading and parsing DXF files into pixel-space polylines.
The DxfData dataclass (pure structure) lives in app.models.dxf_model.
"""

from __future__ import annotations

import cv2
import ezdxf
from ezdxf import path
import numpy as np

from app.models.dxf import Dxf


def _outer_silhouette_area_px(polylines: list[np.ndarray], canvas_shape: tuple[int, int]) -> float:
    """Rasterize polylines, bridge tiny gaps, and return the area enclosed by
    the largest external contour. Holes inside the part are ignored on purpose."""
    if not polylines:
        return 0.0
    H, W = canvas_shape
    canvas = np.zeros((H, W), dtype=np.uint8)
    for poly in polylines:
        if len(poly) < 2:
            continue
        pts = np.round(poly).astype(np.int32)
        cv2.polylines(canvas, [pts], isClosed=False, color=255, thickness=2)
    canvas = cv2.morphologyEx(
        canvas, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    return float(max(cv2.contourArea(c) for c in contours))


def load_dxf(
    filepath: str,
    px_per_mm: float,
    canvas_shape: tuple[int, int] = (3648, 5472),
) -> Dxf:
    """
    Parse *filepath* and return pixel-space polylines centred on the canvas.

    Parameters
    ----------
    filepath : str
        Path to the .dxf file.
    px_per_mm : float
        Camera calibration (pixels per millimetre).
    canvas_shape : (H, W)
        Size of the target image in pixels.

    Returns
    -------
    Dxf
        Dataclass holding pixel-space polylines and the DXF bounding-box centre.
    """
    doc = ezdxf.readfile(filepath)
    msp = doc.modelspace()

    # ── Step 1: Extract geometry per entity ───────────────────────────
    # Each DXF entity (LINE, ARC, SPLINE, CIRCLE, LWPOLYLINE…) becomes its own
    # polyline. We do NOT chain/merge adjacent segments — preserving entity
    # boundaries is crucial for "points of interest" in Refine mode and gives
    # the optimizer many more independently-shaped features to lock onto.
    raw_polylines_mm: list[np.ndarray] = []
    raw_layers: list[str] = []   # layer name per polyline (upper-cased)

    def iter_all_entities(layout):
        for e in layout:
            if e.dxftype() == "INSERT":
                try:
                    yield from e.virtual_entities()
                except Exception:
                    pass
            else:
                yield e

    for e in iter_all_entities(msp):
        try:
            p = path.make_path(e)
            pts = list(p.flattening(distance=0.05))
            if len(pts) < 2:
                continue
            raw_polylines_mm.append(
                np.array([[q.x, q.y] for q in pts], dtype=np.float32)
            )
            layer = getattr(e.dxf, "layer", "GLOBAL").upper()
            raw_layers.append(layer)
        except (TypeError, Exception):
            continue

    if not raw_polylines_mm:
        raise ValueError("No supported geometry found in the DXF file.")

    # ── Step 2: Bounding-box centre (mm space) ────────────────────────
    all_pts_mm = np.vstack(raw_polylines_mm)
    dxf_cx = (all_pts_mm[:, 0].min() + all_pts_mm[:, 0].max()) / 2.0
    dxf_cy = (all_pts_mm[:, 1].min() + all_pts_mm[:, 1].max()) / 2.0

    # ── Step 3: Coordinate transform → pixel space (per polyline) ────
    H, W = canvas_shape
    canvas_cx, canvas_cy = W / 2.0, H / 2.0

    polylines: list[np.ndarray] = []
    polylines_global: list[np.ndarray] = []
    polylines_refine: list[np.ndarray] = []
    polylines_rot: list[np.ndarray] = []
    polylines_pan: list[np.ndarray] = []
    for i, poly_mm in enumerate(raw_polylines_mm):
        x_px = canvas_cx + (poly_mm[:, 0] - dxf_cx) * px_per_mm
        y_px = canvas_cy - (poly_mm[:, 1] - dxf_cy) * px_per_mm  # Y-flip
        poly_px = np.column_stack([x_px, y_px]).astype(np.float32)
        polylines.append(poly_px)
        layer = raw_layers[i]
        if layer == "REFINE":
            polylines_refine.append(poly_px)
        elif layer == "ROT":
            polylines_rot.append(poly_px)
        elif layer == "PAN":
            polylines_pan.append(poly_px)
        else:
            polylines_global.append(poly_px)

    dxf_area_px = _outer_silhouette_area_px(polylines, canvas_shape)

    return Dxf(
        polylines=polylines,
        dxf_center_mm=(float(dxf_cx), float(dxf_cy)),
        doc=doc,
        px_per_mm=px_per_mm,
        canvas_shape=canvas_shape,
        polylines_global=polylines_global,
        polylines_refine=polylines_refine,
        polylines_rot=polylines_rot,
        polylines_pan=polylines_pan,
        dxf_area_px=dxf_area_px,
    )
