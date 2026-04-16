"""
VideoFIT — DXF Service
Business logic for loading and parsing DXF files into pixel-space polylines.
The DxfData dataclass (pure structure) lives in app.models.dxf_model.
"""

from __future__ import annotations

import collections

import ezdxf
from ezdxf import path
import numpy as np

from app.models.dxf import Dxf


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
    DxfData
        Dataclass holding pixel-space polylines and the DXF bounding-box centre.
    """
    doc = ezdxf.readfile(filepath)
    msp = doc.modelspace()

    # ── Step 1: Extract all geometry universally ──────────────────────
    segments = []

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
            for i in range(len(pts) - 1):
                segments.append((pts[i].x, pts[i].y, pts[i + 1].x, pts[i + 1].y))
        except TypeError:
            continue

    if not segments:
        raise ValueError("No supported geometry found in the DXF file.")

    # ── Step 2: Bounding-box centre (mm space) ────────────────────────
    seg_array = np.array(segments, dtype=np.float32)
    x_coords = np.concatenate([seg_array[:, 0], seg_array[:, 2]])
    y_coords = np.concatenate([seg_array[:, 1], seg_array[:, 3]])

    dxf_cx = (x_coords.min() + x_coords.max()) / 2.0
    dxf_cy = (y_coords.min() + y_coords.max()) / 2.0

    # ── Step 3: Coordinate transform → pixel space ───────────────────
    H, W = canvas_shape
    canvas_cx, canvas_cy = W / 2.0, H / 2.0

    x_px = canvas_cx + (x_coords - dxf_cx) * px_per_mm
    y_px = canvas_cy - (y_coords - dxf_cy) * px_per_mm  # Y-flip: DXF Y-up → pixel Y-down

    px_segments = np.column_stack([
        x_px[: len(segments)], y_px[: len(segments)],
        x_px[len(segments) :], y_px[len(segments) :],
    ])

    # ── Step 4: O(N) topology chaining ───────────────────────────────
    adjacency: dict = collections.defaultdict(list)

    def pt_key(x, y):
        return (round(x, 1), round(y, 1))

    for i, (x0, y0, x1, y1) in enumerate(px_segments):
        adjacency[pt_key(x0, y0)].append((i, x1, y1))
        adjacency[pt_key(x1, y1)].append((i, x0, y0))

    polylines: list[np.ndarray] = []
    used_segments: set[int] = set()

    for i, (x0, y0, x1, y1) in enumerate(px_segments):
        if i in used_segments:
            continue

        chain = [(x0, y0), (x1, y1)]
        used_segments.add(i)

        while True:
            last_pt = chain[-1]
            key = pt_key(*last_pt)
            connected = next(
                (seg for seg in adjacency[key] if seg[0] not in used_segments), None
            )
            if connected:
                seg_idx, next_x, next_y = connected
                used_segments.add(seg_idx)
                chain.append((next_x, next_y))
            else:
                break

        polylines.append(np.array(chain, dtype=np.float32))

    return Dxf(
        polylines=polylines,
        dxf_center_mm=(float(dxf_cx), float(dxf_cy)),
        doc=doc,
        px_per_mm=px_per_mm,
        canvas_shape=canvas_shape,
    )
