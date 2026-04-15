from __future__ import annotations

import collections
from dataclasses import dataclass, field

import ezdxf
from ezdxf import path
import numpy as np


@dataclass
class DxfData:
    """Result of parsing a DXF file."""
    polylines: list[np.ndarray] = field(default_factory=list)  # Nx2 float32, pixel coords
    dxf_center_mm: tuple[float, float] = (0.0, 0.0)  # centre of DXF bounding box (mm)


def load_dxf(
        filepath: str,
        px_per_mm: float,
        canvas_shape: tuple[int, int] = (3648, 5472),
) -> DxfData:
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
    """
    doc = ezdxf.readfile(filepath)
    msp = doc.modelspace()

    # ── Step 1: Extract all geometry universally ──────────────────────
    segments = []

    # Helper to unpack blocks (INSERTs) into base geometry
    def iter_all_entities(layout):
        for e in layout:
            if e.dxftype() == 'INSERT':
                try:
                    # Explode the block reference into virtual entities
                    yield from e.virtual_entities()
                except Exception:
                    pass  # Safely ignore broken block references
            else:
                yield e

    for e in iter_all_entities(msp):
        try:
            # make_path handles Lines, Arcs, Circles, Ellipses, Splines, Polylines natively
            p = path.make_path(e)

            # Flatten with a 0.05mm tolerance
            pts = list(p.flattening(distance=0.05))
            if len(pts) < 2:
                continue

            for i in range(len(pts) - 1):
                segments.append((pts[i].x, pts[i].y, pts[i + 1].x, pts[i + 1].y))
        except TypeError:
            # Entity doesn't support path generation (e.g., Text, Hatch, Dimensions)
            continue

    if not segments:
        raise ValueError("No supported geometry found in the DXF file.")

    # ── Step 2: Compute DXF bounding-box efficiently ──────────────────
    # Vectorize the bounds calculation for speed
    seg_array = np.array(segments, dtype=np.float32)  # Shape: (N, 4)
    x_coords = np.concatenate([seg_array[:, 0], seg_array[:, 2]])
    y_coords = np.concatenate([seg_array[:, 1], seg_array[:, 3]])

    dxf_cx = (x_coords.min() + x_coords.max()) / 2.0
    dxf_cy = (y_coords.min() + y_coords.max()) / 2.0

    # ── Step 3: Vectorized coordinate transformation ──────────────────
    H, W = canvas_shape
    canvas_cx, canvas_cy = W / 2.0, H / 2.0

    # Transform all X coordinates
    x_px = canvas_cx + (x_coords - dxf_cx) * px_per_mm
    # Transform all Y coordinates (Y-flip: DXF Y-up → pixel Y-down)
    y_px = canvas_cy - (y_coords - dxf_cy) * px_per_mm

    # Rebuild segments in pixel space: [(x0, y0, x1, y1), ...]
    px_segments = np.column_stack([
        x_px[:len(segments)], y_px[:len(segments)],
        x_px[len(segments):], y_px[len(segments):]
    ])

    # ── Step 4: O(N) Topology Chaining ────────────────────────────────
    # Use a dictionary to map rounded point coordinates to segment indices
    # This completely eliminates the previous O(N^2) search bottleneck.
    adjacency = collections.defaultdict(list)

    def pt_key(x, y):
        return (round(x, 1), round(y, 1))  # 0.1 pixel tolerance for connection

    for i, (x0, y0, x1, y1) in enumerate(px_segments):
        adjacency[pt_key(x0, y0)].append((i, x1, y1))
        adjacency[pt_key(x1, y1)].append((i, x0, y0))

    polylines: list[np.ndarray] = []
    used_segments = set()

    for i, (x0, y0, x1, y1) in enumerate(px_segments):
        if i in used_segments:
            continue

        chain = [(x0, y0), (x1, y1)]
        used_segments.add(i)

        # Grow the chain forward
        while True:
            last_pt = chain[-1]
            key = pt_key(*last_pt)

            # Find an unused segment connected to this point
            connected = next((seg for seg in adjacency[key] if seg[0] not in used_segments), None)

            if connected:
                seg_idx, next_x, next_y = connected
                used_segments.add(seg_idx)
                chain.append((next_x, next_y))
            else:
                break

        polylines.append(np.array(chain, dtype=np.float32))

    return DxfData(
        polylines=polylines,
        dxf_center_mm=(float(dxf_cx), float(dxf_cy)),
    )