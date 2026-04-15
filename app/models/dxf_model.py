"""
Metrology Vision Pro — DXF Model
Parses a DXF file using ezdxf and produces pixel-space polylines.

Track A (math):  Pixel-space polylines used for the fitting engine and display.
Track B (display): Same polylines rendered as QPainterPath with cosmetic pens.

The DXF is converted from mm → pixel coordinates at load time, centred on the
canvas.  This matches the proven POC approach exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import ezdxf
import numpy as np


@dataclass
class DxfData:
    """Result of parsing a DXF file."""
    polylines: list[np.ndarray] = field(default_factory=list)  # Nx2 float32, pixel coords
    dxf_center_mm: tuple[float, float] = (0.0, 0.0)           # centre of DXF bounding box (mm)


def load_dxf(
    path: str,
    px_per_mm: float,
    canvas_shape: tuple[int, int] = (3648, 5472),
) -> DxfData:
    """
    Parse *path* and return pixel-space polylines centred on the canvas.

    Parameters
    ----------
    path : str
        Path to the .dxf file.
    px_per_mm : float
        Camera calibration (pixels per millimetre).
    canvas_shape : (H, W)
        Size of the target image in pixels.

    Returns
    -------
    DxfData
    """
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()

    # ── Step 1: collect raw DXF segments (x0, y0, x1, y1) in mm ─────
    segments: list[tuple[float, float, float, float]] = []

    for e in msp:
        t = e.dxftype()

        if t == "LINE":
            segments.append((
                e.dxf.start.x, e.dxf.start.y,
                e.dxf.end.x, e.dxf.end.y,
            ))

        elif t == "LWPOLYLINE":
            pts = list(e.get_points())
            for i in range(len(pts) - 1):
                segments.append((pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]))
            if e.closed and len(pts) > 1:
                segments.append((pts[-1][0], pts[-1][1], pts[0][0], pts[0][1]))

        elif t == "CIRCLE":
            c, r = e.dxf.center, e.dxf.radius
            pts_c = [
                (c.x + r * np.cos(a * 2 * np.pi / 60),
                 c.y + r * np.sin(a * 2 * np.pi / 60))
                for a in range(61)
            ]
            for i in range(60):
                segments.append((*pts_c[i], *pts_c[i + 1]))

        elif t == "ARC":
            c, r = e.dxf.center, e.dxf.radius
            a0 = np.radians(e.dxf.start_angle)
            a1 = np.radians(e.dxf.end_angle)
            if a1 < a0:
                a1 += 2 * np.pi
            n = max(12, int(abs(a1 - a0) / np.pi * 60))
            angles = np.linspace(a0, a1, n)
            pts_a = [(c.x + r * np.cos(a), c.y + r * np.sin(a)) for a in angles]
            for i in range(len(pts_a) - 1):
                segments.append((*pts_a[i], *pts_a[i + 1]))

        elif t == "SPLINE":
            try:
                pts_s = list(e.flattening(distance=0.05))
            except Exception:
                continue
            if len(pts_s) < 2:
                continue
            for i in range(len(pts_s) - 1):
                segments.append((pts_s[i].x, pts_s[i].y, pts_s[i + 1].x, pts_s[i + 1].y))

    if not segments:
        raise ValueError("No supported geometry found in the DXF file.")

    # ── Step 2: compute DXF bounding-box centre (mm) ────────────────
    all_xy = np.array(
        [(x, y) for x0, y0, x1, y1 in segments for x, y in [(x0, y0), (x1, y1)]]
    )
    min_xy = all_xy.min(axis=0)
    max_xy = all_xy.max(axis=0)
    dxf_cx = (min_xy[0] + max_xy[0]) / 2
    dxf_cy = (min_xy[1] + max_xy[1]) / 2

    # ── Step 3: convert mm → pixel, centred on canvas ────────────────
    H, W = canvas_shape
    canvas_cx, canvas_cy = W / 2.0, H / 2.0
    scale = px_per_mm

    def to_px(x: float, y: float) -> tuple[float, float]:
        return (
            canvas_cx + (x - dxf_cx) * scale,
            canvas_cy - (y - dxf_cy) * scale,     # Y-flip: DXF Y-up → pixel Y-down
        )

    # ── Step 4: chain segments into polylines (POC logic) ────────────
    px_segments = [
        (to_px(x0, y0), to_px(x1, y1))
        for x0, y0, x1, y1 in segments
    ]

    polylines: list[np.ndarray] = []
    remaining = list(px_segments)

    while remaining:
        seg = remaining.pop(0)
        chain = [seg[0], seg[1]]
        changed = True
        while changed:
            changed = False
            for i, s in enumerate(remaining):
                p0, p1 = s
                if np.linalg.norm(np.array(chain[-1]) - np.array(p0)) < 2:
                    chain.append(p1)
                    remaining.pop(i)
                    changed = True
                    break
                elif np.linalg.norm(np.array(chain[-1]) - np.array(p1)) < 2:
                    chain.append(p0)
                    remaining.pop(i)
                    changed = True
                    break
        polylines.append(np.array(chain, dtype=np.float32))

    return DxfData(
        polylines=polylines,
        dxf_center_mm=(dxf_cx, dxf_cy),
    )

