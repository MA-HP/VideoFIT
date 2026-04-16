"""
VideoFIT — DXF Overlay
Draws fitted DXF polylines onto a QGraphicsScene using cosmetic pens.

Two rendering modes:
  • Preview  — cyan outlines (pre-alignment)
  • Heatmap  — segment-by-segment green→yellow→red based on distance
               to the real edge (post-alignment), matching the POC exactly.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QPainterPath, QPen
from PySide6.QtWidgets import (
    QGraphicsItem, QGraphicsLineItem, QGraphicsPathItem, QGraphicsScene,
)

from app.models.fit_result import FitResult


_PREVIEW_COLOR = QColor(0, 200, 255, 255)   # cyan
_PEN_WIDTH = 2                               # cosmetic px


class DxfOverlay:
    """Manages QGraphicsItems representing the DXF overlay."""

    def __init__(self, scene: QGraphicsScene) -> None:
        self._scene = scene
        self._items: list[QGraphicsItem] = []

    # ── Public ───────────────────────────────────────────────────────

    def clear(self) -> None:
        """Remove all overlay items from the scene."""
        for item in self._items:
            self._scene.removeItem(item)
        self._items.clear()

    def draw_preview(self, polylines: list[np.ndarray]) -> None:
        """
        Draw DXF polylines as cyan outlines (before alignment).
        """
        self.clear()

        pen = QPen(_PREVIEW_COLOR, _PEN_WIDTH)
        pen.setCosmetic(True)

        for poly in polylines:
            if len(poly) < 2:
                continue
            path = QPainterPath()
            path.moveTo(QPointF(float(poly[0][0]), float(poly[0][1])))
            for x, y in poly[1:]:
                path.lineTo(QPointF(float(x), float(y)))

            item = QGraphicsPathItem(path)
            item.setPen(pen)
            item.setBrush(QBrush())
            item.setZValue(100)
            self._scene.addItem(item)
            self._items.append(item)

    def draw_heatmap(
        self,
        polylines: list[np.ndarray],
        result: FitResult,
    ) -> None:
        """
        Draw DXF polylines transformed by the alignment result, with each
        segment colored by its distance to the nearest real edge.

        Color logic (matches the POC):
          • ≤ 1 px  →  pure green   (perfect match)
          • 1–3 px  →  green→yellow→red gradient
          • > 3 px  →  pure red     (bad match)

        Uses QGraphicsLineItem per segment for individual coloring with
        cosmetic pens for infinite-zoom support.
        """
        self.clear()

        dist_t = result.dist_t
        if dist_t is None:
            # Fallback: draw plain green if no distance transform available
            self._draw_aligned_plain(polylines, result)
            return

        H, W = dist_t.shape
        a = np.radians(result.angle_deg)
        cos_t = np.cos(a)
        sin_t = np.sin(a)
        cx = result.dxf_cx
        cy = result.dxf_cy
        tx = result.tx
        ty = result.ty

        for poly in polylines:
            if len(poly) < 2:
                continue

            # Transform all vertices (vectorised)
            xs = poly[:, 0] - cx
            ys = poly[:, 1] - cy
            nx = cos_t * xs - sin_t * ys + cx + tx
            ny = sin_t * xs + cos_t * ys + cy + ty
            pts = np.stack([nx, ny], axis=1)

            # Draw segment by segment with heatmap color
            for i in range(len(pts) - 1):
                # Sample distance at the start point of the segment
                px_x = int(np.clip(pts[i][0], 0, W - 1))
                px_y = int(np.clip(pts[i][1], 0, H - 1))
                d = float(dist_t[px_y, px_x])

                # Color mapping: green → yellow → red
                color = _distance_to_color(d)

                line = QGraphicsLineItem(
                    float(pts[i][0]), float(pts[i][1]),
                    float(pts[i + 1][0]), float(pts[i + 1][1]),
                )
                pen = QPen(color, _PEN_WIDTH)
                pen.setCosmetic(True)
                pen.setCapStyle(Qt.RoundCap)
                line.setPen(pen)
                line.setZValue(100)

                self._scene.addItem(line)
                self._items.append(line)

    # ── Private helpers ──────────────────────────────────────────────

    def _draw_aligned_plain(
        self,
        polylines: list[np.ndarray],
        result: FitResult,
    ) -> None:
        """Fallback: draw aligned polylines in solid green."""
        pen = QPen(QColor(0, 255, 0, 255), _PEN_WIDTH)
        pen.setCosmetic(True)

        a = np.radians(result.angle_deg)
        cos_t = np.cos(a)
        sin_t = np.sin(a)
        cx, cy = result.dxf_cx, result.dxf_cy
        tx, ty = result.tx, result.ty

        for poly in polylines:
            if len(poly) < 2:
                continue
            xs = poly[:, 0] - cx
            ys = poly[:, 1] - cy
            nx = cos_t * xs - sin_t * ys + cx + tx
            ny = sin_t * xs + cos_t * ys + cy + ty

            path = QPainterPath()
            path.moveTo(QPointF(float(nx[0]), float(ny[0])))
            for j in range(1, len(nx)):
                path.lineTo(QPointF(float(nx[j]), float(ny[j])))

            item = QGraphicsPathItem(path)
            item.setPen(pen)
            item.setBrush(QBrush())
            item.setZValue(100)
            self._scene.addItem(item)
            self._items.append(item)


def _distance_to_color(d: float) -> QColor:
    """
    Map a distance-transform value to a heatmap color.

      ≤ 1.0 px  →  pure green  (0, 255, 0)
      1–3 px    →  green → yellow → red gradient
      > 3.0 px  →  pure red    (255, 0, 0)
    """
    if d <= 1.0:
        return QColor(0, 255, 0)
    elif d <= 3.0:
        t = (d - 1.0) / 2.0          # 0.0 → 1.0
        r = int(255 * t)
        g = int(255 * (1.0 - t))
        return QColor(r, g, 0)
    else:
        return QColor(255, 0, 0)

