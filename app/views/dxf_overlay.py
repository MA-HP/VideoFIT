"""
Metrology Vision Pro — DXF Overlay
Draws fitted DXF polylines onto a QGraphicsScene using cosmetic pens
for perfect infinite-zoom inspection.

Matches the POC approach: transform polyline vertices mathematically,
then render each polyline as a QGraphicsPathItem with a cosmetic QPen.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF
from PySide6.QtGui import QBrush, QColor, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsPathItem, QGraphicsScene

from app.models.dxf_fitter import FitResult


_OVERLAY_COLOR = QColor(0, 200, 255, 255)   # cyan (matches POC)
_OVERLAY_WIDTH = 2                           # cosmetic px


class DxfOverlay:
    """Manages QGraphicsPathItems representing the DXF overlay."""

    def __init__(self, scene: QGraphicsScene) -> None:
        self._scene = scene
        self._items: list[QGraphicsPathItem] = []

    # ── Public ───────────────────────────────────────────────────────

    def clear(self) -> None:
        """Remove all overlay items from the scene."""
        for item in self._items:
            self._scene.removeItem(item)
        self._items.clear()

    def draw_preview(self, polylines: list[np.ndarray]) -> None:
        """
        Draw DXF polylines onto the scene without alignment (raw pixel coords).
        Used when user loads a DXF before running alignment.
        """
        self.clear()

        pen = QPen(_OVERLAY_COLOR, _OVERLAY_WIDTH)
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

    def draw_aligned(
        self,
        polylines: list[np.ndarray],
        result: FitResult,
    ) -> None:
        """
        Draw DXF polylines transformed by the alignment result.

        Each polyline's vertices are rotated around (dxf_cx, dxf_cy)
        then translated by (tx, ty), exactly as the POC does.
        """
        self.clear()

        pen = QPen(QColor(0, 255, 0, 255), _OVERLAY_WIDTH)
        pen.setCosmetic(True)

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

            path = QPainterPath()
            path.moveTo(QPointF(float(nx[0]), float(ny[0])))
            for i in range(1, len(nx)):
                path.lineTo(QPointF(float(nx[i]), float(ny[i])))

            item = QGraphicsPathItem(path)
            item.setPen(pen)
            item.setBrush(QBrush())
            item.setZValue(100)
            self._scene.addItem(item)
            self._items.append(item)

