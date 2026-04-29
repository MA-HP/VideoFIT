"""
VideoFIT — Measure Overlay
Renders fitted shapes (circles, arcs, lines) onto the shared QGraphicsScene.
"""

from __future__ import annotations

from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsScene

from app.models.measure_result import MeasureResult, ShapeKind

# Visual style for fitted shapes
_PEN_COLOR = QColor(0, 230, 120)   # vivid green
_PEN_WIDTH = 2                      # cosmetic pixels
_CENTRE_CROSS_PX = 12               # half-size of the crosshair arms


class MeasureOverlay:
    """Adds and removes fitted-shape graphics items on the shared scene."""

    def __init__(self, scene: QGraphicsScene) -> None:
        self._scene = scene
        self._items: list = []

    # ── Public API ────────────────────────────────────────────────────

    def draw_shapes(self, results: list[MeasureResult]) -> None:
        """Replace all current shapes with *results*."""
        self.clear()
        for r in results:
            self._draw_one(r)

    def add_shape(self, result: MeasureResult) -> None:
        """Append a single shape without clearing existing ones."""
        self._draw_one(result)

    def clear(self) -> None:
        """Remove all shape graphics items from the scene."""
        for item in self._items:
            self._scene.removeItem(item)
        self._items.clear()

    # ── Internals ────────────────────────────────────────────────────

    def _pen(self) -> QPen:
        pen = QPen(_PEN_COLOR, _PEN_WIDTH)
        pen.setCosmetic(True)   # width stays constant regardless of zoom
        return pen

    def _draw_one(self, r: MeasureResult) -> None:
        pen = self._pen()
        if r.kind == ShapeKind.CIRCLE:
            self._draw_circle(r, pen)
        elif r.kind == ShapeKind.ARC:
            self._draw_arc(r, pen)
        elif r.kind == ShapeKind.LINE:
            self._draw_line(r, pen)

    def _draw_circle(self, r: MeasureResult, pen: QPen) -> None:
        rect = QRectF(
            r.cx - r.radius, r.cy - r.radius,
            r.radius * 2, r.radius * 2,
        )
        self._items.append(self._scene.addEllipse(rect, pen))
        self._add_crosshair(r.cx, r.cy, pen)

    def _draw_arc(self, r: MeasureResult, pen: QPen) -> None:
        """
        Draw an arc using QPainterPath.arcTo.

        Angle convention:
          • Our arc_start_deg / arc_span_deg use screen atan2 convention
            (right=0°, CW-positive on screen since Y is down).
          • Qt's arcTo uses mathematical CCW-positive convention (right=0°,
            CCW-positive = CW-positive visually on Y-down screen).
          • Therefore: qt_start = -arc_start_deg, qt_span = -arc_span_deg.
        """
        rect = QRectF(
            r.cx - r.radius, r.cy - r.radius,
            r.radius * 2, r.radius * 2,
        )
        qt_start = -r.arc_start_deg
        qt_span = -r.arc_span_deg

        path = QPainterPath()
        path.arcMoveTo(rect, qt_start)
        path.arcTo(rect, qt_start, qt_span)
        self._items.append(self._scene.addPath(path, pen))
        self._add_crosshair(r.cx, r.cy, pen)

    def _draw_line(self, r: MeasureResult, pen: QPen) -> None:
        x1, y1 = r.line_p1
        x2, y2 = r.line_p2
        self._items.append(self._scene.addLine(x1, y1, x2, y2, pen))

    def _add_crosshair(self, cx: float, cy: float, pen: QPen) -> None:
        s = _CENTRE_CROSS_PX
        self._items.append(self._scene.addLine(cx - s, cy, cx + s, cy, pen))
        self._items.append(self._scene.addLine(cx, cy - s, cx, cy + s, pen))
