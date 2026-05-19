"""
VideoFIT — Measure Overlay
Renders fitted shapes, magnet-paint preview, and live fit preview onto the
shared QGraphicsScene.
"""

from __future__ import annotations

from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsScene

from app.models.measure_result import MeasureResult, ShapeKind

# ── Visual styles ─────────────────────────────────────────────────────────────
_SHAPE_COLOR = QColor(0, 230, 120)         # vivid green — final shapes
_PREVIEW_COLOR = QColor(0, 180, 255, 160)  # semi-transparent cyan — live preview
_MAGNET_COLOR = QColor(255, 200, 0)        # yellow dots — snapped edge points
_PEN_WIDTH = 2
_MAGNET_DOT_RADIUS = 2.5
_CENTRE_CROSS_PX = 12


class MeasureOverlay:
    """Manages fitted-shape and in-progress visualisation on the scene."""

    def __init__(self, scene: QGraphicsScene) -> None:
        self._scene = scene
        self._shape_items: list = []
        self._preview_items: list = []
        self._magnet_items: list = []

    # ══════════════════════════════════════════════════════════════════════════
    # Final shapes
    # ══════════════════════════════════════════════════════════════════════════

    def draw_shapes(self, results: list[MeasureResult]) -> None:
        """Replace all current shapes with *results*."""
        self.clear_shapes()
        for r in results:
            self._draw_one(r, self._shape_pen(), self._shape_items)

    def add_shape(self, result: MeasureResult) -> None:
        """Append a single shape without clearing existing ones."""
        self._draw_one(result, self._shape_pen(), self._shape_items)

    def clear_shapes(self) -> None:
        self._remove_items(self._shape_items)

    # ══════════════════════════════════════════════════════════════════════════
    # Magnet paint (edge points snapped during drag)
    # ══════════════════════════════════════════════════════════════════════════

    def set_magnet_points(self, pts) -> None:
        """Show yellow dots at snapped edge-point positions (N,2 array)."""
        self._remove_items(self._magnet_items)
        if pts is None or len(pts) == 0:
            return
        pen = QPen(QColor(0, 0, 0, 0))
        brush = _MAGNET_COLOR
        r = _MAGNET_DOT_RADIUS
        for x, y in pts:
            item = self._scene.addEllipse(x - r, y - r, 2 * r, 2 * r, pen, brush)
            item.setZValue(100)
            self._magnet_items.append(item)

    def clear_magnet(self) -> None:
        self._remove_items(self._magnet_items)

    # ══════════════════════════════════════════════════════════════════════════
    # Live fit preview (semi-transparent shape while dragging)
    # ══════════════════════════════════════════════════════════════════════════

    def set_preview_shape(self, result: MeasureResult | None) -> None:
        """Show a semi-transparent preview of the shape being fitted."""
        self._remove_items(self._preview_items)
        if result is None:
            return
        self._draw_one(result, self._preview_pen(), self._preview_items)

    def clear_preview(self) -> None:
        self._remove_items(self._preview_items)

    # ══════════════════════════════════════════════════════════════════════════
    # Clear all
    # ══════════════════════════════════════════════════════════════════════════

    def clear(self) -> None:
        self.clear_shapes()
        self.clear_magnet()
        self.clear_preview()

    # ══════════════════════════════════════════════════════════════════════════
    # Internals
    # ══════════════════════════════════════════════════════════════════════════

    def _shape_pen(self) -> QPen:
        pen = QPen(_SHAPE_COLOR, _PEN_WIDTH)
        pen.setCosmetic(True)
        return pen

    def _preview_pen(self) -> QPen:
        pen = QPen(_PREVIEW_COLOR, _PEN_WIDTH)
        pen.setCosmetic(True)
        pen.setDashPattern([6, 4])
        return pen

    def _remove_items(self, item_list: list) -> None:
        for item in item_list:
            self._scene.removeItem(item)
        item_list.clear()

    def _draw_one(self, r: MeasureResult, pen: QPen, target: list) -> None:
        if r.kind == ShapeKind.CIRCLE:
            self._draw_circle(r, pen, target)
        elif r.kind == ShapeKind.ARC:
            self._draw_arc(r, pen, target)
        elif r.kind == ShapeKind.LINE:
            self._draw_line(r, pen, target)

    def _draw_circle(self, r: MeasureResult, pen: QPen, target: list) -> None:
        rect = QRectF(r.cx - r.radius, r.cy - r.radius, r.radius * 2, r.radius * 2)
        target.append(self._scene.addEllipse(rect, pen))
        self._add_crosshair(r.cx, r.cy, pen, target)

    def _draw_arc(self, r: MeasureResult, pen: QPen, target: list) -> None:
        rect = QRectF(r.cx - r.radius, r.cy - r.radius, r.radius * 2, r.radius * 2)
        qt_start = -r.arc_start_deg
        qt_span = -r.arc_span_deg
        path = QPainterPath()
        path.arcMoveTo(rect, qt_start)
        path.arcTo(rect, qt_start, qt_span)
        target.append(self._scene.addPath(path, pen))
        self._add_crosshair(r.cx, r.cy, pen, target)

    def _draw_line(self, r: MeasureResult, pen: QPen, target: list) -> None:
        x1, y1 = r.line_p1
        x2, y2 = r.line_p2
        target.append(self._scene.addLine(x1, y1, x2, y2, pen))

    def _add_crosshair(self, cx: float, cy: float, pen: QPen, target: list) -> None:
        s = _CENTRE_CROSS_PX
        target.append(self._scene.addLine(cx - s, cy, cx + s, cy, pen))
        target.append(self._scene.addLine(cx, cy - s, cx, cy + s, pen))

