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
import ezdxf
from ezdxf.addons.drawing import Frontend, RenderContext
from ezdxf.addons.drawing.pyqt import PyQtBackend
from ezdxf.addons.drawing.config import Configuration, BackgroundPolicy
from PySide6.QtGui import QBrush, QColor, QImage, QTransform
from PySide6.QtWidgets import (
    QGraphicsItem, QGraphicsLineItem, QGraphicsPathItem, QGraphicsScene,
    QGraphicsView,
)

from app.models.fit_result import FitResult
from app.models.dxf import Dxf


_PREVIEW_COLOR = QColor(0, 200, 255, 255)   # cyan
_PEN_WIDTH = 2                               # cosmetic px


class DxfOverlay:
    """Manages QGraphicsItems representing the DXF overlay."""

    def __init__(self, scene: QGraphicsScene) -> None:
        self._scene = scene
        self._items: list[QGraphicsItem] = []
        # Heatmap brush cache — rebuilding the full numpy RGBA pipeline is
        # expensive; skip it when the distance field and thresholds haven't changed.
        self._brush_cache: QBrush | None = None
        self._brush_cache_key: tuple | None = None

    # ── Viewport helpers ─────────────────────────────────────────────

    def _set_viewport_updates(self, enabled: bool) -> None:
        """Block/restore viewport repaints on all attached views.

        When ezdxf pumps hundreds of items into the scene each one triggers
        a repaint.  Disabling updates during that burst and doing a single
        repaint at the end is significantly faster.
        """
        mode = (QGraphicsView.MinimalViewportUpdate if enabled
                else QGraphicsView.NoViewportUpdate)
        for view in self._scene.views():
            if isinstance(view, QGraphicsView):
                view.setViewportUpdateMode(mode)
                if enabled:
                    view.viewport().update()

    # ── Public ───────────────────────────────────────────────────────

    def clear(self) -> None:
        """Remove all overlay items from the scene."""
        for item in self._items:
            self._scene.removeItem(item)
        self._items.clear()

    def draw_preview(self, dxf: Dxf) -> None:
        """Draw DXF using ezdxf PyQtBackend (preview / pre-alignment)."""
        self.clear()

        if dxf.doc is None:
            return

        ctx = RenderContext(dxf.doc)
        backend = PyQtBackend(scene=self._scene)
        config = Configuration(background_policy=BackgroundPolicy.OFF)

        # Snapshot existing item ids BEFORE rendering so we can identify the
        # new ones afterwards without a second render pass.
        ids_before = {id(i) for i in self._scene.items()}

        self._set_viewport_updates(False)
        try:
            Frontend(ctx, backend, config=config).draw_layout(
                dxf.doc.modelspace(), finalize=True
            )
        finally:
            self._set_viewport_updates(True)

        new_items = [i for i in self._scene.items() if id(i) not in ids_before]

        group = self._scene.createItemGroup([])
        for item in new_items:
            if isinstance(item, (QGraphicsPathItem, QGraphicsLineItem)):
                pen = item.pen()
                pen.setColor(_PREVIEW_COLOR)
                pen.setCosmetic(True)
                pen.setWidth(_PEN_WIDTH)
                item.setPen(pen)
            group.addToGroup(item)

        if dxf.canvas_shape[0] != 0:
            H, W = dxf.canvas_shape
            canvas_cx, canvas_cy = W / 2.0, H / 2.0
            dxf_cx, dxf_cy = dxf.dxf_center_mm
            px_per_mm = dxf.px_per_mm

            t = QTransform()
            t.translate(canvas_cx, canvas_cy)
            t.scale(px_per_mm, -px_per_mm)
            t.translate(-dxf_cx, -dxf_cy)
            group.setTransform(t)

        group.setZValue(100)
        self._items.append(group)

    def draw_heatmap(
        self,
        dxf: Dxf,
        result: FitResult,
        heatmap_min: float = 1.0,
        heatmap_max: float = 3.0,
        color_low: str = "#00FF00",
        color_mid: str = "#FF8000",
        color_high: str = "#FF0000",
    ) -> None:
        """Draw aligned DXF with per-pixel heatmap colouring."""
        self.clear()
        dist_field = result.dist_raw if result.dist_raw is not None else result.dist_t
        self._draw_aligned_native(dxf, result, dist_t=dist_field,
                                  heatmap_min=heatmap_min, heatmap_max=heatmap_max,
                                  color_low=color_low, color_mid=color_mid,
                                  color_high=color_high)

    # ── Private helpers ──────────────────────────────────────────────

    def _draw_aligned_native(self, dxf: Dxf, result: FitResult,
                              dist_t: np.ndarray | None = None,
                              heatmap_min: float = 1.0,
                              heatmap_max: float = 3.0,
                              color_low: str = "#00FF00",
                              color_mid: str = "#FF8000",
                              color_high: str = "#FF0000") -> None:
        if dxf.doc is None:
            return

        ctx = RenderContext(dxf.doc)
        backend = PyQtBackend(scene=self._scene)
        config = Configuration(
            background_policy=BackgroundPolicy.OFF,
            line_policy=ezdxf.addons.drawing.config.LinePolicy.SOLID,
        )

        ids_before = {id(i) for i in self._scene.items()}

        self._set_viewport_updates(False)
        try:
            Frontend(ctx, backend, config=config).draw_layout(
                dxf.doc.modelspace(), finalize=True
            )
        finally:
            self._set_viewport_updates(True)

        new_items = [i for i in self._scene.items() if id(i) not in ids_before]

        if dxf.canvas_shape[0] != 0:
            canvas_H, canvas_W = dxf.canvas_shape
            canvas_cx, canvas_cy = canvas_W / 2.0, canvas_H / 2.0
            dxf_cx_mm, dxf_cy_mm = dxf.dxf_center_mm
            px_per_mm = dxf.px_per_mm

            t = QTransform()
            t.translate(canvas_cx, canvas_cy)
            t.scale(px_per_mm, -px_per_mm)
            t.translate(-dxf_cx_mm, -dxf_cy_mm)

            t_fit = QTransform()
            t_fit.translate(result.tx, result.ty)
            t_fit.translate(result.dxf_cx, result.dxf_cy)
            t_fit.rotate(result.angle_deg)
            t_fit.translate(-result.dxf_cx, -result.dxf_cy)

            final_t = t * t_fit
        else:
            final_t = QTransform()

        if dist_t is not None:
            # Reuse the cached brush when the same distance field + thresholds
            # are used again (e.g. after a simple pan/zoom redraw).
            cache_key = (id(dist_t), heatmap_min, heatmap_max,
                         color_low, color_mid, color_high)
            if cache_key != self._brush_cache_key:
                self._brush_cache = self._create_heatmap_brush(
                    dist_t, final_t,
                    heatmap_min=heatmap_min, heatmap_max=heatmap_max,
                    color_low=color_low, color_mid=color_mid, color_high=color_high,
                )
                self._brush_cache_key = cache_key
            brush = self._brush_cache
        else:
            brush = QBrush(QColor(color_low))

        group = self._scene.createItemGroup([])
        for item in new_items:
            pen = item.pen() if hasattr(item, 'pen') else None
            if pen is not None:
                pen.setBrush(brush)
                pen.setCosmetic(True)
                pen.setWidth(_PEN_WIDTH)
                item.setPen(pen)
            group.addToGroup(item)

        group.setTransform(final_t)
        group.setZValue(100)
        self._items.append(group)

    def _create_heatmap_brush(self, dist_t: np.ndarray, final_t: QTransform,
                               heatmap_min: float = 1.0,
                               heatmap_max: float = 3.0,
                               color_low: str = "#00FF00",
                               color_mid: str = "#FF8000",
                               color_high: str = "#FF0000") -> QBrush:
        H, W = dist_t.shape
        d = dist_t.astype(np.float32)

        ql = QColor(color_low)
        qm = QColor(color_mid)
        qh = QColor(color_high)

        span = max(heatmap_max - heatmap_min, 1e-6)
        t = np.clip((d - heatmap_min) / span, 0.0, 1.0).astype(np.float32)

        out_of_tol = d > heatmap_max
        r_in = (ql.red()   + t * (qm.red()   - ql.red())).astype(np.uint8)
        g_in = (ql.green() + t * (qm.green() - ql.green())).astype(np.uint8)
        b_in = (ql.blue()  + t * (qm.blue()  - ql.blue())).astype(np.uint8)

        r = np.where(out_of_tol, np.uint8(qh.red()),   r_in).astype(np.uint8)
        g = np.where(out_of_tol, np.uint8(qh.green()), g_in).astype(np.uint8)
        b = np.where(out_of_tol, np.uint8(qh.blue()),  b_in).astype(np.uint8)
        a = np.full(d.shape, 255, dtype=np.uint8)

        # Stack into RGBA and pin the array so the QImage buffer stays valid
        self._heatmap_img_data = np.ascontiguousarray(np.stack([r, g, b, a], axis=-1))

        qimg = QImage(
            self._heatmap_img_data.data,
            W, H, W * 4,
            QImage.Format.Format_RGBA8888,
        )

        brush = QBrush(qimg)
        inv_t, invertible = final_t.inverted()
        if invertible:
            brush.setTransform(inv_t)
        return brush


def _distance_to_color(d: float, heatmap_min: float = 1.0, heatmap_max: float = 3.0) -> QColor:
    span = max(heatmap_max - heatmap_min, 1e-6)
    t = max(0.0, min(1.0, (d - heatmap_min) / span))
    r = int(255 * t)
    g = int(255 * (1.0 - t))
    return QColor(r, g, 0)
