"""
VideoFIT — DXF Overlay
Draws fitted DXF polylines onto a QGraphicsScene using cosmetic pens.

Two rendering modes:
  • Preview  — cyan outlines (pre-alignment)
  • Heatmap  — segment-by-segment green→yellow→red based on distance
               to the real edge (post-alignment), matching the POC exactly.

Performance note
────────────────
ezdxf's Frontend.draw_layout() is expensive (~300–600 ms for complex DXFs).
We cache the rendered QGraphicsItems the first time a DXF is drawn and reuse
them on every subsequent heatmap update — only the pen brush changes.
"""

from __future__ import annotations

import numpy as np
import ezdxf
from ezdxf.addons.drawing import Frontend, RenderContext
from ezdxf.addons.drawing.pyqt import PyQtBackend
from ezdxf.addons.drawing.config import Configuration, BackgroundPolicy
from PySide6.QtGui import QBrush, QColor, QImage, QTransform
from PySide6.QtWidgets import (
    QGraphicsItem, QGraphicsItemGroup, QGraphicsLineItem, QGraphicsPathItem,
    QGraphicsScene, QGraphicsView,
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

        # ── Render cache ──────────────────────────────────────────────
        # Key: id of the Dxf object (changes when a new file is loaded).
        # Value: list of (item, original_pen) tuples for fast brush swaps.
        self._cached_dxf_id: int | None = None
        self._cached_items: list[tuple[QGraphicsPathItem | QGraphicsLineItem, object]] = []
        self._cached_group: QGraphicsItemGroup | None = None

        # Heatmap brush cache — skip numpy pipeline when nothing changed.
        self._brush_cache: QBrush | None = None
        self._brush_cache_key: tuple | None = None

    # ── Viewport helpers ─────────────────────────────────────────────

    def _set_viewport_updates(self, enabled: bool) -> None:
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
        # Invalidate render cache when overlay is explicitly cleared
        self._cached_dxf_id = None
        self._cached_items.clear()
        self._cached_group = None
        self._brush_cache = None
        self._brush_cache_key = None

    def draw_preview(self, dxf: Dxf) -> None:
        """Draw DXF using ezdxf PyQtBackend (preview / pre-alignment)."""
        self.clear()
        if dxf.doc is None:
            return

        group, pen_items = self._render_dxf(dxf)

        # Colour all lines cyan for preview
        for item, _ in pen_items:
            pen = item.pen()
            pen.setColor(_PREVIEW_COLOR)
            pen.setCosmetic(True)
            pen.setWidth(_PEN_WIDTH)
            item.setPen(pen)

        if dxf.canvas_shape[0] != 0:
            H, W = dxf.canvas_shape
            t = QTransform()
            t.translate(W / 2.0, H / 2.0)
            t.scale(dxf.px_per_mm, -dxf.px_per_mm)
            t.translate(-dxf.dxf_center_mm[0], -dxf.dxf_center_mm[1])
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
        """Convenience wrapper: compute RGBA then display.
        Call from main thread only (legacy / App mode).
        For Auto mode use compute_heatmap_rgba() on worker + draw_heatmap_from_rgba() on main.
        """
        rgba, shape = DxfOverlay.compute_heatmap_rgba(
            result, heatmap_min=heatmap_min, heatmap_max=heatmap_max,
            color_low=color_low, color_mid=color_mid, color_high=color_high,
        )
        self.draw_heatmap_from_rgba(dxf, result, rgba, shape)

    # ── Split API: compute on worker, display on main thread ─────────

    @staticmethod
    def compute_heatmap_rgba(
        result: FitResult,
        heatmap_min: float = 1.0,
        heatmap_max: float = 3.0,
        color_low: str = "#00FF00",
        color_mid: str = "#FF8000",
        color_high: str = "#FF0000",
    ) -> tuple[bytes, tuple[int, int]]:
        """Pure numpy — safe to call on any thread.
        Returns (rgba_bytes, (H, W)) ready for QImage construction.
        """
        dist_field = result.dist_raw if result.dist_raw is not None else result.dist_t
        H, W = dist_field.shape
        d = dist_field.astype(np.float32)

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

        rgba = np.ascontiguousarray(np.stack([r, g, b, a], axis=-1))
        return rgba.tobytes(), (H, W)

    def draw_heatmap_from_rgba(
        self,
        dxf: Dxf,
        result: FitResult,
        rgba_bytes: bytes,
        shape: tuple[int, int],
    ) -> None:
        """Qt-only — must be called on the main thread.
        Receives pre-computed RGBA bytes from compute_heatmap_rgba().
        Creates QImage + QBrush and applies to cached scene items.
        """
        H, W = shape
        final_t = self._build_transform(dxf, result)

        cache_key = (result.tx, result.ty, result.angle_deg,
                     id(rgba_bytes), H, W)
        if cache_key != self._brush_cache_key or self._brush_cache is None:
            qimg = QImage(rgba_bytes, W, H, W * 4, QImage.Format.Format_RGBA8888)
            # Keep a reference so the buffer isn't GC'd while QImage is alive
            self._heatmap_img_data = rgba_bytes
            brush = QBrush(qimg)
            inv_t, invertible = final_t.inverted()
            if invertible:
                brush.setTransform(inv_t)
            self._brush_cache = brush
            self._brush_cache_key = cache_key
        brush = self._brush_cache

        dxf_id = id(dxf)
        if dxf_id == self._cached_dxf_id and self._cached_group is not None:
            self._apply_brush(self._cached_items, brush)
            self._cached_group.setTransform(final_t)
        else:
            for item in self._items:
                self._scene.removeItem(item)
            self._items.clear()

            group, pen_items = self._render_dxf(dxf)
            self._apply_brush(pen_items, brush)
            group.setTransform(final_t)
            group.setZValue(100)

            self._cached_dxf_id = dxf_id
            self._cached_items = pen_items
            self._cached_group = group
            self._items.append(group)
            self._brush_cache = brush
            self._brush_cache_key = cache_key

        self._set_viewport_updates(True)

    # ── Private helpers ──────────────────────────────────────────────

    def _render_dxf(self, dxf: Dxf) -> tuple[QGraphicsItemGroup,
                                               list[tuple]]:
        """Render dxf.doc into the scene via ezdxf and return (group, pen_items).

        This is the expensive step — call only when the DXF changes.
        """
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
            pass  # viewport re-enable handled by caller

        new_items = [i for i in self._scene.items() if id(i) not in ids_before]

        group = self._scene.createItemGroup([])
        pen_items: list[tuple] = []
        for item in new_items:
            if isinstance(item, (QGraphicsPathItem, QGraphicsLineItem)):
                pen_items.append((item, item.pen()))
            group.addToGroup(item)

        return group, pen_items

    @staticmethod
    def _apply_brush(pen_items: list[tuple], brush: QBrush) -> None:
        """Swap the pen brush on all cached pen items."""
        for item, _ in pen_items:
            pen = item.pen()
            pen.setBrush(brush)
            pen.setCosmetic(True)
            pen.setWidth(_PEN_WIDTH)
            item.setPen(pen)

    @staticmethod
    def _build_transform(dxf: Dxf, result: FitResult) -> QTransform:
        if dxf.canvas_shape[0] == 0:
            return QTransform()

        canvas_H, canvas_W = dxf.canvas_shape
        dxf_cx_mm, dxf_cy_mm = dxf.dxf_center_mm
        px_per_mm = dxf.px_per_mm

        t = QTransform()
        t.translate(canvas_W / 2.0, canvas_H / 2.0)
        t.scale(px_per_mm, -px_per_mm)
        t.translate(-dxf_cx_mm, -dxf_cy_mm)

        t_fit = QTransform()
        t_fit.translate(result.tx, result.ty)
        t_fit.translate(result.dxf_cx, result.dxf_cy)
        t_fit.rotate(result.angle_deg)
        t_fit.translate(-result.dxf_cx, -result.dxf_cy)

        return t * t_fit

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
