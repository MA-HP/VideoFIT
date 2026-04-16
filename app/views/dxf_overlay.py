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
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QImage, QPainterPath, QPen, QTransform
from PySide6.QtWidgets import (
    QGraphicsItem, QGraphicsLineItem, QGraphicsPathItem, QGraphicsScene,
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

    # ── Public ───────────────────────────────────────────────────────

    def clear(self) -> None:
        """Remove all overlay items from the scene."""
        for item in self._items:
            self._scene.removeItem(item)
        self._items.clear()

    def draw_preview(self, dxf: Dxf) -> None:
        """
        Draw DXF using ezdxf PyQtBackend.
        """
        self.clear()

        if dxf.doc is None:
            return

        ctx = RenderContext(dxf.doc)
        backend = PyQtBackend(scene=self._scene)

        config = Configuration(
            background_policy=BackgroundPolicy.OFF,
        )

        frontend = Frontend(ctx, backend, config=config)
        frontend.draw_layout(dxf.doc.modelspace(), finalize=True)

        dxf_group = self._scene.createItemGroup([])
        for item in self._scene.items():
            if item not in self._items and getattr(item, 'zValue', lambda: 0)() == 0 and item.parentItem() is None:
                # We need a proper way to distinguish dxf items. For now we assume new items.
                pass

        # ACTUALLY, we should clear and group properly, let's fix it this way:
        # Collect items before:
        items_before = set(self._scene.items())

        frontend = Frontend(ctx, backend, config=config)
        frontend.draw_layout(dxf.doc.modelspace(), finalize=True)

        items_after = set(self._scene.items())
        new_items = items_after - items_before

        group = self._scene.createItemGroup([])
        for item in new_items:
            # We override colors for preview
            if isinstance(item, QGraphicsPathItem) or isinstance(item, QGraphicsLineItem):
                pen = item.pen()
                pen.setColor(_PREVIEW_COLOR)
                pen.setCosmetic(True)
                pen.setWidth(_PEN_WIDTH)
                item.setPen(pen)
            group.addToGroup(item)

        # Apply the scaling and translation identical to dxf_service.py
        if dxf.canvas_shape[0] != 0:
            H, W = dxf.canvas_shape
            canvas_cx, canvas_cy = W / 2.0, H / 2.0
            dxf_cx, dxf_cy = dxf.dxf_center_mm
            px_per_mm = dxf.px_per_mm

            # transform steps:
            # 1. translate so dxf_center_mm is at (0,0)
            # 2. scale to pixels (and flip Y)
            # 3. translate to canvas center

            t = QTransform()
            t.translate(canvas_cx, canvas_cy)
            t.scale(px_per_mm, -px_per_mm)
            t.translate(-dxf_cx, -dxf_cy)

            group.setTransform(t)

        group.setZValue(100)
        self._items.append(group)
        # We also need to transform it if needed, or if it is already in px we are fine. Wait, dxf.doc is in mm.
        # But we need it in px.

    def draw_heatmap(
        self,
        dxf: Dxf,
        result: FitResult,
    ) -> None:
        """
        Draw DXF natively using ezdxf PyQtBackend, applying a heatmap color
        to each native item based on its average distance to the nearest real edge.
        """
        self.clear()

        self._draw_aligned_native(dxf, result, dist_t=result.dist_t)

    # ── Private helpers ──────────────────────────────────────────────

    def _draw_aligned_native(self, dxf: Dxf, result: FitResult, dist_t: np.ndarray | None = None) -> None:
        if dxf.doc is None:
            return

        ctx = RenderContext(dxf.doc)
        backend = PyQtBackend(scene=self._scene)

        config = Configuration(
            background_policy=BackgroundPolicy.OFF,
            line_policy=ezdxf.addons.drawing.config.LinePolicy.SOLID,
        )

        items_before = set(self._scene.items())

        frontend = Frontend(ctx, backend, config=config)
        frontend.draw_layout(dxf.doc.modelspace(), finalize=True)

        items_after = set(self._scene.items())
        new_items = items_after - items_before

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
            brush = self._create_heatmap_brush(dist_t, final_t)
        else:
            brush = QBrush(QColor(0, 255, 0, 255))

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

    def _create_heatmap_brush(self, dist_t: np.ndarray, final_t: QTransform) -> QBrush:
        H, W = dist_t.shape

        # Vectorized color mapping
        d = dist_t.astype(np.float32)

        # <= 1.0 -> Green (0, 255, 0)
        # 1.0 to 3.0 -> Green to Red (t=(d-1)/2, r=255*t, g=255*(1-t))
        # > 3.0 -> Red (255, 0, 0)

        t = np.clip((d - 1.0) / 2.0, 0.0, 1.0)

        r = (255.0 * t).astype(np.uint8)
        g = (255.0 * (1.0 - t)).astype(np.uint8)
        b = np.zeros_like(r)
        a = np.full_like(r, 255)

        img_data = np.stack([r, g, b, a], axis=-1)

        # Garantir que le bloc de mémoire est contigu pour QImage afin d'éviter tout crash
        self._heatmap_img_data = np.ascontiguousarray(img_data)

        qimg = QImage(
            self._heatmap_img_data.data,
            W, H,
            W * 4,
            QImage.Format.Format_RGBA8888
        )

        brush = QBrush(qimg)
        # The brush texture space is identically the canvas pixel space (0..W, 0..H).
        # The item being drawn is in "DXF native space".
        # When the user draws at local native (0,0), it transforms to scene QTransform(final_t) -> (X,Y) scene pixel.
        # But QBrush uses its own transform from Texture -> Item Local space!
        # If we want Texture Space (Canvas pixels) to exactly align with Scene Space (Canvas pixels),
        # Texture Space = Item Local Space * final_t.
        # Thus the brush transform from Texture Space to Item Local Space is the INVERSE of final_t!

        inv_t, invertible = final_t.inverted()
        if invertible:
            brush.setTransform(inv_t)

        return brush

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