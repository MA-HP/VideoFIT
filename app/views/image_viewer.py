"""
VideoFIT — Image Viewer
QGraphicsView with pan & zoom for live camera frames.

Interaction modes:
  - Normal: left-click drag pans, wheel zooms
  - Stroke: left-click drag draws a freehand path, emits stroke_completed on release
"""

import numpy as np

from PySide6.QtCore import Qt, Signal, Slot, QPointF, QEvent
from PySide6.QtGui import (
    QColor, QImage, QMouseEvent, QPainterPath, QPen, QPixmap,
)
from PySide6.QtWidgets import (
    QGraphicsPathItem, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView,
)

# Visual style of the stroke preview
_STROKE_COLOR = QColor(0, 210, 255)
_STROKE_WIDTH = 2
_STROKE_DASH = [6, 4]
_STROKE_THROTTLE_SQ = 4.0


class ImageViewer(QGraphicsView):
    """
    Zoomable, pannable viewer for camera frames.

    Signals
    -------
    stroke_completed(np.ndarray)
        Emitted when the user finishes a stroke.  (N, 2) float32 array.
    stroke_progress(np.ndarray)
        Emitted during a stroke drag.  (N, 2) float32 array.
    """

    stroke_completed = Signal(object)
    stroke_progress = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._pixmap_item.setPos(-0.5, -0.5)
        self._pixmap_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._scene.addItem(self._pixmap_item)

        # Never use Qt's built-in drag modes — we handle everything manually
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet("background: transparent; border: none;")

        self._current_cv_img: np.ndarray | None = None
        self._last_w: int = -1
        self._last_h: int = -1

        # Interaction mode
        self._stroke_mode: bool = False

        # Pan state (manual implementation)
        self._panning: bool = False
        self._pan_start: QPointF = QPointF()

        # Stroke state
        self._stroking: bool = False  # True while actively drawing a stroke
        self._stroke_pts: list[tuple[float, float]] = []
        self._stroke_path: QPainterPath = QPainterPath()
        self._stroke_item: QGraphicsPathItem | None = None
        self._stroke_last_pt: tuple[float, float] | None = None

    # ── Public ────────────────────────────────────────────────────────

    def clear_view(self) -> None:
        """Remove the current pixmap (e.g. when a camera disconnects)."""
        self._scene.clear()
        self._pixmap_item = QGraphicsPixmapItem()
        self._pixmap_item.setPos(-0.5, -0.5)
        self._pixmap_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._scene.addItem(self._pixmap_item)
        self._last_w = -1
        self._last_h = -1

    @Slot(object)
    def update_image(self, cv_img: np.ndarray) -> None:
        """Receive an RGB numpy array and display it."""
        self._current_cv_img = cv_img
        h, w, ch = cv_img.shape
        if not cv_img.flags['C_CONTIGUOUS']:
            cv_img = np.ascontiguousarray(cv_img)
        bytes_per_line = ch * w
        qimg = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self._pixmap_item.setPixmap(QPixmap.fromImage(qimg, Qt.ImageConversionFlag.NoFormatConversion))
        if w != self._last_w or h != self._last_h:
            self._scene.setSceneRect(-0.5, -0.5, w, h)
            self._last_w, self._last_h = w, h

    def set_roi_mode(self, enabled: bool) -> None:
        """Enable or disable the stroke drawing tool."""
        self._stroke_mode = enabled
        if enabled:
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            self._clear_stroke_items()

    # ── Qt event overrides ────────────────────────────────────────────

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)

    def viewportEvent(self, event: QEvent) -> bool:
        """
        Intercept all viewport events.  We handle mouse press/move/release
        here to guarantee they are processed before QGraphicsView's internal
        scene-dispatch / drag-mode logic can consume them.
        """
        etype = event.type()

        if etype == QEvent.Type.MouseButtonPress:
            me: QMouseEvent = event
            if me.button() == Qt.MouseButton.LeftButton:
                if self._stroke_mode:
                    self._start_stroke(me)
                else:
                    self._start_pan(me)
                return True

        elif etype == QEvent.Type.MouseMove:
            me: QMouseEvent = event
            if self._stroking:
                self._continue_stroke(me)
                return True
            elif self._panning:
                self._continue_pan(me)
                return True

        elif etype == QEvent.Type.MouseButtonRelease:
            me: QMouseEvent = event
            if me.button() == Qt.MouseButton.LeftButton:
                if self._stroking:
                    self._finish_stroke()
                    return True
                elif self._panning:
                    self._finish_pan()
                    return True

        return super().viewportEvent(event)

    # ── Stroke helpers ────────────────────────────────────────────────

    def _start_stroke(self, event: QMouseEvent) -> None:
        self._stroking = True
        pt = self.mapToScene(event.position().toPoint())
        x, y = pt.x(), pt.y()
        self._stroke_pts = [(x, y)]
        self._stroke_last_pt = (x, y)
        self._stroke_path = QPainterPath()
        self._stroke_path.moveTo(pt)
        self._clear_stroke_items()
        pen = QPen(_STROKE_COLOR, _STROKE_WIDTH)
        pen.setCosmetic(True)
        pen.setDashPattern(_STROKE_DASH)
        self._stroke_item = self._scene.addPath(self._stroke_path, pen)

    def _continue_stroke(self, event: QMouseEvent) -> None:
        pt = self.mapToScene(event.position().toPoint())
        px, py = pt.x(), pt.y()
        if self._stroke_last_pt is not None:
            lx, ly = self._stroke_last_pt
            if (px - lx) ** 2 + (py - ly) ** 2 < _STROKE_THROTTLE_SQ:
                return
        self._stroke_pts.append((px, py))
        self._stroke_path.lineTo(pt)
        self._stroke_last_pt = (px, py)
        if self._stroke_item is not None:
            self._stroke_item.setPath(self._stroke_path)
        self.stroke_progress.emit(np.array(self._stroke_pts, dtype=np.float32))

    def _finish_stroke(self) -> None:
        self._stroking = False
        pts_arr = np.array(self._stroke_pts, dtype=np.float32)
        self._clear_stroke_items()
        if len(pts_arr) >= 2:
            self.stroke_completed.emit(pts_arr)

    # ── Pan helpers ───────────────────────────────────────────────────

    def _start_pan(self, event: QMouseEvent) -> None:
        self._panning = True
        self._pan_start = event.position()
        self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)

    def _continue_pan(self, event: QMouseEvent) -> None:
        delta = event.position() - self._pan_start
        self._pan_start = event.position()
        self.horizontalScrollBar().setValue(
            self.horizontalScrollBar().value() - int(delta.x()))
        self.verticalScrollBar().setValue(
            self.verticalScrollBar().value() - int(delta.y()))

    def _finish_pan(self) -> None:
        self._panning = False
        self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)

    # ── Helpers ───────────────────────────────────────────────────────

    def _clear_stroke_items(self) -> None:
        if self._stroke_item is not None:
            self._scene.removeItem(self._stroke_item)
            self._stroke_item = None
        self._stroke_pts = []
        self._stroke_last_pt = None
        self._stroke_path = QPainterPath()

