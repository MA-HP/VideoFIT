"""
VideoFIT — Image Viewer
QGraphicsView with pan & zoom for live camera frames.

Build / stroke mode
───────────────────
When set_roi_mode(True) is called, the viewer switches to a "stroke brush"
interaction: the user clicks and drags a freehand path over edge features.
As the stroke is drawn, a dashed cyan path is rendered on the scene.
On mouse-release, ``stroke_completed`` is emitted with an (N, 2) float32
array of scene-coordinate (= image-pixel) path points, and the mode
automatically resets to normal pan/zoom.
"""

import numpy as np

from PySide6.QtCore import Qt, QPointF, Signal, Slot
from PySide6.QtGui import (
    QColor, QImage, QPainterPath, QPen, QPixmap,
)
from PySide6.QtWidgets import (
    QGraphicsPathItem, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView,
)

# Visual style of the stroke preview
_STROKE_COLOR  = QColor(0, 210, 255)
_STROKE_WIDTH  = 2        # cosmetic (screen) pixels
_STROKE_DASH   = [6, 4]   # dash/gap pattern in screen pixels


class ImageViewer(QGraphicsView):
    """
    Zoomable, pannable viewer for camera frames.

    Signals
    -------
    stroke_completed(np.ndarray)
        Emitted when the user finishes a Build stroke.  Carries an (N, 2)
        float32 array of scene / image-pixel coordinates tracing the path.
        Emitted only when the stroke has at least 2 distinct points.
    """

    stroke_completed = Signal(object)   # np.ndarray (N, 2) float32

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._pixmap_item.setPos(-0.5, -0.5)
        self._scene.addItem(self._pixmap_item)

        # Pan & Zoom
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet("background: transparent; border: none;")

        self._current_cv_img: np.ndarray | None = None

        # Stroke / brush state
        self._stroke_mode: bool = False
        self._stroke_pts: list[tuple[float, float]] = []
        self._stroke_path: QPainterPath = QPainterPath()
        self._stroke_item: QGraphicsPathItem | None = None

    # ── Public ────────────────────────────────────────────────────────

    def clear_view(self) -> None:
        """Remove the current pixmap (e.g. when a camera disconnects)."""
        self._scene.clear()
        self._pixmap_item = QGraphicsPixmapItem()
        self._pixmap_item.setPos(-0.5, -0.5)   # keep pixel-centre alignment
        self._scene.addItem(self._pixmap_item)

    @Slot(object)
    def update_image(self, cv_img: np.ndarray) -> None:
        """Receive an RGB numpy array and display it."""
        self._current_cv_img = cv_img
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        qimg = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self._pixmap_item.setPixmap(QPixmap.fromImage(qimg))
        # Scene rect in scene coordinates: pixel [row=y, col=x] centre sits at
        # integer (x, y) because the pixmap is offset by -0.5 in both axes.
        self._scene.setSceneRect(-0.5, -0.5, w, h)

    def set_roi_mode(self, enabled: bool) -> None:
        """
        Enable or disable the Build stroke tool.

        While enabled the cursor becomes a crosshair and click-drag draws a
        freehand selection stroke over the image.  The mode auto-disables
        after the mouse is released.
        """
        self._stroke_mode = enabled
        if enabled:
            self.setDragMode(QGraphicsView.NoDrag)
            self.setCursor(Qt.CrossCursor)
        else:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.setCursor(Qt.ArrowCursor)
            self._clear_stroke()

    # ── Qt event overrides ────────────────────────────────────────────

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event) -> None:
        if self._stroke_mode and event.button() == Qt.LeftButton:
            pt = self.mapToScene(event.pos())
            self._stroke_pts = [(pt.x(), pt.y())]

            self._stroke_path = QPainterPath()
            self._stroke_path.moveTo(pt)

            pen = QPen(_STROKE_COLOR, _STROKE_WIDTH)
            pen.setCosmetic(True)
            pen.setDashPattern(_STROKE_DASH)
            self._clear_stroke()
            self._stroke_item = self._scene.addPath(self._stroke_path, pen)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._stroke_mode and self._stroke_pts:
            pt = self.mapToScene(event.pos())
            self._stroke_pts.append((pt.x(), pt.y()))
            self._stroke_path.lineTo(pt)
            if self._stroke_item is not None:
                self._stroke_item.setPath(self._stroke_path)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._stroke_mode and event.button() == Qt.LeftButton and self._stroke_pts:
            pts_arr = np.array(self._stroke_pts, dtype=np.float32)
            self.set_roi_mode(False)   # clears stroke visuals, resets cursor
            if len(pts_arr) >= 2:
                self.stroke_completed.emit(pts_arr)
        else:
            super().mouseReleaseEvent(event)

    # ── Helpers ───────────────────────────────────────────────────────

    def _clear_stroke(self) -> None:
        if self._stroke_item is not None:
            self._scene.removeItem(self._stroke_item)
            self._stroke_item = None
        self._stroke_pts = []
        self._stroke_path = QPainterPath()
