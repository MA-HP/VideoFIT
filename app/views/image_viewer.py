"""
VideoFIT — Image Viewer
QGraphicsView with pan & zoom for live camera frames, and optional
rubber-band ROI selection for the Measure mode Build tool.
"""

import numpy as np

from PySide6.QtCore import Qt, QPointF, QRectF, Signal, Slot
from PySide6.QtGui import QColor, QImage, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene, QGraphicsView,
)


class ImageViewer(QGraphicsView):
    """Zoomable, pannable viewer that displays OpenCV BGR→RGB frames.

    When ROI mode is active (set_roi_mode(True)), the user can click-drag
    to draw a selection rectangle.  On release, roi_selected is emitted
    with the rectangle corners in scene (pixel) coordinates, and ROI mode
    is automatically disabled.
    """

    roi_selected = Signal(float, float, float, float)  # x1, y1, x2, y2

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)

        # Pan & Zoom
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        # Sleek look — no scrollbars
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet("background: transparent; border: none;")

        self._current_cv_img: np.ndarray | None = None

        # ROI selection state
        self._roi_mode: bool = False
        self._roi_start: QPointF | None = None
        self._roi_rect_item: QGraphicsRectItem | None = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def clear_view(self) -> None:
        """Remove the current pixmap (e.g. when a camera disconnects)."""
        self._scene.clear()
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)

    @Slot(object)
    def update_image(self, cv_img: np.ndarray) -> None:
        """Receive an RGB numpy array and display it."""
        self._current_cv_img = cv_img
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        qimg = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self._pixmap_item.setPixmap(QPixmap.fromImage(qimg))
        self._scene.setSceneRect(self._pixmap_item.boundingRect())

    def set_roi_mode(self, enabled: bool) -> None:
        """Enable or disable rubber-band ROI selection."""
        self._roi_mode = enabled
        if enabled:
            self.setDragMode(QGraphicsView.NoDrag)
            self.setCursor(Qt.CrossCursor)
        else:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.setCursor(Qt.ArrowCursor)
            self._clear_roi_rect()

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def wheelEvent(self, event) -> None:
        zoom_in_factor = 1.15
        zoom_out_factor = 1.0 / zoom_in_factor
        factor = zoom_in_factor if event.angleDelta().y() > 0 else zoom_out_factor
        self.scale(factor, factor)

    def mousePressEvent(self, event) -> None:
        if self._roi_mode and event.button() == Qt.LeftButton:
            self._roi_start = self.mapToScene(event.pos())
            self._clear_roi_rect()
            pen = QPen(QColor(0, 200, 255), 1)
            pen.setCosmetic(True)
            self._roi_rect_item = self._scene.addRect(
                QRectF(self._roi_start, self._roi_start), pen
            )
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._roi_mode and self._roi_start is not None:
            current = self.mapToScene(event.pos())
            rect = QRectF(self._roi_start, current).normalized()
            if self._roi_rect_item is not None:
                self._roi_rect_item.setRect(rect)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._roi_mode and event.button() == Qt.LeftButton and self._roi_start is not None:
            end = self.mapToScene(event.pos())
            rect = QRectF(self._roi_start, end).normalized()
            self._roi_start = None
            self.set_roi_mode(False)    # clears rect and resets cursor
            if rect.width() > 2 and rect.height() > 2:
                self.roi_selected.emit(
                    rect.x(), rect.y(), rect.right(), rect.bottom()
                )
        else:
            super().mouseReleaseEvent(event)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clear_roi_rect(self) -> None:
        if self._roi_rect_item is not None:
            self._scene.removeItem(self._roi_rect_item)
            self._roi_rect_item = None
