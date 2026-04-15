"""
Metrology Vision Pro — Image Viewer
QGraphicsView with pan & zoom for live camera frames.
"""

import numpy as np

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView


class ImageViewer(QGraphicsView):
    """Zoomable, pannable viewer that displays OpenCV BGR→RGB frames."""

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

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def wheelEvent(self, event) -> None:
        zoom_in_factor = 1.15
        zoom_out_factor = 1.0 / zoom_in_factor
        factor = zoom_in_factor if event.angleDelta().y() > 0 else zoom_out_factor
        self.scale(factor, factor)

