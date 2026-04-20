"""
VideoFIT – Debug Preprocessing Window
Displays all intermediate stages of the edge-detection pipeline in a
scrollable grid so the user can visually diagnose bad frames.
Double-click any card to open a full-screen pan/zoom viewer.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QImage, QPixmap, QWheelEvent, QMouseEvent, QKeyEvent, QPainter
from PySide6.QtWidgets import (
    QDialog, QGridLayout, QLabel, QScrollArea, QSizePolicy,
    QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QFrame,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
)


def _ndarray_to_pixmap(img: np.ndarray, max_size: int = 400) -> QPixmap:
    """Convert a numpy image (gray or BGR) to a scaled QPixmap."""
    if img is None:
        return QPixmap()

    if img.dtype != np.uint8:
        mn, mx = img.min(), img.max()
        if mx > mn:
            img = ((img - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)

    if img.ndim == 2:
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
    elif img.ndim == 3 and img.shape[2] == 3:
        h, w, ch = img.shape
        qimg = QImage(img.data, w, h, ch * w, QImage.Format_RGB888)
    else:
        return QPixmap()

    pixmap = QPixmap.fromImage(qimg.copy())
    if max_size > 0 and (pixmap.width() > max_size or pixmap.height() > max_size):
        pixmap = pixmap.scaled(
            max_size, max_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
    return pixmap


# ---------------------------------------------------------------------------
# Full-screen pan/zoom viewer
# ---------------------------------------------------------------------------

class _ZoomView(QGraphicsView):
    """GraphicsView with mouse-wheel zoom and middle/left-drag pan."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing, False)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setStyleSheet("background: #000; border: none;")
        self._panning = False
        self._pan_start = QPointF()

    # ── zoom ────────────────────────────────────────────────────────
    def wheelEvent(self, event: QWheelEvent) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    # ── pan ─────────────────────────────────────────────────────────
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() in (Qt.MiddleButton, Qt.LeftButton):
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(
                int(self.horizontalScrollBar().value() - delta.x()))
            self.verticalScrollBar().setValue(
                int(self.verticalScrollBar().value() - delta.y()))
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() in (Qt.MiddleButton, Qt.LeftButton):
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)


class FullscreenStageViewer(QDialog):
    """
    Full-screen pan/zoom viewer for a single preprocessing stage image.
    Open with double-click on a StageCard.
    Keyboard shortcuts: Escape → close, F → toggle fullscreen,
    + / - → zoom, 0 → fit in view.
    """

    def __init__(self, title: str, pixmap: QPixmap, parent=None) -> None:
        super().__init__(parent, Qt.Window)
        self.setWindowTitle(f"VideoFIT Debug – {title}")
        self.setStyleSheet("background: #000;")
        self.resize(1200, 800)

        # Scene / view
        self._scene = QGraphicsScene(self)
        self._item = QGraphicsPixmapItem(pixmap)
        self._item.setTransformationMode(Qt.SmoothTransformation)
        self._scene.addItem(self._item)

        self._view = _ZoomView(self)
        self._view.setScene(self._scene)

        # Top bar
        top_bar = QWidget()
        top_bar.setStyleSheet(
            "background:#111; border-bottom:1px solid rgba(255,255,255,20);")
        tl = QHBoxLayout(top_bar)
        tl.setContentsMargins(12, 6, 12, 6)

        lbl = QLabel(title)
        lbl.setStyleSheet("color:white; font-weight:bold; font-size:13px;")
        tl.addWidget(lbl)
        tl.addStretch()

        def _btn(text, cb):
            b = QPushButton(text)
            b.setStyleSheet("""
                QPushButton { background:rgba(255,255,255,20); border:none;
                    border-radius:5px; color:white; padding:5px 12px; }
                QPushButton:hover { background:rgba(255,255,255,40); }
            """)
            b.clicked.connect(cb)
            return b

        tl.addWidget(_btn("⊕ Zoom In",  lambda: self._view.scale(1.25, 1.25)))
        tl.addWidget(_btn("⊖ Zoom Out", lambda: self._view.scale(0.8,  0.8)))
        tl.addWidget(_btn("⤢ Fit",      self._fit))
        tl.addWidget(_btn("⛶ Fullscreen", self._toggle_fullscreen))
        tl.addWidget(_btn("✕ Close",    self.close))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(top_bar)
        layout.addWidget(self._view)

        # Fit on show
        self._fit()

    def _fit(self):
        self._view.fitInView(
            QRectF(self._item.pixmap().rect()), Qt.KeepAspectRatio)

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        k = event.key()
        if k == Qt.Key_Escape:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.close()
        elif k in (Qt.Key_Plus, Qt.Key_Equal):
            self._view.scale(1.25, 1.25)
        elif k == Qt.Key_Minus:
            self._view.scale(0.8, 0.8)
        elif k == Qt.Key_0:
            self._fit()
        elif k == Qt.Key_F:
            self._toggle_fullscreen()
        else:
            super().keyPressEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        self._fit()


# ---------------------------------------------------------------------------
# Stage card  (double-click → fullscreen viewer)
# ---------------------------------------------------------------------------

class _StageCard(QFrame):
    """A titled card displaying one preprocessing stage image."""

    def __init__(self, title: str, img: np.ndarray, max_size: int = 380, parent=None):
        super().__init__(parent)
        self.setObjectName("StageCard")
        self.setStyleSheet("""
            QFrame#StageCard {
                background-color: rgba(25, 25, 25, 240);
                border: 1px solid rgba(255, 255, 255, 20);
                border-radius: 10px;
            }
            QFrame#StageCard:hover {
                border: 1px solid rgba(100, 180, 255, 120);
            }
        """)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Double-click to open full-screen viewer")

        # Store full-resolution pixmap for the viewer
        self._title = title
        self._full_pixmap = _ndarray_to_pixmap(img, max_size=0)  # unlimited

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Title
        lbl_title = QLabel(title)
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setStyleSheet("""
            color: rgba(255, 255, 255, 200);
            font-weight: bold; font-size: 11px;
            background: transparent; border: none;
        """)
        layout.addWidget(lbl_title)

        # Thumbnail
        lbl_img = QLabel()
        lbl_img.setAlignment(Qt.AlignCenter)
        lbl_img.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        pixmap = _ndarray_to_pixmap(img, max_size)
        if not pixmap.isNull():
            lbl_img.setPixmap(pixmap)
            lbl_img.setFixedSize(pixmap.size())
        else:
            lbl_img.setText("(no image)")
            lbl_img.setStyleSheet("color: gray; border: none;")
        layout.addWidget(lbl_img, alignment=Qt.AlignCenter)

        # Hint label
        hint = QLabel("🔍 double-click to zoom")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet(
            "color: rgba(100,180,255,140); font-size: 9px;"
            " background: transparent; border: none;")
        layout.addWidget(hint)

        # Stats
        if img is not None:
            stats = _compute_stats(img)
            lbl_stats = QLabel(stats)
            lbl_stats.setAlignment(Qt.AlignCenter)
            lbl_stats.setStyleSheet("""
                color: rgba(180, 180, 180, 180);
                font-size: 10px; background: transparent; border: none;
            """)
            layout.addWidget(lbl_stats)

    def mouseDoubleClickEvent(self, event) -> None:
        if not self._full_pixmap.isNull():
            viewer = FullscreenStageViewer(self._title, self._full_pixmap, self)
            viewer.exec()
        super().mouseDoubleClickEvent(event)


def _compute_stats(img: np.ndarray) -> str:
    """Return a short human-readable stats string for an image."""
    try:
        arr = img.astype(np.float32)
        mn, mx, mean = arr.min(), arr.max(), arr.mean()
        if img.ndim == 2 and img.dtype == np.uint8:
            nonzero = int(np.count_nonzero(img))
            total = img.size
            pct = 100.0 * nonzero / total if total > 0 else 0.0
            return f"min={mn:.0f}  max={mx:.0f}  mean={mean:.1f}  non-zero={pct:.1f}%"
        return f"min={mn:.2f}  max={mx:.2f}  mean={mean:.2f}"
    except Exception:
        return ""


class DebugPreprocessingWindow(QDialog):
    """
    Floating window showing every intermediate image produced by
    ``compute_edges`` so the user can diagnose bad preprocessing frames.

    Usage
    -----
    Call ``update_stages(stages_dict)`` after each fit to refresh the view.
    ``stages_dict`` keys are stage names, values are numpy arrays.
    Double-click any stage card to open a full-screen pan/zoom viewer.
    """

    _STAGE_ORDER = [
        ("gray",            "① Grayscale"),
        ("blur",            "② Gaussian Blur (mask prep)"),
        ("thresh",          "③ Adaptive Threshold"),
        ("mask",            "④ Silhouette Mask"),
        ("gamma",           "⑤ Gamma Corrected (γ=0.8)"),
        ("clahe",           "⑥ CLAHE"),
        ("edges_raw",       "⑦ Canny-Devernay Edges"),
        ("edges_final",     "⑧ Edges + Silhouette Boundary"),
        ("distance_field",  "⑨ Distance Transform"),
        ("edge_points_viz",      "⑩ Sub-pixel Points (rounded)"),
        ("subpixel_offset_map",  "⑪ Sub-pixel Offset Map  R=ΔX G=ΔY"),
        ("subpixel_overlay",     "⑫ Sub-pixel Overlay on CLAHE (anti-aliased)"),
    ]

    def __init__(self, parent=None) -> None:
        super().__init__(parent, Qt.Window | Qt.WindowStaysOnTopHint)
        self.setWindowTitle("VideoFIT – Debug: Preprocessing Stages")
        self.setMinimumSize(900, 600)
        self.resize(1300, 820)
        self.setStyleSheet("background-color: #0d0d0d;")

        # ── Top bar ───────────────────────────────────────────────────
        top_bar = QWidget()
        top_bar.setStyleSheet(
            "background-color: #131313; border-bottom: 1px solid rgba(255,255,255,20);")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(16, 8, 16, 8)

        self._lbl_status = QLabel("No data yet — run a fit to populate stages.")
        self._lbl_status.setStyleSheet("color: rgba(255,255,255,160); font-size: 12px;")
        top_layout.addWidget(self._lbl_status)
        top_layout.addStretch()

        def _tbtn(text, cb, color="rgba(255,255,255,18)"):
            b = QPushButton(text)
            b.setStyleSheet(f"""
                QPushButton {{
                    background: {color}; border: none; border-radius: 6px;
                    color: white; font-weight: bold; padding: 6px 14px;
                }}
                QPushButton:hover {{ background: rgba(255,255,255,40); }}
            """)
            b.clicked.connect(cb)
            return b

        top_layout.addWidget(_tbtn("⛶  Fullscreen", self._toggle_fullscreen))
        btn_close = _tbtn("✕  Close", self.hide, "rgba(200,50,50,180)")
        btn_close.setStyleSheet("""
            QPushButton {
                background: rgba(200,50,50,180); border: none; border-radius: 6px;
                color: white; font-weight: bold; padding: 6px 14px;
            }
            QPushButton:hover { background: rgba(220,70,70,220); }
        """)
        top_layout.addWidget(btn_close)

        # ── Scroll area ───────────────────────────────────────────────
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet("background-color: #0d0d0d; border: none;")
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self._grid_widget = QWidget()
        self._grid_widget.setStyleSheet("background-color: transparent;")
        self._grid_layout = QGridLayout(self._grid_widget)
        self._grid_layout.setContentsMargins(16, 16, 16, 16)
        self._grid_layout.setSpacing(12)
        self._scroll.setWidget(self._grid_widget)

        # ── Main layout ───────────────────────────────────────────────
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(top_bar)
        main_layout.addWidget(self._scroll)

    # ------------------------------------------------------------------

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Escape and self.isFullScreen():
            self.showNormal()
        elif event.key() == Qt.Key_F:
            self._toggle_fullscreen()
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_stages(self, stages: dict[str, np.ndarray], fit_info: str = "") -> None:
        """
        Refresh the grid with new preprocessing stage images.

        Parameters
        ----------
        stages : dict
            Mapping of stage-key → numpy image.
        fit_info : str
            Optional one-line summary of the fit result shown in the status bar.
        """
        # Clear existing cards
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        cols = 4
        row, col = 0, 0
        for key, title in self._STAGE_ORDER:
            img = stages.get(key)
            if img is None:
                continue
            card = _StageCard(title, img, max_size=300)
            self._grid_layout.addWidget(card, row, col)
            col += 1
            if col >= cols:
                col = 0
                row += 1

        if fit_info:
            self._lbl_status.setText(f"Last fit: {fit_info}")
        else:
            self._lbl_status.setText(
                f"Stages captured — {len(stages)} images.  "
                f"  💡 Double-click any card to open a pan/zoom viewer.")

        if not self.isVisible():
            self.show()
        self.raise_()
