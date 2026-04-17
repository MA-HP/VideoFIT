"""
VideoFIT – Debug Preprocessing Window
Displays all intermediate stages of the edge-detection pipeline in a
scrollable grid so the user can visually diagnose bad frames.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog, QGridLayout, QLabel, QScrollArea, QSizePolicy,
    QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QFrame,
)


def _ndarray_to_pixmap(img: np.ndarray, max_size: int = 400) -> QPixmap:
    """Convert a numpy image (gray or BGR) to a scaled QPixmap."""
    if img is None:
        return QPixmap()

    if img.dtype != np.uint8:
        # Normalise float images to [0, 255]
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
    if pixmap.width() > max_size or pixmap.height() > max_size:
        pixmap = pixmap.scaled(
            max_size, max_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
    return pixmap


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
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Title
        lbl_title = QLabel(title)
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setStyleSheet("""
            color: rgba(255, 255, 255, 200);
            font-weight: bold;
            font-size: 11px;
            background: transparent;
            border: none;
        """)
        layout.addWidget(lbl_title)

        # Image
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

        # Stats label
        if img is not None:
            stats = _compute_stats(img)
            lbl_stats = QLabel(stats)
            lbl_stats.setAlignment(Qt.AlignCenter)
            lbl_stats.setStyleSheet("""
                color: rgba(180, 180, 180, 180);
                font-size: 10px;
                background: transparent;
                border: none;
            """)
            layout.addWidget(lbl_stats)


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
    """

    # Ordered list of (key, display_title) pairs
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
        ("edge_points_viz", "⑩ Sub-pixel Edge Points"),
    ]

    def __init__(self, parent=None) -> None:
        super().__init__(parent, Qt.Window | Qt.WindowStaysOnTopHint)
        self.setWindowTitle("VideoFIT – Debug: Preprocessing Stages")
        self.setMinimumSize(900, 600)
        self.resize(1300, 820)
        self.setStyleSheet("background-color: #0d0d0d;")

        # ── Top bar ───────────────────────────────────────────────────
        top_bar = QWidget()
        top_bar.setStyleSheet("background-color: #131313; border-bottom: 1px solid rgba(255,255,255,20);")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(16, 8, 16, 8)

        self._lbl_status = QLabel("No data yet — run a fit to populate stages.")
        self._lbl_status.setStyleSheet("color: rgba(255,255,255,160); font-size: 12px;")
        top_layout.addWidget(self._lbl_status)
        top_layout.addStretch()

        btn_close = QPushButton("✕  Close")
        btn_close.setStyleSheet("""
            QPushButton {
                background: rgba(200,50,50,180);
                border: none; border-radius: 6px;
                color: white; font-weight: bold;
                padding: 6px 14px;
            }
            QPushButton:hover { background: rgba(220,70,70,220); }
        """)
        btn_close.clicked.connect(self.hide)
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
            self._lbl_status.setText(f"Stages captured — {len(stages)} images.")

        if not self.isVisible():
            self.show()
        self.raise_()

