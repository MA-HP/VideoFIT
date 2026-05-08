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


def _draw_crosshair(img: np.ndarray, cx: float, cy: float,
                    color=(0, 255, 0), radius: int = 28, thickness: int = 3) -> None:
    """Draw a high-contrast crosshair + circle (black outline + coloured fill)."""
    import cv2
    x, y = int(round(cx)), int(round(cy))
    arm = radius + 10
    outline = (0, 0, 0)
    ot = thickness + 3   # outline is thicker
    # Outline pass
    cv2.circle(img, (x, y), radius, outline, ot, cv2.LINE_AA)
    cv2.line(img, (x - arm, y), (x + arm, y), outline, ot, cv2.LINE_AA)
    cv2.line(img, (x, y - arm), (x, y + arm), outline, ot, cv2.LINE_AA)
    # Colour pass
    cv2.circle(img, (x, y), radius, color, thickness, cv2.LINE_AA)
    cv2.line(img, (x - arm, y), (x + arm, y), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x, y - arm), (x, y + arm), color, thickness, cv2.LINE_AA)
    # Small filled dot at centre
    cv2.circle(img, (x, y), max(3, thickness), outline, -1, cv2.LINE_AA)
    cv2.circle(img, (x, y), max(2, thickness - 1), color, -1, cv2.LINE_AA)


def _put_text_with_bg(img: np.ndarray, text: str, org: tuple[int, int],
                      color=(255, 255, 255), font_scale: float = 0.55,
                      thickness: int = 1) -> None:
    """Draw text with a semi-transparent dark background rectangle."""
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    pad = 4
    cv2.rectangle(img,
                  (x - pad, y - th - pad),
                  (x + tw + pad, y + baseline + pad),
                  (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def _make_dxf_preview(dxf_data, fit_result=None,
                      thumb_size: int = 1400) -> np.ndarray | None:
    """
    Render DXF polylines on a dark canvas at high resolution. Mark:
      - orange  : raw DXF pivot (dxf_cx, dxf_cy)
      - cyan    : aligned centroid after transform (if fit_result is provided)
    Returns an RGB uint8 image.
    """
    import cv2
    if dxf_data is None or not dxf_data.polylines:
        return None

    all_pts = np.concatenate(dxf_data.polylines, axis=0)
    if len(all_pts) == 0:
        return None

    mn = all_pts.min(axis=0)
    mx = all_pts.max(axis=0)
    rng = mx - mn
    if rng[0] < 1 or rng[1] < 1:
        return None

    margin = 60
    scale = (thumb_size - 2 * margin) / max(rng[0], rng[1])
    W = int(round(rng[0] * scale)) + 2 * margin
    H = int(round(rng[1] * scale)) + 2 * margin

    canvas = np.full((H, W, 3), 18, dtype=np.uint8)  # very dark grey bg

    def _to_px(pts):
        p = (pts - mn) * scale + margin
        return p.astype(np.int32)

    # Draw polylines with anti-aliasing
    for poly in dxf_data.polylines:
        pts = _to_px(poly)
        cv2.polylines(canvas, [pts], isClosed=False,
                      color=(200, 200, 200), thickness=2, lineType=cv2.LINE_AA)

    # Marker radius scaled to image size
    r = max(20, int(thumb_size * 0.025))

    if fit_result is not None:
        raw_pivot = np.array([[fit_result.dxf_cx, fit_result.dxf_cy]])
        pivot_local = (raw_pivot - mn) * scale + margin
        px, py = pivot_local[0, 0], pivot_local[0, 1]

        # DXF pivot (orange)
        _draw_crosshair(canvas, px, py, color=(0, 165, 255), radius=r, thickness=3)

        # Aligned centroid (cyan)
        ax = px + fit_result.tx * scale
        ay = py + fit_result.ty * scale
        _draw_crosshair(canvas, ax, ay, color=(0, 230, 230), radius=r, thickness=3)

        # Legend
        lh = max(28, int(H * 0.04))
        _put_text_with_bg(canvas, "● DXF pivot",          (12, lh),      (0, 165, 255), 0.7, 2)
        _put_text_with_bg(canvas, "● Aligned center",     (12, lh * 2),  (0, 230, 230), 0.7, 2)
    else:
        mid = _to_px(np.array([[(mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2]]))
        _draw_crosshair(canvas, mid[0, 0], mid[0, 1], color=(0, 165, 255), radius=r, thickness=3)
        _put_text_with_bg(canvas, "● DXF centre", (12, 36), (0, 165, 255), 0.7, 2)

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def _make_centroid_overlay(edge_result, fit_result=None,
                           frame_bgr: np.ndarray | None = None,
                           thumb_size: int = 1400) -> np.ndarray | None:
    """
    Build a full-res overlay image showing:
      - green  : detected shape centroid
      - cyan   : aligned DXF centroid (dxf_cx + tx, dxf_cy + ty)
      - orange : raw DXF pivot (dxf_cx, dxf_cy)
    Returns an RGB uint8 image (downscaled for display, full-res stored for zoom).
    """
    import cv2
    if edge_result is None:
        return None

    edges = edge_result.edges  # uint8 grayscale
    h, w = edges.shape[:2]

    if frame_bgr is not None and frame_bgr.shape[:2] == (h, w):
        base = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).copy()
        # Burn detected edges as bright blue overlay
        mask = edges > 0
        base[mask] = (60, 120, 255)
    else:
        # Fall back to edge map converted to greyscale-on-black RGB
        base = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    # Marker radius: ~1.5 % of the longer image dimension — always legible
    r = max(18, int(max(h, w) * 0.015))
    t = max(2, r // 10)

    # Detected centroid (green)
    cx = float(edge_result.silhouette_centroid[0])
    cy = float(edge_result.silhouette_centroid[1])
    _draw_crosshair(base, cx, cy, color=(0, 230, 80), radius=r, thickness=t)

    if fit_result is not None:
        # Raw DXF pivot (orange)
        _draw_crosshair(base, fit_result.dxf_cx, fit_result.dxf_cy,
                        color=(255, 140, 0), radius=int(r * 0.8), thickness=t)
        # Aligned DXF centroid (cyan)
        acx = fit_result.dxf_cx + fit_result.tx
        acy = fit_result.dxf_cy + fit_result.ty
        _draw_crosshair(base, acx, acy, color=(0, 220, 220), radius=r, thickness=t)

    # Legend — positioned top-left, with bg rectangle for readability
    fs = max(0.6, min(1.4, max(h, w) / 3000))  # font scale relative to image size
    ft = max(1, int(fs * 2))
    lpad = 12
    lh = int(max(h, w) * 0.025)
    _put_text_with_bg(base, "● Detected centroid", (lpad, lh),          (0, 230, 80),  fs, ft)
    if fit_result is not None:
        _put_text_with_bg(base, "● DXF pivot",         (lpad, lh * 2),  (255, 140, 0), fs, ft)
        _put_text_with_bg(base, "● Aligned DXF center",(lpad, lh * 3),  (0, 220, 220), fs, ft)

    # Downscale to thumb_size for display (full-res kept via _StageCard)
    if max(h, w) > thumb_size:
        s = thumb_size / max(h, w)
        base = cv2.resize(base, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    return base


def _info_card(title: str, rows: list[tuple[str, str]], accent: str = "#4eb3f7") -> QFrame:
    """Build a small dark card with a title and key/value rows."""
    card = QFrame()
    card.setObjectName("InfoCard")
    card.setStyleSheet("""
        QFrame#InfoCard {
            background-color: rgba(20, 20, 30, 240);
            border: 1px solid rgba(255, 255, 255, 20);
            border-radius: 10px;
        }
    """)
    layout = QVBoxLayout(card)
    layout.setContentsMargins(12, 10, 12, 10)
    layout.setSpacing(4)

    lbl_title = QLabel(title)
    lbl_title.setStyleSheet(
        f"color: {accent}; font-weight: bold; font-size: 12px;"
        " background: transparent; border: none;")
    layout.addWidget(lbl_title)

    sep = QFrame()
    sep.setFrameShape(QFrame.HLine)
    sep.setStyleSheet("color: rgba(255,255,255,20); border: none;"
                      " border-top: 1px solid rgba(255,255,255,25);")
    layout.addWidget(sep)

    for key, val in rows:
        row_w = QWidget()
        row_w.setStyleSheet("background: transparent;")
        rl = QHBoxLayout(row_w)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(6)
        lk = QLabel(key)
        lk.setStyleSheet(
            "color: rgba(180,180,180,200); font-size: 11px;"
            " background: transparent; border: none;")
        lv = QLabel(val)
        lv.setStyleSheet(
            "color: rgba(255,255,255,220); font-size: 11px; font-weight: bold;"
            " background: transparent; border: none;")
        lv.setTextInteractionFlags(Qt.TextSelectableByMouse)
        rl.addWidget(lk)
        rl.addStretch()
        rl.addWidget(lv)
        layout.addWidget(row_w)

    return card


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
        ("edges_dev",       "② GPU Devernay Sub-pixel Edges"),
        ("edge_points_viz", "③ Sub-pixel Edge Points"),
        ("distance_field",  "④ Distance Transform"),
    ]

    def __init__(self, parent=None) -> None:
        super().__init__(parent, Qt.Window | Qt.WindowStaysOnTopHint)
        self.setWindowTitle("VideoFIT – Debug: Preprocessing Stages")
        self.setMinimumSize(900, 600)
        self.resize(1400, 920)
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

        # ── Info bar (scrollable row of data cards) ───────────────────
        info_scroll = QScrollArea()
        info_scroll.setFixedHeight(220)
        info_scroll.setWidgetResizable(True)
        info_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        info_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        info_scroll.setStyleSheet(
            "background-color: #111116; border: none;"
            " border-bottom: 1px solid rgba(255,255,255,15);")

        self._info_bar_widget = QWidget()
        self._info_bar_widget.setStyleSheet("background: transparent;")
        self._info_bar_layout = QHBoxLayout(self._info_bar_widget)
        self._info_bar_layout.setContentsMargins(14, 10, 14, 10)
        self._info_bar_layout.setSpacing(10)
        info_scroll.setWidget(self._info_bar_widget)

        self._info_scroll = info_scroll
        self._info_scroll.setVisible(False)

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
        main_layout.addWidget(self._info_scroll)
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
    # Info bar
    # ------------------------------------------------------------------

    def _rebuild_info_bar(self, fit_result=None, edge_result=None,
                          dxf_data=None, px_per_mm: float = 1.0) -> None:
        """Populate the info card bar with data from fit/edge results."""
        while self._info_bar_layout.count():
            item = self._info_bar_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if fit_result is None and edge_result is None:
            self._info_scroll.setVisible(False)
            return

        # ── Detected shape centroid ───────────────────────────────────
        if edge_result is not None:
            cx = float(edge_result.silhouette_centroid[0])
            cy = float(edge_result.silhouette_centroid[1])
            n_pts = len(edge_result.edge_points) if edge_result.edge_points is not None else 0
            rows: list[tuple[str, str]] = [
                ("Center X", f"{cx:.1f} px"),
                ("Center Y", f"{cy:.1f} px"),
                ("Edge points", f"{n_pts:,}"),
            ]
            if px_per_mm > 0:
                rows.append(("Center X (mm)", f"{cx / px_per_mm:.3f} mm"))
                rows.append(("Center Y (mm)", f"{cy / px_per_mm:.3f} mm"))
            self._info_bar_layout.addWidget(
                _info_card("📍 Detected Shape Centroid", rows, "#4eb3f7"))

        # ── DXF centroid / pivot ──────────────────────────────────────
        if fit_result is not None:
            dcx, dcy = fit_result.dxf_cx, fit_result.dxf_cy
            rows = [
                ("Pivot X", f"{dcx:.1f} px"),
                ("Pivot Y", f"{dcy:.1f} px"),
            ]
            if px_per_mm > 0:
                rows.append(("Pivot X (mm)", f"{dcx / px_per_mm:.3f} mm"))
                rows.append(("Pivot Y (mm)", f"{dcy / px_per_mm:.3f} mm"))
            if dxf_data is not None and hasattr(dxf_data, 'polylines') and dxf_data.polylines:
                all_pts = np.concatenate(dxf_data.polylines, axis=0)
                if len(all_pts):
                    mn = all_pts.min(axis=0)
                    mx = all_pts.max(axis=0)
                    w_px = mx[0] - mn[0]
                    h_px = mx[1] - mn[1]
                    rows.append(("BBox W", f"{w_px:.1f} px"))
                    rows.append(("BBox H", f"{h_px:.1f} px"))
                    if px_per_mm > 0:
                        rows.append(("BBox W (mm)", f"{w_px / px_per_mm:.2f} mm"))
                        rows.append(("BBox H (mm)", f"{h_px / px_per_mm:.2f} mm"))
            self._info_bar_layout.addWidget(
                _info_card("📐 DXF Pivot / Centroid", rows, "#f7a94e"))

        # ── Alignment transform ───────────────────────────────────────
        if fit_result is not None:
            aligned_cx = fit_result.dxf_cx + fit_result.tx
            aligned_cy = fit_result.dxf_cy + fit_result.ty
            offset_px = float(np.hypot(fit_result.tx, fit_result.ty))
            rows = [
                ("Translation X", f"{fit_result.tx:+.2f} px"),
                ("Translation Y", f"{fit_result.ty:+.2f} px"),
                ("Rotation", f"{fit_result.angle_deg:.3f}°"),
                ("Offset magnitude", f"{offset_px:.2f} px"),
                ("Aligned center X", f"{aligned_cx:.1f} px"),
                ("Aligned center Y", f"{aligned_cy:.1f} px"),
            ]
            if px_per_mm > 0:
                rows.append(("Translation X (mm)", f"{fit_result.tx / px_per_mm:+.4f} mm"))
                rows.append(("Translation Y (mm)", f"{fit_result.ty / px_per_mm:+.4f} mm"))
                rows.append(("Offset (mm)", f"{offset_px / px_per_mm:.4f} mm"))
            self._info_bar_layout.addWidget(
                _info_card("🔄 Alignment Transform", rows, "#82f74e"))

        # ── Fit quality ───────────────────────────────────────────────
        if fit_result is not None:
            inlier_pct = fit_result.inlier_frac * 100.0
            quality = ("✅ Good" if inlier_pct >= 80
                       else ("⚠️ Fair" if inlier_pct >= 50 else "❌ Poor"))
            rows = [
                ("Cost (mean dist)", f"{fit_result.cost:.4f}"),
                ("Inlier fraction", f"{inlier_pct:.1f}%"),
                ("Quality", quality),
            ]
            if edge_result is not None:
                scx = float(edge_result.silhouette_centroid[0])
                scy = float(edge_result.silhouette_centroid[1])
                acx = fit_result.dxf_cx + fit_result.tx
                acy = fit_result.dxf_cy + fit_result.ty
                delta = float(np.hypot(scx - acx, scy - acy))
                rows.append(("Centroid Δ", f"{delta:.2f} px"))
                if px_per_mm > 0:
                    rows.append(("Centroid Δ (mm)", f"{delta / px_per_mm:.4f} mm"))
            self._info_bar_layout.addWidget(
                _info_card("📊 Fit Quality", rows, "#f74e82"))

        self._info_bar_layout.addStretch()
        self._info_scroll.setVisible(True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_stages(self, stages: dict[str, np.ndarray], fit_info: str = "",
                      fit_result=None, edge_result=None,
                      dxf_data=None, px_per_mm: float = 1.0,
                      frame_bgr: np.ndarray | None = None) -> None:
        """
        Refresh the grid with new preprocessing stage images.

        Parameters
        ----------
        stages : dict
            Mapping of stage-key → numpy image.
        fit_info : str
            Optional one-line summary shown in the status bar.
        fit_result : FitResult | None
        edge_result : EdgeResult | None
        dxf_data : Dxf | None
        px_per_mm : float
        frame_bgr : np.ndarray | None
            Original camera BGR frame for centroid overlay background.
        """
        self._rebuild_info_bar(fit_result, edge_result, dxf_data, px_per_mm)

        # Build synthetic visual stages
        extra_stages: dict[str, tuple[str, np.ndarray]] = {}
        dxf_preview = _make_dxf_preview(dxf_data, fit_result)
        if dxf_preview is not None:
            extra_stages["__dxf_preview__"] = ("🗺 DXF Preview + Centroids", dxf_preview)
        centroid_overlay = _make_centroid_overlay(edge_result, fit_result, frame_bgr)
        if centroid_overlay is not None:
            extra_stages["__centroid_overlay__"] = ("🎯 Detected vs Aligned Centroids", centroid_overlay)

        # Clear existing stage cards
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        cols = 4
        row, col = 0, 0

        # Extra visual stages — 2 large cards side-by-side on their own row
        visual_items = list(extra_stages.items())
        for i, (key, (title, img)) in enumerate(visual_items):
            card = _StageCard(title, img, max_size=520)
            # Each card spans 2 columns so the pair fills the 4-column grid
            self._grid_layout.addWidget(card, row, i * 2, 1, 2)
        if visual_items:
            row += 1
            col = 0

        # Standard pipeline stages
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
