"""
VideoFIT — Edge Cache Service
Background edge detection triggered by image stability.

Monitors incoming frames; when the image stabilises (robust pixel-change
percentage stays below threshold for N consecutive frames), runs edge
detection in a background thread and caches the result.

Stability uses the same robust method as AutoPresenter: downscale, noise
threshold, morphological open, then percentage of changed pixels.
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal, Slot

from app.models.edge_result import EdgeResult
from app.services.edge_service import compute_edges

_DETECT_WIDTH = 640
_NOISE_PX_THRESHOLD = 25
_MORPH_KERNEL_SIZE = 15


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────

class _EdgeWorkerSignals(QObject):
    finished = Signal(object)  # EdgeResult
    error = Signal(str)


class _EdgeWorker(QRunnable):
    def __init__(self, frame_bgr: np.ndarray) -> None:
        super().__init__()
        self.frame_bgr = frame_bgr
        self.signals = _EdgeWorkerSignals()

    def run(self) -> None:
        try:
            result = compute_edges(self.frame_bgr)
            self.signals.finished.emit(result)
        except Exception:
            import traceback
            self.signals.error.emit(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# Service
# ─────────────────────────────────────────────────────────────────────────────

class EdgeCacheService(QObject):
    """
    Watches frames and automatically computes edges when the image stabilises.

    Stability detection (same as Auto mode):
      1. Downscale frame to 640px wide
      2. Absolute diff with previous frame
      3. Threshold at 25 to ignore sensor noise
      4. Morphological open to remove speckle
      5. Percentage of remaining changed pixels < threshold
      6. Must stay stable for ``stabilization_frames`` consecutive ticks
    """

    edges_ready = Signal(object)  # EdgeResult

    def __init__(
        self,
        stability_threshold_pct: float = 0.1,
        stabilization_frames: int = 5,
        tick_interval_ms: int = 100,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._threshold_pct = stability_threshold_pct
        self._stabilization_frames = stabilization_frames

        self._prev_gray: np.ndarray | None = None
        self._stable_frame_rgb: np.ndarray | None = None
        self._cached_result: EdgeResult | None = None
        self._computing = False
        self._enabled = False
        self._stable_count = 0
        self._edges_valid = False  # True after edges computed for current stable image

        self._pool = QThreadPool.globalInstance()

    # ── Public API ────────────────────────────────────────────────────

    @property
    def cached_edges(self) -> EdgeResult | None:
        return self._cached_result

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, on: bool) -> None:
        """Enable/disable watching."""
        self._enabled = on
        if not on:
            self._prev_gray = None
            self._stable_count = 0

    def invalidate(self) -> None:
        """Force re-computation on next stable frame."""
        self._cached_result = None
        self._edges_valid = False
        self._prev_gray = None
        self._stable_count = 0

    @Slot(object)
    def on_frame(self, cv_img_rgb: np.ndarray) -> None:
        """Feed a new frame (RGB). Called from camera_service.frame_ready."""
        if not self._enabled or self._computing:
            return

        # Downscale for fast comparison
        h, w = cv_img_rgb.shape[:2]
        scale = _DETECT_WIDTH / w
        small = cv2.resize(cv_img_rgb, (_DETECT_WIDTH, int(h * scale)),
                           interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

        if self._prev_gray is not None and gray.shape == self._prev_gray.shape:
            diff_pct = self._frame_diff_pct(self._prev_gray, gray)

            if diff_pct < self._threshold_pct:
                # Frame is stable
                self._stable_count += 1
                if self._stable_count == 1:
                    # First stable frame — remember the full-res frame
                    self._stable_frame_rgb = cv_img_rgb
                if (self._stable_count >= self._stabilization_frames
                        and not self._edges_valid):
                    # Stable long enough and no valid edges — compute
                    self._launch_edge_compute()
            else:
                # Significant change detected
                self._stable_count = 0
                if self._edges_valid:
                    # Image changed — invalidate cached edges
                    self._cached_result = None
                    self._edges_valid = False
        else:
            self._stable_count = 0

        self._prev_gray = gray

    # ── Internal ──────────────────────────────────────────────────────

    def _frame_diff_pct(self, ref: np.ndarray, cur: np.ndarray) -> float:
        """Robust frame difference as percentage of pixels changed.
        Same algorithm as AutoPresenter."""
        diff = cv2.absdiff(ref, cur)
        _, mask = cv2.threshold(diff, _NOISE_PX_THRESHOLD, 255, cv2.THRESH_BINARY)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        _, mask = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY)
        k = max(3, int(_MORPH_KERNEL_SIZE * _DETECT_WIDTH / 5472) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return (np.count_nonzero(mask) / mask.size) * 100.0

    def _launch_edge_compute(self) -> None:
        if self._stable_frame_rgb is None or self._computing:
            return
        self._computing = True
        frame_bgr = cv2.cvtColor(self._stable_frame_rgb, cv2.COLOR_RGB2BGR)
        worker = _EdgeWorker(frame_bgr.copy())
        worker.signals.finished.connect(self._on_edges_computed)
        worker.signals.error.connect(self._on_error)
        worker.setAutoDelete(True)
        self._pool.start(worker)

    @Slot(object)
    def _on_edges_computed(self, result: EdgeResult) -> None:
        self._computing = False
        self._cached_result = result
        self._edges_valid = True
        self.edges_ready.emit(result)

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self._computing = False
        print(f"[EdgeCacheService] Error:\n{msg}")

