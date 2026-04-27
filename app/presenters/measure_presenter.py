"""
VideoFIT — Measure Presenter
Orchestrates the Measure-mode pipeline:

  Run   → edge detect → geometric auto-fit → overlay (replaces all shapes)
  Build → stroke drawn on viewer → collect nearby sub-pixel edge pts →
          geometric fit of chosen type → overlay (appends shape)

Edge points from the last Run are cached so that Build strokes do not
need to re-run the full edge-detection pipeline.
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

from app.models.settings import AppSettings
from app.services.edge_service import compute_edges
from app.services.shape_fit_service import (
    auto_detect_shapes,
    collect_near_stroke,
    fit_shape,
    interpolate_stroke,
)
from app.views.image_viewer import ImageViewer
from app.views.measure_overlay import MeasureOverlay
from app.views.toolbar import Toolbar


# ─────────────────────────────────────────────────────────────────────────────
# Worker: auto-detect (Run button)
# ─────────────────────────────────────────────────────────────────────────────

class _RunSignals(QObject):
    finished = Signal(object)   # (edge_pts: ndarray, results: list[MeasureResult])
    error    = Signal(str)


class _RunWorker(QRunnable):
    """Edge-detect the frame then fit all shapes via connected-component analysis."""

    def __init__(self, frame_bgr: np.ndarray) -> None:
        super().__init__()
        self.frame_bgr = frame_bgr
        self.signals   = _RunSignals()

    def run(self) -> None:
        try:
            edge_result = compute_edges(self.frame_bgr)
            pts = edge_result.edge_points
            if pts is None or len(pts) == 0:
                self.signals.finished.emit((np.empty((0, 2), np.float32), []))
                return
            results = auto_detect_shapes(edge_result.edges, pts)
            self.signals.finished.emit((pts, results))
        except Exception:
            import traceback
            self.signals.error.emit(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# Worker: stroke fit — fast path (edge points already cached)
# ─────────────────────────────────────────────────────────────────────────────

class _StrokeSignals(QObject):
    finished = Signal(object)   # list[MeasureResult] (0 or 1 item)
    error    = Signal(str)


class _StrokeFitWorker(QRunnable):
    """
    Given pre-computed sub-pixel edge points and the drawn stroke path,
    densify the stroke, collect nearby edge points, and fit the requested shape.
    No edge detection is performed (uses the Run-mode cache).
    """

    def __init__(
        self,
        edge_pts: np.ndarray,
        stroke_pts: np.ndarray,
        shape_kind: str,
    ) -> None:
        super().__init__()
        self.edge_pts   = edge_pts
        self.stroke_pts = stroke_pts
        self.shape_kind = shape_kind
        self.signals    = _StrokeSignals()

    def run(self) -> None:
        try:
            dense = interpolate_stroke(self.stroke_pts, spacing=2.0)
            collected = collect_near_stroke(self.edge_pts, dense)
            if len(collected) < 5:
                self.signals.finished.emit([])
                return
            result = fit_shape(collected, kind=self.shape_kind)
            self.signals.finished.emit([result] if result is not None else [])
        except Exception:
            import traceback
            self.signals.error.emit(traceback.format_exc())


class _EdgeStrokeFitWorker(QRunnable):
    """
    Slow path (no cache): edge-detect the frame first, then stroke-fit.
    Caches edge_pts via the ``finished`` signal so subsequent strokes are fast.
    """

    def __init__(
        self,
        frame_bgr: np.ndarray,
        stroke_pts: np.ndarray,
        shape_kind: str,
    ) -> None:
        super().__init__()
        self.frame_bgr  = frame_bgr
        self.stroke_pts = stroke_pts
        self.shape_kind = shape_kind
        self.signals    = _StrokeSignals()

    def run(self) -> None:
        try:
            edge_result = compute_edges(self.frame_bgr)
            pts = edge_result.edge_points
            if pts is None or len(pts) == 0:
                self.signals.finished.emit(([], None))
                return
            dense = interpolate_stroke(self.stroke_pts, spacing=2.0)
            collected = collect_near_stroke(pts, dense)
            if len(collected) < 5:
                self.signals.finished.emit(([], pts))
                return
            result = fit_shape(collected, kind=self.shape_kind)
            results = [result] if result is not None else []
            # Emit (results, edge_pts) so the presenter can cache edge_pts
            self.signals.finished.emit((results, pts))
        except Exception:
            import traceback
            self.signals.error.emit(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# Presenter
# ─────────────────────────────────────────────────────────────────────────────

class MeasurePresenter(QObject):
    """
    Mediates between the Measure-mode toolbar buttons, shape-fitting services,
    and the overlay view layer.

    Edge points from the last Run are cached in ``_cached_edge_pts`` so that
    Build strokes can skip the expensive edge-detection step.
    """

    def __init__(
        self,
        settings: AppSettings,
        viewer: ImageViewer,
        toolbar: Toolbar,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self._viewer   = viewer
        self._toolbar  = toolbar

        self._overlay          = MeasureOverlay(viewer._scene)
        self._pool             = QThreadPool.globalInstance()
        self._cached_edge_pts: np.ndarray | None = None

        # ── Wiring ────────────────────────────────────────────────────
        toolbar.btn_run_measure.clicked.connect(self._on_run)
        toolbar.btn_build.clicked.connect(self._on_build_clicked)
        viewer.stroke_completed.connect(self._on_stroke_completed)

    # ── Slots ─────────────────────────────────────────────────────────

    @Slot()
    def _on_run(self) -> None:
        """Snap the current frame, run edge detection, auto-detect all shapes."""
        frame = self._viewer._current_cv_img
        if frame is None:
            print("Measure: no image available.")
            return

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self._toolbar.btn_run_measure.setEnabled(False)
        self._toolbar.btn_run_measure.setText(" Running…")

        worker = _RunWorker(frame_bgr.copy())
        worker.signals.finished.connect(self._on_run_done)
        worker.signals.error.connect(self._on_error)
        worker.setAutoDelete(True)
        self._pool.start(worker)

    @Slot(bool)
    def _on_build_clicked(self, checked: bool) -> None:
        """Toggle the stroke-brush mode on the viewer."""
        self._viewer.set_roi_mode(checked)

    @Slot(object)
    def _on_stroke_completed(self, stroke_pts: np.ndarray) -> None:
        """
        Viewer emitted the drawn stroke path.
        Use cached edge points if available; otherwise run edge detection first.
        """
        self._toolbar.btn_build.setChecked(False)   # uncheck Build button

        shape_kind = self._toolbar.current_shape().lower()
        frame = self._viewer._current_cv_img
        if frame is None:
            print("Measure: no image for stroke fit.")
            return

        if self._cached_edge_pts is not None and len(self._cached_edge_pts) > 0:
            # ── Fast path: reuse edge points from last Run ────────────
            worker = _StrokeFitWorker(
                self._cached_edge_pts, stroke_pts, shape_kind
            )
            worker.signals.finished.connect(self._on_stroke_done)
            worker.signals.error.connect(self._on_error)
        else:
            # ── Slow path: edge-detect then fit ──────────────────────
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            worker = _EdgeStrokeFitWorker(frame_bgr.copy(), stroke_pts, shape_kind)
            worker.signals.finished.connect(self._on_edge_stroke_done)
            worker.signals.error.connect(self._on_error)

        worker.setAutoDelete(True)
        self._pool.start(worker)

    # ── Callbacks ─────────────────────────────────────────────────────

    @Slot(object)
    def _on_run_done(self, payload: tuple) -> None:
        edge_pts, results = payload
        self._cached_edge_pts = edge_pts

        kinds: dict[str, int] = {}
        for r in results:
            kinds[r.kind.value] = kinds.get(r.kind.value, 0) + 1
        summary = ", ".join(f"{v}×{k}" for k, v in kinds.items()) if kinds else "none"
        print(f"Measure: {len(results)} shapes — {summary}")

        self._overlay.draw_shapes(results)
        self._toolbar.btn_run_measure.setEnabled(True)
        self._toolbar.btn_run_measure.setText(" Run")

    @Slot(object)
    def _on_stroke_done(self, results: list) -> None:
        if results:
            r = results[0]
            print(
                f"Measure Build: {r.kind.value}  "
                f"residual={r.residual_rms:.3f} px  n={r.n_points}"
            )
            self._overlay.add_shape(r)
        else:
            print("Measure Build: no shape fitted — try a longer stroke or different type.")

    @Slot(object)
    def _on_edge_stroke_done(self, payload: tuple) -> None:
        """Slow-path callback: payload is (results, edge_pts) — cache edge_pts."""
        results, edge_pts = payload
        if edge_pts is not None and len(edge_pts) > 0:
            self._cached_edge_pts = edge_pts
        self._on_stroke_done(results)

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        print(f"Measure error:\n{msg}")
        self._toolbar.btn_run_measure.setEnabled(True)
        self._toolbar.btn_run_measure.setText(" Run")
        self._toolbar.btn_build.setChecked(False)

    # ── Public API ────────────────────────────────────────────────────

    def clear_overlay(self) -> None:
        """Called when switching to Compare mode."""
        self._overlay.clear()
        self._cached_edge_pts = None
        self._toolbar.btn_build.setChecked(False)
        self._viewer.set_roi_mode(False)
