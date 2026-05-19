"""
VideoFIT — Measure Presenter (rewrite)
Two sub-modes driven by the toolbar:

  Auto (Run)  — One click triggers auto_detect_shapes on cached edges.
                Edge detection runs automatically in background via EdgeCacheService
                whenever the image stabilises.

  Manual (Draw) — User selects shape type, drags mouse over an edge.
                  During drag, nearby edge points "magnet-paint" (snap to the
                  detected edges in real time).  On release, the desired shape
                  is fitted to the collected points.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal, Slot
from scipy.spatial import cKDTree

from app.models.edge_result import EdgeResult
from app.models.measure_result import MeasureResult
from app.models.settings import AppSettings
from app.services.edge_cache_service import EdgeCacheService
from app.services.shape_fit_service import (
    auto_detect_shapes,
    collect_near_stroke,
    fit_shape,
    interpolate_stroke,
    STROKE_BRUSH_RADIUS_PX,
)
from app.views.image_viewer import ImageViewer
from app.views.measure_overlay import MeasureOverlay
from app.views.toolbar import Toolbar


# ─────────────────────────────────────────────────────────────────────────────
# Workers
# ─────────────────────────────────────────────────────────────────────────────

class _WorkerSignals(QObject):
    finished = Signal(object)
    error = Signal(str)


class _AutoFitWorker(QRunnable):
    """Fit all shapes from pre-computed edge data."""

    def __init__(self, edge_result: EdgeResult) -> None:
        super().__init__()
        self.edge_result = edge_result
        self.signals = _WorkerSignals()

    def run(self) -> None:
        try:
            pts = self.edge_result.edge_points
            if pts is None or len(pts) == 0:
                self.signals.finished.emit([])
                return
            results = auto_detect_shapes(self.edge_result.edges, pts)
            self.signals.finished.emit(results)
        except Exception:
            import traceback
            self.signals.error.emit(traceback.format_exc())


class _ManualFitWorker(QRunnable):
    """Fit a single shape to collected edge points."""

    def __init__(self, collected_pts: np.ndarray, shape_kind: str) -> None:
        super().__init__()
        self.collected_pts = collected_pts
        self.shape_kind = shape_kind
        self.signals = _WorkerSignals()

    def run(self) -> None:
        try:
            if len(self.collected_pts) < 5:
                self.signals.finished.emit(None)
                return
            result = fit_shape(self.collected_pts, kind=self.shape_kind)
            self.signals.finished.emit(result)
        except Exception:
            import traceback
            self.signals.error.emit(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# Presenter
# ─────────────────────────────────────────────────────────────────────────────

class MeasurePresenter(QObject):
    """
    Orchestrates both Auto and Manual measure modes.

    Relies on EdgeCacheService for background edge detection triggered by
    image stability.
    """

    def __init__(
        self,
        settings: AppSettings,
        viewer: ImageViewer,
        toolbar: Toolbar,
        edge_cache: EdgeCacheService,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self._viewer = viewer
        self._toolbar = toolbar
        self._edge_cache = edge_cache

        self._overlay = MeasureOverlay(viewer._scene)
        self._pool = QThreadPool.globalInstance()

        # Manual mode drag state
        self._dragging = False
        self._collected_pts: np.ndarray | None = None
        self._edge_tree: cKDTree | None = None

        # Throttle live magnet paint
        self._last_magnet_time: float = 0.0
        _MAGNET_THROTTLE_MS = 50  # minimum interval between magnet updates
        self._magnet_throttle_s = _MAGNET_THROTTLE_MS / 1000.0

        # ── Wiring ────────────────────────────────────────────────────
        # Auto mode: Run button
        toolbar.btn_run_measure.clicked.connect(self._on_auto_fit)

        # Manual mode: Draw button toggles stroke interaction
        toolbar.btn_build.clicked.connect(self._on_draw_clicked)

        # Viewer stroke events for manual mode
        viewer.stroke_completed.connect(self._on_stroke_completed)
        viewer.stroke_progress.connect(self._on_stroke_progress)

        # Edge cache ready
        edge_cache.edges_ready.connect(self._on_edges_ready)

        print("[Measure] Presenter initialized — Draw button connected.")

    # ══════════════════════════════════════════════════════════════════════════
    # Auto mode
    # ══════════════════════════════════════════════════════════════════════════

    @Slot()
    def _on_auto_fit(self) -> None:
        """Run button clicked — fit all shapes from cached edges."""
        edge_result = self._edge_cache.cached_edges
        if edge_result is None:
            # No edges yet — force a computation from the current frame
            frame = self._viewer._current_cv_img
            if frame is None:
                print("[Measure Auto] No image available.")
                return
            self._force_edge_compute(frame)
            return

        self._start_auto_fit(edge_result)

    def _start_auto_fit(self, edge_result: EdgeResult) -> None:
        self._toolbar.btn_run_measure.setEnabled(False)
        self._toolbar.btn_run_measure.setText(" Running…")

        worker = _AutoFitWorker(edge_result)
        worker.signals.finished.connect(self._on_auto_fit_done)
        worker.signals.error.connect(self._on_error)
        worker.setAutoDelete(True)
        self._pool.start(worker)

    def _force_edge_compute(self, frame_rgb: np.ndarray) -> None:
        """When no cached edges exist, compute them then auto-fit."""
        from app.services.edge_service import compute_edges

        self._toolbar.btn_run_measure.setEnabled(False)
        self._toolbar.btn_run_measure.setText(" Detecting…")

        class _Worker(QRunnable):
            def __init__(self, frame_bgr, signals):
                super().__init__()
                self.frame_bgr = frame_bgr
                self.signals = signals

            def run(self):
                try:
                    result = compute_edges(self.frame_bgr)
                    self.signals.finished.emit(result)
                except Exception:
                    import traceback
                    self.signals.error.emit(traceback.format_exc())

        signals = _WorkerSignals()
        signals.finished.connect(self._on_forced_edges_done)
        signals.error.connect(self._on_error)

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        worker = _Worker(frame_bgr.copy(), signals)
        worker.setAutoDelete(True)
        self._pool.start(worker)

    @Slot(object)
    def _on_forced_edges_done(self, edge_result: EdgeResult) -> None:
        """Forced edge computation done — now auto-fit."""
        self._edge_cache._cached_result = edge_result
        self._start_auto_fit(edge_result)

    @Slot(object)
    def _on_auto_fit_done(self, results: list) -> None:
        kinds: dict[str, int] = {}
        for r in results:
            kinds[r.kind.value] = kinds.get(r.kind.value, 0) + 1
        summary = ", ".join(f"{v}×{k}" for k, v in kinds.items()) if kinds else "none"
        print(f"[Measure Auto] {len(results)} shapes — {summary}")

        self._overlay.draw_shapes(results)
        self._toolbar.btn_run_measure.setEnabled(True)
        self._toolbar.btn_run_measure.setText(" Run")

    @Slot(object)
    def _on_edges_ready(self, edge_result: EdgeResult) -> None:
        """EdgeCacheService computed new edges in background."""
        # Rebuild the KD-tree for manual mode snapping
        pts = edge_result.edge_points
        if pts is not None and len(pts) > 0:
            self._edge_tree = cKDTree(pts.astype(np.float64))
        else:
            self._edge_tree = None
        print(f"[Measure] Edges ready — {len(pts) if pts is not None else 0} points")

    # ══════════════════════════════════════════════════════════════════════════
    # Manual mode
    # ══════════════════════════════════════════════════════════════════════════

    @Slot()
    def _on_draw_clicked(self) -> None:
        """Draw button clicked — read its checked state and enable/disable stroke mode."""
        checked = self._toolbar.btn_build.isChecked()
        print(f"[Measure] Draw clicked, checked={checked}")
        self._viewer.set_roi_mode(checked)
        if not checked:
            self._overlay.clear_magnet()
            self._overlay.clear_preview()

    @Slot(object)
    def _on_stroke_completed(self, stroke_pts: np.ndarray) -> None:
        """
        User finished dragging on the viewer.
        Collect edge points near the stroke path and fit the chosen shape.
        Stroke mode remains active for continuous drawing.
        """
        self._overlay.clear_magnet()
        self._overlay.clear_preview()

        edge_result = self._edge_cache.cached_edges
        if edge_result is None or edge_result.edge_points is None:
            print("[Measure Manual] No edge data — click Run first or wait for edges.")
            return

        edge_pts = edge_result.edge_points
        shape_kind = self._toolbar.current_shape().lower()

        # Densify stroke and collect nearby edge points
        dense = interpolate_stroke(stroke_pts, spacing=2.0)
        collected = collect_near_stroke(edge_pts, dense)

        if len(collected) < 5:
            print("[Measure Manual] Too few edge points collected — try a longer stroke.")
            return

        # Show collected points as magnet paint
        self._overlay.set_magnet_points(collected)

        # Fit in background
        worker = _ManualFitWorker(collected, shape_kind)
        worker.signals.finished.connect(self._on_manual_fit_done)
        worker.signals.error.connect(self._on_error)
        worker.setAutoDelete(True)
        self._pool.start(worker)

    @Slot(object)
    def _on_manual_fit_done(self, result: MeasureResult | None) -> None:
        self._overlay.clear_magnet()
        if result is not None:
            print(
                f"[Measure Manual] {result.kind.value}  "
                f"residual={result.residual_rms:.3f} px  n={result.n_points}"
            )
            self._overlay.add_shape(result)
        else:
            print("[Measure Manual] No shape fitted — try different type or longer stroke.")

    # ══════════════════════════════════════════════════════════════════════════
    # Live magnet paint during drag
    # ══════════════════════════════════════════════════════════════════════════

    @Slot(object)
    def _on_stroke_progress(self, stroke_pts: np.ndarray) -> None:
        """Called repeatedly during drag — show snapped edge points in real time."""
        now = time.time()
        if now - self._last_magnet_time < self._magnet_throttle_s:
            return
        self._last_magnet_time = now

        edge_result = self._edge_cache.cached_edges
        if edge_result is None or edge_result.edge_points is None:
            return

        # Collect edge points near the current stroke path
        dense = interpolate_stroke(stroke_pts, spacing=4.0)
        collected = collect_near_stroke(edge_result.edge_points, dense)
        self._overlay.set_magnet_points(collected)

    # ══════════════════════════════════════════════════════════════════════════
    # Error handling & public API
    # ══════════════════════════════════════════════════════════════════════════

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        print(f"[Measure] Error:\n{msg}")
        self._toolbar.btn_run_measure.setEnabled(True)
        self._toolbar.btn_run_measure.setText(" Run")
        self._toolbar.btn_build.setChecked(False)

    def clear_overlay(self) -> None:
        """Called when switching away from Measure mode."""
        self._overlay.clear()
        self._toolbar.btn_build.setChecked(False)
        self._viewer.set_roi_mode(False)

    def activate(self) -> None:
        """Called when entering Measure mode — enable edge cache and activate Draw."""
        self._edge_cache.set_enabled(True)
        # Auto-activate Draw mode so user can immediately draw on the canvas
        if not self._toolbar.btn_build.isChecked():
            self._toolbar.btn_build.setChecked(True)
            self._toolbar._on_measure_tool_clicked(self._toolbar.btn_build)
        self._viewer.set_roi_mode(True)

    def deactivate(self) -> None:
        """Called when leaving Measure mode — disable edge cache."""
        self._edge_cache.set_enabled(False)
        self._overlay.clear()

