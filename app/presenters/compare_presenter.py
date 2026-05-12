"""
VideoFIT — Compare Presenter
Orchestrates the DXF-to-piece alignment pipeline:
  Load DXF → Edge detection → Distance transform → Fit → Overlay

Matches the proven POC workflow exactly.
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot, Qt
from PySide6.QtWidgets import QFileDialog

from app.models.dxf import Dxf
from app.services.dxf_service import load_dxf
from app.services.edge_service import compute_edges
from app.services.fit_service import fit, fit_complete, fit_poc
from app.models.fit_result import FitResult
from app.models.settings import AppSettings
from app.views.debug_window import DebugPreprocessingWindow
from app.views.dxf_overlay import DxfOverlay
from app.views.image_viewer import ImageViewer
from app.views.settings_panel import SettingsPanel
from app.views.toolbar import Toolbar


class _FitSignals(QObject):
    """Signals emitted by the background fitting worker."""
    finished = Signal(object)   # (FitResult, rgba_bytes, shape) or (FitResult, rgba_bytes, shape, stages, edge_result, frame_bgr)
    error = Signal(str)


class _FitWorker(QRunnable):
    """Runs the heavy edge-detection + fitting pipeline off the UI thread."""

    def __init__(self, frame_bgr: np.ndarray, dxf_data: Dxf, debug: bool = False,
                 mode: str = "Best Fit", objective: str = "Strict", max_error_px: float = 2.0,
                 heatmap_min: float = 1.0, heatmap_max: float = 3.0,
                 color_low: str = "#00FF00", color_mid: str = "#FF8000",
                 color_high: str = "#FF0000") -> None:
        super().__init__()
        self.frame_bgr = frame_bgr
        self.dxf_data = dxf_data
        self.debug = debug
        self.mode = mode
        self.objective = objective
        self.max_error_px = max_error_px
        self.heatmap_min = heatmap_min
        self.heatmap_max = heatmap_max
        self.color_low = color_low
        self.color_mid = color_mid
        self.color_high = color_high
        self.signals = _FitSignals()

    def run(self) -> None:
        try:
            if self.debug:
                edge_result, stages = compute_edges(self.frame_bgr, capture_stages=True)
            else:
                edge_result = compute_edges(self.frame_bgr)
                stages = None

            if self.mode == "POC":
                result = fit_poc(
                    polylines_all=self.dxf_data.polylines,
                    polylines_rot=self.dxf_data.polylines_rot,
                    polylines_pan=self.dxf_data.polylines_pan,
                    edge_points=edge_result.edge_points,
                    silhouette_mask=edge_result.mask,
                    distance_field=edge_result.distance_field,
                    objective=self.objective,
                    max_error_px=self.max_error_px,
                )
            elif self.mode == "Refine":
                result = fit_complete(
                    polylines_all=self.dxf_data.polylines,
                    polylines_refine=self.dxf_data.polylines_refine,
                    edge_points=edge_result.edge_points,
                    silhouette_mask=edge_result.mask,
                    distance_field=edge_result.distance_field,
                    polylines_rot=self.dxf_data.polylines_rot,
                    polylines_pan=self.dxf_data.polylines_pan,
                    objective=self.objective,
                    max_error_px=self.max_error_px,
                )
            else:
                result = fit(
                    polylines=self.dxf_data.polylines,
                    edge_points=edge_result.edge_points,
                    silhouette_mask=edge_result.mask,
                    distance_field=edge_result.distance_field,
                    objective=self.objective,
                    max_error_px=self.max_error_px,
                )

            # Compute RGBA heatmap here on the worker thread (pure numpy, no Qt)
            # so the main thread only does lightweight Qt brush/scene operations.
            rgba_bytes, shape = DxfOverlay.compute_heatmap_rgba(
                result,
                heatmap_min=self.heatmap_min,
                heatmap_max=self.heatmap_max,
                color_low=self.color_low,
                color_mid=self.color_mid,
                color_high=self.color_high,
            )

            if self.debug:
                self.signals.finished.emit((result, rgba_bytes, shape, stages, edge_result, self.frame_bgr))
            else:
                self.signals.finished.emit((result, rgba_bytes, shape))
        except Exception:
            import traceback
            self.signals.error.emit(traceback.format_exc())


class _ReanalyzeWorker(QRunnable):
    """Re-runs edge detection on a new frame and recomputes the heatmap
    using the *existing* fit transform (no re-fitting)."""

    def __init__(self, frame_bgr: np.ndarray, dxf_data: Dxf, prev_result: FitResult,
                 debug: bool = False, max_error_px: float = 2.0,
                 heatmap_min: float = 1.0, heatmap_max: float = 3.0,
                 color_low: str = "#00FF00", color_mid: str = "#FF8000",
                 color_high: str = "#FF0000") -> None:
        super().__init__()
        self.frame_bgr = frame_bgr
        self.dxf_data = dxf_data
        self.prev_result = prev_result
        self.debug = debug
        self.max_error_px = max_error_px
        self.heatmap_min = heatmap_min
        self.heatmap_max = heatmap_max
        self.color_low = color_low
        self.color_mid = color_mid
        self.color_high = color_high
        self.signals = _FitSignals()

    def run(self) -> None:
        try:
            if self.debug:
                edge_result, stages = compute_edges(self.frame_bgr, capture_stages=True)
            else:
                edge_result = compute_edges(self.frame_bgr)
                stages = None

            r = self.prev_result
            new_result = FitResult(
                tx=r.tx, ty=r.ty, angle_deg=r.angle_deg,
                cost=r.cost, dxf_cx=r.dxf_cx, dxf_cy=r.dxf_cy,
                inlier_frac=r.inlier_frac,
                dist_t=edge_result.distance_field,
                dist_raw=edge_result.distance_field,
            )

            # Compute RGBA on the worker thread (pure numpy)
            rgba_bytes, shape = DxfOverlay.compute_heatmap_rgba(
                new_result,
                heatmap_min=self.heatmap_min,
                heatmap_max=self.heatmap_max,
                color_low=self.color_low,
                color_mid=self.color_mid,
                color_high=self.color_high,
            )

            if self.debug:
                self.signals.finished.emit((new_result, rgba_bytes, shape, stages, edge_result, self.frame_bgr))
            else:
                self.signals.finished.emit((new_result, rgba_bytes, shape))
        except Exception:
            import traceback
            self.signals.error.emit(traceback.format_exc())


class ComparePresenter(QObject):
    """
    Mediates between the Compare-mode toolbar buttons, the fitting
    back-end, and the overlay view layer.
    """

    def __init__(
        self,
        settings: AppSettings,
        viewer: ImageViewer,
        toolbar: Toolbar,
        settings_panel: SettingsPanel,
        debug: bool = False,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self._viewer = viewer
        self._toolbar = toolbar
        self._settings_panel = settings_panel
        self._debug_enabled = debug

        self._dxf_data: Dxf | None = None
        self._last_result: FitResult | None = None
        self._overlay = DxfOverlay(self._viewer._scene)
        self._pool = QThreadPool.globalInstance()
        self._debug_window = DebugPreprocessingWindow() if debug else None
        self._active_signals = None  # prevent GC of worker signals before queued delivery

        # Wire toolbar buttons
        self._toolbar.btn_load.clicked.connect(self._on_load)
        self._toolbar.btn_fit.clicked.connect(self._on_run)
        self._toolbar.btn_reanalyze.clicked.connect(self._on_reanalyze)
        if debug:
            self._settings_panel.chk_debug.toggled.connect(self._on_debug_toggled)

    # ── Slots ────────────────────────────────────────────────────────

    @Slot()
    def _on_load(self) -> None:
        """Open a file dialog to select a DXF file."""
        path, _ = QFileDialog.getOpenFileName(
            self._viewer,
            "Open DXF File",
            "",
            "DXF Files (*.dxf *.DXF);;All Files (*)",
        )
        if not path:
            return

        calibration = self._active_calibration()
        if calibration <= 0:
            print("Invalid calibration value — cannot load DXF.")
            return

        # Determine canvas size from the current frame
        frame = self._viewer._current_cv_img
        if frame is not None:
            canvas_shape = frame.shape[:2]  # (H, W)
        else:
            canvas_shape = (3648, 5472)  # sensible default for 20MP

        try:
            self._dxf_data = load_dxf(path, calibration, canvas_shape)
            n_poly = len(self._dxf_data.polylines)
            n_pts = sum(len(p) for p in self._dxf_data.polylines)
            print(f"DXF loaded: {n_poly} polylines, {n_pts} total points, "
                  f"calibration={calibration:.2f} px/mm")

            # Show a preview overlay (un-aligned, raw pixel coords)
            self._overlay.draw_preview(self._dxf_data)

        except Exception as exc:
            print(f"Failed to load DXF: {exc}")

    @Slot()
    def _on_run(self) -> None:
        """Snap the current frame, run the alignment pipeline, and overlay."""
        if self._dxf_data is None:
            print("No DXF loaded — please load a file first.")
            return

        frame = self._viewer._current_cv_img
        if frame is None:
            print("No image available for comparison.")
            return

        # The camera delivers RGB frames; the POC edge processor expects BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Disable button while running
        self._toolbar.btn_fit.setEnabled(False)
        self._toolbar.btn_fit.setText(" Running…")

        debug = (self._debug_enabled
                 and self._settings_panel.chk_debug.isChecked())
        mode = self._settings_panel.combo_comparison.currentText()
        objective = self._settings_panel.combo_fit_objective.currentText()
        px_per_mm = self._active_calibration()
        max_error_px = self._active_heatmap_max() * px_per_mm if px_per_mm > 0 else 2.0
        heatmap_min = self._active_heatmap_min() * px_per_mm
        heatmap_max = self._active_heatmap_max() * px_per_mm
        worker = _FitWorker(
            frame_bgr.copy(), self._dxf_data, debug=debug, mode=mode,
            objective=objective, max_error_px=max_error_px,
            heatmap_min=heatmap_min, heatmap_max=heatmap_max,
            color_low=self._settings.app_defaults.heatmap_color_low,
            color_mid=self._settings.app_defaults.heatmap_color_mid,
            color_high=self._settings.app_defaults.heatmap_color_high,
        )
        worker.signals.finished.connect(self._on_fit_done, Qt.QueuedConnection)
        worker.signals.error.connect(self._on_fit_error, Qt.QueuedConnection)
        self._active_signals = worker.signals  # prevent GC before queued delivery
        worker.setAutoDelete(True)
        self._pool.start(worker)

    # ── Callbacks ────────────────────────────────────────────────────

    @Slot(object)
    def _on_fit_done(self, payload) -> None:
        # payload: (result, rgba_bytes, shape) or (result, rgba_bytes, shape, stages, edge_result, frame_bgr)
        if len(payload) == 6:
            result, rgba_bytes, shape, stages, edge_result, frame_bgr = payload
        else:
            result, rgba_bytes, shape = payload
            stages, edge_result, frame_bgr = None, None, None

        fit_info = (f"tx={result.tx:+.2f}  ty={result.ty:+.2f}  "
                    f"angle={result.angle_deg:.2f}°  cost={result.cost:.2f}  "
                    f"inlier={result.inlier_frac * 100:.2f}%")
        print(f"Fit complete — {fit_info}")
        self._last_result = result

        # Qt-only ops (no numpy) — fast on main thread
        self._overlay.draw_heatmap_from_rgba(self._dxf_data, result, rgba_bytes, shape)

        if stages is not None and self._debug_window is not None:
            self._debug_window.update_stages(
                stages, fit_info=fit_info, fit_result=result,
                edge_result=edge_result, dxf_data=self._dxf_data,
                px_per_mm=self._dxf_data.px_per_mm, frame_bgr=frame_bgr,
            )

        self._toolbar.btn_fit.setEnabled(True)
        self._toolbar.btn_fit.setText(" Fit")

    @Slot(str)
    def _on_fit_error(self, msg: str) -> None:
        print(f"Fit failed:\n{msg}")
        self._toolbar.btn_fit.setEnabled(True)
        self._toolbar.btn_fit.setText(" Fit")

    @Slot()
    def _on_reanalyze(self) -> None:
        """Re-run edge detection on the current frame and recompute the
        heatmap using the *previous* fit transform (no re-fitting).
        Useful when lighting conditions have changed between DIA / EPI."""
        if self._dxf_data is None:
            print("No DXF loaded — please load a file first.")
            return
        if self._last_result is None:
            print("No previous fit result — run a fit first.")
            return

        frame = self._viewer._current_cv_img
        if frame is None:
            print("No image available for re-analysis.")
            return

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        self._toolbar.btn_reanalyze.setEnabled(False)
        self._toolbar.btn_reanalyze.setText(" Reanalyzing…")

        debug = (self._debug_enabled
                 and self._settings_panel.chk_debug.isChecked())
        px_per_mm = self._active_calibration()
        max_error_px = self._active_heatmap_max() * px_per_mm if px_per_mm > 0 else 2.0
        heatmap_min = self._active_heatmap_min() * px_per_mm
        heatmap_max = self._active_heatmap_max() * px_per_mm

        worker = _ReanalyzeWorker(
            frame_bgr.copy(), self._dxf_data, self._last_result,
            debug=debug, max_error_px=max_error_px,
            heatmap_min=heatmap_min, heatmap_max=heatmap_max,
            color_low=self._settings.app_defaults.heatmap_color_low,
            color_mid=self._settings.app_defaults.heatmap_color_mid,
            color_high=self._settings.app_defaults.heatmap_color_high,
        )
        worker.signals.finished.connect(self._on_reanalyze_done, Qt.QueuedConnection)
        worker.signals.error.connect(self._on_reanalyze_error, Qt.QueuedConnection)
        self._active_signals = worker.signals  # prevent GC before queued delivery
        worker.setAutoDelete(True)
        self._pool.start(worker)

    @Slot(object)
    def _on_reanalyze_done(self, payload) -> None:
        if len(payload) == 6:
            result, rgba_bytes, shape, stages, edge_result, frame_bgr = payload
        else:
            result, rgba_bytes, shape = payload
            stages, edge_result, frame_bgr = None, None, None

        print("Re-analysis complete — heatmap updated with new edge data, same alignment.")
        self._last_result = result

        # Qt-only — fast
        self._overlay.draw_heatmap_from_rgba(self._dxf_data, result, rgba_bytes, shape)

        if stages is not None and self._debug_window is not None:
            self._debug_window.update_stages(
                stages, fit_info="Re-analysis (same transform)",
                fit_result=result, edge_result=edge_result,
                dxf_data=self._dxf_data, px_per_mm=self._dxf_data.px_per_mm,
                frame_bgr=frame_bgr,
            )

        self._toolbar.btn_reanalyze.setEnabled(True)
        self._toolbar.btn_reanalyze.setText(" Reanalyze")

    @Slot(str)
    def _on_reanalyze_error(self, msg: str) -> None:
        print(f"Re-analysis failed:\n{msg}")
        self._toolbar.btn_reanalyze.setEnabled(True)
        self._toolbar.btn_reanalyze.setText(" Reanalyse")

    @Slot(bool)
    def _on_debug_toggled(self, checked: bool) -> None:
        """Show/hide the debug window when the Settings checkbox is toggled."""
        if self._debug_window is None:
            return
        if checked:
            self._debug_window.show()
            self._debug_window.raise_()
        else:
            self._debug_window.hide()

    # ── Helpers ──────────────────────────────────────────────────────

    def _active_calibration(self) -> float:
        """Return the calibration from appsettings for the currently selected camera."""
        try:
            idx = self._settings_panel.combo_camera.currentIndex()
            return float(self._settings.cameras[idx].calibration_px_mm)
        except (IndexError, ValueError):
            pass
        try:
            return float(self._settings.cameras[0].calibration_px_mm)
        except (IndexError, ValueError):
            return 0.0

    def _active_heatmap_min(self) -> float:
        try:
            return float(self._settings_panel.input_heatmap_min.text())
        except (ValueError, AttributeError):
            return 1.0

    def _active_heatmap_max(self) -> float:
        try:
            return float(self._settings_panel.input_heatmap_max.text())
        except (ValueError, AttributeError):
            return 3.0

    def clear_overlay(self) -> None:
        """Remove any existing overlay (e.g. when switching modes)."""
        self._overlay.clear()
