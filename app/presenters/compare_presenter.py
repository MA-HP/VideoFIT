"""
Metrology Vision Pro — Compare Presenter
Orchestrates the DXF-to-piece alignment pipeline:
  Load DXF → Edge detection → Distance transform → Fit → Overlay

Matches the proven POC workflow exactly.
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot
from PySide6.QtWidgets import QFileDialog

from app.models.dxf_fitter import FitResult, fit
from app.models.dxf_model import DxfData, load_dxf
from app.models.edge_processor import compute_edges
from app.models.settings import AppSettings
from app.views.dxf_overlay import DxfOverlay
from app.views.image_viewer import ImageViewer
from app.views.settings_panel import SettingsPanel
from app.views.toolbar import Toolbar


class _FitSignals(QObject):
    """Signals emitted by the background fitting worker."""
    finished = Signal(object)   # FitResult
    error = Signal(str)


class _FitWorker(QRunnable):
    """Runs the heavy edge-detection + Nelder-Mead pipeline off the UI thread."""

    def __init__(self, frame_bgr: np.ndarray, dxf_data: DxfData) -> None:
        super().__init__()
        self.frame_bgr = frame_bgr
        self.dxf_data = dxf_data
        self.signals = _FitSignals()

    def run(self) -> None:
        try:
            # Edge detection (expects BGR)
            edge_result = compute_edges(self.frame_bgr)

            # Fit (polylines are already in pixel space)
            result = fit(
                polylines=self.dxf_data.polylines,
                distance_field=edge_result.distance_field,
                silhouette_mask=edge_result.mask,
            )
            self.signals.finished.emit(result)
        except Exception as exc:
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
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self._viewer = viewer
        self._toolbar = toolbar
        self._settings_panel = settings_panel

        self._dxf_data: DxfData | None = None
        self._overlay = DxfOverlay(self._viewer._scene)
        self._pool = QThreadPool.globalInstance()

        # Wire toolbar buttons
        self._toolbar.btn_load.clicked.connect(self._on_load)
        self._toolbar.btn_run_compare.clicked.connect(self._on_run)

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
            self._overlay.draw_preview(self._dxf_data.polylines)

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
        self._toolbar.btn_run_compare.setEnabled(False)
        self._toolbar.btn_run_compare.setText(" Running…")

        worker = _FitWorker(frame_bgr.copy(), self._dxf_data)
        worker.signals.finished.connect(self._on_fit_done)
        worker.signals.error.connect(self._on_fit_error)
        worker.setAutoDelete(True)
        self._pool.start(worker)

    # ── Callbacks ────────────────────────────────────────────────────

    @Slot(object)
    def _on_fit_done(self, result: FitResult) -> None:
        print(f"Fit complete — tx={result.tx:+.1f}  ty={result.ty:+.1f}  "
              f"angle={result.angle_deg:.2f}°  cost={result.cost:.2f}  "
              f"inlier={result.inlier_frac * 100:.1f}%")

        self._overlay.draw_heatmap(
            self._dxf_data.polylines,
            result,
        )

        self._toolbar.btn_run_compare.setEnabled(True)
        self._toolbar.btn_run_compare.setText(" Run")

    @Slot(str)
    def _on_fit_error(self, msg: str) -> None:
        print(f"Fit failed:\n{msg}")
        self._toolbar.btn_run_compare.setEnabled(True)
        self._toolbar.btn_run_compare.setText(" Run")

    # ── Helpers ──────────────────────────────────────────────────────

    def _active_calibration(self) -> float:
        """Return the calibration value for the currently selected camera."""
        try:
            text = self._settings_panel.input_calibration.text()
            return float(text)
        except (ValueError, AttributeError):
            pass
        # Fallback: first camera in settings
        try:
            return float(self._settings.cameras[0].calibration_px_mm)
        except (IndexError, ValueError):
            return 0.0

    def clear_overlay(self) -> None:
        """Remove any existing overlay (e.g. when switching modes)."""
        self._overlay.clear()

