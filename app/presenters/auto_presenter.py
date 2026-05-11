"""
VideoFIT — Auto Presenter
Automated pipeline execution with movement-based triggering.

State machine:
  IDLE → CLEAN_PLATE → WAITING_MOVEMENT → WAITING_STABILITY → RUNNING_PIPELINE → WAITING_MOVEMENT …
"""

from __future__ import annotations

import json
import os
from enum import Enum, auto

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal, Slot, Qt

from app.models.dxf import Dxf
from app.models.fit_result import FitResult
from app.models.settings import AppSettings
from app.services.dxf_service import load_dxf
from app.services.edge_service import compute_edges
from app.services.fit_service import fit, fit_complete, fit_poc
from app.services.lighting_service import LightingService, CHANNEL_NAMES
from app.views.dxf_overlay import DxfOverlay
from app.views.image_viewer import ImageViewer
from app.views.settings_panel import SettingsPanel
from app.views.toolbar import Toolbar

# Map channel name → channel number  (from CHANNEL_NAMES = {1:"EPI", 2:"-", 3:"COAX", 4:"DIA"})
_NAME_TO_CHANNEL: dict[str, int] = {v: k for k, v in CHANNEL_NAMES.items()}


class _Phase(Enum):
    IDLE = auto()
    CLEAN_PLATE = auto()       # capture reference once at startup
    WAITING_MOVEMENT = auto()  # FOV ≈ clean plate — waiting for a piece to enter
    WAITING_STABILITY = auto() # piece detected — waiting for it to stop moving
    RUNNING_PIPELINE = auto()  # analysis running
    WAITING_CHANGE = auto()    # pipeline done — waiting for anything to change (new piece / repositioned / removed+replaced)


class _StepSignals(QObject):
    """Lives in the main thread; signals are always dispatched via the Qt event loop."""
    done = Signal(object)   # FitResult
    error = Signal(str)


class _CompareStepWorker(QRunnable):
    """Off-thread worker for a single Fit or Reanalyze pipeline step."""

    def __init__(self, task: str, frame_bgr: np.ndarray, dxf_data: Dxf,
                 prev_result: FitResult | None,
                 mode: str = "Best Fit", objective: str = "Strict",
                 max_error_px: float = 2.0) -> None:
        super().__init__()
        self.task = task.lower()
        self.frame_bgr = frame_bgr
        self.dxf_data = dxf_data
        self.prev_result = prev_result
        self.mode = mode
        self.objective = objective
        self.max_error_px = max_error_px
        # _StepSignals is created on the main thread (here, in __init__),
        # so Qt will use QueuedConnection when the worker emits from the pool thread.
        self.signals = _StepSignals()

    def run(self) -> None:
        try:
            edge_result = compute_edges(self.frame_bgr)

            if self.task in ("reanalyze", "reanalyse", "reanalize"):
                if self.prev_result is None:
                    self.signals.error.emit("Reanalyze: no previous fit result.")
                    return
                r = self.prev_result
                result = FitResult(
                    tx=r.tx, ty=r.ty, angle_deg=r.angle_deg,
                    cost=r.cost, dxf_cx=r.dxf_cx, dxf_cy=r.dxf_cy,
                    inlier_frac=r.inlier_frac,
                    dist_t=edge_result.distance_field,
                    dist_raw=edge_result.distance_field,
                )
            else:
                # Fit
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
            self.signals.done.emit(result)
        except Exception:
            import traceback
            self.signals.error.emit(traceback.format_exc())


class AutoPresenter(QObject):
    """
    Drives the automated inspection loop.
    The toolbar in Auto mode shows only Start/Stop + a status label.
    """

    status_changed = Signal(str)

    # Internal signals used to safely marshal worker callbacks to the main thread
    _step_done = Signal(object, str, int)   # (FitResult, task_name, cycle_id)
    _step_error = Signal(str, int)          # (error_msg, cycle_id)

    def __init__(
        self,
        settings: AppSettings,
        viewer: ImageViewer,
        toolbar: Toolbar,
        settings_panel: SettingsPanel,
        overlay: DxfOverlay,
        lighting_service: LightingService,
        app_dir: str,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self._viewer = viewer
        self._toolbar = toolbar
        self._settings_panel = settings_panel
        self._overlay = overlay
        self._lighting = lighting_service
        self._app_dir = app_dir
        self._pool = QThreadPool.globalInstance()

        # State machine
        self._phase = _Phase.IDLE
        self._clean_plate: np.ndarray | None = None
        self._prev_gray: np.ndarray | None = None
        self._last_analysed_gray: np.ndarray | None = None
        self._stable_count = 0
        self._cycle_id = 0

        # Pipeline
        self._pipeline_def: list[dict] = []
        self._pipeline_index = 0
        self._dxf_data: Dxf | None = None
        self._last_result: FitResult | None = None

        # Thresholds (overridden by JSON)
        self._movement_threshold = 0.5    # % coherent pixels changed vs clean plate
        self._stability_threshold = 0.1   # % coherent pixels changed frame-to-frame
        self._stabilization_frames = 5    # consecutive stable frames needed
        self._noise_px_threshold = 25     # gray delta to ignore as sensor noise
        self._morph_kernel_size = 15      # morphological open kernel: filters blobs smaller than this

        # Internal signals always delivered on the main thread via QueuedConnection
        self._step_done.connect(self._on_step_done, Qt.QueuedConnection)
        self._step_error.connect(self._on_step_error, Qt.QueuedConnection)

        # Poll camera at ~15 Hz
        self._timer = QTimer(self)
        self._timer.setInterval(66)
        self._timer.timeout.connect(self._tick)

        # Wire the single Start/Stop button
        self._toolbar.btn_auto_start.clicked.connect(self._on_start_stop)

    # ── Pipeline loading ─────────────────────────────────────────────

    def load_pipeline(self, path: str) -> bool:
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            self._pipeline_def = raw.get("pipeline", [])
            self._movement_threshold = raw.get("movement_threshold", 0.5)
            self._stability_threshold = raw.get("stability_threshold", 0.1)
            self._stabilization_frames = int(raw.get("stabilization_frames", 5))
            self._noise_px_threshold = int(raw.get("noise_px_threshold", 25))
            self._morph_kernel_size = int(raw.get("morph_kernel_size", 15))

            # Pre-load the first DXF referenced in any compare step
            for step in self._pipeline_def:
                dxf_path = step.get("params", {}).get("dxf_file")
                if dxf_path:
                    full = os.path.join(self._app_dir, dxf_path)
                    if os.path.isfile(full):
                        cal = self._active_calibration()
                        frame = self._viewer._current_cv_img
                        canvas = frame.shape[:2] if frame is not None else (3648, 5472)
                        self._dxf_data = load_dxf(full, cal, canvas)
                        print(f"[Auto] DXF loaded: {full}")
                    break

            print(f"[Auto] Pipeline loaded ({len(self._pipeline_def)} steps)")
            return True
        except Exception as exc:
            print(f"[Auto] Pipeline load error: {exc}")
            return False

    # ── Start / Stop ─────────────────────────────────────────────────

    @Slot()
    def _on_start_stop(self) -> None:
        if self._phase == _Phase.IDLE:
            self._start()
        else:
            self._stop()

    def _start(self) -> None:
        if not self._pipeline_def:
            path = os.path.join(self._app_dir, "pipeline.json")
            if not self.load_pipeline(path):
                self._update_status("No pipeline.json found — cannot start.")
                return

        self._phase = _Phase.CLEAN_PLATE   # capture clean plate once, then never again
        self._clean_plate = None
        self._prev_gray = None
        self._last_analysed_gray = None
        self._stable_count = 0
        self._last_result = None
        self._cycle_id = 0

        self._toolbar.btn_auto_start.setText(" Stop")
        self._update_status("Capturing clean plate…")
        self._timer.start()

    def _stop(self) -> None:
        self._timer.stop()
        self._phase = _Phase.IDLE
        self._cycle_id = -1  # invalidate any in-flight worker callbacks
        self._toolbar.btn_auto_start.setText(" Start")
        self._update_status("Idle")

    # ── Timer tick ───────────────────────────────────────────────────

    @Slot()
    def _tick(self) -> None:
        frame = self._viewer._current_cv_img
        if frame is None:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        if self._phase == _Phase.CLEAN_PLATE:
            self._handle_clean_plate(gray)
        elif self._phase == _Phase.WAITING_MOVEMENT:
            self._handle_movement(gray)
        elif self._phase == _Phase.WAITING_STABILITY:
            self._handle_stability(gray)
        elif self._phase == _Phase.WAITING_CHANGE:
            self._handle_change(gray)
        # RUNNING_PIPELINE: driven by callbacks, tick is a no-op

    # ── Phase handlers ────────────────────────────────────────────────

    def _handle_clean_plate(self, gray: np.ndarray) -> None:
        if self._prev_gray is None:
            # First frame — just store and wait for next tick
            self._prev_gray = gray.copy()
            return
        # Wait until consecutive frames are identical (sensor noise settled)
        if self._frame_diff_pct(self._prev_gray, gray) < self._stability_threshold:
            self._clean_plate = gray.copy()
            # Transition — set phase FIRST to avoid re-entering this handler
            self._phase = _Phase.WAITING_MOVEMENT
            self._update_status("Clean plate ready — waiting for piece…")
            print("[Auto] Clean plate captured.")
        else:
            self._prev_gray = gray.copy()

    def _handle_movement(self, gray: np.ndarray) -> None:
        diff = self._frame_diff_pct(self._clean_plate, gray)
        if diff > self._movement_threshold:
            self._phase = _Phase.WAITING_STABILITY
            self._prev_gray = gray.copy()
            self._stable_count = 0
            self._update_status(f"Piece detected ({diff:.1f}%) — waiting for stability…")
            print(f"[Auto] Movement: {diff:.2f}%")

    def _handle_stability(self, gray: np.ndarray) -> None:
        # Piece left FOV before we could stabilise — go back to waiting
        if self._frame_diff_pct(self._clean_plate, gray) < self._movement_threshold:
            self._phase = _Phase.WAITING_MOVEMENT
            self._stable_count = 0
            self._update_status("Piece removed — waiting for piece…")
            return

        diff = self._frame_diff_pct(self._prev_gray, gray)
        self._prev_gray = gray.copy()

        if diff < self._stability_threshold:
            self._stable_count += 1
            self._update_status(
                f"Stabilizing… {self._stable_count}/{self._stabilization_frames}")
            if self._stable_count >= self._stabilization_frames:
                self._update_status("Stable — running pipeline…")
                print(f"[Auto] Stable after {self._stable_count} frames.")
                self._run_pipeline()
        else:
            self._stable_count = 0

    def _handle_change(self, gray: np.ndarray) -> None:
        """After a pipeline completes, wait for the scene to change meaningfully
        compared to the frame we analysed — so a static piece never re-triggers."""
        diff = self._frame_diff_pct(self._last_analysed_gray, gray)
        if diff > self._movement_threshold:
            # Something changed — but first check it is not just the clean plate
            diff_vs_clean = self._frame_diff_pct(self._clean_plate, gray)
            if diff_vs_clean < self._movement_threshold:
                # Piece was removed and nothing new yet — go to WAITING_MOVEMENT
                self._phase = _Phase.WAITING_MOVEMENT
                self._update_status("Piece removed — waiting for piece…")
                print("[Auto] Scene returned to clean plate — waiting for next piece.")
            else:
                # New content detected — go straight to stability check
                self._phase = _Phase.WAITING_STABILITY
                self._prev_gray = gray.copy()
                self._stable_count = 0
                self._update_status(f"Change detected ({diff:.1f}%) — waiting for stability…")
                print(f"[Auto] Scene changed: {diff:.2f}% — checking stability.")

    # ── Pipeline execution ────────────────────────────────────────────

    def _run_pipeline(self) -> None:
        self._phase = _Phase.RUNNING_PIPELINE
        self._pipeline_index = 0
        self._cycle_id += 1
        self._execute_next_step()

    def _execute_next_step(self) -> None:
        if self._pipeline_index >= len(self._pipeline_def):
            self._on_pipeline_complete()
            return

        step = self._pipeline_def[self._pipeline_index]
        action = step.get("action", "").lower()
        params = step.get("params", {})
        n = self._pipeline_index + 1
        total = len(self._pipeline_def)
        self._update_status(f"Step {n}/{total}: {action}…")

        if action == "lighting":
            self._exec_lighting(params)
        elif action == "compare":
            self._exec_compare(params)
        else:
            print(f"[Auto] Unknown action '{action}' — skipping.")
            self._pipeline_index += 1
            self._execute_next_step()

    def _exec_lighting(self, params: dict) -> None:
        """Send lighting commands then wait 300 ms for hardware + sensor to settle."""
        for name, intensity in params.items():
            ch = _NAME_TO_CHANNEL.get(name.upper())
            if ch is not None:
                self._lighting.set_intensity(ch, float(intensity))
                print(f"[Auto] Light {name}={intensity} → ch{ch}")
            else:
                print(f"[Auto] Unknown lighting channel name: '{name}'")
        self._pipeline_index += 1
        QTimer.singleShot(300, self._execute_next_step)

    def _exec_compare(self, params: dict) -> None:
        if self._dxf_data is None:
            print("[Auto] No DXF data — skipping compare step.")
            self._pipeline_index += 1
            self._execute_next_step()
            return

        frame = self._viewer._current_cv_img
        if frame is None:
            print("[Auto] No frame — skipping compare step.")
            self._pipeline_index += 1
            self._execute_next_step()
            return

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        task = params.get("task", "Fit")
        mode = params.get("mode", "Best Fit")
        objective = params.get("objective", "Strict")
        px_per_mm = self._active_calibration()
        max_error_px = self._active_heatmap_max() * px_per_mm if px_per_mm > 0 else 2.0

        cycle = self._cycle_id

        worker = _CompareStepWorker(
            task=task, frame_bgr=frame_bgr.copy(),
            dxf_data=self._dxf_data, prev_result=self._last_result,
            mode=mode, objective=objective, max_error_px=max_error_px,
        )

        # Two-stage connection:
        # 1. DirectConnection  → lambda runs on the pool thread, INSIDE run(),
        #    while worker.signals is still alive (before setAutoDelete destroys it).
        # 2. The lambda emits self._step_done which is connected QueuedConnection
        #    → callback is posted to the main-thread event loop, safe for Qt Graphics.
        worker.signals.done.connect(
            lambda result, t=task, c=cycle: self._step_done.emit(result, t, c),
            Qt.DirectConnection,
        )
        worker.signals.error.connect(
            lambda msg, c=cycle: self._step_error.emit(msg, c),
            Qt.DirectConnection,
        )
        worker.setAutoDelete(True)
        self._pool.start(worker)

    # ── Step callbacks (always on main thread via QueuedConnection) ──

    @Slot(object, str, int)
    def _on_step_done(self, result: FitResult, task: str, cycle: int) -> None:
        if cycle != self._cycle_id:
            return  # stale callback from a superseded cycle

        self._last_result = result
        print(f"[Auto] {task} done — tx={result.tx:+.1f} ty={result.ty:+.1f} "
              f"angle={result.angle_deg:.2f}° inlier={result.inlier_frac*100:.1f}%")

        # Safe: we are on the main thread here
        px_per_mm = self._dxf_data.px_per_mm
        self._overlay.draw_heatmap(
            self._dxf_data, result,
            heatmap_min=self._active_heatmap_min() * px_per_mm,
            heatmap_max=self._active_heatmap_max() * px_per_mm,
            color_low=self._settings.app_defaults.heatmap_color_low,
            color_mid=self._settings.app_defaults.heatmap_color_mid,
            color_high=self._settings.app_defaults.heatmap_color_high,
        )

        self._pipeline_index += 1
        self._execute_next_step()

    @Slot(str, int)
    def _on_step_error(self, msg: str, cycle: int) -> None:
        if cycle != self._cycle_id:
            return
        print(f"[Auto] Step error: {msg}")
        self._pipeline_index += 1
        self._execute_next_step()

    def _on_pipeline_complete(self) -> None:
        print(f"[Auto] Pipeline complete (cycle {self._cycle_id}).")
        # Snapshot the frame we just analysed — WAITING_CHANGE will compare
        # future frames against this, not against the clean plate.
        frame = self._viewer._current_cv_img
        if frame is not None:
            self._last_analysed_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        self._update_status("Done — waiting for scene change…")
        self._phase = _Phase.WAITING_CHANGE
        self._stable_count = 0

    # ── Helpers ───────────────────────────────────────────────────────

    def _frame_diff_pct(self, ref: np.ndarray | None, cur: np.ndarray) -> float:
        """
        Percentage of the frame covered by *coherent* changes — i.e. real
        objects, not sensor/lighting noise.

        Pipeline:
          1. Absolute difference + threshold at `_noise_px_threshold`
          2. Gaussian blur to merge nearby responses into blobs
          3. Morphological OPEN (erode then dilate) to kill isolated noise pixels
          4. Count surviving pixels → fraction of frame area

        Scattered noise produces only tiny isolated clusters that are wiped out
        by the opening.  A real piece creates large contiguous blobs that survive.
        """
        if ref is None:
            return 0.0
        if ref.shape != cur.shape:
            return 100.0

        diff = cv2.absdiff(ref, cur)

        # Threshold: ignore anything below the noise floor
        _, mask = cv2.threshold(diff, self._noise_px_threshold, 255, cv2.THRESH_BINARY)

        # Blur first to bridge tiny gaps between adjacent changed pixels
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)

        # Morphological opening: erodes isolated specks, then restores real blobs.
        # Kernel size controls the minimum "object" size we care about.
        # 15×15 at typical resolutions ≈ a few mm² → ignores dust/noise, catches pieces.
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self._morph_kernel_size, self._morph_kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        changed = np.count_nonzero(mask)
        return (changed / mask.size) * 100.0

    def _active_calibration(self) -> float:
        try:
            idx = self._settings_panel.combo_camera.currentIndex()
            return float(self._settings.cameras[idx].calibration_px_mm)
        except (IndexError, ValueError):
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

    def _update_status(self, text: str) -> None:
        self.status_changed.emit(text)
        self._toolbar.lbl_auto_status.setText(text)
