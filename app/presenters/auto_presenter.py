"""
VideoFIT — Auto Presenter

Pipeline execution model
────────────────────────
Three-level parallelism:

1. MOVEMENT WARMUP — as soon as a piece is detected, apply the first lighting
   step from the pipeline. Stability checking takes ≥500 ms; the 150 ms
   settle is fully hidden inside that wait. By the time the piece is stable,
   the sensor is already under the correct lighting.

2. INTER-STEP LOOKAHEAD — when a Compare (Fit/Reanalyze) is submitted to the
   thread pool, immediately scan ahead and pre-apply any upcoming Lighting
   steps + pre-capture + pre-run edge detection for the next Compare step.
   All in parallel with the running Fit.

3. FRAME PRE-CAPTURE — frame is always captured BEFORE any lighting change,
   so Fit always sees the correct lighting even though EPI is already being
   applied concurrently.

Result: total time ≈ max(Fit, 150 ms + Edge detection)
instead of Fit + 150 ms + Edge detection.
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

_NAME_TO_CHANNEL: dict[str, int] = {v: k for k, v in CHANNEL_NAMES.items()}
_DETECT_WIDTH = 640


class _Phase(Enum):
    IDLE = auto()
    CLEAN_PLATE = auto()
    WAITING_MOVEMENT = auto()
    WAITING_STABILITY = auto()
    RUNNING_PIPELINE = auto()
    WAITING_CHANGE = auto()


class _WorkerSignals(QObject):
    done = Signal(object)
    error = Signal(str)


class _EdgeOnlyWorker(QRunnable):
    def __init__(self, frame_bgr: np.ndarray) -> None:
        super().__init__()
        self.frame_bgr = frame_bgr
        self.signals = _WorkerSignals()

    def run(self) -> None:
        try:
            self.signals.done.emit(compute_edges(self.frame_bgr))
        except Exception:
            import traceback
            self.signals.error.emit(traceback.format_exc())


class _CompareStepWorker(QRunnable):
    """Fit or Reanalyze — optionally receives a pre-computed EdgeResult."""

    def __init__(self, task: str, frame_bgr: np.ndarray, dxf_data: Dxf,
                 prev_result: FitResult | None,
                 mode: str = "Best Fit", objective: str = "Strict",
                 max_error_px: float = 2.0,
                 heatmap_min: float = 1.0, heatmap_max: float = 3.0,
                 color_low: str = "#00FF00", color_mid: str = "#FF8000",
                 color_high: str = "#FF0000",
                 precomputed_edge=None) -> None:
        super().__init__()
        self.task = task.lower()
        self.frame_bgr = frame_bgr
        self.dxf_data = dxf_data
        self.prev_result = prev_result
        self.mode = mode
        self.objective = objective
        self.max_error_px = max_error_px
        self.heatmap_min = heatmap_min
        self.heatmap_max = heatmap_max
        self.color_low = color_low
        self.color_mid = color_mid
        self.color_high = color_high
        self.precomputed_edge = precomputed_edge
        self.signals = _WorkerSignals()

    def run(self) -> None:
        try:
            edge_result = self.precomputed_edge or compute_edges(self.frame_bgr)

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
                if self.mode == "POC":
                    result = fit_poc(
                        polylines_all=self.dxf_data.polylines,
                        polylines_rot=self.dxf_data.polylines_rot,
                        polylines_pan=self.dxf_data.polylines_pan,
                        edge_points=edge_result.edge_points,
                        silhouette_mask=edge_result.mask,
                        distance_field=edge_result.distance_field,
                        objective=self.objective, max_error_px=self.max_error_px,
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
                        objective=self.objective, max_error_px=self.max_error_px,
                    )
                else:
                    result = fit(
                        polylines=self.dxf_data.polylines,
                        edge_points=edge_result.edge_points,
                        silhouette_mask=edge_result.mask,
                        distance_field=edge_result.distance_field,
                        objective=self.objective, max_error_px=self.max_error_px,
                    )

            rgba_bytes, shape = DxfOverlay.compute_heatmap_rgba(
                result,
                heatmap_min=self.heatmap_min, heatmap_max=self.heatmap_max,
                color_low=self.color_low, color_mid=self.color_mid, color_high=self.color_high,
            )
            self.signals.done.emit((result, rgba_bytes, shape))
        except Exception:
            import traceback
            self.signals.error.emit(traceback.format_exc())


class AutoPresenter(QObject):

    status_changed = Signal(str)
    _step_done = Signal(object, str, int)
    _step_error = Signal(str, int)
    _lookahead_ready = Signal(object, int)

    def __init__(self, settings: AppSettings, viewer: ImageViewer, toolbar: Toolbar,
                 settings_panel: SettingsPanel, overlay: DxfOverlay,
                 lighting_service: LightingService, app_dir: str,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._viewer = viewer
        self._toolbar = toolbar
        self._settings_panel = settings_panel
        self._overlay = overlay
        self._lighting = lighting_service
        self._app_dir = app_dir
        self._pool = QThreadPool.globalInstance()

        self._phase = _Phase.IDLE
        self._clean_plate: np.ndarray | None = None
        self._prev_gray: np.ndarray | None = None
        self._last_analysed_gray: np.ndarray | None = None
        self._stable_count = 0
        self._cycle_id = 0

        self._pipeline_def: list[dict] = []
        self._pipeline_index = 0
        self._dxf_data: Dxf | None = None
        self._last_result: FitResult | None = None

        # Warmup: first pipeline lighting pre-applied during stability wait
        self._warmup_lighting_applied = False  # first step pre-applied at movement detection

        # Lookahead: lighting + edge pre-computed while Compare runs
        self._lookahead_edge = None
        self._lookahead_frame_bgr = None
        self._lookahead_lighting_applied = 0

        self._movement_threshold = 0.5
        self._stability_threshold = 0.1
        self._stabilization_frames = 5
        self._noise_px_threshold = 25
        self._morph_kernel_size = 15

        self._step_done.connect(self._on_step_done, Qt.QueuedConnection)
        self._step_error.connect(self._on_step_error, Qt.QueuedConnection)
        self._lookahead_ready.connect(self._on_lookahead_ready, Qt.QueuedConnection)

        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._tick)

        self._toolbar.btn_auto_start.clicked.connect(self._on_start_stop)

    # ── Pipeline loading ──────────────────────────────────────────────

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

            for step in self._pipeline_def:
                dxf_path = step.get("params", {}).get("dxf_file")
                if dxf_path:
                    full = dxf_path if os.path.isabs(dxf_path) \
                        else os.path.join(self._app_dir, dxf_path)
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

    # ── Start / Stop ──────────────────────────────────────────────────

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

        self._phase = _Phase.CLEAN_PLATE
        self._clean_plate = None
        self._prev_gray = None
        self._last_analysed_gray = None
        self._stable_count = 0
        self._last_result = None
        self._cycle_id = 0
        self._warmup_lighting_applied = False
        self._reset_lookahead()

        self._toolbar.btn_auto_start.setText(" Stop")
        self._update_status("Capturing clean plate…")
        self._timer.start()

    def _stop(self) -> None:
        self._timer.stop()
        self._phase = _Phase.IDLE
        self._cycle_id = -1
        self._warmup_lighting_applied = False
        self._reset_lookahead()
        self._toolbar.btn_auto_start.setText(" Start")
        self._update_status("Idle")

    def _reset_lookahead(self) -> None:
        self._lookahead_edge = None
        self._lookahead_frame_bgr = None
        self._lookahead_lighting_applied = 0

    # ── Timer tick ────────────────────────────────────────────────────

    @Slot()
    def _tick(self) -> None:
        frame = self._viewer._current_cv_img
        if frame is None:
            return
        h, w = frame.shape[:2]
        scale = _DETECT_WIDTH / w
        small = cv2.resize(frame, (_DETECT_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

        if self._phase == _Phase.CLEAN_PLATE:
            self._handle_clean_plate(gray)
        elif self._phase == _Phase.WAITING_MOVEMENT:
            self._handle_movement(gray)
        elif self._phase == _Phase.WAITING_STABILITY:
            self._handle_stability(gray)
        elif self._phase == _Phase.WAITING_CHANGE:
            self._handle_change(gray)

    # ── Phase handlers ────────────────────────────────────────────────

    def _handle_clean_plate(self, gray: np.ndarray) -> None:
        if self._prev_gray is None:
            self._prev_gray = gray.copy()
            return
        if self._frame_diff_pct(self._prev_gray, gray) < self._stability_threshold:
            self._clean_plate = gray.copy()
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
            self._warmup_lighting_applied = False
            self._update_status(f"Piece detected ({diff:.1f}%) — stabilizing…")
            print(f"[Auto] Movement detected: {diff:.2f}%")

            # ── WARMUP: pre-apply first lighting step NOW ──────────────
            # Stability checking takes ≥500 ms. The 150 ms lighting settle
            # is fully hidden inside that wait — no extra delay at pipeline start.
            first_step = self._pipeline_def[0] if self._pipeline_def else None
            if first_step and first_step.get("action", "").lower() == "lighting":
                self._apply_lighting(first_step.get("params", {}))
                self._warmup_lighting_applied = True
                print("[Auto] Warmup: first lighting pre-applied during stability wait.")

    def _handle_stability(self, gray: np.ndarray) -> None:
        if self._frame_diff_pct(self._clean_plate, gray) < self._movement_threshold:
            self._phase = _Phase.WAITING_MOVEMENT
            self._stable_count = 0
            self._warmup_lighting_applied = False
            self._update_status("Piece removed — waiting for piece…")
            return

        diff = self._frame_diff_pct(self._prev_gray, gray)
        self._prev_gray = gray.copy()

        if diff < self._stability_threshold:
            self._stable_count += 1
            self._update_status(f"Stabilizing… {self._stable_count}/{self._stabilization_frames}")
            if self._stable_count >= self._stabilization_frames:
                self._update_status("Stable — running pipeline…")
                print(f"[Auto] Stable after {self._stable_count} frames.")
                self._run_pipeline()
        else:
            self._stable_count = 0

    def _handle_change(self, gray: np.ndarray) -> None:
        diff = self._frame_diff_pct(self._last_analysed_gray, gray)
        if diff > self._movement_threshold:
            diff_vs_clean = self._frame_diff_pct(self._clean_plate, gray)
            if diff_vs_clean < self._movement_threshold:
                self._phase = _Phase.WAITING_MOVEMENT
                self._warmup_lighting_applied = False
                self._update_status("Piece removed — waiting for piece…")
                print("[Auto] Returned to clean plate.")
            else:
                self._phase = _Phase.WAITING_STABILITY
                self._prev_gray = gray.copy()
                self._stable_count = 0
                self._warmup_lighting_applied = False
                self._update_status(f"Change detected ({diff:.1f}%) — stabilizing…")
                print(f"[Auto] Scene changed: {diff:.2f}%")

                # Warmup for next cycle too
                first_step = self._pipeline_def[0] if self._pipeline_def else None
                if first_step and first_step.get("action", "").lower() == "lighting":
                    self._apply_lighting(first_step.get("params", {}))
                    self._warmup_lighting_applied = True
                    print("[Auto] Warmup: first lighting re-applied for next cycle.")

    # ── Pipeline execution ────────────────────────────────────────────

    def _run_pipeline(self) -> None:
        self._phase = _Phase.RUNNING_PIPELINE
        self._pipeline_index = 0
        self._cycle_id += 1
        self._reset_lookahead()
        self._execute_next_step()

    def _execute_next_step(self) -> None:
        if self._pipeline_index >= len(self._pipeline_def):
            self._on_pipeline_complete()
            return

        step = self._pipeline_def[self._pipeline_index]
        action = step.get("action", "").lower()
        params = step.get("params", {})
        n, total = self._pipeline_index + 1, len(self._pipeline_def)
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
        """Apply lighting + settle.

        Three cases:
        - Warmup pre-applied this step during stability wait → skip hardware +
          skip settle (settle already happened, >500 ms elapsed).
        - Lookahead pre-applied this step while a Compare was running → skip
          hardware + skip settle (settle happened during Fit).
        - Normal: send commands + wait 150 ms.
        """
        if self._warmup_lighting_applied and self._pipeline_index == 0:
            # First step was pre-applied at movement detection — already settled
            print(f"[Auto] Step {self._pipeline_index + 1} (lighting): already settled via warmup — skipping.")
            self._warmup_lighting_applied = False
            self._pipeline_index += 1
            self._execute_next_step()

        elif self._lookahead_lighting_applied > 0:
            # Pre-applied by lookahead during previous Compare
            print(f"[Auto] Step {self._pipeline_index + 1} (lighting): already pre-applied by lookahead — skipping settle.")
            self._lookahead_lighting_applied -= 1
            self._pipeline_index += 1
            self._execute_next_step()

        else:
            # Normal path
            self._apply_lighting(params)
            self._pipeline_index += 1
            QTimer.singleShot(150, self._execute_next_step)

    def _apply_lighting(self, params: dict) -> None:
        for name, intensity in params.items():
            ch = _NAME_TO_CHANNEL.get(name.upper())
            if ch is not None:
                self._lighting.set_intensity(ch, float(intensity))
                print(f"[Auto] Light {name}={intensity} → ch{ch}")

    def _exec_compare(self, params: dict) -> None:
        if self._dxf_data is None:
            print("[Auto] No DXF — skipping.")
            self._pipeline_index += 1
            self._execute_next_step()
            return

        # ── Step 1: capture frame (BEFORE any lighting change) ────────
        if self._lookahead_frame_bgr is not None:
            frame_bgr = self._lookahead_frame_bgr
            precomputed_edge = self._lookahead_edge
            self._lookahead_frame_bgr = None
            self._lookahead_edge = None
            print(f"[Auto] Using lookahead frame "
                  f"(edge {'ready ✓' if precomputed_edge else 'computing…'}).")
        else:
            frame = self._viewer._current_cv_img
            if frame is None:
                print("[Auto] No frame — skipping.")
                self._pipeline_index += 1
                self._execute_next_step()
                return
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).copy()
            precomputed_edge = None

        task = params.get("task", "Fit")
        mode = params.get("mode", "Best Fit")
        objective = params.get("objective", "Strict")
        px_per_mm = self._active_calibration()
        hm_max = self._active_heatmap_max() * px_per_mm if px_per_mm > 0 else 2.0
        hm_min = self._active_heatmap_min() * px_per_mm
        cycle = self._cycle_id

        # ── Step 2: scan ahead, apply next lighting NOW, start edge capture ──
        # This happens BEFORE submitting the Fit worker, ensuring the lighting
        # command goes out at the earliest possible moment.
        self._schedule_lookahead(after_index=self._pipeline_index + 1, cycle=cycle)

        # ── Step 3: submit Fit/Reanalyze worker ───────────────────────
        print(f"[Auto] Starting {task} (frame captured, lighting pre-changed for next step).")
        worker = _CompareStepWorker(
            task=task, frame_bgr=frame_bgr, dxf_data=self._dxf_data,
            prev_result=self._last_result, mode=mode, objective=objective,
            max_error_px=hm_max, heatmap_min=hm_min, heatmap_max=hm_max,
            color_low=self._settings.app_defaults.heatmap_color_low,
            color_mid=self._settings.app_defaults.heatmap_color_mid,
            color_high=self._settings.app_defaults.heatmap_color_high,
            precomputed_edge=precomputed_edge,
        )
        worker.signals.done.connect(
            lambda p, t=task, c=cycle: self._step_done.emit(p, t, c),
            Qt.DirectConnection)
        worker.signals.error.connect(
            lambda m, c=cycle: self._step_error.emit(m, c),
            Qt.DirectConnection)
        worker.setAutoDelete(True)
        self._pool.start(worker)

    def _schedule_lookahead(self, after_index: int, cycle: int) -> None:
        """Apply upcoming lighting steps immediately, then pre-capture the
        next compare frame after settle — all before the current Fit finishes."""
        idx = after_index
        lighting_steps = []

        while idx < len(self._pipeline_def):
            action = self._pipeline_def[idx].get("action", "").lower()
            if action == "lighting":
                lighting_steps.append(self._pipeline_def[idx].get("params", {}))
                idx += 1
            else:
                break

        has_next_compare = (idx < len(self._pipeline_def) and
                            self._pipeline_def[idx].get("action", "").lower() == "compare")

        if not lighting_steps and not has_next_compare:
            return

        # Apply all upcoming lighting steps RIGHT NOW
        for lp in lighting_steps:
            self._apply_lighting(lp)
        self._lookahead_lighting_applied = len(lighting_steps)

        if has_next_compare:
            print(f"[Auto] Lookahead: {len(lighting_steps)} lighting step(s) pre-applied, "
                  f"capturing frame after 150 ms settle (parallel with Fit).")
            QTimer.singleShot(150, lambda c=cycle: self._lookahead_capture(c))

    def _lookahead_capture(self, cycle: int) -> None:
        if cycle != self._cycle_id:
            return
        frame = self._viewer._current_cv_img
        if frame is None:
            return
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).copy()
        self._lookahead_frame_bgr = frame_bgr
        print("[Auto] Lookahead: EPI frame captured — starting edge detection in parallel.")

        worker = _EdgeOnlyWorker(frame_bgr)
        worker.signals.done.connect(
            lambda edge, c=cycle: self._lookahead_ready.emit(edge, c),
            Qt.DirectConnection)
        worker.signals.error.connect(
            lambda msg: print(f"[Auto] Lookahead edge error: {msg}"),
            Qt.DirectConnection)
        worker.setAutoDelete(True)
        self._pool.start(worker)

    @Slot(object, int)
    def _on_lookahead_ready(self, edge_result, cycle: int) -> None:
        if cycle != self._cycle_id:
            return
        self._lookahead_edge = edge_result
        print("[Auto] Lookahead: edge result ready ✓")

    # ── Step callbacks ────────────────────────────────────────────────

    @Slot(object, str, int)
    def _on_step_done(self, payload, task: str, cycle: int) -> None:
        if cycle != self._cycle_id:
            return
        result, rgba_bytes, shape = payload
        self._last_result = result
        print(f"[Auto] {task} done — tx={result.tx:+.1f} ty={result.ty:+.1f} "
              f"angle={result.angle_deg:.2f}° inlier={result.inlier_frac*100:.1f}%")
        self._overlay.draw_heatmap_from_rgba(self._dxf_data, result, rgba_bytes, shape)
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
        frame = self._viewer._current_cv_img
        if frame is not None:
            h, w = frame.shape[:2]
            scale = _DETECT_WIDTH / w
            small = cv2.resize(frame, (_DETECT_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
            self._last_analysed_gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        self._update_status("Done — waiting for scene change…")
        self._phase = _Phase.WAITING_CHANGE
        self._stable_count = 0
        self._warmup_lighting_applied = False
        self._reset_lookahead()

    # ── Helpers ───────────────────────────────────────────────────────

    def _frame_diff_pct(self, ref: np.ndarray | None, cur: np.ndarray) -> float:
        if ref is None:
            return 0.0
        if ref.shape != cur.shape:
            return 100.0
        diff = cv2.absdiff(ref, cur)
        _, mask = cv2.threshold(diff, self._noise_px_threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        _, mask = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY)
        k = max(3, int(self._morph_kernel_size * _DETECT_WIDTH / 5472) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return (np.count_nonzero(mask) / mask.size) * 100.0

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
