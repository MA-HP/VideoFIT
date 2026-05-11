"""
VideoFIT — Orchestrator
Application-level wiring: creates models, views, and presenters,
connects signals, and manages the lifecycle.
"""

from __future__ import annotations

import os
import sys

from app.models.settings import AppSettings
from app.services.camera_service import CameraService
from app.presenters.camera_presenter import CameraPresenter
from app.presenters.compare_presenter import ComparePresenter
from app.presenters.auto_presenter import AutoPresenter
from app.presenters.lighting_presenter import LightingPresenter
from app.presenters.measure_presenter import MeasurePresenter
from app.presenters.settings_presenter import SettingsPresenter
from app.views.main_window import MetrologyWindow


class AppOrchestrator:
    """
    Single entry-point that assembles the entire application graph.

    Instantiate after QApplication is created but before ``app.exec()``.
    """

    def __init__(self, debug: bool = False) -> None:
        # Resolve the directory where the executable / script lives
        if getattr(sys, "frozen", False):
            self._app_dir = os.path.dirname(sys.executable)
        else:
            self._app_dir = os.path.dirname(os.path.abspath(__file__))
            # When running from the app/ package, go up one level to project root
            self._app_dir = os.path.dirname(self._app_dir)

        self._debug = debug

        # --- Models ---
        settings_path = os.path.join(self._app_dir, "appsettings.json")
        self._settings = AppSettings.from_json(settings_path)
        self._camera_service = CameraService()
        self._camera_service.refresh_device_list()

        # Determine UI mode from settings ("App" or "Auto")
        app_mode = self._settings.app_defaults.mode  # "App" | "Auto"
        print(f"[Orchestrator] UI mode: {app_mode}")

        # --- View ---
        self._window = MetrologyWindow(toolbar_mode=app_mode)

        # Connect camera frames → viewer
        self._camera_service.frame_ready.connect(self._window.viewer.update_image)

        # --- Presenters always created ---
        self._settings_presenter = SettingsPresenter(
            settings=self._settings,
            panel=self._window.settings_panel,
            toggle_button=self._window.btn_settings,
        )

        self._lighting_presenter = LightingPresenter(
            settings=self._settings,
            panel=self._window.lighting_panel,
            toggle_button=self._window.btn_lighting,
        )

        self._window.settings_panel.combo_camera.currentTextChanged.connect(
            self._lighting_presenter.load_camera_preset
        )

        self._camera_presenter = CameraPresenter(
            settings=self._settings,
            camera_service=self._camera_service,
            panel=self._window.settings_panel,
            viewer=self._window.viewer,
            app_dir=self._app_dir,
        )

        # --- Mode-specific wiring ---
        if app_mode.strip().lower() == "auto":
            self._wire_auto_mode()
        else:
            self._wire_app_mode()

        if self._debug:
            self._window.settings_panel.enable_debug_option()

        self._window.destroyed.connect(self._cleanup)
        self._camera_presenter.activate_default_camera()

    # ------------------------------------------------------------------
    # Mode wiring
    # ------------------------------------------------------------------

    def _wire_auto_mode(self) -> None:
        """Auto mode: single Start/Stop toolbar, pipeline drives everything."""
        # ComparePresenter is not connected to toolbar buttons in auto mode
        # but we still need its overlay. Create it without toolbar wiring.
        from app.views.dxf_overlay import DxfOverlay

        self._overlay = DxfOverlay(self._window.viewer._scene)

        self._auto_presenter = AutoPresenter(
            settings=self._settings,
            viewer=self._window.viewer,
            toolbar=self._window.toolbar,
            settings_panel=self._window.settings_panel,
            overlay=self._overlay,
            lighting_service=self._lighting_presenter._service,
            app_dir=self._app_dir,
        )

        # Auto-load pipeline.json on startup
        pipeline_path = os.path.join(self._app_dir, "pipeline.json")
        if os.path.isfile(pipeline_path):
            self._auto_presenter.load_pipeline(pipeline_path)
        else:
            print(f"[Orchestrator] No pipeline.json found at {pipeline_path}")

    def _wire_app_mode(self) -> None:
        """App mode: Measure/Compare toolbar with full presenter wiring."""
        self._compare_presenter = ComparePresenter(
            settings=self._settings,
            viewer=self._window.viewer,
            toolbar=self._window.toolbar,
            settings_panel=self._window.settings_panel,
            debug=self._debug,
        )

        self._measure_presenter = MeasurePresenter(
            settings=self._settings,
            viewer=self._window.viewer,
            toolbar=self._window.toolbar,
        )

        # Toolbar repositioning after mode switch
        self._window.toolbar.btn_measure.clicked.connect(self._reposition_toolbar)
        self._window.toolbar.btn_compare.clicked.connect(self._reposition_toolbar)

        # Clear overlays on mode switch
        self._window.toolbar.btn_measure.clicked.connect(
            self._compare_presenter.clear_overlay
        )
        self._window.toolbar.btn_compare.clicked.connect(
            self._measure_presenter.clear_overlay
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def window(self) -> MetrologyWindow:
        return self._window

    def show(self) -> None:
        self._window.show()

    def cleanup(self) -> None:
        """Explicitly release all IC4 resources — call before ic4.Library.exit()."""
        self._camera_service.release()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _reposition_toolbar(self) -> None:
        self._window.toolbar.adjustSize()
        self._window._position_floating_elements()

    def _cleanup(self) -> None:
        self._camera_service.disconnect()
