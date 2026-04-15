"""
Metrology Vision Pro — Orchestrator
Application-level wiring: creates models, views, and presenters,
connects signals, and manages the lifecycle.
"""

from __future__ import annotations

import os
import sys

from app.models.camera import CameraManager
from app.models.settings import AppSettings
from app.presenters.camera_presenter import CameraPresenter
from app.presenters.compare_presenter import ComparePresenter
from app.presenters.settings_presenter import SettingsPresenter
from app.views.main_window import MetrologyWindow


class AppOrchestrator:
    """
    Single entry-point that assembles the entire application graph.

    Instantiate after QApplication is created but before ``app.exec()``.
    """

    def __init__(self) -> None:
        # Resolve the directory where the executable / script lives
        if getattr(sys, "frozen", False):
            self._app_dir = os.path.dirname(sys.executable)
        else:
            self._app_dir = os.path.dirname(os.path.abspath(__file__))
            # When running from the app/ package, go up one level to project root
            self._app_dir = os.path.dirname(self._app_dir)

        # --- Models ---
        settings_path = os.path.join(self._app_dir, "appsettings.json")
        self._settings = AppSettings.from_json(settings_path)
        self._camera_manager = CameraManager()
        self._camera_manager.refresh_device_list()

        # --- View ---
        self._window = MetrologyWindow()

        # Connect camera frames → viewer
        self._camera_manager.frame_ready.connect(self._window.viewer.update_image)

        # --- Presenters ---
        self._settings_presenter = SettingsPresenter(
            settings=self._settings,
            panel=self._window.settings_panel,
            toggle_button=self._window.btn_settings,
        )

        self._camera_presenter = CameraPresenter(
            settings=self._settings,
            camera_manager=self._camera_manager,
            panel=self._window.settings_panel,
            viewer=self._window.viewer,
            app_dir=self._app_dir,
        )

        self._compare_presenter = ComparePresenter(
            settings=self._settings,
            viewer=self._window.viewer,
            toolbar=self._window.toolbar,
            settings_panel=self._window.settings_panel,
        )

        # Toolbar repositioning after mode switch
        self._window.toolbar.btn_measure.clicked.connect(self._reposition_toolbar)
        self._window.toolbar.btn_compare.clicked.connect(self._reposition_toolbar)

        # Clear DXF overlay when switching back to Measure mode
        self._window.toolbar.btn_measure.clicked.connect(
            self._compare_presenter.clear_overlay
        )

        # Clean shutdown
        self._window.destroyed.connect(self._cleanup)

        # Kick off the first camera
        self._camera_presenter.activate_default_camera()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def window(self) -> MetrologyWindow:
        return self._window

    def show(self) -> None:
        self._window.show()

    def cleanup(self) -> None:
        """Explicitly stop streaming — call before ic4.Library.exit()."""
        self._camera_manager.disconnect()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _reposition_toolbar(self) -> None:
        self._window.toolbar.adjustSize()
        self._window._position_floating_elements()

    def _cleanup(self) -> None:
        self._camera_manager.disconnect()

