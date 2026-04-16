"""
VideoFIT — Camera Presenter
Reacts to camera-combo changes, drives CameraService, and updates the view.
"""

from __future__ import annotations

import os

from PySide6.QtCore import QObject

from app.models.settings import AppSettings
from app.services.camera_service import CameraService
from app.views.image_viewer import ImageViewer
from app.views.settings_panel import SettingsPanel


class CameraPresenter(QObject):
    """Mediates between the camera combo box, the CameraService, and the viewer."""

    def __init__(
        self,
        settings: AppSettings,
        camera_service: CameraService,
        panel: SettingsPanel,
        viewer: ImageViewer,
        app_dir: str,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self._camera = camera_service
        self._panel = panel
        self._viewer = viewer
        self._app_dir = app_dir

        # Wire UI → presenter
        self._panel.combo_camera.currentIndexChanged.connect(self._on_camera_changed)

    # ------------------------------------------------------------------
    # Public — called once by the orchestrator to kick off the first camera
    # ------------------------------------------------------------------

    def activate_default_camera(self) -> None:
        """Select index 0 in the combo and trigger connection."""
        if self._settings.cameras:
            self._on_camera_changed(0)

    # ------------------------------------------------------------------
    # Slot
    # ------------------------------------------------------------------

    def _on_camera_changed(self, index: int) -> None:
        cameras = self._settings.cameras
        if not cameras or index < 0 or index >= len(cameras):
            return

        cam_info = cameras[index]

        # Update calibration in the view
        self._panel.input_calibration.setText(str(cam_info.calibration_px_mm))

        # Resolve config path
        config_path = None
        if cam_info.config_file:
            config_path = os.path.join(self._app_dir, cam_info.config_file)

        # Try connecting
        try:
            self._camera.open_camera(cam_info.serial, config_path)
        except RuntimeError as e:
            print(f"Camera '{cam_info.name}' (Serial: {cam_info.serial}) is disconnected.")
            self._viewer.clear_view()
        except Exception as e:
            print(f"Failed to open camera stream: {e}")
            self._viewer.clear_view()

