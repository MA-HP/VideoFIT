"""
VideoFIT — Settings Presenter
Manages the settings panel toggle and applies default values from AppSettings.
"""

from __future__ import annotations

from PySide6.QtCore import QObject

from app.models.settings import AppSettings
from app.views.settings_panel import SettingsPanel


class SettingsPresenter(QObject):
    """Binds the settings panel to the application settings model."""

    def __init__(
        self,
        settings: AppSettings,
        panel: SettingsPanel,
        toggle_button,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self._panel = panel
        self._toggle_btn = toggle_button

        # Apply defaults from the model into the view
        self._panel.combo_comparison.setCurrentText(settings.app_defaults.comparison_mode)

        # Populate camera combo from settings (names only — presenter decides content)
        self._panel.combo_camera.clear()
        if settings.cameras:
            for cam in settings.cameras:
                self._panel.combo_camera.addItem(cam.name)
        else:
            self._panel.combo_camera.addItem("No cameras in settings")

        # Wire toggle
        self._toggle_btn.clicked.connect(self._on_toggle)

    # ------------------------------------------------------------------

    def _on_toggle(self) -> None:
        self._panel.setVisible(self._toggle_btn.isChecked())

