"""
VideoFIT — Lighting Presenter
Single background thread serialises all TCP commands.
For each channel only the latest pending value is sent — no queue buildup,
no parallel connection conflicts.
"""

from __future__ import annotations

import threading

from PySide6.QtCore import QObject
from PySide6.QtWidgets import QPushButton

from app.models.settings import AppDefaults, AppSettings, CameraInfo
from app.services.lighting_service import LightingService
from app.views.lighting_panel import LightingPanel


class _LightingWorker:
    """
    Single daemon thread that serialises every TCP command.

    send(channel, value) overwrites any pending value for that channel.
    The thread drains the pending dict one command at a time — so during
    a burst (e.g. slider drag) intermediate values are dropped and only
    the latest per channel is ever transmitted.
    """

    def __init__(self, service: LightingService) -> None:
        self._service = service
        self._pending: dict[int, float] = {}   # channel → latest value
        self._lock = threading.Lock()
        self._event = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def send(self, channel: int, value: float) -> None:
        with self._lock:
            self._pending[channel] = value     # overwrite — only latest matters
        self._event.set()

    def _run(self) -> None:
        while True:
            self._event.wait()
            self._event.clear()

            while True:
                with self._lock:
                    if not self._pending:
                        break
                    # Pop one channel to send (FIFO on insertion order)
                    channel, value = next(iter(self._pending.items()))
                    del self._pending[channel]

                # TCP call outside the lock so new send() can arrive freely
                self._service.set_intensity(channel, value)


class LightingPresenter(QObject):

    def __init__(
        self,
        settings: AppSettings,
        panel: LightingPanel,
        toggle_button: QPushButton,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self._panel = panel
        self._toggle_btn = toggle_button

        ip = settings.app_defaults.lighting_ip
        port = settings.app_defaults.lighting_port
        self._service = LightingService(ip, port)
        self._worker = _LightingWorker(self._service)

        active = self._active_channels_from_defaults(settings.app_defaults)
        self._rebuild_panel(active)

        if settings.cameras:
            self._apply_camera_preset(settings.cameras[0], send=True)

        self._toggle_btn.clicked.connect(self._on_toggle)
        self._panel.intensity_changed.connect(self._on_intensity_changed)

    # ------------------------------------------------------------------

    def load_camera_preset(self, camera_name: str) -> None:
        for cam in self._settings.cameras:
            if cam.name == camera_name:
                self._apply_camera_preset(cam, send=True)
                return

    # ------------------------------------------------------------------

    @staticmethod
    def _active_channels_from_defaults(defaults: AppDefaults) -> set[int]:
        active = {ch for ch, flag in [
            (1, defaults.light_ch1_active),
            (2, defaults.light_ch2_active),
            (3, defaults.light_ch3_active),
            (4, defaults.light_ch4_active),
        ] if flag}
        return active or {1, 2, 3, 4}

    def _rebuild_panel(self, active_channels: set[int]) -> None:
        try:
            self._panel.intensity_changed.disconnect(self._on_intensity_changed)
        except RuntimeError:
            pass
        self._panel._active_channels = active_channels
        self._panel._sliders.clear()
        self._panel._value_labels.clear()
        old_layout = self._panel.layout()
        if old_layout:
            while old_layout.count():
                item = old_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
        self._panel._init_content(active_channels)
        self._panel.set_toggle_button(self._toggle_btn)
        self._panel.adjustSize()
        self._panel.intensity_changed.connect(self._on_intensity_changed)

    def _apply_camera_preset(self, cam: CameraInfo, send: bool = False) -> None:
        for ch, val in [
            (1, cam.light_ch1_intensity),
            (2, cam.light_ch2_intensity),
            (3, cam.light_ch3_intensity),
            (4, cam.light_ch4_intensity),
        ]:
            self._panel.set_channel_intensity(ch, val)
            if send:
                self._worker.send(ch, val)

    def _on_toggle(self) -> None:
        self._panel.setVisible(self._toggle_btn.isChecked())

    def _on_intensity_changed(self, channel: int, intensity: float) -> None:
        self._worker.send(channel, intensity)
