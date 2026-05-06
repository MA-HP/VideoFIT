"""
VideoFIT — Lighting Panel
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QGridLayout, QLabel, QSlider, QWidget

from app.services.lighting_service import CHANNEL_NAMES
from app.views.glass_panel import GlassPanel


class LightingPanel(GlassPanel):
    """Floating panel with one slider per active lighting channel."""

    intensity_changed = Signal(int, float)  # (channel, intensity 0–100)

    def __init__(self, active_channels: set[int] | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)

        self._sliders: dict[int, QSlider] = {}
        self._value_labels: dict[int, QLabel] = {}
        self._active_channels: set[int] = active_channels or {1, 2, 3, 4}

        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(15, 15, 15, 15)
        self._layout.setSpacing(10)

        self._init_content(self._active_channels)
        self.adjustSize()
        self.hide()

    def _init_content(self, active_channels: set[int]) -> None:
        self._active_channels = active_channels

        row = 1
        for ch in [1, 2, 3, 4]:
            if ch not in active_channels:
                continue

            name = CHANNEL_NAMES.get(ch, f"CH{ch}")
            lbl = QLabel(f"CH{ch} – {name}:")

            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(0)
            slider.setFixedWidth(160)
            slider.setFixedHeight(30)   # match QComboBox/QLineEdit row height
            slider.setProperty("channel", ch)
            slider.valueChanged.connect(self._on_value_changed)

            val_lbl = QLabel("0")
            val_lbl.setFixedWidth(32)
            val_lbl.setFixedHeight(30)  # match row height
            val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

            self._layout.addWidget(lbl,     row, 0)
            self._layout.addWidget(slider,  row, 1)
            self._layout.addWidget(val_lbl, row, 2)

            self._sliders[ch] = slider
            self._value_labels[ch] = val_lbl
            row += 1

    def set_channel_intensity(self, channel: int, intensity: float) -> None:
        slider = self._sliders.get(channel)
        if slider is None:
            return
        slider.blockSignals(True)
        slider.setValue(int(round(intensity)))
        slider.blockSignals(False)
        if channel in self._value_labels:
            self._value_labels[channel].setText(str(int(round(intensity))))

    def _on_value_changed(self, value: int) -> None:
        ch = self.sender().property("channel")
        if ch in self._value_labels:
            self._value_labels[ch].setText(str(value))
        self.intensity_changed.emit(ch, float(value))
