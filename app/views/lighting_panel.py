"""
VideoFIT — Lighting Panel
"""

from __future__ import annotations

import os

from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtWidgets import QGridLayout, QLabel, QPushButton, QSlider, QWidget

from app.services.lighting_service import CHANNEL_NAMES
from app.views.glass_panel import GlassPanel

_ON_OFF_STYLE = """
    QPushButton {
        background-color: rgba(255, 255, 255, 10);
        border: 1px solid rgba(255, 255, 255, 40);
        border-radius: 4px;
        padding: 0px;
    }
    QPushButton:checked {
        background-color: rgba(255, 255, 255, 30);
        border: 1px solid rgba(255, 255, 255, 120);
    }
    QPushButton:hover {
        background-color: rgba(255, 255, 255, 20);
        border: 1px solid rgba(255, 255, 255, 80);
    }
"""

def _power_icon() -> "QIcon":
    from PySide6.QtGui import QIcon
    svg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "..", "icons", "power.svg")
    svg_path = os.path.normpath(svg_path)
    return QIcon(svg_path)


class LightingPanel(GlassPanel):
    """Floating panel with one slider + ON/OFF button per active lighting channel."""

    intensity_changed = Signal(int, float)  # (channel, intensity 0–100)
    channel_toggled = Signal(int, bool)     # (channel, is_on)

    def __init__(self, active_channels: set[int] | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)

        self._sliders: dict[int, QSlider] = {}
        self._value_labels: dict[int, QLabel] = {}
        self._toggle_btns: dict[int, QPushButton] = {}
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
            lbl = QLabel(f"{name}:")

            # ON/OFF toggle button with power icon
            btn = QPushButton()
            btn.setIcon(_power_icon())
            btn.setIconSize(QSize(16, 16))
            btn.setCheckable(True)
            btn.setChecked(True)
            btn.setFixedSize(30, 30)
            btn.setStyleSheet(_ON_OFF_STYLE)
            btn.setToolTip("Toggle channel on/off")
            btn.setProperty("channel", ch)
            btn.toggled.connect(self._on_toggle_changed)

            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(0)
            slider.setFixedWidth(140)
            slider.setFixedHeight(30)
            slider.setProperty("channel", ch)
            slider.valueChanged.connect(self._on_value_changed)

            val_lbl = QLabel("0")
            val_lbl.setFixedWidth(32)
            val_lbl.setFixedHeight(30)
            val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

            self._layout.addWidget(lbl,     row, 0)
            self._layout.addWidget(btn,     row, 1)
            self._layout.addWidget(slider,  row, 2)
            self._layout.addWidget(val_lbl, row, 3)

            self._sliders[ch] = slider
            self._value_labels[ch] = val_lbl
            self._toggle_btns[ch] = btn
            row += 1

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def set_channel_intensity(self, channel: int, intensity: float) -> None:
        """Programmatically set slider value (does NOT emit signals)."""
        slider = self._sliders.get(channel)
        if slider is None:
            return
        slider.blockSignals(True)
        slider.setValue(int(round(intensity)))
        slider.blockSignals(False)
        if channel in self._value_labels:
            self._value_labels[channel].setText(str(int(round(intensity))))

    def set_channel_on(self, channel: int, on: bool) -> None:
        """Programmatically set the ON/OFF button state (does NOT emit signals)."""
        btn = self._toggle_btns.get(channel)
        if btn is None:
            return
        btn.blockSignals(True)
        btn.setChecked(on)
        btn.blockSignals(False)
        if channel in self._sliders:
            self._sliders[channel].setEnabled(on)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_value_changed(self, value: int) -> None:
        ch = self.sender().property("channel")
        if ch in self._value_labels:
            self._value_labels[ch].setText(str(value))
        # Only emit if the channel is ON
        btn = self._toggle_btns.get(ch)
        if btn and btn.isChecked():
            self.intensity_changed.emit(ch, float(value))

    def _on_toggle_changed(self, checked: bool) -> None:
        ch = self.sender().property("channel")
        if ch in self._sliders:
            self._sliders[ch].setEnabled(checked)
        self.channel_toggled.emit(ch, checked)


