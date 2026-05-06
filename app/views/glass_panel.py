"""
VideoFIT — GlassPanel base
Shared base for all floating glass panels.
Closes automatically when the user clicks outside the panel.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QFrame, QWidget

from app.constants import GLASS_STYLE


class GlassPanel(QFrame):
    """
    Glassmorphism panel that:
    - applies GLASS_STYLE automatically
    - hides itself (and unchecks its toggle button) when the user
      clicks anywhere outside the panel
    """

    def __init__(self, toggle_button=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("GlassPanel")
        self.setStyleSheet(GLASS_STYLE)
        self._toggle_btn = toggle_button

        # Install a global event filter to detect outside clicks
        QApplication.instance().installEventFilter(self)

    def set_toggle_button(self, btn) -> None:
        self._toggle_btn = btn

    # ------------------------------------------------------------------

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if (
            self.isVisible()
            and event.type() == QEvent.MouseButtonPress
        ):
            # Map the click position to our coordinate space
            mouse_event: QMouseEvent = event
            global_pos = mouse_event.globalPosition().toPoint()
            local_pos = self.mapFromGlobal(global_pos)

            # Close if click is outside this panel (and outside the toggle btn)
            if not self.rect().contains(local_pos):
                btn_absorbs = (
                    self._toggle_btn is not None
                    and self._toggle_btn.isVisible()
                    and self._toggle_btn.rect().contains(
                        self._toggle_btn.mapFromGlobal(global_pos)
                    )
                )
                if not btn_absorbs:
                    self.hide()
                    if self._toggle_btn is not None:
                        self._toggle_btn.setChecked(False)

        return super().eventFilter(watched, event)

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        if self._toggle_btn is not None:
            self._toggle_btn.setChecked(False)

