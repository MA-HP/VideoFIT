"""
VideoFIT — Title Bar
Frameless window title bar with drag-to-move, minimize, maximize and close.
"""

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QSpacerItem, QWidget

from app.constants import TITLE_BAR_STYLE


class TitleBar(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(40)
        self.setStyleSheet(TITLE_BAR_STYLE)
        self._drag_pos = QPoint()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 5, 0)
        layout.setSpacing(5)

        self.title_label = QLabel("VideoFIT")
        layout.addWidget(self.title_label)
        layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        for text, name, slot in [
            ("—", None, lambda: self.window().showMinimized()),
            ("☐", None, self._toggle_max_restore),
            ("✕", "btnClose", lambda: self.window().close()),
        ]:
            btn = QPushButton(text)
            btn.setFixedSize(30, 30)
            if name:
                btn.setObjectName(name)
            btn.clicked.connect(slot)
            layout.addWidget(btn)

    # ------------------------------------------------------------------

    def _toggle_max_restore(self) -> None:
        win = self.window()
        win.showNormal() if win.isMaximized() else win.showMaximized()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() == Qt.LeftButton:
            delta = event.globalPosition().toPoint() - self._drag_pos
            self.window().move(self.window().pos() + delta)
            self._drag_pos = event.globalPosition().toPoint()

