"""
VideoFIT — Toolbar
Mode-switching toolbar (Measure / Compare) with contextual tool pages.
"""

from PySide6.QtWidgets import (
    QButtonGroup, QFrame, QHBoxLayout, QPushButton, QStackedWidget, QWidget,
)

from app.constants import GLASS_STYLE
from app.views.icon_manager import IconManager


def _make_tool_page(
    buttons: list[tuple[str, str, str]],
    return_buttons: bool = False,
) -> QWidget | tuple[QWidget, list[QPushButton]]:
    """Create a horizontal row of tool buttons.

    If *return_buttons* is ``True``, returns ``(page, [btn, …])`` so the
    caller can keep named references.
    """
    page = QWidget()
    page.setStyleSheet("background-color: transparent;")
    layout = QHBoxLayout(page)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(5)
    btn_list: list[QPushButton] = []
    for label, icon_name, fallback in buttons:
        btn = QPushButton(label)
        btn.setIcon(IconManager.get_icon(icon_name, fallback))
        layout.addWidget(btn)
        btn_list.append(btn)
    if return_buttons:
        return page, btn_list
    return page


class Toolbar(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("GlassPanel")
        self.setStyleSheet(GLASS_STYLE)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(5)

        self._mode_group = QButtonGroup(self)
        self.btn_measure = QPushButton(" Measure")
        self.btn_measure.setIcon(IconManager.get_icon("measure", "📐"))
        self.btn_measure.setCheckable(True)
        self.btn_measure.setChecked(True)

        self.btn_compare = QPushButton(" Compare")
        self.btn_compare.setIcon(IconManager.get_icon("compare", "↔️"))
        self.btn_compare.setCheckable(True)

        for btn in (self.btn_measure, self.btn_compare):
            self._mode_group.addButton(btn)
            layout.addWidget(btn)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("background-color: rgba(255, 255, 255, 20); width: 1px;")
        layout.addWidget(sep)

        self.tool_stack = QStackedWidget()
        self.tool_stack.setStyleSheet("background-color: transparent;")
        self.tool_stack.addWidget(_make_tool_page([
            (" Run", "run", "▶️"),
            (" Draw", "draw", "🔨"),
            (" Distances", "distance", "📏"),
        ]))

        # Compare page — keep named references for the presenter
        compare_page, compare_btns = _make_tool_page([
            (" Load", "loadfile", "📄"),
            (" Run", "run", "▶️"),
        ], return_buttons=True)
        self.btn_load = compare_btns[0]
        self.btn_run_compare = compare_btns[1]
        self.tool_stack.addWidget(compare_page)

        layout.addWidget(self.tool_stack)
        self._mode_group.buttonClicked.connect(self._on_mode_changed)
        self.adjustSize()

    def _on_mode_changed(self, button: QPushButton) -> None:
        index = 0 if button is self.btn_measure else 1
        self.tool_stack.setCurrentIndex(index)
        self.adjustSize()

