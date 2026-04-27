"""
VideoFIT — Toolbar
Mode-switching toolbar (Measure / Compare) with contextual tool pages.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QActionGroup
from PySide6.QtWidgets import (
    QButtonGroup, QFrame, QHBoxLayout, QMenu, QPushButton,
    QStackedWidget, QToolButton, QWidget,
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

        # ── Measure page ──────────────────────────────────────────────
        measure_page = QWidget()
        measure_page.setStyleSheet("background-color: transparent;")
        measure_layout = QHBoxLayout(measure_page)
        measure_layout.setContentsMargins(0, 0, 0, 0)
        measure_layout.setSpacing(5)

        self.btn_run_measure = QPushButton(" Run")
        self.btn_run_measure.setIcon(IconManager.get_icon("run", "▶️"))

        # Build button: main area toggles ROI mode; arrow opens shape-type menu
        self.btn_build = QToolButton()
        self.btn_build.setText(" Build")
        self.btn_build.setIcon(IconManager.get_icon("draw", "🔨"))
        self.btn_build.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_build.setPopupMode(QToolButton.MenuButtonPopup)
        self.btn_build.setCheckable(True)

        shape_menu = QMenu(self.btn_build)
        self._shape_group = QActionGroup(shape_menu)
        self._shape_group.setExclusive(True)
        for name in ("Auto", "Circle", "Arc", "Line"):
            action = shape_menu.addAction(name)
            action.setCheckable(True)
            self._shape_group.addAction(action)
        self._shape_group.actions()[0].setChecked(True)   # default: Auto
        shape_menu.triggered.connect(self._on_shape_selected)
        self.btn_build.setMenu(shape_menu)

        self.btn_distances = QPushButton(" Distances")
        self.btn_distances.setIcon(IconManager.get_icon("distance", "📏"))
        self.btn_distances.setEnabled(False)  # future feature

        for widget in (self.btn_run_measure, self.btn_build, self.btn_distances):
            measure_layout.addWidget(widget)

        self.tool_stack.addWidget(measure_page)

        # ── Compare page ──────────────────────────────────────────────
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

    # ── Slots ─────────────────────────────────────────────────────────

    def _on_mode_changed(self, button: QPushButton) -> None:
        index = 0 if button is self.btn_measure else 1
        self.tool_stack.setCurrentIndex(index)
        self.adjustSize()

    def _on_shape_selected(self, action) -> None:
        """Update the Build button label when a shape type is chosen."""
        self.btn_build.setText(f" {action.text()}")
        self.btn_build.adjustSize()
        self.adjustSize()

    # ── Helpers ───────────────────────────────────────────────────────

    def current_shape(self) -> str:
        """Return the currently selected shape type ('Auto', 'Circle', 'Arc', 'Line')."""
        checked = self._shape_group.checkedAction()
        return checked.text() if checked else "Auto"
