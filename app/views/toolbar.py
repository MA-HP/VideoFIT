"""
VideoFIT — Toolbar
Mode-switching toolbar (Measure / Compare) with contextual tool pages.
"""

from PySide6.QtWidgets import (
    QButtonGroup, QFrame, QHBoxLayout, QVBoxLayout, QPushButton,
    QStackedWidget, QWidget
)
from PySide6.QtCore import Qt

from app.constants import GLASS_STYLE
from app.views.icon_manager import IconManager


def _make_tool_page(
    buttons: list[tuple[str, str, str] | str],
    return_buttons: bool = False,
) -> QWidget | tuple[QWidget, list[QPushButton]]:
    """Create a horizontal row of tool buttons.

    Pass a string "|" to insert a vertical separator.
    """
    page = QWidget()
    page.setStyleSheet("background-color: transparent;")
    layout = QHBoxLayout(page)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(5)
    btn_list: list[QPushButton] = []

    for item in buttons:
        if item == "|":
            sep = QFrame()
            sep.setFrameShape(QFrame.VLine)
            sep.setStyleSheet("background-color: rgba(255, 255, 255, 20); width: 1px; margin: 0 4px;")
            layout.addWidget(sep)
            continue

        label, icon_name, fallback = item
        btn = QPushButton(label)
        btn.setIcon(IconManager.get_icon(icon_name, fallback))
        layout.addWidget(btn)
        btn_list.append(btn)

    if return_buttons:
        return page, btn_list
    return page


class Toolbar(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(8)
        # Ensure internal elements align to the bottom of the container
        main_layout.setAlignment(Qt.AlignBottom | Qt.AlignHCenter)

        # ==========================================
        # 1. TOP ROW: SUB-TOOLBAR (Separate Frame)
        # ==========================================
        self.sub_toolbar = QFrame()
        self.sub_toolbar.setObjectName("GlassPanel")
        self.sub_toolbar.setStyleSheet(GLASS_STYLE)

        sub_layout = QHBoxLayout(self.sub_toolbar)
        sub_layout.setContentsMargins(10, 5, 10, 5)
        sub_layout.setSpacing(5)

        self.sub_tool_stack = QStackedWidget()
        self.sub_tool_stack.setStyleSheet("background-color: transparent;")

        # Page 0: Empty (fallback)
        self.sub_tool_stack.addWidget(QWidget())

        # Page 1: Draw Mode Sub-tools (Includes the RUN button)
        draw_page, draw_btns = _make_tool_page([
            (" Run", "run", "▶️"),
            "|",  # Separator between action and shapes
            (" Circle", "circle", "⭕"),
            (" Arc", "arc", "🌙"),
            (" Line", "line", "➖"),
        ], return_buttons=True)

        self.btn_run_measure = draw_btns[0]

        self.draw_group = QButtonGroup(self)
        for b in draw_btns[1:]:
            b.setCheckable(True)
            self.draw_group.addButton(b)
        draw_btns[1].setChecked(True)
        self.sub_tool_stack.addWidget(draw_page)

        # Page 2: Distance Mode Sub-tools
        dist_page, dist_btns = _make_tool_page([
            (" Point-to-Point", "p2p", "📍"),
            (" Point-to-Line", "p2l", "📐"),
            (" Line-to-Line", "l2l", "⏸️"),
        ], return_buttons=True)
        self.dist_group = QButtonGroup(self)
        for b in dist_btns:
            b.setCheckable(True)
            self.dist_group.addButton(b)
        dist_btns[0].setChecked(True)
        self.sub_tool_stack.addWidget(dist_page)

        sub_layout.addWidget(self.sub_tool_stack)
        main_layout.addWidget(self.sub_toolbar, alignment=Qt.AlignHCenter)

        self.sub_toolbar.hide()


        # ==========================================
        # 2. BOTTOM ROW: MAIN TOOLBAR (Separate Frame)
        # ==========================================
        self.main_toolbar = QFrame()
        self.main_toolbar.setObjectName("GlassPanel")
        self.main_toolbar.setStyleSheet(GLASS_STYLE)

        bottom_layout = QHBoxLayout(self.main_toolbar)
        bottom_layout.setContentsMargins(10, 5, 10, 5)
        bottom_layout.setSpacing(5)

        # Measure / Compare Mode Switcher
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
            bottom_layout.addWidget(btn)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("background-color: rgba(255, 255, 255, 20); width: 1px;")
        bottom_layout.addWidget(sep)

        # Main Tool Stack
        self.tool_stack = QStackedWidget()
        self.tool_stack.setStyleSheet("background-color: transparent;")

        # ----------------------------------------------------
        # Measure page (Explicit instantiation to match styles)
        # ----------------------------------------------------
        measure_page = QWidget()
        measure_page.setStyleSheet("background-color: transparent;")
        measure_layout = QHBoxLayout(measure_page)
        measure_layout.setContentsMargins(0, 0, 0, 0)
        measure_layout.setSpacing(5)

        # Build button (Draw)
        self.btn_build = QPushButton(" Draw")
        self.btn_build.setIcon(IconManager.get_icon("draw", "🔨"))
        self.btn_build.setCheckable(True)
        self.btn_build.clicked.connect(lambda: self._on_measure_tool_clicked(self.btn_build))

        # Distances button
        self.btn_distances = QPushButton(" Distances")
        self.btn_distances.setIcon(IconManager.get_icon("distance", "📏"))
        self.btn_distances.setCheckable(True)
        self.btn_distances.clicked.connect(lambda: self._on_measure_tool_clicked(self.btn_distances))

        # Group them but allow unchecking (so sub-toolbar can hide)
        self._measure_tools_group = QButtonGroup(self)
        self._measure_tools_group.setExclusive(False)

        for btn in (self.btn_build, self.btn_distances):
            self._measure_tools_group.addButton(btn)
            measure_layout.addWidget(btn)

        self.tool_stack.addWidget(measure_page)
        # ----------------------------------------------------

        # Compare page
        compare_page, compare_btns = _make_tool_page([
            (" Load", "loadfile", "📄"),
            (" Run", "run", "▶️"),
        ], return_buttons=True)
        self.btn_load = compare_btns[0]
        self.btn_run_compare = compare_btns[1]
        self.tool_stack.addWidget(compare_page)

        bottom_layout.addWidget(self.tool_stack)
        main_layout.addWidget(self.main_toolbar, alignment=Qt.AlignHCenter)

        self._mode_group.buttonClicked.connect(self._on_mode_changed)
        self.adjustSize()

    def _on_measure_tool_clicked(self, clicked_btn: QPushButton) -> None:
        """Manage mutual exclusivity of the sub-tools in Measure mode."""
        if clicked_btn is self.btn_build and clicked_btn.isChecked():
            self.btn_distances.setChecked(False)
        elif clicked_btn is self.btn_distances and clicked_btn.isChecked():
            self.btn_build.setChecked(False)

        self._update_sub_toolbar()

    def _update_sub_toolbar(self) -> None:
        """Update visibility and keep the main toolbar anchored to the bottom."""
        # Capture the current bottom position before making changes
        old_bottom = self.geometry().bottom()

        if self.btn_measure.isChecked():
            if self.btn_build.isChecked():
                self.sub_tool_stack.setCurrentIndex(1) # Draw page
                self.sub_toolbar.show()
            elif self.btn_distances.isChecked():
                self.sub_tool_stack.setCurrentIndex(2) # Distance page
                self.sub_toolbar.show()
            else:
                self.sub_toolbar.hide()
        else:
            self.sub_toolbar.hide()

        # Force layout to recalculate its new size
        self.adjustSize()

        # Shift the widget so the bottom edge stays exactly where it was
        new_geo = self.geometry()
        new_geo.moveBottom(old_bottom)
        self.setGeometry(new_geo)

    def _on_mode_changed(self, button: QPushButton) -> None:
        """Toggle between Measure and Compare stacks."""
        index = 0 if button is self.btn_measure else 1
        self.tool_stack.setCurrentIndex(index)
        self._update_sub_toolbar()

    def current_shape(self) -> str:
        """Return the currently selected shape name from the Draw sub-toolbar."""
        btn = self.draw_group.checkedButton()
        if btn:
            return btn.text().strip()
        return "Circle"

    def current_distance_mode(self) -> str:
        """Return the currently selected distance logic from the Distance sub-toolbar."""
        btn = self.dist_group.checkedButton()
        if btn:
            return btn.text().strip()
        return "Point-to-Point"