"""
VideoFIT — Toolbar
Mode-switching toolbar.

  mode="App"  → Measure / Compare switcher with contextual tool pages
  mode="Auto" → Single Start/Stop button + status label only
"""

from PySide6.QtWidgets import (
    QButtonGroup, QFrame, QHBoxLayout, QLabel, QVBoxLayout, QPushButton,
    QStackedWidget, QWidget
)
from PySide6.QtCore import Qt

from app.constants import GLASS_STYLE
from app.views.icon_manager import IconManager


def _make_tool_page(
    buttons: list[tuple[str, str, str] | str],
    return_buttons: bool = False,
) -> QWidget | tuple[QWidget, list[QPushButton]]:
    """Create a horizontal row of tool buttons. Pass "|" to insert a separator."""
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
            sep.setStyleSheet("background-color: rgba(255, 255, 255, 45); width: 1px; margin: 0 4px;")
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
    def __init__(self, mode: str = "App", parent: QWidget | None = None) -> None:
        super().__init__(parent)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(8)
        main_layout.setAlignment(Qt.AlignBottom | Qt.AlignHCenter)

        self._app_mode = mode.strip().lower()  # "app" or "auto"

        if self._app_mode == "auto":
            self._build_auto_toolbar(main_layout)
        else:
            self._build_sub_toolbar(main_layout)
            self._build_main_toolbar(main_layout)

        self.adjustSize()

    # ── Auto mode ────────────────────────────────────────────────────

    def _build_auto_toolbar(self, main_layout: QVBoxLayout) -> None:
        """Single glass panel: Start/Stop + status label."""
        self.main_toolbar = QFrame()
        self.main_toolbar.setObjectName("GlassPanel")
        self.main_toolbar.setStyleSheet(GLASS_STYLE)

        row = QHBoxLayout(self.main_toolbar)
        row.setContentsMargins(14, 7, 14, 7)
        row.setSpacing(12)

        self.btn_auto_start = QPushButton(" Start")
        self.btn_auto_start.setIcon(IconManager.get_icon("run", "▶️"))
        row.addWidget(self.btn_auto_start)

        self.lbl_auto_status = QLabel("Idle")
        self.lbl_auto_status.setStyleSheet(
            "color: rgba(255,255,255,180); font-size: 12px; background: transparent;")
        row.addWidget(self.lbl_auto_status)

        main_layout.addWidget(self.main_toolbar, alignment=Qt.AlignHCenter)

        # Stubs so orchestrator code that references these doesn't crash
        self.sub_toolbar = QFrame()   # hidden, never shown

    # ── App mode ─────────────────────────────────────────────────────

    def _build_sub_toolbar(self, main_layout: QVBoxLayout) -> None:
        """Top row: context-sensitive sub-tools (Draw shapes, Distance modes)."""
        self.sub_toolbar = QFrame()
        self.sub_toolbar.setObjectName("GlassPanel")
        self.sub_toolbar.setStyleSheet(GLASS_STYLE)

        sub_layout = QHBoxLayout(self.sub_toolbar)
        sub_layout.setContentsMargins(10, 5, 10, 5)
        sub_layout.setSpacing(5)

        self.sub_tool_stack = QStackedWidget()
        self.sub_tool_stack.setStyleSheet("background-color: transparent;")

        # Page 0: empty fallback
        self.sub_tool_stack.addWidget(QWidget())

        # Page 1: Draw sub-tools
        draw_page, draw_btns = _make_tool_page([
            (" Run", "run", "▶️"),
            "|",
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

        # Page 2: Distance sub-tools
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

    def _build_main_toolbar(self, main_layout: QVBoxLayout) -> None:
        """Bottom row: Measure / Compare switcher + tool pages."""
        self.main_toolbar = QFrame()
        self.main_toolbar.setObjectName("GlassPanel")
        self.main_toolbar.setStyleSheet(GLASS_STYLE)

        bottom_layout = QHBoxLayout(self.main_toolbar)
        bottom_layout.setContentsMargins(10, 5, 10, 5)
        bottom_layout.setSpacing(5)

        # Mode switcher
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
        sep.setStyleSheet("background-color: rgba(255, 255, 255, 45); width: 1px;")
        bottom_layout.addWidget(sep)

        # Tool stack
        self.tool_stack = QStackedWidget()
        self.tool_stack.setStyleSheet("background-color: transparent;")

        # Measure page
        measure_page = QWidget()
        measure_page.setStyleSheet("background-color: transparent;")
        measure_layout = QHBoxLayout(measure_page)
        measure_layout.setContentsMargins(0, 0, 0, 0)
        measure_layout.setSpacing(5)

        self.btn_build = QPushButton(" Draw")
        self.btn_build.setIcon(IconManager.get_icon("draw", "🔨"))
        self.btn_build.setCheckable(True)
        self.btn_build.clicked.connect(lambda: self._on_measure_tool_clicked(self.btn_build))

        self.btn_distances = QPushButton(" Distances")
        self.btn_distances.setIcon(IconManager.get_icon("distance", "📏"))
        self.btn_distances.setCheckable(True)
        self.btn_distances.clicked.connect(lambda: self._on_measure_tool_clicked(self.btn_distances))

        self._measure_tools_group = QButtonGroup(self)
        self._measure_tools_group.setExclusive(False)
        for btn in (self.btn_build, self.btn_distances):
            self._measure_tools_group.addButton(btn)
            measure_layout.addWidget(btn)
        self.tool_stack.addWidget(measure_page)

        # Compare page
        compare_page, compare_btns = _make_tool_page([
            (" Load", "loadfile", "📄"),
            (" Fit", "run", "▶️"),
            (" Reanalyze", "magic", "🔄"),
        ], return_buttons=True)
        self.btn_load = compare_btns[0]
        self.btn_fit = compare_btns[1]
        self.btn_reanalyze = compare_btns[2]
        self.tool_stack.addWidget(compare_page)

        bottom_layout.addWidget(self.tool_stack)
        main_layout.addWidget(self.main_toolbar, alignment=Qt.AlignHCenter)

        self._mode_group.buttonClicked.connect(self._on_mode_changed)

    # ── Shared helpers ────────────────────────────────────────────────

    def _on_measure_tool_clicked(self, clicked_btn: QPushButton) -> None:
        if clicked_btn is self.btn_build and clicked_btn.isChecked():
            self.btn_distances.setChecked(False)
        elif clicked_btn is self.btn_distances and clicked_btn.isChecked():
            self.btn_build.setChecked(False)
        self._update_sub_toolbar()

    def _update_sub_toolbar(self) -> None:
        old_bottom = self.geometry().bottom()

        if self.btn_measure.isChecked():
            if self.btn_build.isChecked():
                self.sub_tool_stack.setCurrentIndex(1)
                self.sub_toolbar.show()
            elif self.btn_distances.isChecked():
                self.sub_tool_stack.setCurrentIndex(2)
                self.sub_toolbar.show()
            else:
                self.sub_toolbar.hide()
        else:
            self.sub_toolbar.hide()

        self.adjustSize()
        new_geo = self.geometry()
        new_geo.moveBottom(old_bottom)
        self.setGeometry(new_geo)

    def _on_mode_changed(self, button: QPushButton) -> None:
        index = 0 if button is self.btn_measure else 1
        self.tool_stack.setCurrentIndex(index)
        self._update_sub_toolbar()

    def current_shape(self) -> str:
        btn = self.draw_group.checkedButton()
        return btn.text().strip() if btn else "Circle"

    def current_distance_mode(self) -> str:
        btn = self.dist_group.checkedButton()
        return btn.text().strip() if btn else "Point-to-Point"
