"""
VideoFIT — Settings Panel
"""

from PySide6.QtWidgets import QCheckBox, QComboBox, QGridLayout, QLabel, QLineEdit, QWidget

from app.views.glass_panel import GlassPanel


class SettingsPanel(GlassPanel):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)

        layout = QGridLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        self.combo_comparison = QComboBox()
        self.combo_comparison.addItems(["Best Fit", "Refine", "POC"])
        self.combo_fit_objective = QComboBox()
        self.combo_fit_objective.addItems(["Strict", "Tolerance"])
        self.combo_fit_objective.setToolTip(
            "Strict: minimise mean distance to edges.\n"
            "Tolerance: maximise points within max error threshold."
        )
        self.combo_camera = QComboBox()
        self.input_heatmap_min = QLineEdit("0.1")
        self.input_heatmap_max = QLineEdit("0.5")

        _ROW_H = 30
        for w in (self.combo_comparison, self.combo_fit_objective,
                  self.combo_camera, self.input_heatmap_min, self.input_heatmap_max):
            w.setFixedHeight(_ROW_H)

        rows = [
            ("Camera:", self.combo_camera),
            ("Comparison Mode:", self.combo_comparison),
            ("Fit Objective:", self.combo_fit_objective),
            ("Min error (mm):", self.input_heatmap_min),
            ("Max error (mm):", self.input_heatmap_max),
        ]
        for row, (label_text, widget) in enumerate(rows):
            layout.addWidget(QLabel(label_text), row, 0)
            layout.addWidget(widget, row, 1)

        self._debug_row = len(rows)
        self._lbl_debug = QLabel("Debug preprocessing:")
        self.chk_debug = QCheckBox()
        self.chk_debug.setToolTip("Show intermediate preprocessing images after each fit")
        layout.addWidget(self._lbl_debug, self._debug_row, 0)
        layout.addWidget(self.chk_debug, self._debug_row, 1)
        self._lbl_debug.hide()
        self.chk_debug.hide()

        self.adjustSize()
        self.hide()

    def enable_debug_option(self) -> None:
        self._lbl_debug.show()
        self.chk_debug.show()
        self.adjustSize()
