"""
VideoFIT — Settings Panel
Glass-morphism floating panel with calibration, comparison and camera controls.
"""

from PySide6.QtWidgets import QCheckBox, QComboBox, QFrame, QGridLayout, QLabel, QLineEdit, QWidget

from app.constants import GLASS_STYLE


class SettingsPanel(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("GlassPanel")
        self.setStyleSheet(GLASS_STYLE)

        layout = QGridLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        # Expose widgets for presenters to manipulate
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

        # Debug checkbox — only made visible when DEBUG=True in main.py
        self._debug_row = len(rows)
        self._lbl_debug = QLabel("Debug preprocessing:")
        self.chk_debug = QCheckBox()
        self.chk_debug.setToolTip(
            "Show the intermediate preprocessing images after each fit"
        )
        layout.addWidget(self._lbl_debug, self._debug_row, 0)
        layout.addWidget(self.chk_debug, self._debug_row, 1)
        # Hidden until explicitly enabled
        self._lbl_debug.hide()
        self.chk_debug.hide()

        self.adjustSize()
        self.hide()

    # ------------------------------------------------------------------

    def enable_debug_option(self) -> None:
        """Make the debug checkbox visible (called when DEBUG=True in main.py)."""
        self._lbl_debug.show()
        self.chk_debug.show()
        self.adjustSize()
