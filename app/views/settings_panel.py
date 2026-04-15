"""
Metrology Vision Pro — Settings Panel
Glass-morphism floating panel with calibration, comparison and camera controls.
"""

from PySide6.QtWidgets import QComboBox, QFrame, QGridLayout, QLabel, QLineEdit, QWidget

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
        self.input_calibration = QLineEdit("0.000")
        self.combo_comparison = QComboBox()
        self.combo_comparison.addItems(["Best Fit", "Complete"])
        self.combo_camera = QComboBox()

        rows = [
            ("Calibration (px/mm):", self.input_calibration),
            ("Comparison Mode:", self.combo_comparison),
            ("Camera:", self.combo_camera),
        ]
        for row, (label_text, widget) in enumerate(rows):
            layout.addWidget(QLabel(label_text), row, 0)
            layout.addWidget(widget, row, 1)

        self.adjustSize()
        self.hide()

