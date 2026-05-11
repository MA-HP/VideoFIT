"""
VideoFIT — Main Window (View)
Frameless glassmorphism shell. Contains only layout and positioning —
no business logic.
"""

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtWidgets import QFrame, QPushButton, QSizeGrip, QVBoxLayout, QWidget

from app.constants import MARGIN, SETTINGS_BTN_STYLE
from app.views.icon_manager import IconManager
from app.views.image_viewer import ImageViewer
from app.views.lighting_panel import LightingPanel
from app.views.settings_panel import SettingsPanel
from app.views.title_bar import TitleBar
from app.views.toolbar import Toolbar


class MetrologyWindow(QWidget):
    """
    The application shell. Owns every visual sub-component but delegates
    all behaviour to presenters wired by the orchestrator.
    """

    def __init__(self, toolbar_mode: str = "App") -> None:
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowSystemMenuHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(1100, 800)

        self._toolbar_mode = toolbar_mode
        self._build_shell()
        self._build_floating_ui()

        QTimer.singleShot(0, self._position_floating_elements)

    # ------------------------------------------------------------------
    # Shell construction
    # ------------------------------------------------------------------

    def _build_shell(self) -> None:
        self._container = QFrame(self)
        self._container.setObjectName("MainContainer")
        self._container.setStyleSheet("""
            QFrame#MainContainer {
                background-color: #0d0d0d;
                border: 1px solid rgba(255, 255, 255, 30);
                border-radius: 15px;
            }
        """)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._container)

        inner = QVBoxLayout(self._container)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.setSpacing(0)

        self.title_bar = TitleBar(self)
        inner.addWidget(self.title_bar)

        self._work_area = QFrame()
        self._work_area.setStyleSheet("""
            background-color: qradialgradient(
                cx:0.5, cy:0.5, radius:0.8, fx:0.5, fy:0.5,
                stop:0 #1a1a1a, stop:1 #050505
            );
            border-bottom-left-radius: 15px;
            border-bottom-right-radius: 15px;
        """)
        inner.addWidget(self._work_area)

        wa_layout = QVBoxLayout(self._work_area)
        wa_layout.setContentsMargins(0, 0, 0, 0)
        self.viewer = ImageViewer()
        wa_layout.addWidget(self.viewer)

    def _build_floating_ui(self) -> None:
        BTN_SIZE = 45
        ICON_SIZE = BTN_SIZE  # pixmap fills the button exactly; margin is baked into the pixmap

        self.btn_settings = QPushButton(self._work_area)
        self.btn_settings.setIcon(IconManager.get_icon("settings", "⚙️", size=ICON_SIZE, inset_factor=0.25))
        self.btn_settings.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self.btn_settings.setFixedSize(BTN_SIZE, BTN_SIZE)
        self.btn_settings.setCheckable(True)
        self.btn_settings.setStyleSheet(SETTINGS_BTN_STYLE)

        self.btn_lighting = QPushButton(self._work_area)
        self.btn_lighting.setIcon(IconManager.get_icon("light", "💡", size=ICON_SIZE, inset_factor=0.25))
        self.btn_lighting.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self.btn_lighting.setFixedSize(BTN_SIZE, BTN_SIZE)
        self.btn_lighting.setCheckable(True)
        self.btn_lighting.setStyleSheet(SETTINGS_BTN_STYLE)

        self.settings_panel = SettingsPanel(self._work_area)
        self.lighting_panel = LightingPanel(parent=self._work_area)  # channels set later by presenter
        self.toolbar = Toolbar(mode=self._toolbar_mode, parent=self._work_area)

        self._size_grip = QSizeGrip(self._container)
        self._size_grip.setStyleSheet("background-color: transparent;")

    # ------------------------------------------------------------------
    # Positioning helpers
    # ------------------------------------------------------------------

    def _position_floating_elements(self) -> None:
        wa_w = self._work_area.width()
        wa_h = self._work_area.height()

        # Settings button — top-right
        self.btn_settings.move(wa_w - self.btn_settings.width() - MARGIN, MARGIN)
        self.settings_panel.move(
            wa_w - self.settings_panel.width() - MARGIN,
            MARGIN + self.btn_settings.height() + 10,
        )

        # Lighting button — below settings button
        light_btn_y = MARGIN + self.btn_settings.height() + 10
        self.btn_lighting.move(wa_w - self.btn_lighting.width() - MARGIN, light_btn_y)
        self.lighting_panel.move(
            wa_w - self.lighting_panel.width() - MARGIN,
            light_btn_y + self.btn_lighting.height() + 10,
        )

        self.toolbar.move(
            (wa_w - self.toolbar.width()) // 2,
            wa_h - self.toolbar.height() - MARGIN,
        )
        self._size_grip.move(
            self.width() - self._size_grip.width(),
            self.height() - self._size_grip.height(),
        )

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_floating_elements()

