"""
VideoFIT — UI Constants & Shared Styles
"""

MARGIN = 20

GLASS_STYLE = """
    QFrame#GlassPanel {
        background-color: rgba(15, 15, 15, 210);
        border: 1px solid rgba(255, 255, 255, 25);
        border-radius: 12px;
    }
    QLabel {
        color: rgba(255, 255, 255, 220);
        font-weight: bold;
        background-color: transparent;
    }
    QPushButton {
        background-color: transparent;
        border: 1px solid transparent;
        border-radius: 6px;
        padding: 8px 12px;
        color: rgba(255, 255, 255, 220);
        font-size: 13px;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: rgba(255, 255, 255, 15);
        border: 1px solid rgba(255, 255, 255, 30);
    }
    QPushButton:pressed {
        background-color: rgba(255, 255, 255, 5);
    }
    QPushButton:checked {
        background-color: rgba(255, 255, 255, 20);
        border: 1px solid rgba(255, 255, 255, 60);
        color: white;
    }
    QToolButton {
        background-color: transparent;
        border: 1px solid transparent;
        border-radius: 6px;
        padding: 8px 12px;
        color: rgba(255, 255, 255, 220);
        font-size: 13px;
        font-weight: bold;
    }
    QToolButton:hover {
        background-color: rgba(255, 255, 255, 15);
        border: 1px solid rgba(255, 255, 255, 30);
    }
    QToolButton:pressed {
        background-color: rgba(255, 255, 255, 5);
    }
    QToolButton:checked {
        background-color: rgba(255, 255, 255, 20);
        border: 1px solid rgba(255, 255, 255, 60);
        color: white;
    }
    QToolButton::menu-button {
        border: none;
        border-left: 1px solid rgba(255, 255, 255, 20);
        border-top-right-radius: 6px;
        border-bottom-right-radius: 6px;
        width: 14px;
    }
    QToolButton::menu-arrow { image: none; }
    QLineEdit, QComboBox {
        background-color: rgba(0, 0, 0, 210);
        border: 1px solid rgba(255, 255, 255, 20);
        border-radius: 5px;
        padding: 5px;
        color: white;
    }
    QComboBox::drop-down { border: none; }
"""

TITLE_BAR_STYLE = """
    QFrame {
        background-color: rgba(15, 15, 15, 200);
        border-top-left-radius: 15px;
        border-top-right-radius: 15px;
        border-bottom: 1px solid rgba(255, 255, 255, 20);
    }
    QLabel {
        color: rgba(255, 255, 255, 200);
        font-weight: bold;
        font-size: 14px;
        padding-left: 10px;
    }
    QPushButton {
        background-color: transparent;
        border: none;
        border-radius: 5px;
        color: white;
        font-weight: bold;
    }
    QPushButton:hover { background-color: rgba(255, 255, 255, 30); }
    QPushButton#btnClose:hover { background-color: rgba(255, 50, 50, 200); }
"""

SETTINGS_BTN_STYLE = GLASS_STYLE + """
    QPushButton {
        border-radius: 22px;
        background-color: rgba(15, 15, 15, 210);
        border: 1px solid rgba(255, 255, 255, 25);
    }
    QPushButton:checked { background-color: rgba(255, 255, 255, 30); }
"""

