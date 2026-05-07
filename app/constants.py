"""
VideoFIT — UI Constants & Shared Styles
"""

MARGIN = 20

GLASS_STYLE = """
    QFrame#GlassPanel {
        background-color: rgba(10, 10, 10, 210);
        border: 1px solid rgba(255, 255, 255, 45);
        border-radius: 12px;
    }
    QLabel {
        background: transparent;
        color: rgba(255, 255, 255, 220);
        font-size: 13px;
        font-weight: bold;
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
        background-color: rgba(255, 255, 255, 20);
        border: 1px solid rgba(255, 255, 255, 50);
    }
    QPushButton:pressed { background-color: rgba(255, 255, 255, 8); }
    QPushButton:checked {
        background-color: rgba(255, 255, 255, 30);
        border: 1px solid rgba(255, 255, 255, 80);
        color: white;
    }
    QLineEdit, QComboBox {
        background-color: rgba(0, 0, 0, 160);
        border: 1px solid rgba(255, 255, 255, 35);
        border-radius: 5px;
        padding: 5px 7px;
        color: rgba(255, 255, 255, 230);
        font-size: 13px;
        selection-background-color: rgba(255, 255, 255, 50);
    }
    QComboBox::drop-down { border: none; background: transparent; }
    QComboBox QAbstractItemView {
        background-color: rgba(20, 20, 20, 240);
        border: 1px solid rgba(255, 255, 255, 40);
        color: rgba(255, 255, 255, 220);
        selection-background-color: rgba(255, 255, 255, 40);
    }
    QCheckBox {
        background: transparent;
        color: rgba(255, 255, 255, 220);
        font-size: 13px;
        font-weight: bold;
    }
    QCheckBox::indicator {
        width: 15px;
        height: 15px;
        border: 1px solid rgba(255, 255, 255, 60);
        border-radius: 3px;
        background: rgba(255, 255, 255, 15);
    }
    QCheckBox::indicator:checked { background: rgba(255, 255, 255, 150); }
    QSlider { background: transparent; }
    QSlider::groove:horizontal {
        height: 5px;
        background: rgba(255, 255, 255, 40);
        border-radius: 2px;
    }
    QSlider::sub-page:horizontal {
        background: rgba(255, 255, 255, 180);
        border-radius: 2px;
    }
    QSlider::handle:horizontal {
        background: white;
        border: none;
        width: 16px;
        height: 16px;
        margin: -6px 0;
        border-radius: 8px;
    }
    QSlider::handle:horizontal:hover { background: rgba(200, 200, 200, 255); }
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
        background-color: rgba(15, 15, 15, 180);
        border: 1px solid rgba(255, 255, 255, 25);
        padding: 0px;
    }
    QPushButton:checked { background-color: rgba(255, 255, 255, 30); }
"""
