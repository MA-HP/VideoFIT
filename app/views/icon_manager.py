# python
import os

from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPen, QPixmap, QGuiApplication
from PySide6.QtSvg import QSvgRenderer


class IconManager:
    """Utility for loading icons with automatic text-based fallbacks.
    Rasterise SVG into a QPixmap of the requested size (handles high-DPI)."""

    @staticmethod
    def get_icon(name: str, fallback_text: str, size: int = 64) -> QIcon:
        path = f"icons/{name}.svg"

        # get device pixel ratio for high-DPI screens
        screen = QGuiApplication.primaryScreen()
        dpr = float(screen.devicePixelRatio()) if screen else 1.0

        # content tuning
        Y_OFFSET_FACTOR = 0.1  # move content down by this fraction of the pixmap height
        RIGHT_PADDING_FACTOR = 0  # leave some space at the right of the icon

        # logical (device-independent) size and physical size
        logical_w = max(1, int(size))
        logical_h = max(1, int(size))
        phys_w = max(1, int(logical_w * dpr))
        phys_h = max(1, int(logical_h * dpr))

        pixmap = QPixmap(phys_w, phys_h)
        # set DPR before painting so painter uses logical coordinates
        pixmap.setDevicePixelRatio(dpr)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        right_pad = int(logical_w * RIGHT_PADDING_FACTOR)
        y_offset = int(logical_h * Y_OFFSET_FACTOR)

        logical_rect = QRect(0, 0, logical_w, logical_h)

        if os.path.exists(path):
            renderer = QSvgRenderer(path)
            if renderer.isValid():
                # compute available area in logical coords (respect right padding and vertical offset)
                available_w = max(1, logical_w - right_pad)
                available_h = max(1, logical_h - y_offset)

                svg_size = renderer.defaultSize()
                if svg_size.width() > 0 and svg_size.height() > 0:
                    svg_w = svg_size.width()
                    svg_h = svg_size.height()
                    scale = min(available_w / svg_w, available_h / svg_h)
                    target_w = max(1, int(svg_w * scale))
                    target_h = max(1, int(svg_h * scale))
                else:
                    target_w = available_w
                    target_h = available_h

                # center the rendered SVG inside the available area, apply vertical offset
                x = (available_w - target_w) // 2
                y = y_offset + (available_h - target_h) // 2
                render_rect = QRect(x, y, target_w, target_h)

                renderer.render(painter, render_rect)
            else:
                # fallback to text if SVG fails to load
                painter.setPen(QPen(QColor(255, 255, 255, 220), 1))
                font = QFont("Segoe UI", max(1, int(size * 0.5)), QFont.Bold)
                painter.setFont(font)
                text_rect = logical_rect.adjusted(0, y_offset, -right_pad, 0)
                painter.drawText(text_rect, Qt.AlignCenter, fallback_text)
        else:
            painter.setPen(QPen(QColor(255, 255, 255, 220), 1))
            font = QFont("Segoe UI", max(1, int(size * 0.5)), QFont.Bold)
            painter.setFont(font)
            text_rect = logical_rect.adjusted(0, y_offset, -right_pad, 0)
            painter.drawText(text_rect, Qt.AlignCenter, fallback_text)

        painter.end()

        return QIcon(pixmap)
