import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QLabel, QComboBox,
                               QFileDialog, QSlider, QGroupBox, QFormLayout)
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap


class EdgeDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Industrial Edge Detection: EPI & DIA")
        self.resize(1000, 700)

        self.current_image_path = None
        self.original_img = None

        self.init_ui()

    def init_ui(self):
        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- Left Panel: Controls ---
        control_panel = QWidget()
        control_panel.setFixedWidth(300)
        control_layout = QVBoxLayout(control_panel)

        # Load Button
        self.btn_load = QPushButton("Load Image")
        self.btn_load.clicked.connect(self.load_image)
        control_layout.addWidget(self.btn_load)

        # Image Type Selector
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["DIA (Backlight)", "EPI (Toplight)", "EPI + DIA (Mixed)"])
        self.mode_selector.currentIndexChanged.connect(self.process_image)
        control_layout.addWidget(QLabel("Select Image Type:"))
        control_layout.addWidget(self.mode_selector)

        # Tuning Sliders Group
        slider_group = QGroupBox("Live Tuning (Canny Thresholds)")
        slider_layout = QFormLayout(slider_group)

        self.slider_min = QSlider(Qt.Horizontal)
        self.slider_min.setRange(0, 255)
        self.slider_min.setValue(50)
        self.slider_min.valueChanged.connect(self.process_image)

        self.slider_max = QSlider(Qt.Horizontal)
        self.slider_max.setRange(0, 255)
        self.slider_max.setValue(150)
        self.slider_max.valueChanged.connect(self.process_image)

        slider_layout.addRow("Min Threshold:", self.slider_min)
        slider_layout.addRow("Max Threshold:", self.slider_max)
        control_layout.addWidget(slider_group)

        control_layout.addStretch()
        main_layout.addWidget(control_panel)

        # --- Right Panel: Image Display ---
        self.image_label = QLabel("Load an image to begin.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #222; color: #fff;")
        main_layout.addWidget(self.image_label, 1)  # 1 is the stretch factor

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.bmp)")
        if file_name:
            self.current_image_path = file_name
            self.original_img = cv2.imread(file_name)
            self.process_image()

    def process_image(self):
        if self.original_img is None:
            return

        mode = self.mode_selector.currentText()
        min_thresh = self.slider_min.value()
        max_thresh = self.slider_max.value()

        edges = None

        # 1. DIA Logic (Resilient to dust/scratches on glass)
        if "DIA" in mode and "Mixed" not in mode:
            # Extract Green channel for best contrast
            b, g, r = cv2.split(self.original_img)

            # Otsu's thresholding
            _, binary = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Morphological Opening to remove small noise/scratches
            kernel = np.ones((5, 5), np.uint8)
            clean_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

            edges = cv2.Canny(clean_mask, min_thresh, max_thresh)

        # 2. EPI Logic (Resilient to metal surface texture)
        elif "EPI" in mode and "Mixed" not in mode:
            gray = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)

            # Bilateral filter removes texture noise but keeps sharp edges
            smooth = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

            edges = cv2.Canny(smooth, min_thresh, max_thresh)

        # 3. Mixed EPI+DIA Logic
        else:
            # For the mixed image, we apply a heavier bilateral filter due to flaring
            gray = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)
            smooth = cv2.bilateralFilter(gray, d=11, sigmaColor=100, sigmaSpace=100)
            edges = cv2.Canny(smooth, min_thresh, max_thresh)

        self.display_image(edges)

    def display_image(self, img_array):
        # Convert OpenCV grayscale to QImage
        height, width = img_array.shape
        bytes_per_line = width

        q_img = QImage(img_array.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)

        # Scale pixmap to fit the label while keeping aspect ratio
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        # Ensure image resizes dynamically when the window is resized
        super().resizeEvent(event)
        self.process_image()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Optional: Set a dark theme style for a more industrial look
    app.setStyle("Fusion")

    window = EdgeDetectorApp()
    window.show()
    sys.exit(app.exec())