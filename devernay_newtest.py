import sys
import time
import cv2
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as ndi

from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget,
                               QFileDialog, QGraphicsView, QGraphicsScene,
                               QGraphicsItem, QGraphicsPixmapItem, QPushButton, QLabel, QHBoxLayout)
from PySide6.QtCore import QThread, Signal, Qt, QRectF, QPointF
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor

# =====================================================================
# 1. GPU BILATERAL FILTER KERNEL
# =====================================================================
bilateral_kernel_code = r'''
extern "C" __global__
void bilateral_kernel(
    const float* input, float* output, 
    int width, int height, 
    float sigma_s, float sigma_r, int radius) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float center_val = input[y * width + x];
    float sum = 0.0f;
    float norm = 0.0f;

    float var_s = 2.0f * sigma_s * sigma_s;
    float var_r = 2.0f * sigma_r * sigma_r;

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            int nx = max(0, min(width - 1, x + dx));
            int ny = max(0, min(height - 1, y + dy));

            float val = input[ny * width + nx];
            float space_dist2 = (float)(dx * dx + dy * dy);
            float color_dist = val - center_val;

            float weight = expf(-(space_dist2 / var_s) - ((color_dist * color_dist) / var_r));

            sum += val * weight;
            norm += weight;
        }
    }
    output[y * width + x] = sum / norm;
}
'''
fused_bilateral = cp.RawKernel(bilateral_kernel_code, 'bilateral_kernel', options=('-use_fast_math',))

# =====================================================================
# 2. FUSED SOBEL GRADIENT KERNEL
# =====================================================================
sobel_kernel_code = r'''
extern "C" __global__
void fused_sobel_kernel(
    const float* input, float* Gx, float* Gy, float* G_mag, 
    int width, int height) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) return;

    float p00 = input[(y-1)*width + (x-1)];
    float p01 = input[(y-1)*width + x];
    float p02 = input[(y-1)*width + (x+1)];
    float p10 = input[y*width + (x-1)];
    float p12 = input[y*width + (x+1)];
    float p20 = input[(y+1)*width + (x-1)];
    float p21 = input[(y+1)*width + x];
    float p22 = input[(y+1)*width + (x+1)];

    float gx = (p02 - p00) + 2.0f*(p12 - p10) + (p22 - p20);
    float gy = (p20 - p00) + 2.0f*(p21 - p01) + (p22 - p02);

    int idx = y * width + x;
    Gx[idx] = gx;
    Gy[idx] = gy;
    // Optimized: Replaced hypotf with raw hardware-accelerated sqrtf
    G_mag[idx] = sqrtf(gx * gx + gy * gy); 
}
'''
fused_sobel = cp.RawKernel(sobel_kernel_code, 'fused_sobel_kernel', options=('-use_fast_math',))

# =====================================================================
# 3. FUSED DEVERNAY CUDA KERNEL (WITH CURVATURE REJECTION)
# =====================================================================
devernay_kernel_code = r'''
extern "C" __global__
void fast_devernay_kernel(
    const float* Gx, const float* Gy, const float* G_mag,
    float* out_x, float* out_y, bool* out_mask,
    int width, int height, float low_thresh, float min_curvature) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) return;

    int idx = y * width + x;
    float mag0 = G_mag[idx];

    if (mag0 < low_thresh) {
        out_mask[idx] = false;
        return;
    }

    float gx = Gx[idx];
    float gy = Gy[idx];

    float abs_gx = abs(gx);
    float abs_gy = abs(gy);

    float mag_plus = 0.0;
    float mag_minus = 0.0;
    float dx_norm = 0.0;
    float dy_norm = 0.0;

    if (abs_gx > abs_gy) {
        float weight = gy / (gx + 1e-6f); 
        mag_plus  = G_mag[y * width + (x + 1)]; 
        mag_minus = G_mag[y * width + (x - 1)];
        dx_norm = 1.0f; 
        dy_norm = weight;
    } else {
        float weight = gx / (gy + 1e-6f);
        mag_plus  = G_mag[(y + 1) * width + x];
        mag_minus = G_mag[(y - 1) * width + x];
        dx_norm = weight;
        dy_norm = 1.0f;
    }

    if (mag0 >= mag_plus && mag0 >= mag_minus) {
        // SECOND DERIVATIVE (CURVATURE) CHECK
        float curvature = abs(mag_minus - 2.0f * mag0 + mag_plus);
        if (curvature < min_curvature) {
            out_mask[idx] = false;
            return;
        }

        // Sub-pixel parabola math
        float denom = 2.0f * (mag_minus - 2.0f * mag0 + mag_plus) + 1e-6f;
        float delta = (mag_minus - mag_plus) / denom;
        delta = fmaxf(-0.5f, fminf(delta, 0.5f));

        out_x[idx] = (float)x + (delta * dx_norm);
        out_y[idx] = (float)y + (delta * dy_norm);
        out_mask[idx] = true;
    } else {
        out_mask[idx] = false;
    }
}
'''
fused_devernay = cp.RawKernel(devernay_kernel_code, 'fast_devernay_kernel', options=('-use_fast_math',))


# =====================================================================
# 4. FAST VECTOR RENDERER
# =====================================================================
class SubpixelOverlay(QGraphicsItem):
    def __init__(self, qpoints, width, height):
        super().__init__()
        self.qpoints = qpoints
        self.rect = QRectF(0, 0, width, height)

    def boundingRect(self):
        return self.rect

    def paint(self, painter, option, widget):
        pen = QPen(QColor(255, 0, 0, 200))
        pen.setWidth(0)
        painter.setPen(pen)
        painter.drawPoints(self.qpoints)


# =====================================================================
# 5. INFINITE ZOOM VIEWPORT
# =====================================================================
class InteractiveView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    def wheelEvent(self, event):
        zoom_in_factor = 1.15
        zoom_out_factor = 1.0 / zoom_in_factor
        zoom_factor = zoom_in_factor if event.angleDelta().y() > 0 else zoom_out_factor
        self.scale(zoom_factor, zoom_factor)


# =====================================================================
# 6. OPTIMIZED GPU MATH ENGINE
# =====================================================================
class DevernayProcessor(QThread):
    finished_processing = Signal(np.ndarray, list, float)

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path
        self.structure_3x3 = cp.ones((3, 3), dtype=cp.int32)

    def run(self):
        img_bgr = cv2.imread(self.image_path)
        if img_bgr is None:
            print("Failed to load image.")
            return

        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        height, width = img_gray.shape

        cp.cuda.Stream.null.synchronize()
        start_time = time.perf_counter()

        cp_frame = cp.asarray(img_gray, dtype=cp.float32)

        block_dim = (16, 16)
        grid_dim = ((width + block_dim[0] - 1) // block_dim[0],
                    (height + block_dim[1] - 1) // block_dim[1])

        # --- STEP A: PURE PREPROCESSING ---
        f_min = cp_frame.min()
        f_max = cp_frame.max()
        cp_frame = (cp_frame - f_min) * (255.0 / (f_max - f_min + 1e-6))

        filtered_frame = cp.empty_like(cp_frame)
        fused_bilateral(
            grid_dim, block_dim,
            (cp_frame, filtered_frame, width, height, cp.float32(2.0), cp.float32(25.0), 2)
        )

        # --- STEP B: FUSED GRADIENTS ---
        Gx = cp.empty_like(cp_frame)
        Gy = cp.empty_like(cp_frame)
        G_mag = cp.empty_like(cp_frame)

        fused_sobel(
            grid_dim, block_dim,
            (filtered_frame, Gx, Gy, G_mag, width, height)
        )

        # --- STEP C: FUSED DEVERNAY (LOW THRESHOLD + CURVATURE FILTER) ---
        out_x = cp.empty_like(G_mag)
        out_y = cp.empty_like(G_mag)
        out_mask = cp.zeros_like(G_mag, dtype=cp.bool_)

        fused_devernay(
            grid_dim, block_dim,
            (Gx, Gy, G_mag, out_x, out_y, out_mask, width, height, cp.float32(15.0), cp.float32(2.0))
        )

        # --- STEP D: STRICT GPU HYSTERESIS ---
        closed_mask = ndi.binary_closing(out_mask, structure=self.structure_3x3)
        labeled_mask, num_features = ndi.label(closed_mask, structure=self.structure_3x3)

        if num_features > 0:
            component_sizes = cp.bincount(labeled_mask.ravel())

            high_thresh = 50.0
            strong_pixels = out_mask & (G_mag > high_thresh)
            strong_labels_present = cp.unique(labeled_mask[strong_pixels])

            has_strong_anchor = cp.zeros(num_features + 1, dtype=cp.bool_)
            has_strong_anchor[strong_labels_present] = True

            min_edge_length = 15
            valid_labels = (component_sizes >= min_edge_length) & has_strong_anchor
            valid_labels[0] = False

            clean_mask = out_mask & valid_labels[labeled_mask]
        else:
            clean_mask = out_mask

        # --- STEP E: EXTRACT DATA (Optimized PCIe Handover) ---
        # Stack into a single 2D array on the GPU
        valid_coords_gpu = cp.column_stack((out_x[clean_mask], out_y[clean_mask]))

        # Single block transfer to the CPU
        valid_coords_cpu = cp.asnumpy(valid_coords_gpu)

        cp.cuda.Stream.null.synchronize()
        compute_time_ms = (time.perf_counter() - start_time) * 1000

        # Unpack directly from the unified CPU array
        qpoints = [QPointF(x, y) for x, y in valid_coords_cpu]
        self.finished_processing.emit(img_gray, qpoints, compute_time_ms)


# =====================================================================
# 7. MAIN WINDOW
# =====================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pro Devernay | High-Precision Curvature Filter")
        self.resize(1200, 800)

        self.scene = QGraphicsScene()
        self.view = InteractiveView(self.scene)

        self.btn_load = QPushButton("Load Image (.bmp, .png, .jpg)")
        self.btn_load.clicked.connect(self.load_image)

        self.lbl_stats = QLabel("Compute Time: N/A")
        self.lbl_stats.setStyleSheet("font-weight: bold; font-size: 14px; color: #333;")

        top_bar = QHBoxLayout()
        top_bar.addWidget(self.btn_load)
        top_bar.addStretch()
        top_bar.addWidget(self.lbl_stats)

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_bar)
        main_layout.addWidget(self.view)

        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.bmp *.png *.jpg)")
        if file_path:
            self.lbl_stats.setText("Processing on GPU...")
            self.btn_load.setEnabled(False)

            self.worker = DevernayProcessor(file_path)
            self.worker.finished_processing.connect(self.display_results)
            self.worker.start()

    def display_results(self, img_array, qpoints, compute_time_ms):
        self.scene.clear()

        h, w = img_array.shape
        bytes_per_line = w
        qimg = QImage(img_array.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(qimg))
        self.scene.addItem(pixmap_item)

        overlay_item = SubpixelOverlay(qpoints, w, h)
        self.scene.addItem(overlay_item)

        self.lbl_stats.setText(f"GPU Compute Time: {compute_time_ms:.2f} ms | Edge Points: {len(qpoints)}")
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.btn_load.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())