"""Main application window for the PotHole Detector."""

import cv2
import numpy as np
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QFileDialog, QMessageBox, QSizePolicy,
)

from app.detector import PotholeDetector
from app.image_processor import ImageProcessor
from app.video_thread import VideoThread
from app.widgets import ControlPanel, StatsWidget

# --------------------------------------------------------------------------- #
#  Dark theme stylesheet (Catppuccin Mocha palette)                           #
# --------------------------------------------------------------------------- #
DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 13px;
}

QPushButton {
    background-color: #45475a;
    color: #cdd6f4;
    border: none;
    border-radius: 6px;
    padding: 8px 12px;
    font-weight: bold;
}
QPushButton:hover   { background-color: #585b70; }
QPushButton:pressed { background-color: #313244; }
QPushButton:disabled { background-color: #313244; color: #6c7086; }

QPushButton#primaryBtn {
    background-color: #89b4fa;
    color: #1e1e2e;
}
QPushButton#primaryBtn:hover { background-color: #b4befe; }

QPushButton#dangerBtn {
    background-color: #f38ba8;
    color: #1e1e2e;
}
QPushButton#dangerBtn:hover  { background-color: #eba0ac; }
QPushButton#dangerBtn:disabled { background-color: #313244; color: #6c7086; }

QPushButton#toggleBtn {
    background-color: #313244;
    color: #cdd6f4;
    font-size: 11px;
    padding: 4px 10px;
}
QPushButton#toggleBtn:checked {
    background-color: #89b4fa;
    color: #1e1e2e;
}

QCheckBox { spacing: 8px; }
QCheckBox::indicator {
    width: 16px; height: 16px;
    border: 2px solid #6c7086;
    border-radius: 3px;
    background: #313244;
}
QCheckBox::indicator:checked {
    background-color: #89b4fa;
    border-color: #89b4fa;
}

QSlider::groove:horizontal {
    height: 6px;
    background: #45475a;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #89b4fa;
    width: 16px; height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
QSlider::sub-page:horizontal {
    background: #89b4fa;
    border-radius: 3px;
}

QGroupBox {
    border: 1px solid #45475a;
    border-radius: 8px;
    margin-top: 12px;
    padding: 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    color: #89b4fa;
    font-weight: bold;
}

QProgressBar {
    background: #313244;
    border-radius: 4px;
    height: 10px;
}
QProgressBar::chunk {
    background: #a6e3a1;
    border-radius: 4px;
}

QStatusBar {
    background-color: #181825;
    color: #6c7086;
}

QLabel#imageLabel {
    background-color: #11111b;
    border: 2px solid #313244;
    border-radius: 8px;
    color: #6c7086;
}
"""


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PotHole Detector — Detecção de Irregularidades em Vias")
        self.setMinimumSize(1100, 680)
        self.resize(1280, 780)

        # Core objects (logic wired in next batch)
        self.detector = PotholeDetector()
        self.processor = ImageProcessor()
        self.video_thread: VideoThread | None = None
        self._current_frame: np.ndarray | None = None
        self._current_annotated: np.ndarray | None = None

        self._build_ui()
        self.setStyleSheet(DARK_STYLE)
        self.statusBar().showMessage("Pronto. Abra uma imagem ou vídeo para começar.")

    # ------------------------------------------------------------------ layout
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setSpacing(8)
        root.setContentsMargins(8, 8, 8, 8)

        # Left: control panel
        self.ctrl = ControlPanel()
        root.addWidget(self.ctrl)

        # Center: image display
        center = QVBoxLayout()
        center.setSpacing(6)

        header = QHBoxLayout()
        view_title = QLabel("VISUALIZAÇÃO")
        view_title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        view_title.setStyleSheet("color: #89b4fa;")

        self.btn_toggle = QPushButton("Mostrar Original")
        self.btn_toggle.setObjectName("toggleBtn")
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.setEnabled(False)

        header.addWidget(view_title)
        header.addStretch()
        header.addWidget(self.btn_toggle)
        center.addLayout(header)

        self.image_label = QLabel()
        self.image_label.setObjectName("imageLabel")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setText(
            "Nenhuma imagem carregada\n\n"
            "Abra uma imagem, vídeo ou ative a webcam"
        )
        self.image_label.setFont(QFont("Segoe UI", 13))
        center.addWidget(self.image_label)
        root.addLayout(center, stretch=1)

        # Right: stats panel
        self.stats = StatsWidget()
        self.stats.setFixedWidth(195)
        root.addWidget(self.stats)

        # Status bar
        self.lbl_fps = QLabel("FPS: --")
        self.lbl_fps.setStyleSheet("color: #a6e3a1; margin-right: 8px;")
        self.statusBar().addPermanentWidget(self.lbl_fps)

    # ------------------------------------------------------------------ resize
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._current_annotated is not None:
            frame = (
                self._current_frame
                if self.btn_toggle.isChecked()
                else self._current_annotated
            )
            self._display_frame(frame)

    # ------------------------------------------------------------------ display helper
    def _display_frame(self, frame: np.ndarray):
        lw = max(self.image_label.width(), 1)
        lh = max(self.image_label.height(), 1)
        h, w = frame.shape[:2]
        scale = min(lw / w, lh / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        h2, w2, ch = rgb.shape
        qimg = QImage(rgb.data, w2, h2, ch * w2, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))

    def _set_status(self, msg: str):
        self.statusBar().showMessage(msg)

    def closeEvent(self, event):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
        super().closeEvent(event)
