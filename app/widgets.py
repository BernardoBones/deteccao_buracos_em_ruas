"""Reusable PyQt5 widgets for the PotHole Detector GUI."""

from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QSlider,
    QCheckBox, QGroupBox, QProgressBar,
)


class ControlPanel(QWidget):
    """Left panel with input controls, model loading and filter options."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(220)
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(8, 8, 8, 8)

        title = QLabel("CONTROLES")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        title.setStyleSheet("color: #89b4fa; margin-bottom: 4px;")
        layout.addWidget(title)

        # --- Input group ---
        input_group = QGroupBox("Entrada")
        input_layout = QVBoxLayout(input_group)

        self.btn_image = QPushButton("Abrir Imagem")
        self.btn_image.setObjectName("primaryBtn")
        self.btn_video = QPushButton("Abrir Vídeo")
        self.btn_webcam = QPushButton("Webcam")
        self.btn_stop = QPushButton("Parar")
        self.btn_stop.setObjectName("dangerBtn")
        self.btn_stop.setEnabled(False)

        for btn in (self.btn_image, self.btn_video, self.btn_webcam, self.btn_stop):
            input_layout.addWidget(btn)
        layout.addWidget(input_group)

        # --- Model group ---
        model_group = QGroupBox("Modelo YOLO")
        model_layout = QVBoxLayout(model_group)
        self.btn_load_model = QPushButton("Carregar Modelo")
        self.lbl_model = QLabel("Padrão: yolov8n")
        self.lbl_model.setWordWrap(True)
        self.lbl_model.setStyleSheet("color: #a6e3a1; font-size: 11px;")
        model_layout.addWidget(self.btn_load_model)
        model_layout.addWidget(self.lbl_model)
        layout.addWidget(model_group)

        # --- Filters group ---
        filter_group = QGroupBox("Filtros de Pré-processamento")
        filter_layout = QVBoxLayout(filter_group)

        self.chk_gaussian = QCheckBox("Suavização Gaussiana")
        self.chk_clahe = QCheckBox("Realce CLAHE")
        self.chk_morphological = QCheckBox("Morfológico")
        self.chk_edges = QCheckBox("Detecção de Bordas")
        self.chk_grayscale = QCheckBox("Escala de Cinza")

        for chk in (
            self.chk_gaussian, self.chk_clahe, self.chk_morphological,
            self.chk_edges, self.chk_grayscale,
        ):
            filter_layout.addWidget(chk)
        layout.addWidget(filter_group)

        # --- Confidence threshold ---
        conf_group = QGroupBox("Limiar de Confiança")
        conf_layout = QVBoxLayout(conf_group)
        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(1, 99)
        self.slider_conf.setValue(25)
        self.lbl_conf = QLabel("0.25")
        self.lbl_conf.setAlignment(Qt.AlignCenter)
        self.lbl_conf.setStyleSheet("font-weight: bold; color: #89b4fa;")
        conf_layout.addWidget(self.slider_conf)
        conf_layout.addWidget(self.lbl_conf)
        layout.addWidget(conf_group)

        # --- Export ---
        export_group = QGroupBox("Exportar")
        export_layout = QVBoxLayout(export_group)
        self.btn_save = QPushButton("Salvar Imagem")
        self.btn_save.setEnabled(False)
        export_layout.addWidget(self.btn_save)
        layout.addWidget(export_group)

        layout.addStretch()

    def get_active_filters(self) -> dict:
        return {
            "gaussian":      self.chk_gaussian.isChecked(),
            "clahe":         self.chk_clahe.isChecked(),
            "morphological": self.chk_morphological.isChecked(),
            "edges":         self.chk_edges.isChecked(),
            "grayscale":     self.chk_grayscale.isChecked(),
        }


class StatsWidget(QWidget):
    """Right panel displaying live detection statistics."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)

        title = QLabel("ESTATÍSTICAS")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        title.setStyleSheet("color: #89b4fa; margin-bottom: 4px;")
        layout.addWidget(title)

        total_group = QGroupBox("Total de Detecções")
        total_layout = QVBoxLayout(total_group)
        self._total_lbl = QLabel("0")
        self._total_lbl.setAlignment(Qt.AlignCenter)
        self._total_lbl.setFont(QFont("Segoe UI", 28, QFont.Bold))
        self._total_lbl.setStyleSheet("color: #cdd6f4;")
        total_layout.addWidget(self._total_lbl)
        layout.addWidget(total_group)

        sev_group = QGroupBox("Severidade")
        sev_layout = QVBoxLayout(sev_group)
        self._severo_lbl = QLabel("Severo:    0")
        self._moderado_lbl = QLabel("Moderado:  0")
        self._leve_lbl = QLabel("Leve:      0")
        self._severo_lbl.setStyleSheet("color: #f38ba8; font-weight: bold;")
        self._moderado_lbl.setStyleSheet("color: #fab387; font-weight: bold;")
        self._leve_lbl.setStyleSheet("color: #a6e3a1; font-weight: bold;")
        for lbl in (self._severo_lbl, self._moderado_lbl, self._leve_lbl):
            lbl.setFont(QFont("Consolas", 11))
            sev_layout.addWidget(lbl)
        layout.addWidget(sev_group)

        conf_group = QGroupBox("Confiança Média")
        conf_layout = QVBoxLayout(conf_group)
        self._conf_avg_lbl = QLabel("--")
        self._conf_avg_lbl.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self._conf_avg_lbl.setAlignment(Qt.AlignCenter)
        conf_layout.addWidget(self._conf_avg_lbl)
        self._conf_bar = QProgressBar()
        self._conf_bar.setRange(0, 100)
        self._conf_bar.setValue(0)
        self._conf_bar.setTextVisible(False)
        self._conf_bar.setStyleSheet(
            "QProgressBar { background: #313244; border-radius: 4px; height: 10px; }"
            "QProgressBar::chunk { background: #a6e3a1; border-radius: 4px; }"
        )
        conf_layout.addWidget(self._conf_bar)
        layout.addWidget(conf_group)

        layout.addStretch()

    @pyqtSlot(list)
    def update_stats(self, detections: list):
        total = len(detections)
        severo = sum(1 for d in detections if d["severity"] == "Severo")
        moderado = sum(1 for d in detections if d["severity"] == "Moderado")
        leve = sum(1 for d in detections if d["severity"] == "Leve")

        self._total_lbl.setText(str(total))
        self._severo_lbl.setText(f"Severo:    {severo}")
        self._moderado_lbl.setText(f"Moderado:  {moderado}")
        self._leve_lbl.setText(f"Leve:      {leve}")

        if detections:
            avg_conf = sum(d["confidence"] for d in detections) / total
            self._conf_avg_lbl.setText(f"{avg_conf:.1%}")
            self._conf_bar.setValue(int(avg_conf * 100))
        else:
            self._conf_avg_lbl.setText("--")
            self._conf_bar.setValue(0)
