"""YOLOv8-based pothole detector with severity classification."""

import cv2
import numpy as np
from ultralytics import YOLO

# BGR colors per severity level
_COLORS = {
    "Severo":   (0,  50, 220),   # red
    "Moderado": (0, 140, 255),   # orange
    "Leve":     (80, 210,  80),  # green
}


class PotholeDetector:
    DEFAULT_MODEL = "yolov8n.pt"

    def __init__(self):
        self.conf_threshold: float = 0.25
        self._model: YOLO | None = None
        self._load_default()

    # ------------------------------------------------------------------ model

    def _load_default(self):
        try:
            self._model = YOLO(self.DEFAULT_MODEL)
        except Exception:
            self._model = None

    def load_model(self, path: str) -> bool:
        try:
            self._model = YOLO(path)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------ inference

    def detect(self, frame: np.ndarray) -> list[dict]:
        if self._model is None or frame is None:
            return []

        results = self._model.predict(
            frame,
            conf=self.conf_threshold,
            device="cpu",
            verbose=False,
        )

        h, w = frame.shape[:2]
        total_px = w * h or 1
        detections: list[dict] = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                conf = float(box.conf[0])
                area_ratio = (x2 - x1) * (y2 - y1) / total_px
                detections.append({
                    "bbox":       (x1, y1, x2, y2),
                    "confidence": conf,
                    "severity":   self._classify(area_ratio),
                })

        return detections

    # ------------------------------------------------------------------ drawing

    def draw_detections(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        if frame is None:
            return frame
        out = frame.copy()
        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            sev  = det["severity"]
            conf = det["confidence"]
            color = _COLORS.get(sev, (200, 200, 200))

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            label = f"{sev} {conf:.0%}"
            font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            (tw, th), bl = cv2.getTextSize(label, font, scale, thick)
            ty = max(y1 - bl - 2, th + bl)
            cv2.rectangle(out, (x1, ty - th - bl), (x1 + tw + 4, ty), color, -1)
            cv2.putText(out, label, (x1 + 2, ty - bl), font, scale, (0, 0, 0), thick, cv2.LINE_AA)

        return out

    # ------------------------------------------------------------------ helper

    @staticmethod
    def _classify(area_ratio: float) -> str:
        if area_ratio >= 0.04:
            return "Severo"
        if area_ratio >= 0.01:
            return "Moderado"
        return "Leve"
