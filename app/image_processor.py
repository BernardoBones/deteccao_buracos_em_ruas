"""OpenCV image filters for pre-detection and display."""

import cv2
import numpy as np


class ImageProcessor:
    """
    process_for_detection  → applies pre-detection filters (Gaussian, CLAHE, Morphological)
    process_for_display    → applies view-only filters (Canny edges, Grayscale)
    """

    def process_for_detection(self, frame: np.ndarray, filters: dict) -> np.ndarray:
        out = frame.copy()
        if filters.get("gaussian"):
            out = cv2.GaussianBlur(out, (5, 5), 0)
        if filters.get("clahe"):
            out = self._clahe(out)
        if filters.get("morphological"):
            out = self._morphological(out)
        return out

    def process_for_display(self, frame: np.ndarray, filters: dict) -> np.ndarray:
        out = frame.copy()
        if len(out.shape) == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        if filters.get("edges"):
            out = self._canny(out)
        if filters.get("grayscale"):
            out = self._grayscale(out)
        return out

    # ------------------------------------------------------------------ filters

    @staticmethod
    def _clahe(frame: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if frame.ndim == 3:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return clahe.apply(frame)

    @staticmethod
    def _morphological(frame: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        return cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def _canny(frame: np.ndarray) -> np.ndarray:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        edges = cv2.Canny(gray, 50, 150)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _grayscale(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return frame
