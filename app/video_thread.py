"""QThread that reads video/webcam frames and runs YOLOv8 detection."""

import time

import cv2
from PyQt5.QtCore import QThread, pyqtSignal


class VideoThread(QThread):
    # original frame, detection-processed frame, list[dict] detections
    frame_ready = pyqtSignal(object, object, object)
    fps_updated = pyqtSignal(float)

    def __init__(self, source, detector, processor, filters: dict):
        super().__init__()
        self._source   = source
        self._detector = detector
        self._processor = processor
        self._filters  = dict(filters)
        self._running  = False

    def update_filters(self, filters: dict):
        self._filters = dict(filters)

    def stop(self):
        self._running = False
        self.wait(500)

    def run(self):
        cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            return

        self._running = True
        t0 = time.monotonic()
        frame_count = 0

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    break

                filters    = dict(self._filters)
                det_frame  = self._processor.process_for_detection(frame, filters)
                detections = self._detector.detect(det_frame)
                self.frame_ready.emit(frame.copy(), det_frame, detections)

                frame_count += 1
                elapsed = time.monotonic() - t0
                if elapsed >= 1.0:
                    self.fps_updated.emit(frame_count / elapsed)
                    frame_count = 0
                    t0 = time.monotonic()
        finally:
            cap.release()
            self._running = False
