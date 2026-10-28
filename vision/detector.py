"""
Face Detection Component for ClassPulse AI.
MediaPipe Face Detection as the primary detector, with OpenCV Haar Cascade
as the robust secondary fallback, and adaptive low-light enhancement.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import os
import cv2
import numpy as np

from config import BASE_DIR, VISION_CONFIG

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
except Exception:
    MEDIAPIPE_AVAILABLE = False


@dataclass
class DetectionResult:
    """Represents a single detected face in a frame."""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height) in original frame coordinates
    confidence: float
    centroid: Tuple[int, int]         # (center_x, center_y)
    method: str                       # 'mediapipe', 'haar', or 'skin_contour_fallback'


class FaceDetector:
    """
    Multimodal face detector with MediaPipe BlazeFace as primary,
    OpenCV Haar Cascade as fallback, and low-light adaptive pre-processing.
    Uses a process-wide singleton for the MediaPipe engine to ensure stability on Windows.
    """

    _shared_mp_detector = None
    _mp_init_attempted: bool = False

    def __init__(
        self,
        min_confidence: float = VISION_CONFIG.get("min_detection_confidence", 0.65),
        scale_factor: float = 1.15,
        min_neighbors: int = 6,
        min_size: Tuple[int, int] = (35, 35),
    ):
        self.min_confidence = min_confidence
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

        self.models_dir = BASE_DIR / "data" / "models"
        self.blazeface_path = self.models_dir / "blaze_face_short_range.tflite"
        self.haar_path = self.models_dir / "haarcascade_frontalface_default.xml"
        self.haar_alt_path = self.models_dir / "haarcascade_frontalface_alt2.xml"

        self.haar_cascade = None

        self._init_mediapipe()
        self._init_haar()

    def _init_mediapipe(self) -> None:
        """Initializes the MediaPipe Tasks FaceDetector singleton."""
        if FaceDetector._shared_mp_detector is not None:
            self.mp_detector = FaceDetector._shared_mp_detector
            return

        if FaceDetector._mp_init_attempted:
            self.mp_detector = None
            return

        FaceDetector._mp_init_attempted = True

        if not MEDIAPIPE_AVAILABLE:
            self.mp_detector = None
            return

        if self.blazeface_path.exists():
            try:
                base_options = mp_python.BaseOptions(model_asset_path=str(self.blazeface_path))
                options = mp_vision.FaceDetectorOptions(
                    base_options=base_options,
                    min_detection_confidence=self.min_confidence,
                )
                FaceDetector._shared_mp_detector = mp_vision.FaceDetector.create_from_options(options)
                self.mp_detector = FaceDetector._shared_mp_detector
            except Exception as e:
                self.mp_detector = None
        else:
            self.mp_detector = None

    def _init_haar(self) -> None:
        """Initializes OpenCV Haar Cascade as fallback."""
        if self.haar_path.exists():
            self.haar_cascade = cv2.CascadeClassifier(str(self.haar_path))
        elif self.haar_alt_path.exists():
            self.haar_cascade = cv2.CascadeClassifier(str(self.haar_alt_path))
        else:
            default_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            if os.path.exists(default_path):
                self.haar_cascade = cv2.CascadeClassifier(default_path)

    def is_ready(self) -> bool:
        """Returns True if at least one detector backend is loaded."""
        return (self.mp_detector is not None) or (self.haar_cascade is not None and not self.haar_cascade.empty())

    def _enhance_low_light(self, frame: np.ndarray) -> np.ndarray:
        """Applies adaptive contrast enhancement (CLAHE) if the frame is underexposed/dark."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            mean_brightness = np.mean(gray)
            if mean_brightness < 55.0:
                # Underexposed frame: apply CLAHE in LAB color space
                if len(frame.shape) == 3:
                    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    cl = clahe.apply(l)
                    enhanced = cv2.merge((cl, a, b))
                    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
                else:
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    return clahe.apply(frame)
        except Exception:
            pass
        return frame

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        Executes face detection:
        1. MediaPipe Tasks FaceDetector (Primary)
        2. OpenCV Haar Cascade (Secondary Fallback)
        3. Adaptive Skin-Contour (Tertiary Fallback for Avatars/Simulations)
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return []

        # Enhance low light if needed
        proc_frame = self._enhance_low_light(frame)
        h, w = proc_frame.shape[:2]

        # -------------------------------------------------------------
        # 1. Primary: MediaPipe FaceDetector
        # -------------------------------------------------------------
        if self.mp_detector is not None:
            try:
                rgb_frame = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB) if len(proc_frame.shape) == 3 else proc_frame
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                mp_result = self.mp_detector.detect(mp_img)

                if mp_result and mp_result.detections:
                    results: List[DetectionResult] = []
                    for det in mp_result.detections:
                        bb = det.bounding_box
                        x = max(0, int(bb.origin_x))
                        y = max(0, int(bb.origin_y))
                        bw = min(w - x, int(bb.width))
                        bh = min(h - y, int(bb.height))

                        if bw < self.min_size[0] or bh < self.min_size[1]:
                            continue

                        cx = x + bw // 2
                        cy = y + bh // 2
                        score = float(det.categories[0].score) if det.categories else 0.85
                        if score < self.min_confidence:
                            continue

                        results.append(
                            DetectionResult(
                                bbox=(x, y, bw, bh),
                                confidence=score,
                                centroid=(cx, cy),
                                method="mediapipe",
                            )
                        )

                    if results:
                        return self._suppress_overlapping_detections(results)
            except Exception as e:
                # Graceful fallback without stack traces
                pass

        # -------------------------------------------------------------
        # 2. Secondary Fallback: OpenCV Haar Cascade
        # -------------------------------------------------------------
        if self.haar_cascade is not None and not self.haar_cascade.empty():
            try:
                gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY) if len(proc_frame.shape) == 3 else proc_frame
                gray = cv2.equalizeHist(gray)

                faces = self.haar_cascade.detectMultiScale(
                    gray,
                    scaleFactor=self.scale_factor,
                    minNeighbors=self.min_neighbors,
                    minSize=self.min_size,
                    flags=cv2.CASCADE_SCALE_IMAGE,
                )

                if len(faces) > 0:
                    results = []
                    for (x, y, bw, bh) in faces:
                        cx = int(x + bw / 2)
                        cy = int(y + bh / 2)
                        results.append(
                            DetectionResult(
                                bbox=(int(x), int(y), int(bw), int(bh)),
                                confidence=0.85,
                                centroid=(cx, cy),
                                method="haar",
                            )
                        )
                    return self._suppress_overlapping_detections(results)
            except Exception:
                pass

        # -------------------------------------------------------------
        # 3. Tertiary Fallback: Skin-contour / avatar detection
        # -------------------------------------------------------------
        fallback_res = self._detect_synthetic_face_fallback(proc_frame)
        return self._suppress_overlapping_detections(fallback_res)

    @staticmethod
    def _suppress_overlapping_detections(
        detections: List[DetectionResult], iou_threshold: float = 0.35, min_dist: float = 35.0
    ) -> List[DetectionResult]:
        """Suppresses duplicate detections of the same face in the same frame."""
        if len(detections) <= 1:
            return detections
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept: List[DetectionResult] = []
        for det in sorted_dets:
            duplicate = False
            for k in kept:
                cx1, cy1 = det.centroid
                cx2, cy2 = k.centroid
                dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
                x1, y1, w1, h1 = det.bbox
                x2, y2, w2, h2 = k.bbox
                xi1, yi1 = max(x1, x2), max(y1, y2)
                xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
                inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                union = w1 * h1 + w2 * h2 - inter
                iou = (inter / union) if union > 0 else 0.0
                if dist < min_dist or iou > iou_threshold:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(det)
        return kept

    def _detect_synthetic_face_fallback(self, frame: np.ndarray) -> List[DetectionResult]:
        """Fallback detector identifying face-like contours in synthetic simulations/avatars."""
        if len(frame.shape) != 3:
            return []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 15, 60]), np.array([28, 255, 255]))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fallback_results: List[DetectionResult] = []

        min_area = 700
        max_area = frame.shape[0] * frame.shape[1] * 0.45

        for c in contours:
            area = cv2.contourArea(c)
            if min_area <= area <= max_area:
                x, y, bw, bh = cv2.boundingRect(c)
                aspect_ratio = float(bw) / float(bh)
                if 0.5 <= aspect_ratio <= 2.0:
                    cx = int(x + bw / 2)
                    cy = int(y + bh / 2)
                    fallback_results.append(
                        DetectionResult(
                            bbox=(int(x), int(y), int(bw), int(bh)),
                            confidence=0.75,
                            centroid=(cx, cy),
                            method="skin_contour_fallback",
                        )
                    )

        fallback_results.sort(key=lambda r: r.centroid[0])
        return fallback_results

    @staticmethod
    def draw_detections(
        frame: np.ndarray,
        detections: List[DetectionResult],
        color: Tuple[int, int, int] = (0, 255, 128),
    ) -> np.ndarray:
        """Renders bounding boxes and detection centroids onto a copy of the frame."""
        if frame is None or frame.size == 0:
            return frame

        annotated = frame.copy()
        for idx, det in enumerate(detections):
            x, y, bw, bh = det.bbox
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)
            cv2.circle(annotated, det.centroid, 4, (0, 0, 255), -1)
            label = f"Face {idx + 1} ({det.method})"
            cv2.putText(
                annotated,
                label,
                (x, max(y - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        return annotated
