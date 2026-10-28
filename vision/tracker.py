"""
Temporary Face Tracker for ClassPulse AI.
Maintains anonymous participant identifiers across video frames using
a hybrid Centroid Euclidean Distance and IoU (Intersection over Union) association.
Handles temporary occlusions, overlapping faces, and face re-entry.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2

from vision.detector import DetectionResult
from config import VISION_CONFIG


@dataclass
class TrackedParticipant:
    """Represents an anonymously tracked participant across video frames."""
    tracking_id: str
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[int, int]
    first_seen_timestamp: float
    last_seen_timestamp: float
    disappeared_frames: int = 0
    recent_movement: float = 0.0          # Movement displacement in pixels between latest frames
    accumulated_movement: float = 0.0     # Total accumulated pixel movement
    trajectory: List[Tuple[float, Tuple[int, int]]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "tracking_id": self.tracking_id,
            "bbox": self.bbox,
            "centroid": self.centroid,
            "first_seen_timestamp": round(self.first_seen_timestamp, 2),
            "last_seen_timestamp": round(self.last_seen_timestamp, 2),
            "recent_movement": round(self.recent_movement, 2),
            "accumulated_movement": round(self.accumulated_movement, 2),
        }


class FaceTracker:
    """
    Associates face detections across frames using a hybrid IoU and Euclidean centroid tracker.
    Prevents assigning new IDs on every frame, tolerates momentary occlusions,
    and supports graceful participant re-entry.
    """

    def __init__(
        self,
        max_disappeared: int = VISION_CONFIG.get("tracker_max_disappeared", 25),
        max_distance: float = VISION_CONFIG.get("tracker_max_distance", 105.0),
        min_iou_threshold: float = 0.15,
        reenter_tolerance: int = 25,
        reentry_window_seconds: float = VISION_CONFIG.get("reentry_window_seconds", 12.0),
        min_track_seconds: float = VISION_CONFIG.get("min_track_seconds", 2.0),
    ):
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.min_iou_threshold = min_iou_threshold
        self.reenter_tolerance = reenter_tolerance
        self.reentry_window_seconds = reentry_window_seconds
        self.min_track_seconds = min_track_seconds
        self.next_id: int = 1
        self.tracks: Dict[str, TrackedParticipant] = {}
        # Dormant tracks: recently deregistered tracks that may re-enter
        self.dormant_tracks: Dict[str, TrackedParticipant] = {}
        # Track history of all registered tracks
        self.all_tracks_history: Dict[str, TrackedParticipant] = {}

    def reset(self) -> None:
        """Reset all tracking state."""
        self.next_id = 1
        self.tracks.clear()
        self.dormant_tracks.clear()
        self.all_tracks_history.clear()

    @staticmethod
    def compute_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Computes Intersection over Union (IoU) between two bounding boxes (x, y, w, h)."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        inter_w = max(0, xi2 - xi1)
        inter_h = max(0, yi2 - yi1)
        inter_area = inter_w * inter_h

        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        if union_area <= 0:
            return 0.0
        return float(inter_area) / float(union_area)

    def _register(self, detection: DetectionResult, timestamp: float) -> TrackedParticipant:
        """
        Registers a new participant, checking first if it matches a recently dormant participant
        that left and re-entered within the re-entry window.
        """
        best_dormant_id = None
        best_match_cost = 999.0

        for did, dtrack in list(self.dormant_tracks.items()):
            time_since_seen = timestamp - dtrack.last_seen_timestamp
            if time_since_seen > self.reentry_window_seconds:
                continue

            dcx, dcy = dtrack.centroid
            cx, cy = detection.centroid
            dist = float(np.sqrt((cx - dcx) ** 2 + (cy - dcy) ** 2))
            iou = self.compute_iou(dtrack.bbox, detection.bbox)

            # Eligible for re-entry if within expanded distance or significant IoU
            if dist <= self.max_distance * 1.6 or iou > 0.15:
                norm_dist = min(1.0, dist / (self.max_distance * 1.6))
                cost = (1.0 - iou) * 0.5 + norm_dist * 0.5
                if cost < best_match_cost:
                    best_match_cost = cost
                    best_dormant_id = did

        if best_dormant_id is not None:
            # Re-activate dormant track with its original tracking_id
            re_track = self.dormant_tracks.pop(best_dormant_id)
            prev_cx, prev_cy = re_track.centroid
            curr_cx, curr_cy = detection.centroid
            displacement = float(np.sqrt((curr_cx - prev_cx) ** 2 + (curr_cy - prev_cy) ** 2))

            re_track.bbox = detection.bbox
            re_track.centroid = detection.centroid
            re_track.last_seen_timestamp = timestamp
            re_track.disappeared_frames = 0
            re_track.recent_movement = displacement
            re_track.accumulated_movement += displacement
            re_track.trajectory.append((timestamp, detection.centroid))
            self.tracks[best_dormant_id] = re_track
            self.all_tracks_history[best_dormant_id] = re_track
            return re_track

        # Otherwise register a new participant ID
        tracking_id = f"Participant_{self.next_id:02d}"
        self.next_id += 1

        track = TrackedParticipant(
            tracking_id=tracking_id,
            bbox=detection.bbox,
            centroid=detection.centroid,
            first_seen_timestamp=timestamp,
            last_seen_timestamp=timestamp,
            disappeared_frames=0,
            recent_movement=0.0,
            accumulated_movement=0.0,
            trajectory=[(timestamp, detection.centroid)],
        )
        self.tracks[tracking_id] = track
        self.all_tracks_history[tracking_id] = track
        return track

    def _deregister(self, tracking_id: str) -> None:
        """Moves a lost track to dormant history for re-entry tolerance."""
        if tracking_id in self.tracks:
            lost = self.tracks.pop(tracking_id)
            if len(self.dormant_tracks) > 20:
                # Remove oldest dormant track
                oldest = next(iter(self.dormant_tracks))
                del self.dormant_tracks[oldest]
            self.dormant_tracks[tracking_id] = lost

    def update(
        self,
        detections: List[DetectionResult],
        timestamp: float,
    ) -> List[TrackedParticipant]:
        """
        Updates participant tracks given new detections using hybrid IoU + Centroid matching.
        """
        # Case 1: No detections in this frame
        if len(detections) == 0:
            tracks_to_remove = []
            for tracking_id, track in self.tracks.items():
                track.disappeared_frames += 1
                track.recent_movement = 0.0
                if track.disappeared_frames > self.max_disappeared:
                    tracks_to_remove.append(tracking_id)

            for tid in tracks_to_remove:
                self._deregister(tid)

            return [t for t in self.tracks.values() if t.disappeared_frames <= self.max_disappeared]

        # Case 2: No existing tracks, register all detections
        if len(self.tracks) == 0:
            for det in detections:
                self._register(det, timestamp)
            return list(self.tracks.values())

        # Case 3: Hybrid matching (IoU + Centroid Distance)
        track_ids = list(self.tracks.keys())
        num_tracks = len(track_ids)
        num_dets = len(detections)

        # Build composite cost matrix
        cost_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)

        for i, tid in enumerate(track_ids):
            track = self.tracks[tid]
            tcx, tcy = track.centroid
            for j, det in enumerate(detections):
                dcx, dcy = det.centroid
                euc_dist = float(np.sqrt((tcx - dcx) ** 2 + (tcy - dcy) ** 2))
                iou = self.compute_iou(track.bbox, det.bbox)

                # Normalized Euclidean cost [0, 1]
                norm_dist = min(1.0, euc_dist / max(1.0, self.max_distance))
                # Combined cost: heavy weight on IoU overlap when present
                cost = (1.0 - iou) * 0.60 + norm_dist * 0.40
                if euc_dist > self.max_distance and iou < self.min_iou_threshold:
                    cost = 999.0  # Infeasible match

                cost_matrix[i, j] = cost

        # Match tracks to detections globally by minimum cost
        match_candidates = []
        for r in range(num_tracks):
            for c in range(num_dets):
                c_val = cost_matrix[r, c]
                if c_val < 999.0:
                    match_candidates.append((c_val, r, c))
        match_candidates.sort(key=lambda x: x[0])

        used_rows = set()
        used_cols = set()

        for cost, row, col in match_candidates:
            if row in used_rows or col in used_cols:
                continue

            tracking_id = track_ids[row]
            matched_track = self.tracks[tracking_id]
            det = detections[col]

            prev_cx, prev_cy = matched_track.centroid
            curr_cx, curr_cy = det.centroid
            displacement = float(np.sqrt((curr_cx - prev_cx) ** 2 + (curr_cy - prev_cy) ** 2))

            matched_track.bbox = det.bbox
            matched_track.centroid = det.centroid
            matched_track.last_seen_timestamp = timestamp
            matched_track.disappeared_frames = 0
            matched_track.recent_movement = displacement
            matched_track.accumulated_movement += displacement
            matched_track.trajectory.append((timestamp, det.centroid))

            if len(matched_track.trajectory) > 500:
                matched_track.trajectory = matched_track.trajectory[-500:]

            used_rows.add(row)
            used_cols.add(col)

        # Handle unmatched existing tracks
        unused_rows = set(range(num_tracks)) - used_rows
        tracks_to_remove = []
        for row in unused_rows:
            tid = track_ids[row]
            track = self.tracks[tid]
            track.disappeared_frames += 1
            track.recent_movement = 0.0
            if track.disappeared_frames > self.max_disappeared:
                tracks_to_remove.append(tid)

        for tid in tracks_to_remove:
            self._deregister(tid)

        # Handle unmatched detections -> register new/re-entered tracks
        unused_cols = set(range(num_dets)) - used_cols
        for col in unused_cols:
            self._register(detections[col], timestamp)

        return [t for t in self.tracks.values() if t.disappeared_frames <= self.max_disappeared]

    @staticmethod
    def draw_tracks(
        frame: np.ndarray,
        tracks: List[TrackedParticipant],
        attendance_status: Optional[Dict[str, str]] = None,
    ) -> np.ndarray:
        """
        Renders active tracks, bounding boxes, participant labels, and attendance status onto the frame.
        """
        if frame is None or frame.size == 0:
            return frame

        annotated = frame.copy()
        for track in tracks:
            x, y, w, h = track.bbox
            is_present = False
            if attendance_status and track.tracking_id in attendance_status:
                is_present = (attendance_status[track.tracking_id] == "PRESENT")

            color = (0, 220, 100) if is_present else (255, 165, 0)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            status_text = "PRESENT" if is_present else "ACCUMULATING"
            label = f"{track.tracking_id} [{status_text}]"
            font_scale = 0.5
            thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            tag_y1 = max(y - text_h - 8, 0)
            tag_y2 = y
            cv2.rectangle(annotated, (x, tag_y1), (x + text_w + 6, tag_y2), color, -1)
            cv2.putText(
                annotated,
                label,
                (x + 3, max(y - 4, text_h + 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA,
            )

            # Draw movement trail
            if len(track.trajectory) > 1:
                trail_pts = [pt for _, pt in track.trajectory[-8:]]
                for i in range(1, len(trail_pts)):
                    cv2.line(annotated, trail_pts[i - 1], trail_pts[i], color, 1, cv2.LINE_AA)

        return annotated
