# File: tracker/tracker.py
"""
Multi-person Tracker using IoU-based association.
"""
import numpy as np


class Track:
    """
    Represents a single tracked object with an ID and history of bounding boxes.
    """
    def __init__(self, track_id: int, initial_box: list, max_history: int = 30):
        self.id = track_id
        self.boxes = [initial_box]  # list of [x1, y1, x2, y2]
        self.lost_frames = 0        # consecutive frames without association
        self.max_history = max_history

    def predict_box(self) -> list:
        """
        Return the most recent bounding box as the predicted location.
        """
        return self.boxes[-1]

    def update(self, new_box: list) -> None:
        """
        Update the track with a new bounding box and reset lost_frames.
        """
        self.boxes.append(new_box)
        if len(self.boxes) > self.max_history:
            self.boxes.pop(0)
        self.lost_frames = 0

    def mark_missed(self) -> None:
        """
        Increment the lost_frames counter when no association occurs.
        """
        self.lost_frames += 1


class Tracker:
    """
    IoU-based multi-object tracker that assigns consistent IDs to detections.
    """
    def __init__(self, iou_threshold: float = 0.3, max_lost: int = 5):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.next_id = 0
        self.tracks: list[Track] = []

    @staticmethod
    def _iou(boxA: list, boxB: list) -> float:
        """
        Compute Intersection over Union (IoU) between two bounding boxes.
        Boxes are [x1, y1, x2, y2].
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH

        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        unionArea = areaA + areaB - interArea

        return interArea / unionArea if unionArea > 0 else 0.0

    def update(self, detections: list[list]) -> list[tuple]:
        """
        Update tracks with the latest detections.

        Args:
            detections: list of [x1, y1, x2, y2, conf, cls]
        Returns:
            List of (track_id, box) for all active tracks.
        """
        # 1. Predict current positions of existing tracks
        predicted_boxes = [track.predict_box() for track in self.tracks]

        # 2. Compute IoU matrix between predicted and new detections
        iou_matrix = np.zeros((len(predicted_boxes), len(detections)), dtype=np.float32)
        for t_idx, p_box in enumerate(predicted_boxes):
            for d_idx, det in enumerate(detections):
                iou_matrix[t_idx, d_idx] = self._iou(p_box, det[:4])

        # 3. Greedy matching based on highest IoU
        matched_tracks, matched_detections = set(), set()
        while True:
            if iou_matrix.size == 0:
                break
            t_idx, d_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            if iou_matrix[t_idx, d_idx] < self.iou_threshold:
                break
            # Associate and update
            self.tracks[t_idx].update(detections[d_idx][:4])
            matched_tracks.add(t_idx)
            matched_detections.add(d_idx)
            iou_matrix[t_idx, :] = -1
            iou_matrix[:, d_idx] = -1

        # 4. Mark unmatched tracks as missed
        for idx, track in enumerate(self.tracks):
            if idx not in matched_tracks:
                track.mark_missed()

        # 5. Create new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_detections:
                new_track = Track(self.next_id, det[:4])
                self.next_id += 1
                self.tracks.append(new_track)

        # 6. Remove tracks that have been lost for too long
        self.tracks = [t for t in self.tracks if t.lost_frames <= self.max_lost]

        # 7. Return active track states
        return [(t.id, t.predict_box()) for t in self.tracks]